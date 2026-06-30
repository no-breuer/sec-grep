//! Optional abstract enrichment.
//!
//! Two tiers, each a fallback for the previous:
//!   1. API by DOI: OpenAlex -> Semantic Scholar -> Crossref
//!   2. Static HTML scrape (publisher page)
//!
//! The pure parsing/extraction helpers are unit-tested; the networked
//! orchestration is exercised end-to-end via the CLI.

use std::{
    collections::{BTreeSet, HashMap},
    net::IpAddr,
    sync::OnceLock,
    time::Duration,
};

use futures::stream::{self, StreamExt};
use reqwest::{header, Url};
use scraper::{ElementRef, Html, Selector};
use serde_json::Value;

use crate::config::{AbstractSource, Secrets};
use crate::{Paper, Result};

const MAX_JSON_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_HTML_BODY_BYTES: usize = 4 * 1024 * 1024;
const MAX_STATIC_REDIRECTS: usize = 5;
const MAX_RATE_LIMIT_SLEEP: Duration = Duration::from_secs(65);
const OPENREVIEW_PAGE_SIZE: usize = 500;
const OPENREVIEW_LOGIN_EXPIRES_IN: u64 = 7 * 24 * 60 * 60;

/// Reconstruct plain text from an OpenAlex `abstract_inverted_index`,
/// or read a plain `abstract` string if present.
pub fn abstract_from_openalex(work: &Value) -> Option<String> {
    if let Some(s) = work.get("abstract").and_then(|v| v.as_str()) {
        if !s.trim().is_empty() {
            return Some(s.trim().to_string());
        }
    }
    let idx = work.get("abstract_inverted_index")?.as_object()?;
    let mut positioned: Vec<(u64, &str)> = Vec::new();
    for (word, positions) in idx {
        for p in positions.as_array()? {
            if let Some(pos) = p.as_u64() {
                positioned.push((pos, word.as_str()));
            }
        }
    }
    if positioned.is_empty() {
        return None;
    }
    positioned.sort_by_key(|(p, _)| *p);
    Some(
        positioned
            .into_iter()
            .map(|(_, w)| w)
            .collect::<Vec<_>>()
            .join(" "),
    )
}

pub fn abstract_from_semantic_scholar(paper: &Value) -> Option<String> {
    paper
        .get("abstract")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn abstracts_from_openreview_notes(notes: &Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(notes) = notes.get("notes").and_then(|v| v.as_array()) else {
        return out;
    };
    for note in notes {
        let Some(content) = note.get("content") else {
            continue;
        };
        let Some(abs) = content.get("abstract").and_then(openreview_content_value) else {
            continue;
        };
        for key in ["id", "forum"] {
            if let Some(id) = note.get(key).and_then(|v| v.as_str()) {
                out.insert(id.to_string(), abs.clone());
            }
        }
    }
    out
}

fn openreview_content_value(value: &Value) -> Option<String> {
    value
        .as_str()
        .or_else(|| value.get("value").and_then(|v| v.as_str()))
        .and_then(non_empty_text)
}

/// Crossref returns JATS-flavored XML in `message.abstract`; strip tags.
pub fn abstract_from_crossref(message: &Value) -> Option<String> {
    let raw = message.get("abstract").and_then(|v| v.as_str())?;
    let doc = Html::parse_fragment(raw);
    element_text(doc.root_element())
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract an abstract from a publisher HTML page, trying source-specific
/// selectors first, then generic `og:description` / `description` meta tags.
pub fn extract_abstract_html(html: &str, source: Option<AbstractSource>) -> Option<String> {
    let doc = Html::parse_document(html);

    let source_hit = match source {
        Some(AbstractSource::Acm) => {
            first_selector_text(&doc, &["div.abstractInFull", "div.abstractSection"])
        }
        Some(AbstractSource::Ieee) => first_selector_text(&doc, &["div.abstract-text"]),
        Some(AbstractSource::Ndss) => extract_ndss_abstract(&doc),
        Some(AbstractSource::Neurips) => extract_neurips_abstract(&doc),
        Some(AbstractSource::Openreview) => first_selector_text(&doc, &[".abstract-text-inner"]),
        Some(AbstractSource::Pmlr) => first_selector_text(&doc, &["div#abstract"]),
        Some(AbstractSource::Springer) => first_selector_text(
            &doc,
            &[
                "section[data-title='Abstract'] div.c-article-section__content",
                "#Abs1-content",
            ],
        ),
        Some(AbstractSource::Usenix) => first_selector_text(
            &doc,
            &[
                "div.field-name-field-paper-description",
                "div.field-type-text-with-summary",
            ],
        ),
        _ => None,
    };

    source_hit.or_else(|| first_meta_content(&doc))
}

fn extract_ndss_abstract(doc: &Html) -> Option<String> {
    let mut text = first_selector_text(doc, &["div.paper-data"])?;
    if let Some(authors) = first_selector_text(doc, &["div.paper-data strong"]) {
        if let Some(rest) = text.strip_prefix(&authors) {
            text = rest
                .trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ':' | '-'))
                .to_string();
        }
    }
    non_empty_text(text)
}

fn extract_neurips_abstract(doc: &Html) -> Option<String> {
    let selector = Selector::parse("section.paper-section").ok()?;
    for section in doc.select(&selector) {
        let text = element_text(section)?;
        if let Some(abstract_text) = text.strip_prefix("Abstract") {
            return non_empty_text(abstract_text);
        }
    }
    None
}

fn first_selector_text(doc: &Html, selectors: &[&str]) -> Option<String> {
    for selector in selectors {
        let Ok(selector) = Selector::parse(selector) else {
            continue;
        };
        for element in doc.select(&selector) {
            if let Some(text) = element_text(element) {
                return Some(text);
            }
        }
    }
    None
}

fn first_meta_content(doc: &Html) -> Option<String> {
    let selectors = [
        "meta[name='citation_abstract']",
        "meta[property='og:description']",
        "meta[name='description']",
    ];

    for selector in selectors {
        let Ok(selector) = Selector::parse(selector) else {
            continue;
        };
        for element in doc.select(&selector) {
            if let Some(content) = element.value().attr("content") {
                if let Some(text) = non_empty_text(content) {
                    if is_boilerplate_abstract(&text) {
                        continue;
                    }
                    return Some(text);
                }
            }
        }
    }
    None
}

fn is_boilerplate_abstract(text: &str) -> bool {
    text.starts_with("Promoting openness in scientific communication and the peer-review process")
}

fn element_text(element: ElementRef) -> Option<String> {
    non_empty_text(element.text().collect::<Vec<_>>().join(" "))
}

fn non_empty_text(text: impl AsRef<str>) -> Option<String> {
    let decoded = decode_html_entities(text.as_ref());
    let decoded = decode_html_entities(&decoded);
    let text = collapse_ws(&decoded);
    (!text.is_empty()).then_some(text)
}

fn decode_html_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;

    while let Some(idx) = rest.find('&') {
        out.push_str(&rest[..idx]);
        let entity_start = idx + 1;
        let Some(entity_len) = rest[entity_start..].find(';') else {
            out.push_str(&rest[idx..]);
            return out;
        };
        let entity_end = entity_start + entity_len;
        let entity = &rest[entity_start..entity_end];
        match decode_entity(entity) {
            Some(decoded) => out.push_str(&decoded),
            None => out.push_str(&rest[idx..=entity_end]),
        }
        rest = &rest[entity_end + 1..];
    }

    out.push_str(rest);
    out
}

fn decode_entity(entity: &str) -> Option<String> {
    if let Some(hex) = entity
        .strip_prefix("#x")
        .or_else(|| entity.strip_prefix("#X"))
    {
        let code = u32::from_str_radix(hex, 16).ok()?;
        return char::from_u32(code).map(|c| c.to_string());
    }
    if let Some(dec) = entity.strip_prefix('#') {
        let code = dec.parse::<u32>().ok()?;
        return char::from_u32(code).map(|c| c.to_string());
    }
    match entity {
        "amp" => Some("&".to_string()),
        "apos" => Some("'".to_string()),
        "gt" => Some(">".to_string()),
        "lt" => Some("<".to_string()),
        "nbsp" => Some(" ".to_string()),
        "quot" => Some("\"".to_string()),
        _ => None,
    }
}

/// Networked abstract enrichment using the configured API keys.
pub struct Enricher {
    client: reqwest::Client,
    secrets: Secrets,
    openreview_cache: HashMap<(String, i32), HashMap<String, String>>,
    openreview_login_token: OnceLock<String>,
}

pub enum EnrichResult {
    Found(String),
    Missing(String),
}

impl Enricher {
    pub fn new(secrets: Secrets) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(concat!("sec-grep/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("reqwest client");
        Self {
            client,
            secrets,
            openreview_cache: HashMap::new(),
            openreview_login_token: OnceLock::new(),
        }
    }

    /// Try, in order: API-by-DOI, then static scrape. Returns the first hit.
    pub async fn enrich(
        &self,
        paper: &Paper,
        source: Option<AbstractSource>,
    ) -> Result<EnrichResult> {
        if source == Some(AbstractSource::Openreview) {
            if let Some(url) = &paper.url {
                if let Some(abs) = self.openreview_api_abstract(url).await {
                    return Ok(EnrichResult::Found(abs));
                }
                return self.static_scrape(url, source).await;
            }
            return Ok(EnrichResult::Missing("OpenReview paper has no URL".into()));
        }
        if let Some(doi) = &paper.doi {
            if let Some(abs) = self.api_by_doi(doi).await? {
                return Ok(EnrichResult::Found(abs));
            }
        }
        if let Some(url) = &paper.url {
            return self.static_scrape(url, source).await;
        }
        Ok(EnrichResult::Missing("paper has no DOI or URL".into()))
    }

    pub async fn enrich_many(
        &mut self,
        inputs: Vec<(Paper, Option<AbstractSource>)>,
        jobs: usize,
    ) -> (Vec<(Paper, Result<EnrichResult>)>, Vec<String>) {
        let mut source_warnings = Vec::new();
        let mut needed = BTreeSet::new();
        for (paper, source) in &inputs {
            if *source != Some(AbstractSource::Openreview) {
                continue;
            }
            let key = (paper.venue.clone(), paper.year);
            if !self.openreview_cache.contains_key(&key) {
                needed.insert(key);
            }
        }

        for key in needed {
            match self.openreview_accepted_abstracts(&key.0, key.1).await {
                Ok(abstracts) => {
                    self.openreview_cache.insert(key, abstracts);
                }
                Err(reason) => {
                    source_warnings.push(format!("{} {}: {reason}", key.0, key.1));
                    self.openreview_cache.insert(key, HashMap::new());
                }
            }
        }

        let cache = &self.openreview_cache;
        let enricher = &*self;
        let results = stream::iter(inputs.into_iter().map(|(paper, source)| {
            let enricher = enricher;
            async move {
                let result = match cached_openreview_abstract(cache, &paper, source) {
                    Some(abs) => Ok(EnrichResult::Found(abs)),
                    None => enricher.enrich(&paper, source).await,
                };
                (paper, result)
            }
        }))
        .buffer_unordered(jobs.max(1))
        .collect()
        .await;

        (results, source_warnings)
    }

    async fn api_by_doi(&self, doi: &str) -> Result<Option<String>> {
        let openalex_req = {
            let req = self
                .client
                .get(format!("https://api.openalex.org/works/doi:{doi}"));
            match &self.secrets.openalex_api_key {
                Some(key) => req.query(&[("api_key", key.as_str())]),
                None => req,
            }
        };
        if let EnrichResult::Found(abs) = self
            .fetch_abstract(openalex_req, abstract_from_openalex)
            .await
        {
            return Ok(Some(abs));
        }

        let s2 =
            format!("https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract");
        let s2_req = {
            let req = self.client.get(&s2);
            match &self.secrets.semantic_scholar_key {
                Some(key) => req.header("x-api-key", key),
                None => req,
            }
        };
        if let EnrichResult::Found(abs) = self
            .fetch_abstract(s2_req, abstract_from_semantic_scholar)
            .await
        {
            return Ok(Some(abs));
        }

        let cr = self
            .client
            .get(format!("https://api.crossref.org/works/{doi}"));
        Ok(
            match self
                .fetch_abstract(cr, |json| {
                    json.get("message").and_then(abstract_from_crossref)
                })
                .await
            {
                EnrichResult::Found(abs) => Some(abs),
                EnrichResult::Missing(_) => None,
            },
        )
    }

    async fn openreview_accepted_abstracts(
        &self,
        venue: &str,
        year: i32,
    ) -> std::result::Result<HashMap<String, String>, String> {
        let venue_id = format!("{venue}.cc/{year}/Conference");
        for base in [
            "https://api2.openreview.net/notes",
            "https://api.openreview.net/notes",
        ] {
            let abstracts = self
                .openreview_accepted_abstracts_from(base, &venue_id)
                .await?;
            if !abstracts.is_empty() {
                return Ok(abstracts);
            }
        }
        Ok(HashMap::new())
    }

    async fn openreview_accepted_abstracts_from(
        &self,
        base: &str,
        venue_id: &str,
    ) -> std::result::Result<HashMap<String, String>, String> {
        let mut out = HashMap::new();
        let mut offset = 0usize;
        loop {
            let req = self.openreview_get(base).await.query(&[
                ("content.venueid", venue_id),
                ("limit", &OPENREVIEW_PAGE_SIZE.to_string()),
                ("offset", &offset.to_string()),
            ]);
            let json = self.fetch_json(req).await?;
            let page_len = json
                .get("notes")
                .and_then(|v| v.as_array())
                .map_or(0, Vec::len);
            out.extend(abstracts_from_openreview_notes(&json));
            if page_len < OPENREVIEW_PAGE_SIZE {
                break;
            }
            offset += OPENREVIEW_PAGE_SIZE;
        }
        Ok(out)
    }

    async fn openreview_api_abstract(&self, url: &str) -> Option<String> {
        let forum_id = openreview_forum_id(url)?;
        for base in [
            "https://api2.openreview.net/notes",
            "https://api.openreview.net/notes",
        ] {
            let req = self
                .openreview_get(base)
                .await
                .query(&[("id", forum_id.as_str())]);
            let Ok(json) = self.fetch_json(req).await else {
                continue;
            };
            if let Some(abs) = abstracts_from_openreview_notes(&json)
                .get(&forum_id)
                .cloned()
            {
                return Some(abs);
            }
        }
        None
    }

    async fn openreview_get(&self, url: &str) -> reqwest::RequestBuilder {
        let req = self.client.get(url);
        if url.contains("api2.openreview.net") {
            if let Some(token) = self.openreview_login_token().await {
                return req.header(header::AUTHORIZATION, format!("Bearer {token}"));
            }
        }
        req
    }

    async fn openreview_login_token(&self) -> Option<String> {
        if let Some(token) = self.openreview_login_token.get() {
            return Some(token.clone());
        }
        let token = self.openreview_login().await;
        if let Some(token) = &token {
            let _ = self.openreview_login_token.set(token.clone());
        }
        token
    }

    async fn openreview_login(&self) -> Option<String> {
        let username = self.secrets.openreview_username.as_deref()?;
        let password = self.secrets.openreview_password.as_deref()?;
        let req = self
            .client
            .post("https://api2.openreview.net/login")
            .json(&serde_json::json!({
                "id": username,
                "password": password,
                "expiresIn": OPENREVIEW_LOGIN_EXPIRES_IN,
            }));
        let json = self.fetch_json(req).await.ok()?;
        json.get("token")
            .and_then(|token| token.as_str())
            .map(str::to_string)
    }

    /// Send a request and run `extract` over the JSON body.
    async fn fetch_abstract(
        &self,
        req: reqwest::RequestBuilder,
        extract: impl Fn(&Value) -> Option<String>,
    ) -> EnrichResult {
        let json = match self.fetch_json(req).await {
            Ok(json) => json,
            Err(reason) => return EnrichResult::Missing(reason),
        };
        match extract(&json) {
            Some(abs) => EnrichResult::Found(abs),
            None => EnrichResult::Missing("response had no extractable abstract".into()),
        }
    }

    async fn fetch_json(&self, req: reqwest::RequestBuilder) -> std::result::Result<Value, String> {
        let retry_req = req.try_clone();
        let mut resp = match req.send().await {
            Ok(resp) => resp,
            Err(_) => return Err("request failed".into()),
        };
        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            if let Some(req) = retry_req {
                tokio::time::sleep(rate_limit_delay(&resp)).await;
                resp = match req.send().await {
                    Ok(resp) => resp,
                    Err(_) => return Err("request failed after rate limit".into()),
                };
            }
        }
        if !resp.status().is_success() {
            return Err(http_status_reason("HTTP", resp.status()));
        }
        let Some(bytes) = read_body_limited(resp, MAX_JSON_BODY_BYTES).await else {
            return Err("response too large or unreadable".into());
        };
        serde_json::from_slice::<Value>(&bytes).map_err(|_| "invalid JSON response".into())
    }

    async fn static_scrape(
        &self,
        url: &str,
        source: Option<AbstractSource>,
    ) -> Result<EnrichResult> {
        let Some(mut url) = allowed_static_url(url).await else {
            return Ok(EnrichResult::Missing(
                "URL rejected by static scraper".into(),
            ));
        };

        for _ in 0..=MAX_STATIC_REDIRECTS {
            let resp = match self.client.get(url.clone()).send().await {
                Ok(resp) => resp,
                Err(_) => return Ok(EnrichResult::Missing("static scrape request failed".into())),
            };
            if resp.status().is_redirection() {
                let Some(next_url) = redirect_url(&url, &resp) else {
                    return Ok(EnrichResult::Missing(
                        "redirect without valid location".into(),
                    ));
                };
                let Some(next_url) = allowed_static_url(next_url.as_str()).await else {
                    return Ok(EnrichResult::Missing("redirect URL rejected".into()));
                };
                url = next_url;
                continue;
            }
            if !resp.status().is_success() {
                return Ok(EnrichResult::Missing(http_status_reason(
                    "static scrape HTTP",
                    resp.status(),
                )));
            }
            let Some(html) = read_text_limited(resp, MAX_HTML_BODY_BYTES).await else {
                return Ok(EnrichResult::Missing(
                    "static page too large or unreadable".into(),
                ));
            };
            return Ok(match extract_abstract_html(&html, source) {
                Some(abs) => EnrichResult::Found(abs),
                None => EnrichResult::Missing("static page had no extractable abstract".into()),
            });
        }

        Ok(EnrichResult::Missing("too many redirects".into()))
    }
}

fn cached_openreview_abstract(
    cache: &HashMap<(String, i32), HashMap<String, String>>,
    paper: &Paper,
    source: Option<AbstractSource>,
) -> Option<String> {
    if source != Some(AbstractSource::Openreview) {
        return None;
    }
    let forum_id = openreview_forum_id(paper.url.as_deref()?)?;
    cache
        .get(&(paper.venue.clone(), paper.year))?
        .get(&forum_id)
        .cloned()
}

fn http_status_reason(prefix: &str, status: reqwest::StatusCode) -> String {
    let label = match status.as_u16() {
        401 | 403 => "blocked",
        429 => "rate limited",
        500..=599 => "server error",
        _ => "request failed",
    };
    format!("{prefix} {} ({label})", status.as_u16())
}

fn rate_limit_delay(resp: &reqwest::Response) -> Duration {
    let headers = resp.headers();
    let secs = header_seconds(headers, header::RETRY_AFTER)
        .or_else(|| header_seconds(headers, "ratelimit-reset"))
        .unwrap_or(60);
    Duration::from_secs(secs.clamp(1, MAX_RATE_LIMIT_SLEEP.as_secs()))
}

fn header_seconds(headers: &header::HeaderMap, name: impl header::AsHeaderName) -> Option<u64> {
    headers.get(name)?.to_str().ok()?.parse().ok()
}

async fn allowed_static_url(raw: &str) -> Option<Url> {
    let url = parse_static_url(raw)?;
    let host = url.host_str()?;
    let port = url.port_or_known_default()?;
    if let Some(ip) = parse_host_ip(host) {
        return is_public_ip(ip).then_some(url);
    }
    let addrs = tokio::net::lookup_host((host, port)).await.ok()?;
    let mut has_addr = false;
    for addr in addrs {
        has_addr = true;
        if !is_public_ip(addr.ip()) {
            return None;
        }
    }
    has_addr.then_some(url)
}

fn parse_static_url(raw: &str) -> Option<Url> {
    let url = Url::parse(raw).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    if !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    let host = url.host_str()?;
    if is_localhost(host) {
        return None;
    }
    if let Some(ip) = parse_host_ip(host) {
        return is_public_ip(ip).then_some(url);
    }
    Some(url)
}

fn openreview_forum_id(raw: &str) -> Option<String> {
    let url = parse_static_url(raw)?;
    let host = url.host_str()?.trim_end_matches('.').to_ascii_lowercase();
    if host != "openreview.net" && host != "www.openreview.net" {
        return None;
    }
    if url.path() != "/forum" {
        return None;
    }
    url.query_pairs()
        .find_map(|(key, value)| (key == "id" && !value.is_empty()).then(|| value.into_owned()))
}

fn parse_host_ip(host: &str) -> Option<IpAddr> {
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    host.parse().ok()
}

fn is_localhost(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "localhost" || host.ends_with(".localhost")
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_broadcast()
                || ip.is_multicast()
                || ip.is_documentation())
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            let first = segments[0];
            let is_unique_local = (first & 0xfe00) == 0xfc00;
            let is_link_local = (first & 0xffc0) == 0xfe80;
            let is_documentation = segments[0] == 0x2001 && segments[1] == 0x0db8;
            !(ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || is_unique_local
                || is_link_local
                || is_documentation)
        }
    }
}

fn redirect_url(base: &Url, resp: &reqwest::Response) -> Option<Url> {
    let location = resp.headers().get(header::LOCATION)?.to_str().ok()?;
    base.join(location).ok()
}

async fn read_text_limited(resp: reqwest::Response, limit: usize) -> Option<String> {
    let bytes = read_body_limited(resp, limit).await?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

async fn read_body_limited(mut resp: reqwest::Response, limit: usize) -> Option<Vec<u8>> {
    if resp.content_length().is_some_and(|len| len > limit as u64) {
        return None;
    }

    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.ok()? {
        if body.len().checked_add(chunk.len())? > limit {
            return None;
        }
        body.extend_from_slice(&chunk);
    }
    Some(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn http_status_reason_labels_blocked_and_rate_limited() {
        assert_eq!(
            http_status_reason("HTTP", reqwest::StatusCode::TOO_MANY_REQUESTS),
            "HTTP 429 (rate limited)"
        );
        assert_eq!(
            http_status_reason("static scrape HTTP", reqwest::StatusCode::FORBIDDEN),
            "static scrape HTTP 403 (blocked)"
        );
    }

    #[test]
    fn header_seconds_parses_retry_after() {
        let mut headers = header::HeaderMap::new();
        headers.insert(header::RETRY_AFTER, header::HeaderValue::from_static("120"));
        assert_eq!(header_seconds(&headers, header::RETRY_AFTER), Some(120));
    }

    #[test]
    fn openalex_inverted_index() {
        let work = json!({
            "abstract_inverted_index": {
                "We": [0], "fuzz": [1], "the": [2], "kernel": [3]
            }
        });
        assert_eq!(
            abstract_from_openalex(&work).as_deref(),
            Some("We fuzz the kernel")
        );
    }

    #[test]
    fn openalex_plain_abstract() {
        let work = json!({ "abstract": "  direct text  " });
        assert_eq!(
            abstract_from_openalex(&work).as_deref(),
            Some("direct text")
        );
    }

    #[test]
    fn semantic_scholar_abstract() {
        assert_eq!(
            abstract_from_semantic_scholar(&json!({"abstract": "hello"})).as_deref(),
            Some("hello")
        );
        assert!(abstract_from_semantic_scholar(&json!({"abstract": null})).is_none());
    }

    #[test]
    fn openreview_batch_abstracts_indexes_id_and_forum() {
        let notes = json!({
            "notes": [
                {"id": "v2-id", "forum": "v2-forum", "content": {"abstract": {"value": "V2 abstract."}}},
                {"id": "v1-id", "forum": "v1-forum", "content": {"abstract": "V1 abstract."}}
            ]
        });
        let abstracts = abstracts_from_openreview_notes(&notes);
        assert_eq!(
            abstracts.get("v2-id").map(String::as_str),
            Some("V2 abstract.")
        );
        assert_eq!(
            abstracts.get("v2-forum").map(String::as_str),
            Some("V2 abstract.")
        );
        assert_eq!(
            abstracts.get("v1-id").map(String::as_str),
            Some("V1 abstract.")
        );
        assert_eq!(
            abstracts.get("v1-forum").map(String::as_str),
            Some("V1 abstract.")
        );
    }

    #[test]
    fn openreview_forum_id_from_url() {
        assert_eq!(
            openreview_forum_id("https://openreview.net/forum?id=8EtSBX41mt").as_deref(),
            Some("8EtSBX41mt")
        );
        assert!(openreview_forum_id("https://example.com/forum?id=8EtSBX41mt").is_none());
    }

    #[test]
    fn crossref_strips_jats() {
        let msg = json!({"abstract": "<jats:p>A <jats:bold>bold</jats:bold> claim.</jats:p>"});
        assert_eq!(
            abstract_from_crossref(&msg).as_deref(),
            Some("A bold claim.")
        );
    }

    #[test]
    fn html_acm_source_specific_selector() {
        let html = r#"<html><body>
            <div class="abstractInFull"><p>This is the ACM abstract.</p></div>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Acm)).as_deref(),
            Some("This is the ACM abstract.")
        );
    }

    #[test]
    fn html_ndss_strips_authors_from_paper_data() {
        let html = r#"<html><body>
            <div class="paper-data">
                <p><strong><p>Alice A (Example U), Bob B (Example Labs)</p></strong></p>
                <p>
                    <p>First abstract paragraph.</p>
                    <p>Second abstract paragraph.</p>
                </p>
            </div>
        </body></html>"#;
        let abstract_text = extract_abstract_html(html, Some(AbstractSource::Ndss)).unwrap();
        assert_eq!(
            abstract_text,
            "First abstract paragraph. Second abstract paragraph."
        );
        assert!(!abstract_text.contains("Alice A"));
        assert!(!abstract_text.contains("Bob B"));
    }

    #[test]
    fn html_usenix_source_specific_selector() {
        let html = r#"<html><body>
            <div class="field field-name-field-paper-people-text"><p>Alice A and Bob B</p></div>
            <div class="field field-name-field-paper-description"><p>This is the USENIX abstract.</p></div>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Usenix)).as_deref(),
            Some("This is the USENIX abstract.")
        );
    }

    #[test]
    fn html_neurips_source_specific_selector() {
        let html = r#"<html><body>
            <section class="paper-section">
                <h2 class="section-label">Abstract</h2>
                <p class="paper-abstract"><p>NeurIPS abstract text.</p></p>
            </section>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Neurips)).as_deref(),
            Some("NeurIPS abstract text.")
        );
    }

    #[test]
    fn html_pmlr_prefers_full_page_abstract_over_truncated_meta() {
        let html = r#"<html><head>
            <meta name="description" content="Truncated PMLR abstract...">
        </head><body>
            <h4>Abstract</h4>
            <div id="abstract" class="abstract">
                First full ICML abstract sentence. Second full sentence.
            </div>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Pmlr)).as_deref(),
            Some("First full ICML abstract sentence. Second full sentence.")
        );
    }

    #[test]
    fn html_openreview_source_reads_iclr_virtual_abstract() {
        let html = r#"<html><body>
            <div class="abstract-section">
                <div class="abstract-text-inner">
                    <p>ICLR virtual abstract text.</p>
                </div>
            </div>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Openreview)).as_deref(),
            Some("ICLR virtual abstract text.")
        );
    }

    #[test]
    fn html_rejects_openreview_boilerplate_description() {
        let html = r#"<html><head>
            <meta name="description" content="Promoting openness in scientific communication and the peer-review process">
        </head></html>"#;
        assert!(extract_abstract_html(html, Some(AbstractSource::Openreview)).is_none());
    }

    #[test]
    fn html_ieee_prefers_abstract_text_selector() {
        let html = r#"<html><head>
            <meta property="og:description" content="IEEE fallback abstract.">
        </head><body>
            <div class="abstract-text">This is the IEEE abstract.</div>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Ieee)).as_deref(),
            Some("This is the IEEE abstract.")
        );
    }

    #[test]
    fn html_springer_prefers_full_abstract_section_over_truncated_meta() {
        let html = r#"<html><head>
            <meta property="og:description" content="Truncated Springer abstract...">
        </head><body>
            <section data-title="Abstract">
                <div class="c-article-section__content">
                    <p>This is the full Springer abstract.</p>
                    <p>It has a second sentence.</p>
                </div>
            </section>
        </body></html>"#;
        assert_eq!(
            extract_abstract_html(html, Some(AbstractSource::Springer)).as_deref(),
            Some("This is the full Springer abstract. It has a second sentence.")
        );
    }

    #[test]
    fn html_generic_meta_fallback() {
        let html = r#"<html><head>
            <meta property="og:description" content="Fallback abstract here.">
        </head><body></body></html>"#;
        assert_eq!(
            extract_abstract_html(html, None).as_deref(),
            Some("Fallback abstract here.")
        );
    }

    #[test]
    fn html_meta_fallback_decodes_entities() {
        let html = r#"<html><head>
            <meta property="og:description" content="A&amp;#160;B &amp;amp; C &#8217;">
        </head><body></body></html>"#;
        assert_eq!(
            extract_abstract_html(html, None).as_deref(),
            Some("A B & C ’")
        );
    }

    #[test]
    fn html_no_abstract() {
        assert!(extract_abstract_html("<html></html>", None).is_none());
    }

    #[test]
    fn static_url_rejects_local_or_non_http_targets() {
        assert!(parse_static_url("https://example.com/paper").is_some());
        assert!(parse_static_url("file:///etc/passwd").is_none());
        assert!(parse_static_url("http://localhost/paper").is_none());
        assert!(parse_static_url("http://127.0.0.1/paper").is_none());
        assert!(parse_static_url("http://[::1]/paper").is_none());
        assert!(parse_static_url("https://user@example.com/paper").is_none());
    }
}
