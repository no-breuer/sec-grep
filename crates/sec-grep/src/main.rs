mod tui;

use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use sec_grep_core::abstracts::{EnrichResult, Enricher};
use sec_grep_core::config::{Config, Paths, Secrets};
use sec_grep_core::db::{Database, Search, Sort};
use sec_grep_core::dblp::Dblp;
use sec_grep_core::output::{self, Column, Format};
use sec_grep_core::query;
use sec_grep_core::Paper;

/// Upper bound for dblp year filters; papers never exceed this.
const MAX_YEAR: i32 = 2100;

#[derive(Parser)]
#[command(
    name = "sec-grep",
    about = "Search computer science research literature",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Query string (default action is search). Supports AND/OR/NOT, "phrases",
    /// title:/author:/abstract: text fields, metadata filters
    /// (venue:/year:/rank:/tag:/doi:), and prefix*.
    #[arg(value_name = "QUERY")]
    query: Vec<String>,

    /// Restrict to venues (id or alias), comma- or space-separated.
    #[arg(long, value_delimiter = ',')]
    venue: Vec<String>,

    /// Restrict by year: 2020, 2018-2024, 2020-, or -2019.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true, value_parser = parse_year_arg)]
    year: Vec<query::YearRange>,

    /// Restrict by rank label (e.g. A*, A, B).
    #[arg(long, value_delimiter = ',')]
    rank: Vec<String>,

    /// Restrict by tag (e.g. crypto, systems).
    #[arg(long, value_delimiter = ',')]
    tag: Vec<String>,

    /// Result ordering (default: relevance).
    #[arg(long, value_enum)]
    sort: Option<SortMode>,

    /// Output format (default: table).
    #[arg(long, value_parser = parse_format_arg)]
    format: Option<Format>,

    /// Limit number of results.
    #[arg(long)]
    limit: Option<usize>,

    /// Columns for table/csv output (comma-separated).
    #[arg(long, value_delimiter = ',')]
    fields: Vec<String>,

    /// Launch the interactive TUI.
    #[arg(long)]
    tui: bool,

    /// Override database path.
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    /// Override user config.yaml path.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
}

impl Cli {
    fn has_search_args(&self) -> bool {
        !self.query.is_empty()
            || !self.venue.is_empty()
            || !self.year.is_empty()
            || !self.rank.is_empty()
            || !self.tag.is_empty()
            || self.sort.is_some()
            || self.format.is_some()
            || self.limit.is_some()
            || !self.fields.is_empty()
            || self.tui
    }
}

#[derive(Subcommand)]
enum Command {
    /// Create the data/config directories and an empty database.
    Init,
    /// Fetch paper metadata from dblp (incremental, idempotent).
    Update(UpdateArgs),
    /// Fill missing abstracts on the existing database (no dblp re-fetch).
    Enrich(EnrichArgs),
}

/// Default number of concurrent abstract fetches.
const DEFAULT_JOBS: usize = 8;
const MIN_ENRICH_BATCH: usize = 64;
const MAX_ENRICH_BATCH: usize = 512;
const ENRICH_PROGRESS_INTERVAL: usize = 500;

#[derive(clap::Args)]
struct UpdateArgs {
    /// Only ingest from these venues (id or alias).
    #[arg(long, value_delimiter = ',')]
    venue: Vec<String>,
    /// Use these bundled venue sets for this update (overrides config bundles).
    #[arg(long, value_delimiter = ',')]
    bundle: Vec<String>,
    /// Minimum year (overrides config default).
    #[arg(long)]
    since: Option<i32>,
    /// Also fetch abstracts (slower; uses API keys, then static scrapers).
    #[arg(long)]
    abstracts: bool,
    /// Concurrent abstract fetches (only with --abstracts).
    #[arg(long, default_value_t = DEFAULT_JOBS)]
    jobs: usize,
}

#[derive(clap::Args)]
struct EnrichArgs {
    /// Only enrich these venues (id or alias); default is all.
    #[arg(long, value_delimiter = ',')]
    venue: Vec<String>,
    /// Use these bundled venue sets for this enrich run (overrides config bundles).
    #[arg(long, value_delimiter = ',')]
    bundle: Vec<String>,
    /// Only enrich papers from this year onward.
    #[arg(long)]
    since: Option<i32>,
    /// Concurrent abstract fetches.
    #[arg(long, default_value_t = DEFAULT_JOBS)]
    jobs: usize,
    /// Stop after this many papers (useful for sampling / validation).
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum SortMode {
    Relevance,
    Year,
    Venue,
    Rank,
}

pub(crate) struct SearchOptions<'a> {
    pub(crate) venues: &'a [String],
    pub(crate) ranks: &'a [String],
    pub(crate) tags: &'a [String],
    pub(crate) years: &'a [query::YearRange],
    pub(crate) sort: SortMode,
    pub(crate) limit: Option<usize>,
    pub(crate) offset: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_target(false)
        .without_time()
        .init();

    let cli = Cli::parse();
    reject_search_args_for_subcommands(&cli)?;
    let paths = Paths::resolve()?;
    let config = load_config_with_bundles(&cli, &paths, None)?;

    match &cli.command {
        Some(Command::Init) => cmd_init(&cli, &paths),
        Some(Command::Update(args)) => cmd_update(args, &cli, &paths, &config).await,
        Some(Command::Enrich(args)) => cmd_enrich(args, &cli, &paths, &config).await,
        None if cli.tui => {
            let db = open_db(&cli, &paths)?;
            tui::run(db, config)
        }
        None => cmd_search(&cli, &paths, &config),
    }
}

fn reject_search_args_for_subcommands(cli: &Cli) -> Result<()> {
    if cli.command.is_some() && cli.has_search_args() {
        anyhow::bail!(
            "search query/options cannot be used with subcommands; put command-specific options after the subcommand"
        );
    }
    Ok(())
}

fn log_header(title: &str) {
    eprintln!("{title}");
}

fn log_field(label: &str, value: impl std::fmt::Display) {
    eprintln!("  {label:<10} {value}");
}

fn log_blank() {
    eprintln!();
}

fn load_config_with_bundles(
    cli: &Cli,
    paths: &Paths,
    bundle_override: Option<&[String]>,
) -> Result<Config> {
    let user_path = config_path(cli, paths);
    Config::load_with_bundles(Some(&user_path), bundle_override).context("loading venue config")
}

fn config_path(cli: &Cli, paths: &Paths) -> PathBuf {
    cli.config
        .clone()
        .unwrap_or_else(|| paths.user_config_path())
}

fn db_path(cli: &Cli, paths: &Paths) -> PathBuf {
    cli.db.clone().unwrap_or_else(|| paths.db_path())
}

fn open_db(cli: &Cli, paths: &Paths) -> Result<Database> {
    let path = db_path(cli, paths);
    Database::open_existing(&path).with_context(|| {
        format!(
            "no database at {}; run `sec-grep init` then `sec-grep update`",
            path.display()
        )
    })
}

fn cmd_init(cli: &Cli, paths: &Paths) -> Result<()> {
    paths.ensure_dirs()?;
    let path = db_path(cli, paths);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Database::open(&path).context("creating database")?;
    let config_path = config_path(cli, paths);
    write_default_config(&config_path)?;
    log_header("sec-grep initialized");
    log_field("database", path.display());
    log_field("config", config_path.display());
    log_blank();
    log_field("next", "`sec-grep update`");
    Ok(())
}

fn write_default_config(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, Config::default_user_config_yaml()?)?;
    Ok(())
}

fn cmd_search(cli: &Cli, paths: &Paths, config: &Config) -> Result<()> {
    let db = open_db(cli, paths)?;

    let raw = cli.query.join(" ");
    let search = build_search(
        &raw,
        config,
        SearchOptions {
            venues: &cli.venue,
            ranks: &cli.rank,
            tags: &cli.tag,
            years: &cli.year,
            sort: cli.sort.unwrap_or(SortMode::Relevance),
            limit: cli.limit,
            offset: None,
        },
    )?;
    let papers = db.search(&search)?;

    let columns = parse_columns(&cli.fields)?;
    let format = cli.format.unwrap_or(Format::Table);
    let out =
        output::render(&papers, format, columns.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    if !out.is_empty() {
        print!("{out}");
        if !out.ends_with('\n') {
            println!();
        }
    }
    if matches!(format, Format::Table) {
        eprintln!("results    {}", papers.len());
    }
    Ok(())
}

pub(crate) fn build_search(
    raw_query: &str,
    config: &Config,
    options: SearchOptions<'_>,
) -> sec_grep_core::Result<Search> {
    let parsed = query::parse(raw_query)?;
    let mut venue_selectors = parsed.venue_selectors;
    venue_selectors.extend_from_slice(options.venues);
    let mut rank_selectors = parsed.rank_selectors;
    rank_selectors.extend_from_slice(options.ranks);
    let mut tag_selectors = parsed.tag_selectors;
    tag_selectors.extend_from_slice(options.tags);
    let mut year_ranges = parsed.year_ranges;
    year_ranges.extend_from_slice(options.years);
    let venue_filter =
        config.resolve_venue_filter(&venue_selectors, &rank_selectors, &tag_selectors)?;

    let sort = match options.sort {
        SortMode::Relevance => Sort::Relevance,
        SortMode::Year => Sort::Year,
        SortMode::Venue => Sort::Venue,
        SortMode::Rank => Sort::Rank(config.rank_sort_order()),
    };

    Ok(Search {
        fts: parsed.fts,
        venue_filter,
        doi_terms: parsed.doi_terms,
        year_ranges,
        sort,
        limit: options.limit,
        offset: options.offset,
    })
}

fn parse_columns(fields: &[String]) -> Result<Option<Vec<Column>>> {
    if fields.is_empty() {
        return Ok(None);
    }
    let cols = fields
        .iter()
        .map(|f| f.parse::<Column>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;
    Ok(Some(cols))
}

fn parse_format_arg(value: &str) -> std::result::Result<Format, String> {
    value.parse::<Format>().map_err(|e| e.to_string())
}

fn parse_year_arg(value: &str) -> std::result::Result<query::YearRange, String> {
    query::parse_year_range(value).map_err(|e| e.to_string())
}

async fn cmd_update(args: &UpdateArgs, cli: &Cli, paths: &Paths, config: &Config) -> Result<()> {
    paths.ensure_dirs()?;
    let path = db_path(cli, paths);
    let mut db = Database::open(&path).context("opening database")?;
    let override_config;
    let config = if args.bundle.is_empty() {
        config
    } else {
        override_config = load_config_with_bundles(cli, paths, Some(&args.bundle))?;
        &override_config
    };

    let venue_ids = if args.venue.is_empty() {
        config.all_venue_ids()
    } else {
        config.resolve_venues(&args.venue)?
    };
    let min_year = args.since.unwrap_or(config.defaults.min_year);

    log_header("sec-grep update");
    log_field("bundles", config.bundles.join(", "));
    log_field("venues", venue_ids.len());
    log_field("since", min_year);
    log_blank();

    let dblp = Dblp::default();
    let mut total = 0usize;
    let mut failed = Vec::new();
    for id in &venue_ids {
        let venue = config.venue(id).expect("resolved venue");
        eprint!("  {id:<12} ");
        let _ = std::io::stderr().flush();
        match dblp.fetch_venue(venue, min_year, MAX_YEAR).await {
            Ok(papers) => {
                let n = db.upsert_papers(&papers)?;
                total += papers.len();
                eprintln!("fetched {:>5} papers, {:>5} upserted", papers.len(), n);
            }
            Err(e) => {
                eprintln!("failed   {e}");
                failed.push(id.clone());
            }
        }
    }
    log_blank();
    log_header("summary");
    log_field("fetched", format_args!("{total} papers"));
    log_field("failed", failed.len());
    log_field("database", format_args!("{} papers", db.count()?));

    if !failed.is_empty() {
        anyhow::bail!("failed to fetch venue(s): {}", failed.join(", "));
    }

    if args.abstracts {
        log_blank();
        enrich_abstracts(
            &mut db,
            &venue_ids,
            &[query::YearRange::new(Some(min_year), None)?],
            args.jobs,
            None,
        )
        .await?;
    }
    Ok(())
}

async fn cmd_enrich(args: &EnrichArgs, cli: &Cli, paths: &Paths, config: &Config) -> Result<()> {
    let mut db = open_db(cli, paths)?;
    let override_config;
    let config = if args.bundle.is_empty() {
        config
    } else {
        override_config = load_config_with_bundles(cli, paths, Some(&args.bundle))?;
        &override_config
    };
    let venue_ids = if args.venue.is_empty() {
        if args.bundle.is_empty() {
            Vec::new()
        } else {
            config.all_venue_ids()
        }
    } else {
        config.resolve_venues(&args.venue)?
    };
    let years = match args.since {
        Some(year) => vec![query::YearRange::new(Some(year), None)?],
        None => Vec::new(),
    };
    enrich_abstracts(&mut db, &venue_ids, &years, args.jobs, args.limit).await
}

/// Fill missing abstracts, running up to `jobs` fetches concurrently.
/// `venue_ids` empty means all venues; `limit` caps how many are attempted.
async fn enrich_abstracts(
    db: &mut Database,
    venue_ids: &[String],
    years: &[query::YearRange],
    jobs: usize,
    limit: Option<usize>,
) -> Result<()> {
    let mut enricher = Enricher::new(Secrets::load());
    let pending = db.count_missing_abstracts(venue_ids, years)?;
    let total = limit.map_or(pending, |limit| limit.min(pending));
    let jobs = jobs.max(1);
    let batch_size = enrich_batch_size(jobs);
    log_header("abstract enrichment");
    log_field("pending", format_args!("{pending} abstracts"));
    if !years.is_empty() {
        log_field("years", format_args!("{}", format_year_ranges(years)));
    }
    if total != pending {
        log_field("selected", format_args!("{total} abstracts"));
    }
    log_field("jobs", jobs);
    log_blank();

    let mut filled = 0usize;
    let mut processed = 0usize;
    let mut misses: BTreeMap<String, usize> = BTreeMap::new();
    let mut after_id = 0;
    while processed < total {
        let remaining = total - processed;
        let batch = db.papers_missing_abstract_batch(
            venue_ids,
            years,
            after_id,
            remaining.min(batch_size),
        )?;
        let Some(next_after_id) = batch.last().map(|paper| paper.id) else {
            break;
        };
        after_id = next_after_id;

        let papers = batch
            .into_iter()
            .map(|missing| missing.paper)
            .collect::<Vec<_>>();

        let batch_len = papers.len();
        log_field(
            "batch",
            format_args!(
                "{}-{} / {} ({})",
                processed + 1,
                processed + batch_len,
                total,
                batch_venue_summary(&papers)
            ),
        );
        let results = enricher.enrich_many(papers, jobs).await;

        let mut abstract_updates = Vec::new();
        for (paper, res) in results {
            processed += 1;
            let dblp_key = paper.dblp_key;
            match res {
                Ok(EnrichResult::Found(abs)) => {
                    abstract_updates.push((dblp_key, abs));
                    filled += 1;
                }
                Ok(EnrichResult::Missing(reason)) => {
                    *misses.entry(reason).or_default() += 1;
                }
                Err(e) => tracing::warn!("abstract fetch failed for {dblp_key}: {e}"),
            }
            if processed.is_multiple_of(ENRICH_PROGRESS_INTERVAL) {
                log_field(
                    "progress",
                    format_args!("{processed}/{total} processed, {filled} filled"),
                );
            }
        }
        log_field(
            "batch done",
            format_args!(
                "{} filled, {} missed",
                abstract_updates.len(),
                batch_len - abstract_updates.len()
            ),
        );
        db.set_abstracts(&abstract_updates)?;
    }
    log_blank();
    log_header("summary");
    log_field("filled", format_args!("{filled}/{processed} abstracts"));
    if !misses.is_empty() {
        log_field("missed", processed - filled);
        for (reason, count) in misses {
            eprintln!("  {count:>10} {reason}");
        }
    }
    Ok(())
}

fn format_year_ranges(years: &[query::YearRange]) -> String {
    years
        .iter()
        .map(|year| match year.bounds() {
            (Some(min), Some(max)) if min == max => min.to_string(),
            (Some(min), Some(max)) => format!("{min}-{max}"),
            (Some(min), None) => format!("{min}-"),
            (None, Some(max)) => format!("-{max}"),
            (None, None) => String::new(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn enrich_batch_size(jobs: usize) -> usize {
    jobs.saturating_mul(4)
        .clamp(MIN_ENRICH_BATCH, MAX_ENRICH_BATCH)
}

fn batch_venue_summary(inputs: &[Paper]) -> String {
    let mut counts = BTreeMap::new();
    for paper in inputs {
        *counts.entry(paper.venue.as_str()).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .map(|(venue, count)| format!("{venue}:{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_path(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("sec-grep-{name}-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn build_search_preserves_empty_cli_venue_filter() {
        let config = Config::defaults().unwrap();
        let ranks = vec!["does-not-exist".to_string()];
        let search = build_search(
            "",
            &config,
            SearchOptions {
                venues: &[],
                ranks: &ranks,
                tags: &[],
                years: &[],
                sort: SortMode::Relevance,
                limit: None,
                offset: None,
            },
        )
        .unwrap();

        assert!(search.venue_filter.is_empty());
    }

    #[test]
    fn cli_and_inline_metadata_filters_have_same_semantics() {
        let config = Config::defaults().unwrap();
        let ranks = vec!["A*".to_string()];
        let tags = vec!["crypto".to_string()];

        let inline = build_search(
            "rank:A* tag:crypto",
            &config,
            SearchOptions {
                venues: &[],
                ranks: &[],
                tags: &[],
                years: &[],
                sort: SortMode::Relevance,
                limit: None,
                offset: None,
            },
        )
        .unwrap();

        let cli = build_search(
            "",
            &config,
            SearchOptions {
                venues: &[],
                ranks: &ranks,
                tags: &tags,
                years: &[],
                sort: SortMode::Relevance,
                limit: None,
                offset: None,
            },
        )
        .unwrap();

        assert_eq!(inline.venue_filter, cli.venue_filter);
    }

    #[test]
    fn cli_and_inline_same_kind_metadata_filters_are_ored() {
        let config = Config::defaults().unwrap();
        let ranks = vec!["A*".to_string()];

        let mixed = build_search(
            "rank:A",
            &config,
            SearchOptions {
                venues: &[],
                ranks: &ranks,
                tags: &[],
                years: &[],
                sort: SortMode::Relevance,
                limit: None,
                offset: None,
            },
        )
        .unwrap();

        let inline = build_search(
            "rank:A rank:A*",
            &config,
            SearchOptions {
                venues: &[],
                ranks: &[],
                tags: &[],
                years: &[],
                sort: SortMode::Relevance,
                limit: None,
                offset: None,
            },
        )
        .unwrap();

        assert_eq!(mixed.venue_filter, inline.venue_filter);
    }

    #[test]
    fn repeated_year_cli_flags_are_accepted() {
        let cli = Cli::try_parse_from(["sec-grep", "--year", "2018", "--year", "2029"]).unwrap();
        assert_eq!(
            cli.year,
            vec![
                query::YearRange::single(2018),
                query::YearRange::single(2029)
            ]
        );
    }

    #[test]
    fn update_bundle_flags_are_accepted() {
        let cli = Cli::try_parse_from(["sec-grep", "update", "--bundle", "se,ml"]).unwrap();
        let Some(Command::Update(args)) = cli.command else {
            panic!("expected update command");
        };
        assert_eq!(args.bundle, vec!["se".to_string(), "ml".to_string()]);
    }

    #[test]
    fn enrich_bundle_flags_are_accepted() {
        let cli = Cli::try_parse_from(["sec-grep", "enrich", "--bundle", "se,ml"]).unwrap();
        let Some(Command::Enrich(args)) = cli.command else {
            panic!("expected enrich command");
        };
        assert_eq!(args.bundle, vec!["se".to_string(), "ml".to_string()]);
    }

    #[test]
    fn bundle_override_limits_venue_resolution() {
        let cli = Cli::try_parse_from(["sec-grep", "update", "--bundle", "se", "--venue", "ndss"])
            .unwrap();
        let paths = Paths {
            data_dir: temp_test_path("data"),
            config_dir: temp_test_path("config"),
        };
        let Some(Command::Update(args)) = &cli.command else {
            panic!("expected update command");
        };
        let config = load_config_with_bundles(&cli, &paths, Some(&args.bundle)).unwrap();
        assert!(config.resolve_venues(&args.venue).is_err());
    }

    #[test]
    fn enrich_since_flag_is_accepted() {
        let cli = Cli::try_parse_from(["sec-grep", "enrich", "--since", "2025"]).unwrap();
        let Some(Command::Enrich(args)) = cli.command else {
            panic!("expected enrich command");
        };
        assert_eq!(args.since, Some(2025));
    }

    #[test]
    fn write_default_config_creates_but_does_not_overwrite() {
        let path = temp_test_path("config.yaml");
        let _ = std::fs::remove_file(&path);
        write_default_config(&path).unwrap();
        let default = std::fs::read_to_string(&path).unwrap();
        assert!(default.contains("security"));
        assert!(default.contains("min_year: 2000"));

        std::fs::write(&path, "bundles: []\n").unwrap();
        write_default_config(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "bundles: []\n");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn open_start_year_cli_flag_is_accepted() {
        let cli = Cli::try_parse_from(["sec-grep", "--year", "-2019"]).unwrap();
        assert_eq!(
            cli.year,
            vec![query::YearRange::new(None, Some(2019)).unwrap()]
        );
    }

    #[test]
    fn comma_separated_open_start_year_cli_flag_is_accepted() {
        let cli = Cli::try_parse_from(["sec-grep", "--year", "-2019,2029"]).unwrap();
        assert_eq!(
            cli.year,
            vec![
                query::YearRange::new(None, Some(2019)).unwrap(),
                query::YearRange::single(2029)
            ]
        );
    }

    #[test]
    fn cli_and_inline_same_kind_year_filters_are_ored() {
        let config = Config::defaults().unwrap();
        let years = vec![query::YearRange::single(2029)];

        let search = build_search(
            "year:2018",
            &config,
            SearchOptions {
                venues: &[],
                ranks: &[],
                tags: &[],
                years: &years,
                sort: SortMode::Relevance,
                limit: None,
                offset: None,
            },
        )
        .unwrap();

        assert_eq!(
            search.year_ranges,
            vec![
                query::YearRange::single(2018),
                query::YearRange::single(2029)
            ]
        );
    }
}
