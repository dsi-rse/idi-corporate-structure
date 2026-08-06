# IDI Corporate Structure Pipeline

Automated pipeline for extracting subsidiary information from SEC 10-K (Exhibit 21) and 20-F (Exhibit 8) filings and building hierarchical corporate structure trees.

## Pipeline Overview

This processor consumes SEC filing data already collected by the upstream **sec-scraper** (stored in S3), and performs three stages:

1. **Load** — read the scraper's `manifest.parquet` from the SEC bucket for the requested date range (or the most recent filings via `--daily`) to enumerate 10-K-family (Exhibit 21) and 20-F-family (Exhibit 8) filings, capturing each filing's submission date and its reporting period (`period_of_report`); fetch per-company metadata (state of incorporation, business address, tickers, exchanges) from the SEC submissions API
2. **Retrieval** — load each filing's already-scraped exhibit content from S3 (HTML, plain text, or PDF)
3. **Extraction** — pass exhibit content to `gpt-4.1-nano` using structured output to parse subsidiary names and incorporation locations. Each subsidiary name is grounded against the exhibit text — exact match, then a punctuation-insensitive match, then a windowed in-order token match for names split across table columns — and names not found in the source are dropped to guard against hallucinations. Output is structured `Subsidiary` records written to Parquet.

Processing tracks permanent failures to disk so interrupted runs do not re-attempt filings that will always fail.

### Output Layout

```
{output_file}   # Parquet — one row per subsidiary, with parent-company metadata
                # columns: parent_cik, filing_date, period_of_report, form_type,
                #   exhibit_type, accession_number, exhibit_url, name, location,
                #   parent_name, parent_state_of_incorporation, parent_business_*
                #   (street/city/state/zip/country/country_code), parent_tickers,
                #   parent_exchanges, source_quote, date_added
failures.json   # permanent failures keyed by (cik, accession_number)
```

Output and failure paths support local directories or S3 URLs (`s3://bucket/path`). SEC input is always read from S3 via `--sec-bucket-prefix`.

#### `filing_date` vs `period_of_report`

These are different dates and are not interchangeable:

- **`filing_date`** — when the filing was submitted to EDGAR.
- **`period_of_report`** — the fiscal period the exhibit actually describes (ISO `YYYY-MM-DD`, taken from the filing index page's *Period of Report*).

Delinquent filers submit several years of 10-Ks on the **same day**, each with its own Exhibit 21, so `filing_date` cannot identify which reporting year a subsidiary list covers. Accession number does not resolve it either — its prefix is a filer ID, not a sequence. For CIK 1583994, both accessions were filed 2017-02-24, and the *higher* one (`0001583994-17-000009`) is the older FY2014 exhibit, while `0001574540-17-000007` is FY2016. Select on `period_of_report`.

Rows written before this column was added carry an empty string rather than a null; backfilling them is tracked separately.

---

## Quick Start

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync              # Production
uv sync --all-groups # Development (includes tests and linting tools)
```

### Credentials

```bash
export OPENAI_API_KEY='your-key'
```

| Credential | Source |
|---|---|
| OpenAI API key | [platform.openai.com](https://platform.openai.com/api-keys) |

AWS credentials are always required: SEC input is read from S3 (the sec-scraper's bucket), and output/failures may also be S3 paths.

### Run

```bash
uv run python3 -m src.idi_corporate_structure.orchestrator \
    --sec-bucket-prefix "my-bucket/sec" \
    --output-file "/local/output/latest.parquet" \
    --failure-file "/local/failures/failures.json" \
    --start-date "2026-05-01" \
    --end-date "2026-05-31" \
    --sec-user-agent "Your Name you@example.com" \
    --rate-limit 0.2 \
    --num-workers 10
```

- `OPENAI_API_KEY` is read from the environment (or pass `--openai-api-key`).
- To process the most recent filings instead of an explicit range, replace `--start-date`/`--end-date` with `--daily` (optionally `--look-back N`, default 7). Daily mode reads the latest `filing_date` from `{sec-bucket-prefix}/manifest.parquet` and processes the trailing window.

### Configuration Reference

| Flag | Default | Description |
|---|---|---|
| `--sec-bucket-prefix` | — | Required. `bucket-name/prefix` where the sec-scraper wrote SEC data + `manifest.parquet` |
| `--output-file` | — | Required. Path for Parquet output (local or `s3://`) |
| `--failure-file` | — | Required. Path to failures JSON; parent directory created if missing |
| `--daily` / `--start-date` | — | Required (mutually exclusive). Daily mode, or an explicit `--start-date`/`--end-date` range |
| `--look-back` | `7` | Daily mode only: days to look back from the latest filing date |
| `--sec-user-agent` | env `SEC_USER_AGENT` | Required. SEC EDGAR contact string (`Name email`) |
| `--openai-api-key` | env `OPENAI_API_KEY` | Required. OpenAI API key |
| `--model` | env `OPENAI_MODEL`, then `gpt-4.1-nano` | OpenAI model ID for extraction |
| `--rate-limit` | `0.2` | Seconds between SEC HTTP requests (SEC limit: 10 req/s) |
| `--num-workers` | `10` | Number of concurrent GPT extraction worker threads |

---

## Container Usage

The pipeline ships with a multi-stage Dockerfile and Docker Compose files for running the orchestrator in a container.

### Files

| File | Purpose |
|---|---|
| `dockerfiles/Dockerfile.orchestrator` | Multi-stage `python:3.13-slim` image; non-root `pipeline` user |
| `compose.yml` | Service definition; pulls image from registry |
| `compose.override.yml` | Adds `build:` block for local development; merged automatically by `docker compose` |

### Environment Variables

All required variables must be set before running. Optional variables fall back to the listed defaults.

The container runs `--daily` mode by default (see `compose.yml`), reading SEC input from S3, so AWS credentials must be available in the container (an instance role on EC2, or pass `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` through locally).

#### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for GPT extraction |
| `SEC_BUCKET_PREFIX` | `bucket-name/prefix` where the sec-scraper wrote SEC data + `manifest.parquet` (e.g. `my-bucket/sec`) |
| `SEC_USER_AGENT` | SEC EDGAR contact string (`Name email`) |
| `OUTPUT_MOUNT_SOURCE` | Host directory for Parquet output (e.g. `/data/output`) |
| `FAILURE_MOUNT_SOURCE` | Host directory for failures JSON (e.g. `/data/failures`) |
| `LOG_DIR` | Host directory for log files (e.g. `/data/logs`) |

#### Optional

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4.1-nano` | OpenAI model ID for extraction |
| `OUTPUT_FILE` | `/data/output/latest.parquet` | Container-side path for Parquet output |
| `FAILURE_FILE` | `/data/failures/failures.json` | Container-side path for failures JSON |
| `RATE_LIMIT` | `0.2` | Seconds between SEC HTTP requests |
| `NUM_WORKERS` | `10` | Number of concurrent GPT extraction worker threads |
| `INPUT_SAMPLE_SIZE` | `0` | Limit input to N filings for testing (`0` = no limit) |
| `AWS_REGION` | `us-east-2` | AWS region for S3 and CloudWatch |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` | — | AWS credentials for S3 (omit when using an instance role) |
| `CLOUDWATCH_LOGS_ENABLED` | `false` | Enable CloudWatch log shipping |
| `ORCHESTRATOR_IMAGE` | `ghcr.io/dsi-clinic/idi-corporate-structure-orchestrator:latest` | Image to pull on EC2 (ignored when building locally) |

### Run

**Local — build from source and run:**

```bash
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxx"
export SEC_BUCKET_PREFIX="my-bucket/sec"         # sec-scraper output + manifest.parquet
export SEC_USER_AGENT="Your Name you@example.com"
export OUTPUT_MOUNT_SOURCE="/path/to/output"
export FAILURE_MOUNT_SOURCE="/path/to/failures"
export LOG_DIR="/path/to/logs"
# AWS credentials for S3 (omit if the host provides an instance role)
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...

docker compose up --build orchestrator
```

`compose.override.yml` is merged automatically when running locally, which adds the `build:` block so the image is built from source rather than pulled.

**Detached (background):**

```bash
docker compose up -d --build orchestrator
docker compose logs -f orchestrator   # tail logs
docker compose down                   # stop
```

---

## AWS ECS Architecture

The pipeline runs as an **ECS Fargate task** scheduled by **EventBridge Scheduler**. Infrastructure is defined in `pulumi/` using Pulumi (Python).

### Design Decisions

| Decision | Rationale |
|---|---|
| **Fargate** (not EC2) | No instance management — container runs and exits; portable image |
| **EventBridge Scheduler** (not Step Functions) | Processors are independent; no workflow orchestration needed |
| **Public subnet** (no NAT Gateway) | Task needs outbound internet for SEC EDGAR and OpenAI |
| **`awslogs` driver only** | Captures all stdout/stderr; linked directly to the task in the ECS console; app-level CloudWatch handler disabled (`CLOUDWATCH_LOGS_ENABLED=false`) |
| **ECR** | No pull credential configuration required for Fargate |

### Resources

| Module | Resources |
|---|---|
| `config.py` | Shared name prefix (`{project}-{stack}-{app}`), tags, AWS caller identity |
| `networking.py` | Default VPC, single-AZ public subnet, egress-only security group |
| `iam.py` | Task execution role (ECR pull, CloudWatch Logs, Secrets Manager) + task role (S3, ECS Exec) |
| `ecr.py` | ECR repository + lifecycle policy (retains last 5 images) |
| `ecs.py` | ECS cluster (Fargate, Container Insights), CloudWatch log group (30-day retention), task definition (1 vCPU / 4 GB) |
| `secrets.py` | Secrets Manager secret for OpenAI API key; injected as `OPENAI_API_KEY` env var at task startup |
| `scheduling.py` | EventBridge Scheduler (cron, starts disabled), SQS dead-letter queue for failed invocations, scheduler IAM role |

### S3 File Layout

Everything lives in a single externally-managed S3 bucket (name from SSM). SEC input is written by the upstream sec-scraper under `{sec_prefix}/` (default `sec/`); this processor writes its output under `{app}/`:

```
{bucket}/
  {sec_prefix}/                 ← input, written by the sec-scraper
    manifest.parquet            ← filing index the orchestrator reads (--sec-bucket-prefix)
    ...                         ← scraped exhibit documents
  {app}/
    output/latest.parquet ← output
    failures/failures.json      ← permanent failure registry
```

### Deployment

```bash
cd pulumi/

# First-time setup
uv run --group pulumi pulumi stack init dev
uv run --group pulumi pulumi config set aws:region us-east-2
uv run --group pulumi pulumi config set --secret idi:openai_api_key <key>

# Deploy
uv run --group pulumi pulumi up
```

#### Configuration Reference

| Config | Default | Description |
|---|---|---|
| `aws:region` | `us-east-2` | AWS region |
| `idi:app_name` | `corporate-structure` | Application name used in resource naming |
| `idi:openai_api_key` | — | OpenAI API key (secret; stored in Secrets Manager) |
| `idi:sec_user_agent` | — | Required. SEC EDGAR contact string (`Name email`) |
| `idi:sec_prefix` | `sec` | Prefix within the shared bucket where the sec-scraper wrote SEC data; combined with the bucket name to form `--sec-bucket-prefix` |
| `idi:openai_model` | `gpt-4.1-nano` | OpenAI model ID for extraction |
| `idi:cron_corporate_structure` | `cron(0 2 * * ? *)` | EventBridge schedule expression |
| `idi:schedule_enabled` | `false` | Enable the EventBridge schedule |
| `idi:cpu` | `1024` | Fargate task CPU units |
| `idi:memory` | `4096` | Fargate task memory (MiB) |
| `idi:rate_limit` | `0.2` | Seconds between SEC API requests |
| `idi:num_workers` | `10` | GPT extraction worker threads |
| `idi:input_sample_size` | `0` | Filings to process (`0` = all; set `>0` for testing) |

### Manual Task Execution

```bash
aws ecs run-task \
    --cluster <cluster-name> \
    --task-definition <task-definition> \
    --launch-type FARGATE \
    --propagate-tags TASK_DEFINITION \
    --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}" \
    --overrides '{
        "containerOverrides": [{
            "name": "corporate-structure-orchestrator",
            "environment": [{"name": "INPUT_SAMPLE_SIZE", "value": "5"}]
        }]
    }'
```

Use `pulumi stack output` to retrieve cluster name, subnet ID, and security group ID.

### Monitoring

- **Logs**: CloudWatch → Log groups → `/ecs/{name_prefix}` → stream per task run
- **ECS console**: Tasks tab shows stopped tasks for up to 1 hour after completion
- **Scheduling failures**: Check the SQS dead-letter queue (`pulumi stack output dlq_url`)
- **ECS Exec** (interactive debug into running task):
  ```bash
  aws ecs execute-command \
    --cluster <cluster> \
    --task <task-id> \
    --container corporate-structure-orchestrator \
    --interactive \
    --command "/bin/sh"
  ```

### Building and Pushing the Container Image

```bash
# Set ECR repo URL
ECR_REPO=$(cd pulumi && uv run --group pulumi pulumi stack output ecr_repo_url)

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-2.amazonaws.com

# Build for linux/amd64 (required on Apple Silicon) and push
docker buildx build --platform linux/amd64 \
  -f dockerfiles/Dockerfile.orchestrator \
  -t $ECR_REPO \
  --push .
```

---

## Data Flow

The pipeline consumes the sec-scraper's S3 output and walks down to individual exhibit documents:

```
manifest.parquet  (sec-scraper index in S3 — one row per scraped filing/document)
  │
  │  iter_filings_by_form_type(): filter to 10-K (Ex. 21) / 20-F (Ex. 8) in the date range
  ▼
Filing  (CIK, accession number, filing date, exhibit document s3_keys)
  │
  │  fetch company metadata from the SEC submissions API (state of incorporation, address, …)
  ▼
Exhibit Document  (HTML, plain text, or PDF — already scraped)
  │
  │  load_content(s3_key) from S3, extract text (PDF → pdfplumber, HTML → text)
  ▼
Exhibit Content  (raw text listing subsidiaries)
  │
  │  POST to OpenAI with structured JSON schema (chunked for large exhibits)
  ▼
GPT Extraction  (gpt-4.1-nano, structured output)
  │
  │  dedupe by name, ground each name against the exhibit text, attach parent metadata
  ▼
Subsidiaries List  (name, location, parent CIK + metadata, accession number, …)
  │
  │  write via pandas
  ▼
Parquet Output  ({output_file})
```

Each filing that cannot be processed (missing exhibit, empty content, document too long, etc.) is recorded in `failures.json` as a permanent failure and skipped on subsequent runs.

---

## Architecture

```
pipeline.py
  └── SubsidiaryPipeline.run()
        ├── load_input()       — read scraper manifest from S3 → list[Filing]
        └── process()
              ├── Main thread (producer)
              │     for each Filing:
              │       SecClient.query()  ← fetch directory index (rate-limited)
              │       SecClient.query()  ← fetch Exhibit 21 content
              │       work_queue.put()   ← blocks if all worker slots full
              │
              ├── N extract workers (daemon threads)
              │     work_queue.get()     ← blocks until work available
              │     GptExtractor.extract() → list[Subsidiary]
              │     results_queue.put()
              │
              └── 1 results worker (daemon thread)
                    results_queue.get()  ← accumulates Subsidiary records
```

**Producer-consumer design**:

- The SEC fetcher runs serially on the main thread (respecting EDGAR's 10 req/s rate limit).
- Exhibit content is pushed onto a bounded `queue.Queue(maxsize=num_workers * 2)`, which blocks the producer if workers fall behind.
- GPT extraction runs concurrently across `num_workers` daemon threads, draining the queue as fast as OpenAI responds.
- The two stages overlap — SEC fetching continues while earlier exhibits are being summarized.

**Resumability**:

-`FailureRegistry` persists non-retryable failures to disk after every `failure_flush_every` entries. On re-run, filings whose failures are classified as permanent are skipped without making network requests.

### Modules

This package (`src/idi_corporate_structure/`):

| Module | Purpose |
|---|---|
| `orchestrator.py` | CLI entrypoint — parses arguments, resolves the date range, wires up the pipeline |
| `pipeline.py` | `SubsidiaryPipeline` — orchestrates load, retrieval, and extraction; grounds names, dedupes, and writes Parquet output |
| `extractor.py` | `GptExtractor` — calls OpenAI with a structured JSON schema (chunked for large exhibits) to parse subsidiary names and locations; grounds names against the exhibit text |
| `api.py` | `OpenAiClient` for GPT extraction (subclass of the shared `ApiClient`) |
| `failures.py` | `FailureType` enum and `CorporateStructureFailureClassifier` (maps HTTP responses to retryable vs permanent failures) |
| `normalization.py` | Parent/subsidiary location normalization helpers |
| `types.py` | `Filing`, `Subsidiary`, `CompanyMeta`, `PipelineConfig`, and `PipelineStats` dataclasses |

Shared infrastructure lives in the [`idi-ftm2j-shared`](https://github.com/dsi-clinic/idi-ftm2j-shared) package: `api.ApiClient`/`SecClient` (retries, rate limiting), `failures.FailureRegistry`, `logs` (CloudWatch), `storage.load_content`, and `sec.iter_filings_by_form_type` (reads the scraper manifest).

### Failure Types

| Type | Retryable | Description |
|---|---|---|
| `mismatched_lengths` | No | Parallel filing arrays have unequal lengths |
| `no_form_data` | No | Filing arrays are empty |
| `no_10k_filings` | No | CIK has no 10-K forms |
| `no_overflow_filings` | No | CIK has no overflow filing entries |
| `no_filing_directory` | No | SEC returned no directory listing for the filing |
| `no_exhibit_found` | No | No exhibit file found in the filing directory |
| `no_exhibit_content` | No | Exhibit returned no content |
| `document_error` | No | Exhibit document is too long to process |
| `no_subsidiaries` | No | No subsidiaries found for the filing |
| `missing_period_of_report` | No | Neither the scraped manifest nor the SEC submissions JSON dates the filing |
| `truncated_error` | No | Model response cut off at the output token limit |
| `extraction_failed` | Yes | GPT returned no structured data |
| `timeout_error` | Yes | OpenAI API timed out |
| `api_error` | Yes | Transient HTTP failure |
| `rate_limit` | Yes | SEC 429 — retried with backoff |

---

## Development cycle

Documentation governing all processors: https://github.com/dsi-clinic/idi-ftm2j-shared/tree/main#development--contributing

### CI/CD specifics

**Docs-only pushes are skipped.** `deploy.yml` runs on every push to `dev` or `main` except those touching only `**.md` / `docs/**` (`paths-ignore`); any code or infra change triggers the full job sequence.

**Strict job sequence.** Jobs run in order: `version` → `docker` → `deploy-pulumi` → `sync-ecr`. Each job gates the next, so a Docker build failure will block the Pulumi deploy and ECR sync.

**Single Docker image.** Only `idi-corporate-structure-orchestrator` is built and published — to GHCR first, then re-tagged and pushed to ECR by `sync-ecr`.

**Manual dispatch on issue branches.** `deploy-pulumi` and `sync-ecr` include `issue-*` in their `if` conditions. Pushes to issue branches do not trigger the workflow, but `workflow_dispatch` can be used to manually run a deploy from a feature branch — useful for testing infra changes before merging.

**Required GitHub secrets.** The `deploy-pulumi` job requires the following secrets to be set in the repository. Secrets passed via `[ -n "$VAR" ]` are optional and skip silently if unset; all others are required for the deploy to succeed.

| Secret | Required | Notes |
|---|---|---|
| `AWS_ROLE_ARN_DEPLOY` | Yes | IAM role for Pulumi and ECR |
| `AWS_REGION` | No | Defaults to `us-east-2` |
| `PULUMI_ACCESS_TOKEN` | Yes | |
| `PULUMI_CONFIG_PASSPHRASE` | Yes | |
| `PULUMI_STATE_BUCKET` | Yes | S3 bucket for Pulumi state |
| `ECR_REPOSITORY_PREFIX` | No | Defaults to `{pulumi_project}-{env}-{app_name}` |
| `BUCKET_NAME` | No | S3 bucket for pipeline I/O |
| `SEC_PREFIX` | No | Prefix within the bucket holding the sec-scraper output; defaults to `sec` |
| `ECS_TASK_CPU` | No | |
| `ECS_TASK_MEMORY` | No | |
| `API_RATE_LIMIT` | No | Seconds between SEC HTTP requests |
| `NUM_WORKERS` | No | GPT extraction worker threads |
| `INPUT_SAMPLE_SIZE` | No | Limit input to N filings for testing |
| `OPENAI_API_KEY` | No | Set as a Pulumi secret in Secrets Manager |
| `OPENAI_MODEL` | No | |
| `SEC_USER_AGENT` | No | Set as a Pulumi secret; required by SEC EDGAR to avoid 429s |
| `CRON` | No | EventBridge schedule expression |

