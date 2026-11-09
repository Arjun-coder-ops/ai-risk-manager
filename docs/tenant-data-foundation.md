# Milestone 2: durable tenant data and investigation foundation

This milestone preserves the Flask modular monolith and introduces a separate
authenticated `/api/v1` surface. The legacy `/api/risk` endpoints remain an
explicitly synthetic backwards-compatible demo and are not a multi-tenant API.

## Decisions

| Choice | Why | Alternative and trade-off |
|---|---|---|
| SQLAlchemy 2 | One ORM contract for SQLite tests and PostgreSQL production. | Raw SQL is lighter initially but makes migrations and tenant constraints harder to maintain. |
| Alembic | Versioned, repeatable schema migration path. | `create_all` alone cannot safely evolve deployed schemas. |
| PostgreSQL in Compose | Durable relational storage, constraints, indexes, and transactional ingestion. | SQLite remains test-only; it is not suitable for concurrent multi-user investigations. |
| PyJWT + short-lived bearer tokens | Adds a clear authentication and organization-scoping boundary without a separate identity service. | Secure server sessions are viable; an external IdP should replace bootstrap registration for public production. |
| Atomic batch ingestion | Prevents a partly accepted merchant-day batch from silently contaminating feature history. | Row-by-row partial writes would need a more complex error/retry contract. |
| Database-backed transaction worker | Persists immutable pseudonymous transaction facts, then safely rebuilds affected daily aggregates on retry. | Redis/queue workers improve high-volume operations but are not justified for this small modular-monolith milestone. |
| Database-backed assessments | Scores are reproducible from the organization’s persisted observations and retain model/policy/evidence traceability. | Reading the legacy CSV would mix demonstration data with tenant data. |
| Explicit case state machine | Constrains investigation transitions and requires a decision plus note to close a case. | Free-form statuses are easier initially but make audit and operating metrics unreliable. |
| Verified analyst feedback | Captures human outcomes separately from model decisions, then requires an admin to verify them. | Feeding unverified labels directly into training would create a poisoning and feedback-loop risk. |
| Existing-account membership administration | Lets tenant admins grant, change, or revoke RBAC membership without collecting another user's password. | Public invitations need email delivery, expiry, and account-recovery controls that are intentionally deferred. |
| Correlation IDs and bounded request controls | Gives every response a safe request ID and gives local deployments basic protection against bursts of auth and write traffic. | The fixed-window limiter is deliberately per-process; a shared gateway or durable limiter is required before horizontal scaling. |

## Schema ownership

All tenant-owned operational tables carry `organization_id`. Database and API
constraints cover organizations, users, memberships, merchants,
merchant-day observations, immutable pseudonymous transaction records,
ingestion batches/jobs and idempotency keys, model and feature records, risk
assessments, investigation cases and events, feedback, audit events, and
monitoring snapshots.

`merchant_day_observations` is unique on `(merchant_id, observation_date)`;
merchants are unique on `(organization_id, external_id)`; an ingestion key is
unique per organization. API reads always scope by the token organization.

## Local setup

```powershell
Copy-Item .env.example .env
# Replace POSTGRES_PASSWORD and JWT_SECRET in .env.
docker compose up --build
```

For an externally managed database, set `DATABASE_URL` using the
`postgresql+psycopg://` driver form and execute:

```powershell
alembic upgrade head
```

The container runs Uvicorn against the FastAPI/ASGI gateway, which mounts the
existing Flask WSGI compatibility surface while route groups migrate. It
executes migrations before booting only to keep local Compose reproducible;
deployed environments should run migrations as a dedicated release operation.

The bootstrap endpoint `POST /api/v1/auth/register` is disabled unless
`ALLOW_BOOTSTRAP_REGISTRATION=true` is explicitly set. Use it only long enough
to create an initial local administrator, then set it back to `false` and
restart the service. Do not expose it publicly; production onboarding must use
an invitation or enterprise identity-provider flow.

`GET /health` is a liveness endpoint for the loaded inference artifact.
`GET /ready` additionally performs a `SELECT 1` against the configured
database and is therefore the Compose/container readiness probe. Neither
endpoint exposes tenant data.

## Current authenticated API

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `GET`/`POST /api/v1/members` (admin only; grants access to an existing active account)
- `PATCH`/`DELETE /api/v1/members/{user_id}` (admin only)
- `POST /api/v1/ingestions/merchant-days` (admin/analyst; requires `Idempotency-Key`)
- `POST /api/v1/ingestions/merchant-days/csv` (admin/analyst; multipart `file` and `Idempotency-Key`)
- `POST /api/v1/ingestions/transactions` (admin/analyst; asynchronous daily aggregation; requires `Idempotency-Key`)
- `GET /api/v1/ingestions/{id}`
- `GET /api/v1/merchants`
- `GET /api/v1/audit-events`
- `POST /api/v1/risk-assessments` (admin/analyst; accepts only `merchant_id` and `date`)
- `GET /api/v1/risk-assessments/{id}`
- `GET /api/v1/cases` and `GET /api/v1/cases/{id}`
- `GET /api/v1/cases/{id}/events`
- `POST /api/v1/cases/{id}/assignment` (admin/analyst/investigator)
- `POST /api/v1/cases/{id}/transitions` (admin/analyst/investigator)
- `POST`/`GET /api/v1/risk-assessments/{id}/feedback` (write: admin/analyst/investigator; read: all members)
- `POST /api/v1/feedback/{id}/verify` (admin only)

List endpoints (`/merchants`, `/cases`, and `/audit-events`) accept a one-based
`page` and a bounded `limit` (1–100; default 50). They retain their existing
collection key and add a `pagination` object containing `page`, `limit`,
`total`, and `total_pages` so existing consumers remain compatible.

Every response includes `X-Request-ID`. A caller may provide an 8–128
character safe value (`[A-Za-z0-9][A-Za-z0-9._:-]+`); unsafe or absent values
are replaced with a server-generated UUID. Error envelopes include the same
`request_id`. Request completion logs are structured JSON and deliberately
omit request/response bodies, credentials, and query strings.
`LOG_LEVEL` defaults to `INFO`, so these completion events are emitted by the
container instead of remaining below the default logger threshold.

Authentication endpoints and state-changing `/api/v1` endpoints receive local
fixed-window limits, configured by `AUTH_RATE_LIMIT_PER_MINUTE`,
`WRITE_RATE_LIMIT_PER_MINUTE`, and `RATE_LIMIT_WINDOW_SECONDS`. These limits
are in-process only: they do **not** coordinate Uvicorn workers or replicas.
Before production horizontal scaling, enforce equivalent limits at a trusted
API gateway or install a shared rate-limit store. Rate limiting is a
defense-in-depth control, not authentication, authorization, tenant isolation,
or abuse-prevention proof.

`POST /api/v1/risk-assessments` reads only the authenticated organization’s
persisted merchant-day observations. It stores the model version, policy
version, score, level, status, and explanation evidence. Repeated requests for
the same observation/model/policy return the existing assessment rather than
creating duplicate cases. A `HIGH` or `REVIEW` assessment opens a case and
records its creation as an event; lower-risk and insufficient-evidence results
do not open one automatically.

## Ingestion contract

JSON merchant-day ingestion is deliberately atomic: any invalid record rejects
the entire request. CSV ingestion accepts a UTF-8 `.csv` file whose header is
exactly the merchant-day aggregate schema. It validates rows independently and
creates a durable batch with `SUCCEEDED`, `PARTIAL`, or `REJECTED` status.
Valid rows commit together; invalid, duplicate-in-file, and already-persisted
merchant-days are recorded in a bounded `error_summary` (the first 100 row
errors) rather than silently dropped. A matching `Idempotency-Key` and file
bytes replay the existing batch; reuse with different bytes returns `409`.

CSV processing is synchronous and has a conservative, configurable
`MAX_CSV_INGESTION_ROWS=1000` application limit in addition to the request-body
limit.

Transaction ingestion is deliberately **asynchronous batch processing, not a
real-time streaming system**. `POST /api/v1/ingestions/transactions` accepts
at most `MAX_TRANSACTION_INGESTION_ROWS=1000` records and returns `202` only
after it atomically persists immutable raw records and a `QUEUED` ingestion
job. Each transaction requires exactly: `transaction_id`, `merchant_id`, a
timezone-aware RFC 3339 `occurred_at`, non-negative finite `amount`,
`outcome` (`SUCCEEDED` or `FAILED`), opaque `user_token`, opaque
`device_token`, and a normalized `region_code`. Tokens must not contain PII.
Duplicate provider transaction IDs are rejected per organization.

The Compose `ingestion-worker` claims queued jobs from PostgreSQL. It rebuilds
each affected merchant-day aggregate from transaction facts (`n_transactions`,
failures, amounts, users, first-observed users, devices, and regions). A job
retry never adds a second increment: recomputation replaces the aggregate.
Jobs use bounded exponential retry (three attempts) and expose the batch state
through `GET /api/v1/ingestions/{id}`. Direct merchant-day uploads and
transaction-derived observations cannot occupy the same merchant/day, which
prevents incompatible partial histories from being silently combined.

For a manually managed deployment, run the worker as a separate long-lived
process after migrations:

```powershell
python -m backend.ingestion_worker --poll-seconds 5
```

`python -m backend.ingestion_worker --once` is useful for a controlled local
smoke test. The database worker is suitable for bounded demonstration batches;
it is not a substitute for an independently operated, high-throughput queue.

Cases support `OPEN`, `IN_REVIEW`, `ESCALATED`, `RESOLVED`, and
`FALSE_POSITIVE`. The allowed transitions are deliberately one-way from the
terminal states. Resolving a case requires both a final decision and an
analyst note. Assignment and every status transition create immutable event
records as well as a tenant-scoped audit event. Admins can assign any active
member of the organization; analysts and investigators can assign only
themselves or unassign. `investigator` is an explicit case-work role: it can
assign/transition its own cases and submit feedback, but cannot administer
memberships, verify feedback, create assessments, or ingest data. `viewer` has
read-only access to tenant data.

Analysts and admins may append a `CONFIRMED_FRAUD`, `FALSE_POSITIVE`, or
`INCONCLUSIVE` outcome with a required note to an assessment. The submission
outcome and note are tenant-scoped and immutable through the API; verification
is a one-way admin action that records the verifier and timestamp. Submissions and
verifications also appear in audit logs and the linked case event history.
Feedback does not alter an assessment, case decision, threshold, or model, and
is not consumed by training automatically.

An admin can grant an existing active account the `admin`, `analyst`,
`investigator`, or `viewer` role, change that role, or revoke the membership. The API enforces
organization scope and cannot remove or demote the final admin. It deliberately
does not create accounts, accept another user's password, or issue public
invitation tokens; production onboarding still needs an invitation/SSO flow.

This milestone has a background worker only for retry-safe transaction-to-day
aggregation. It still has no notification delivery, account
provisioning/invitation flow, retraining workflow, or case concurrency version
token. No real merchant data should enter the legacy demo routes, and the
bootstrap registration endpoint must not be publicly exposed.

## Phase 7 incremental productization boundary

The current backend is an incremental FastAPI/ASGI gateway over the Flask
modular monolith while its durable API
contracts are stabilized. It has separate authentication, persistence,
ingestion-worker, request-control, tenant API, and model/policy components.
`GET /api/v1/openapi.json` publishes the source-controlled OpenAPI 3.1
contract for the tenant routes; the synthetic legacy routes are intentionally
excluded. The contract exposes JSON Schema request/response shapes, roles,
pagination, bearer authentication, and idempotency semantics, while existing
runtime validation remains explicit and covered by integration tests.

`backend/asgi.py` serves Swagger UI at `/api/v1/docs`, implements the typed
`/health` and `/ready` operational routes with a safe `X-Request-ID`
correlation header, and mounts the compatibility application through the
maintained `a2wsgi` adapter. The native health route preserves the legacy
response shape when the model is unavailable. The first native authentication
group—`POST /api/v1/auth/register`, `POST /api/v1/auth/login`, and
`GET /api/v1/auth/me`—uses strict Pydantic contracts and shared
framework-neutral service logic. The principal route validates a signed bearer
token *and* its current active membership before returning a strict principal
schema. Its CORS policy uses only the same explicit `ALLOWED_ORIGINS` list as
the compatibility API; native authentication requests retain the existing
bounded process-local rate limit and controlled error envelope. Bootstrap
registration remains a deliberately disabled-by-default local setup escape
hatch, not public production onboarding.

The service does not yet claim a completed FastAPI migration: ingestion,
scoring, case workflow, membership management, and audit route groups still
run via the compatibility adapter. Replacing every route before it has typed
contract coverage would create a risky parallel API surface. The next backend
migration increment must move route groups behind the published contract with
compatibility tests; it must not silently weaken tenant/RBAC controls.

## Reproducible local API baseline

Run `python evaluation/run_api_benchmark.py` to regenerate
`evaluation/api_benchmark.json`. The committed baseline uses 100 sequential
requests after 10 warmups against Flask's test client and a temporary SQLite
database seeded with 50 merchants. It measures `/health`, the OpenAPI document,
and authenticated `/api/v1/merchants?page=1&limit=50`; it is **not** a TCP,
Uvicorn, PostgreSQL, concurrent-load, or production capacity benchmark. On
the recorded Windows/Python 3.10.8 run, the tenant list route measured 9.0631
ms p50, 13.0810 ms p95, and 107.9842 sequential requests/sec. Re-run it on the
target deployment stack before setting capacity or latency objectives.
