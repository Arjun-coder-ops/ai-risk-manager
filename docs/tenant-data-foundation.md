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
| Database-backed assessments | Scores are reproducible from the organization’s persisted observations and retain model/policy/evidence traceability. | Reading the legacy CSV would mix demonstration data with tenant data. |
| Explicit case state machine | Constrains investigation transitions and requires a decision plus note to close a case. | Free-form statuses are easier initially but make audit and operating metrics unreliable. |

## Schema ownership

All tenant-owned operational tables carry `organization_id`. Database and API
constraints cover organizations, users, memberships, merchants,
merchant-day observations, ingestion batches and idempotency keys, model and
feature records, risk assessments, investigation cases and events, feedback,
audit events, and monitoring snapshots.

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

The bootstrap endpoint `POST /api/v1/auth/register` exists for local
development. Do not expose it publicly; production onboarding must use an
invitation or enterprise identity-provider flow.

## Current authenticated API

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/ingestions/merchant-days` (admin/analyst; requires `Idempotency-Key`)
- `GET /api/v1/ingestions/{id}`
- `GET /api/v1/merchants`
- `GET /api/v1/audit-events`
- `POST /api/v1/risk-assessments` (admin/analyst; accepts only `merchant_id` and `date`)
- `GET /api/v1/risk-assessments/{id}`
- `GET /api/v1/cases` and `GET /api/v1/cases/{id}`
- `GET /api/v1/cases/{id}/events`
- `POST /api/v1/cases/{id}/assignment` (admin/analyst)
- `POST /api/v1/cases/{id}/transitions` (admin/analyst)

`POST /api/v1/risk-assessments` reads only the authenticated organization’s
persisted merchant-day observations. It stores the model version, policy
version, score, level, status, and explanation evidence. Repeated requests for
the same observation/model/policy return the existing assessment rather than
creating duplicate cases. A `HIGH` or `REVIEW` assessment opens a case and
records its creation as an event; lower-risk and insufficient-evidence results
do not open one automatically.

Cases support `OPEN`, `IN_REVIEW`, `ESCALATED`, `RESOLVED`, and
`FALSE_POSITIVE`. The allowed transitions are deliberately one-way from the
terminal states. Resolving a case requires both a final decision and an
analyst note. Assignment and every status transition create immutable event
records as well as a tenant-scoped audit event. Admins can assign any active
member of the organization; analysts can assign only themselves or unassign.

The workflow is synchronous in this milestone. It has no background workers,
notification delivery, user-provisioning flow, feedback/retraining loop, or
concurrency version token yet. No real merchant data should enter the legacy
demo routes, and the bootstrap registration endpoint must not be publicly
exposed.
