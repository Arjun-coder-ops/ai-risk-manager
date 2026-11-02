# Milestone 2A: durable tenant data foundation

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

## Schema ownership

All tenant-owned operational tables carry `organization_id`. Database and API
constraints cover organizations, users, memberships, merchants,
merchant-day observations, ingestion batches and idempotency keys, model and
feature records, risk assessments, future investigation tables, feedback,
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

Database-backed model scoring, cases, workers, and analyst feedback workflows
remain later milestones. No real merchant data should enter the legacy demo
routes.
