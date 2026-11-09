"""Versioned, machine-readable contract for the authenticated tenant API.

The compatibility application remains Flask while durable endpoint contracts
are stabilized and selected route groups move to FastAPI. Keeping this OpenAPI
document in source control makes request/response shapes, authentication and
pagination independently reviewable throughout that incremental migration.
"""

from __future__ import annotations

from typing import Any


API_VERSION = "1.0.0"


def _response(description: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    content = {"application/json": {"schema": schema or {"type": "object"}}}
    return {"description": description, "content": content}


def _operation(summary: str, *, tags: list[str], auth: bool = True,
               request_schema: dict[str, Any] | None = None,
               success_code: str = "200", success_schema: dict[str, Any] | None = None,
               extra_responses: dict[str, Any] | None = None,
               parameters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    operation: dict[str, Any] = {
        "summary": summary,
        "tags": tags,
        "responses": {
            success_code: _response("Successful response.", success_schema),
            "400": {"$ref": "#/components/responses/BadRequest"},
            "401": {"$ref": "#/components/responses/Unauthorized"},
            "429": {"$ref": "#/components/responses/RateLimited"},
            "503": {"$ref": "#/components/responses/Unavailable"},
        },
    }
    if auth:
        operation["security"] = [{"bearerAuth": []}]
    if request_schema is not None:
        operation["requestBody"] = {
            "required": True,
            "content": {"application/json": {"schema": request_schema}},
        }
    if parameters:
        operation["parameters"] = parameters
    if extra_responses:
        operation["responses"].update(extra_responses)
    return operation


def openapi_document() -> dict[str, Any]:
    """Return the stable OpenAPI 3.1 contract for `/api/v1` only.

    Legacy `/api/risk` endpoints intentionally remain out of this document:
    they are synthetic demo routes and must not be mistaken for tenant APIs.
    """
    schema_ref = lambda name: {"$ref": f"#/components/schemas/{name}"}
    parameter_ref = lambda name: {"$ref": f"#/components/parameters/{name}"}
    error = {"$ref": "#/components/schemas/Error"}
    pagination_parameters = [parameter_ref("Page"), parameter_ref("Limit")]
    idempotency_parameters = [parameter_ref("IdempotencyKey")]
    paths: dict[str, Any] = {
        "/api/v1/auth/register": {
            "post": _operation("Bootstrap a local organization administrator.", tags=["authentication"], auth=False,
                               request_schema=schema_ref("RegistrationRequest"), success_code="201", success_schema=schema_ref("TokenResponse"),
                               extra_responses={"409": _response("Email or organization slug already exists.", error)}),
        },
        "/api/v1/auth/login": {
            "post": _operation("Obtain a short-lived bearer token.", tags=["authentication"], auth=False,
                               request_schema=schema_ref("LoginRequest"), success_schema=schema_ref("TokenResponse")),
        },
        "/api/v1/auth/me": {"get": _operation("Get the active tenant membership.", tags=["authentication"], success_schema=schema_ref("Principal"))},
        "/api/v1/members": {
            "get": _operation("List active organization memberships.", tags=["members"], success_schema=schema_ref("MemberList")),
            "post": _operation("Grant an existing account a tenant role.", tags=["members"], request_schema=schema_ref("MembershipRequest"), success_code="201", success_schema=schema_ref("Member"), extra_responses={"403": _response("Administrator role is required.", error), "409": _response("Membership already exists.", error)}),
        },
        "/api/v1/members/{user_id}": {
            "patch": _operation("Change a membership role.", tags=["members"], request_schema=schema_ref("RoleUpdateRequest"), success_schema=schema_ref("Member"), extra_responses={"403": _response("Administrator role is required.", error), "404": _response("Membership was not found.", error)}),
            "delete": _operation("Revoke a membership.", tags=["members"], success_code="204", success_schema={}),
        },
        "/api/v1/ingestions/merchant-days": {
            "post": _operation("Atomically ingest validated merchant-day observations.", tags=["ingestion"], request_schema=schema_ref("MerchantDayIngestionRequest"), success_code="201", success_schema=schema_ref("IngestionBatch"), extra_responses={"409": _response("Idempotency conflict.", error), "422": _response("Observation validation failed.", error)}, parameters=idempotency_parameters),
        },
        "/api/v1/ingestions/merchant-days/csv": {
            "post": {
                **_operation("Ingest a UTF-8 merchant-day CSV.", tags=["ingestion"], success_code="201", success_schema=schema_ref("IngestionBatch"), extra_responses={"409": _response("Idempotency conflict.", error), "422": _response("CSV validation failed.", error)}, parameters=idempotency_parameters),
                "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": {"file": {"type": "string", "format": "binary"}}}}}},
            },
        },
        "/api/v1/ingestions/transactions": {
            "post": _operation("Persist a bounded transaction batch for asynchronous aggregation.", tags=["ingestion"], request_schema=schema_ref("TransactionIngestionRequest"), success_code="202", success_schema=schema_ref("IngestionBatch"), extra_responses={"409": _response("Idempotency conflict.", error), "422": _response("Transaction validation failed.", error)}, parameters=idempotency_parameters),
        },
        "/api/v1/ingestions/{batch_id}": {"get": _operation("Get an ingestion batch and job state.", tags=["ingestion"], success_schema=schema_ref("IngestionBatch"), extra_responses={"404": _response("Batch was not found.", error)})},
        "/api/v1/merchants": {"get": _operation("List tenant merchants with pagination.", tags=["merchants"], success_schema=schema_ref("MerchantList"), parameters=pagination_parameters)},
        "/api/v1/risk-assessments": {
            "post": _operation("Create or return an idempotent model/policy assessment.", tags=["assessments"], request_schema=schema_ref("AssessmentRequest"), success_code="201", success_schema=schema_ref("RiskAssessment"), extra_responses={"404": _response("Merchant-day observation was not found.", error), "422": _response("Assessment request is invalid.", error)}),
        },
        "/api/v1/risk-assessments/{assessment_id}": {"get": _operation("Get a tenant-scoped risk assessment.", tags=["assessments"], success_schema=schema_ref("RiskAssessment"), extra_responses={"404": _response("Assessment was not found.", error)})},
        "/api/v1/risk-assessments/{assessment_id}/feedback": {
            "get": _operation("List human feedback for an assessment.", tags=["feedback"], success_schema=schema_ref("FeedbackList"), extra_responses={"404": _response("Assessment was not found.", error)}),
            "post": _operation("Append immutable assessment feedback.", tags=["feedback"], request_schema=schema_ref("FeedbackRequest"), success_code="201", success_schema=schema_ref("Feedback"), extra_responses={"403": _response("A case-work role is required.", error), "404": _response("Assessment was not found.", error)}),
        },
        "/api/v1/feedback/{feedback_id}/verify": {"post": _operation("Verify human feedback without retraining the model.", tags=["feedback"], success_schema=schema_ref("Feedback"), extra_responses={"403": _response("Administrator role is required.", error), "404": _response("Feedback was not found.", error)})},
        "/api/v1/cases": {"get": _operation("List tenant investigation cases with pagination.", tags=["cases"], success_schema=schema_ref("CaseList"), parameters=pagination_parameters + [{"name": "status", "in": "query", "schema": {"enum": ["OPEN", "IN_REVIEW", "ESCALATED", "RESOLVED", "FALSE_POSITIVE"]}}])},
        "/api/v1/cases/{case_id}": {"get": _operation("Get a tenant investigation case.", tags=["cases"], success_schema=schema_ref("InvestigationCase"), extra_responses={"404": _response("Case was not found.", error)})},
        "/api/v1/cases/{case_id}/events": {"get": _operation("List immutable investigation case events.", tags=["cases"], success_schema=schema_ref("CaseEventList"), extra_responses={"404": _response("Case was not found.", error)})},
        "/api/v1/cases/{case_id}/assignment": {"post": _operation("Assign or unassign a case.", tags=["cases"], request_schema=schema_ref("CaseAssignmentRequest"), success_schema=schema_ref("InvestigationCase"), extra_responses={"403": _response("Assignment is not permitted.", error), "404": _response("Case was not found.", error)})},
        "/api/v1/cases/{case_id}/transitions": {"post": _operation("Transition a case through the constrained workflow.", tags=["cases"], request_schema=schema_ref("CaseTransitionRequest"), success_schema=schema_ref("InvestigationCase"), extra_responses={"403": _response("Transition is not permitted.", error), "404": _response("Case was not found.", error), "409": _response("Transition is not allowed from the current state.", error)})},
        "/api/v1/audit-events": {"get": _operation("List tenant audit events with pagination.", tags=["audit"], success_schema=schema_ref("AuditEventList"), parameters=pagination_parameters)},
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "AI Risk Intelligence Platform tenant API",
            "version": API_VERSION,
            "description": "Authenticated, organization-scoped operational API. Legacy synthetic demo routes are intentionally excluded.",
        },
        "servers": [{"url": "/", "description": "Current deployment origin"}],
        "tags": [{"name": name} for name in ("authentication", "members", "ingestion", "merchants", "assessments", "feedback", "cases", "audit")],
        "paths": paths,
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}},
            "parameters": {
                "Page": {"name": "page", "in": "query", "schema": {"type": "integer", "minimum": 1, "default": 1}},
                "Limit": {"name": "limit", "in": "query", "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50}},
                "IdempotencyKey": {"name": "Idempotency-Key", "in": "header", "required": True, "schema": {"type": "string", "pattern": "^[A-Za-z0-9._:-]{8,128}$"}},
            },
            "responses": {
                "BadRequest": _response("Malformed or invalid request.", error),
                "Unauthorized": _response("Bearer access token is missing, invalid, or expired.", error),
                "RateLimited": _response("Request limit exceeded; observe Retry-After.", error),
                "Unavailable": _response("A required dependency is unavailable.", error),
            },
            "schemas": _schemas(),
        },
    }


def _schemas() -> dict[str, Any]:
    """Schemas shared by the versioned tenant contract."""
    identifier = {"type": "string", "minLength": 1}
    nonempty = {"type": "string", "minLength": 1}
    return {
        "Error": {"type": "object", "required": ["status", "reason", "request_id"], "properties": {"status": {"const": "error"}, "reason": {"type": "string"}, "request_id": {"type": "string"}}},
        "Pagination": {"type": "object", "required": ["page", "limit", "total", "total_pages"], "properties": {"page": {"type": "integer", "minimum": 1}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}, "total": {"type": "integer", "minimum": 0}, "total_pages": {"type": "integer", "minimum": 0}}},
        "Principal": {"type": "object", "required": ["user_id", "organization_id", "role"], "properties": {"user_id": identifier, "organization_id": identifier, "role": {"enum": ["admin", "analyst", "investigator", "viewer"]}}},
        "TokenResponse": {"type": "object", "required": ["access_token", "token_type", "organization", "user"], "properties": {"access_token": {"type": "string"}, "token_type": {"const": "Bearer"}, "organization": {"$ref": "#/components/schemas/Organization"}, "user": {"$ref": "#/components/schemas/User"}}},
        "Organization": {"type": "object", "required": ["id", "slug", "name"], "properties": {"id": identifier, "slug": nonempty, "name": nonempty}},
        "User": {"type": "object", "required": ["id", "email", "role"], "properties": {"id": identifier, "email": {"type": "string", "format": "email"}, "role": {"enum": ["admin", "analyst", "investigator", "viewer"]}}},
        "RegistrationRequest": {"type": "object", "additionalProperties": False, "required": ["email", "password", "organization_name", "organization_slug"], "properties": {"email": {"type": "string", "format": "email"}, "password": {"type": "string", "minLength": 12}, "organization_name": nonempty, "organization_slug": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{2,62}$"}}},
        "LoginRequest": {"type": "object", "additionalProperties": False, "required": ["email", "password", "organization_slug"], "properties": {"email": {"type": "string", "format": "email"}, "password": nonempty, "organization_slug": nonempty}},
        "MembershipRequest": {"type": "object", "additionalProperties": False, "required": ["email", "role"], "properties": {"email": {"type": "string", "format": "email"}, "role": {"enum": ["admin", "analyst", "investigator", "viewer"]}}},
        "RoleUpdateRequest": {"type": "object", "additionalProperties": False, "required": ["role"], "properties": {"role": {"enum": ["admin", "analyst", "investigator", "viewer"]}}},
        "Member": {"type": "object", "required": ["user_id", "email", "role", "is_active", "created_at"], "properties": {"user_id": identifier, "email": {"type": "string", "format": "email"}, "role": {"enum": ["admin", "analyst", "investigator", "viewer"]}, "is_active": {"type": "boolean"}, "created_at": {"type": "string", "format": "date-time"}}},
        "MemberList": {"type": "object", "required": ["members"], "properties": {"members": {"type": "array", "items": {"$ref": "#/components/schemas/Member"}}}},
        "MerchantDayObservation": {"type": "object", "additionalProperties": False, "required": ["merchant_id", "date", "n_transactions", "n_failed", "avg_amount", "amount_std", "n_unique_users", "n_new_users", "n_devices", "distinct_regions_seen"], "properties": {"merchant_id": identifier, "date": {"type": "string", "format": "date"}, "n_transactions": {"type": "integer", "minimum": 0}, "n_failed": {"type": "integer", "minimum": 0}, "avg_amount": {"type": "number", "minimum": 0}, "amount_std": {"type": "number", "minimum": 0}, "n_unique_users": {"type": "integer", "minimum": 0}, "n_new_users": {"type": "integer", "minimum": 0}, "n_devices": {"type": "integer", "minimum": 0}, "distinct_regions_seen": {"type": "integer", "minimum": 0}}},
        "MerchantDayIngestionRequest": {"type": "object", "additionalProperties": False, "required": ["observations"], "properties": {"observations": {"type": "array", "minItems": 1, "items": {"$ref": "#/components/schemas/MerchantDayObservation"}}}},
        "TransactionIngestionRequest": {"type": "object", "additionalProperties": False, "required": ["transactions"], "properties": {"transactions": {"type": "array", "minItems": 1, "items": {"type": "object"}}}},
        "IngestionBatch": {"type": "object", "required": ["id", "status", "received_count", "accepted_count", "rejected_count", "error_summary", "created_at", "idempotent_replay"], "properties": {"id": identifier, "status": {"type": "string"}, "received_count": {"type": "integer", "minimum": 0}, "accepted_count": {"type": "integer", "minimum": 0}, "rejected_count": {"type": "integer", "minimum": 0}, "error_summary": {"type": "array", "items": {"type": "object"}}, "created_at": {"type": "string", "format": "date-time"}, "idempotent_replay": {"type": "boolean"}}},
        "Merchant": {"type": "object", "required": ["id", "merchant_id"], "properties": {"id": identifier, "merchant_id": identifier, "display_name": {"type": ["string", "null"]}}},
        "MerchantList": {"type": "object", "required": ["merchants", "pagination"], "properties": {"merchants": {"type": "array", "items": {"$ref": "#/components/schemas/Merchant"}}, "pagination": {"$ref": "#/components/schemas/Pagination"}}},
        "AssessmentRequest": {"type": "object", "additionalProperties": False, "required": ["merchant_id", "date"], "properties": {"merchant_id": identifier, "date": {"type": "string", "format": "date"}}},
        "RiskAssessment": {"type": "object", "required": ["id", "merchant_id", "date", "status", "risk_score", "risk_level", "model_version", "policy_version", "evidence", "policy_decision", "case_id", "idempotent_replay", "created_at"], "properties": {"id": identifier, "merchant_id": identifier, "date": {"type": "string", "format": "date"}, "status": {"type": "string"}, "risk_score": {"type": ["number", "null"]}, "risk_level": {"type": ["string", "null"]}, "model_version": {"type": "string"}, "policy_version": {"type": "string"}, "evidence": {"type": "object"}, "policy_decision": {"type": "object"}, "case_id": {"type": ["string", "null"]}, "idempotent_replay": {"type": "boolean"}, "created_at": {"type": "string", "format": "date-time"}}},
        "FeedbackRequest": {"type": "object", "additionalProperties": False, "required": ["label", "notes"], "properties": {"label": {"enum": ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "INCONCLUSIVE"]}, "notes": {"type": "string", "minLength": 1, "maxLength": 2000}}},
        "Feedback": {"type": "object", "required": ["id", "risk_assessment_id", "label", "notes", "submitted_by_user_id", "verified", "verified_by_user_id", "verified_at", "created_at"], "properties": {"id": identifier, "risk_assessment_id": identifier, "label": {"type": "string"}, "notes": {"type": "string"}, "submitted_by_user_id": identifier, "verified": {"type": "boolean"}, "verified_by_user_id": {"type": ["string", "null"]}, "verified_at": {"type": ["string", "null"], "format": "date-time"}, "created_at": {"type": "string", "format": "date-time"}}},
        "FeedbackList": {"type": "object", "required": ["feedback"], "properties": {"feedback": {"type": "array", "items": {"$ref": "#/components/schemas/Feedback"}}}},
        "CaseAssignmentRequest": {"type": "object", "additionalProperties": False, "properties": {"assigned_user_id": {"type": ["string", "null"]}}},
        "CaseTransitionRequest": {"type": "object", "additionalProperties": False, "required": ["status", "note"], "properties": {"status": {"enum": ["OPEN", "IN_REVIEW", "ESCALATED", "RESOLVED", "FALSE_POSITIVE"]}, "note": {"type": "string", "minLength": 1}, "final_decision": {"type": ["string", "null"]}}},
        "InvestigationCase": {"type": "object", "properties": {"id": identifier, "status": {"type": "string"}, "merchant_id": identifier, "risk_assessment_id": identifier, "assigned_user_id": {"type": ["string", "null"]}}},
        "CaseList": {"type": "object", "required": ["cases", "pagination"], "properties": {"cases": {"type": "array", "items": {"$ref": "#/components/schemas/InvestigationCase"}}, "pagination": {"$ref": "#/components/schemas/Pagination"}}},
        "CaseEventList": {"type": "object", "required": ["events"], "properties": {"events": {"type": "array", "items": {"type": "object"}}}},
        "AuditEventList": {"type": "object", "required": ["events", "pagination"], "properties": {"events": {"type": "array", "items": {"type": "object"}}, "pagination": {"$ref": "#/components/schemas/Pagination"}}},
    }
