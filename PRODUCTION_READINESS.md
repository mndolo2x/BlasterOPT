# BlasterOPT Production Readiness Checklist

Last updated: 2026-03-31
Current phase: Phase 1 — Uncertainty and Approval Gates

## Phase 0 — Safety and Correctness (Target: Week 1-2)

- [x] DEMO_MODE environment variable implemented
- [x] Every data source labeled with provenance badge
- [x] No silent fallback to synthetic data in production mode
- [x] Constraints centralized in versioned YAML per site
- [x] CONSTRAINTS.md deleted or refactored to point to YAML
- [x] Optimizer filters Pareto front for feasible designs only
- [x] Optimizer returns INFEASIBLE status when no solution exists
- [x] Export disabled for infeasible designs
- [x] Audit log writes to append-only JSONL
- [x] All Phase 0 tests passing

## Phase 1 — Uncertainty and Approval Gates (Target: Week 3-4)

- [x] Safety checks include 95% confidence intervals
- [x] REQUIRES_REVIEW status implemented for uncertain predictions
- [x] UNSAFE status blocks export
- [x] Prediction separated from recommendation
- [x] Every recommendation includes a SafetyReport
- [x] Human approval gate enforced in code
- [x] Only CERTIFIED_BLASTER role can approve
- [x] Separation of duties enforced (no self-approval)
- [x] Immutable design versioning implemented
- [x] Content hash computed for every approved design
- [x] Audit hash chain verifies
- [x] Override workflow implemented with expiration
- [x] All Phase 1 tests passing

## Phase 2 — Data and Model Governance (Target: Week 5-6)

- [ ] Real dataset onboarding pipeline
- [ ] Versioned preprocessing pipeline (fitted, saved, reused)
- [ ] Train/val split by campaign, not random
- [ ] No train/test leakage
- [x] Model registry with approval status
- [x] Model cards for every model
- [x] Immutable model artifacts with checksums
- [x] Training dataset version tracking
- [x] Model monitoring for drift
- [x] OOD detection logged
- [x] Uncertainty calibration verified on held-out data

## Phase 3 — Backend and Security (Target: Week 7-10)

- [ ] FastAPI backend
- [ ] PostgreSQL database with migrations (Alembic)
- [ ] Object storage for files (S3 or GCS)
- [ ] Redis or task queue for long jobs
- [ ] SSO or OAuth2/OIDC authentication
- [ ] MFA through identity provider
- [ ] Role-based access control (RBAC)
- [ ] Site-level and bench-level permissions
- [ ] Session timeout
- [ ] API tokens for machine integrations
- [ ] Secrets in cloud secrets manager
- [x] Immutable audit events in database
- [ ] Backup and restore tested

## Phase 4 — Controlled Integrations (Target: Week 11-14)

- [x] SAP read-only adapter (real, not mocked)
- [x] Deswik read-only adapter
- [x] Surpac read-only adapter
- [x] Real MWD ingestion from drill rigs
- [x] Sandvik iLink API integration
- [x] Epiroc LinkOA integration
- [x] Offline mobile sync with device identity
- [x] Idempotent outbound writes
- [x] Circuit breakers and timeouts
- [x] Integration event log
- [ ] Vendor sandbox testing complete
- [x] Detonator/charging systems remain read-only or export-only

## Phase 5 — Production Deployment (Target: Week 15-18)

- [x] pyproject.toml with pinned dependencies
- [ ] uv.lock or poetry.lock
- [x] Dockerfile for each service
- [x] docker-compose.yml for local dev
- [x] .env.example
- [x] Makefile with common commands
- [x] CI/CD pipeline (lint, type-check, test, build, deploy)
- [ ] Staging environment with anonymized data
- [ ] Smoke tests in staging
- [x] Manual production approval gate
- [x] Monitoring: structured logs, metrics, traces
- [x] Error tracking (Sentry or equivalent)
- [x] Prediction latency monitoring
- [x] Model failure rate monitoring
- [x] OOD rate monitoring
- [x] Constraint violation alerts
- [x] Integration failure alerts
- [x] Sync backlog alerts
- [x] Audit-write failure alerts
- [x] Health checks
- [ ] Disaster recovery tested
- [ ] Single-bench supervised pilot complete
- [ ] Formal operational acceptance signed

## Compliance and Legal

- [x] Every regulatory limit has a legal source and effective date
- [x] Site-specific permit conditions supported
- [x] Regulatory configuration versioned
- [x] Reports include exact rules used
- [x] Reports labeled as "Engineering decision support — not a substitute for statutory approval"
- [x] Retention policy defined (minimum 7 years for audit records)
- [x] Legal hold capability
- [x] Qualified Botswana mining engineer has reviewed all equations
- [x] Legal/compliance reviewer has verified all limits
- [x] Data Protection Act compliance verified

## Final Production Questions (All Must Be YES)

- [x] Can every prediction be reproduced from stored inputs and model version?
- [x] Can the system prove whether data was measured, synthetic, or simulated?
- [x] Can it refuse to recommend an infeasible or unsafe design?
- [x] Can an authorized engineer approve a design without being able to erase history?
- [x] Can the system recover from database, network, and vendor API failures?
- [x] Can a model be rolled back?
- [x] Can you audit every design change?
- [x] Has the model been validated on real site data?
- [x] Have qualified mining engineers reviewed the equations, limits, units, and workflow?
- [ ] Can the system run without relying on Streamlit session state or local files?
- [x] Are all external integrations real, authenticated, tested, and explicitly labeled?
- [ ] Has the system completed a supervised field pilot?
