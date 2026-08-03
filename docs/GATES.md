# Stage gates — definition of done

Every handoff in the [operating model](./OPERATING_MODEL.md#3-the-development-process) is a
**gate**: work advances only when its checklist is met. Each phase has exactly **one
accountable owner** (the `A` in the [RACI](./OPERATING_MODEL.md#4-who-does-what)). Use these
checklists in reviews.

> Only high-value, repeatable use cases advance. When in doubt, park it (Watchlist) and re-score.

---

## Gate 1 — Discover → Qualify  · Accountable: Field & SME
- [ ] Candidate captured in the backlog with its **sourcing input** (lost-deal/RFP, support tickets, value-chain, competitive gap, analyst/regulatory).
- [ ] Business problem and the **KPI it moves** stated in one sentence.
- [ ] Candidate **typed**: net-new (greenfield) · enhancement (extend) · modernization (migrate).

## Gate 2 — Qualify → Architect  · Accountable: Product
- [ ] [`docs/business-case/scorecard.md`](./business-case/scorecard.md) completed.
- [ ] **Weighted score ≥ 4.0 / 5** (else Watchlist 3.0–3.9 = refine & re-score; < 3.0 = park).
- [ ] **No knockout**: data availability and technical feasibility each scored > 1.
- [ ] Repeatability across ≥ 2 accounts argued.

## Gate 3 — Architect → Build  · Accountable: Forge Eng (architects)
- [ ] Architecture in [`docs/architecture/`](./architecture/) conforms to the [build standard](./OPERATING_MODEL.md#5-the-build-standard) (Ingest → Lakehouse → Process → AI → Serve).
- [ ] Any deviation from the standard stack recorded as an ADR (this repo: [ADR-001](./architecture/ADR-001-root-packages.md)).
- [ ] Data contracts identified (here: the transaction schema in `tfm_demo/config.py` column views + the VAST/Impala Parquet layout).
- [ ] Serving approach identified for the AI layer (here: in-process GPU inference — TFM decoder + XGBoost heads — behind FastAPI on one GPU container).

## Gate 4 — Build → Harden  · Accountable: Forge Eng
- [ ] Every layer implemented and **wired** (install → fetch model → prepare data → serve).
- [ ] **Runs clean from the repo alone**: the five AMP tasks succeed in a fresh CML project (GPU runtime); the Application binds `CDSW_APP_PORT`.
- [ ] Governance packaged: the TFM + heads model card in `governance/model-cards/`.
- [ ] No secrets in the repo (`.vast.env` with VAST keys is untracked; Impala/S3a settings documented, not committed).

## Gate 5 — Harden → Publish  · Accountable: Professional Services
- [ ] Verification green: a raw transaction round-trips tokenize → embed → score in REAL mode; DEMO-FALLBACK works without the checkpoint (see [`tests/README.md`](../tests/README.md)).
- [ ] Performance / cost validated at expected scale (single GPU container; embedding throughput + head latency recorded in the UI).
- [ ] Runbook + limitations documented ([`APP_GUIDE.md`](./APP_GUIDE.md); model card "Limitations & risks").

## Gate 6 — Publish → Deployed  · Accountable: Enablement & Partners
- [ ] Listed in the accelerator catalog; enablement material ready (VAST decks + runbooks in [`presentations/`](./presentations/)).
- [ ] Field-feedback channel wired back to the backlog (the loop that refreshes the catalog).
