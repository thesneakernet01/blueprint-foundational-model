# FSI demo runbook — run of show, talk tracks, and the two-worlds framing

Presenter guide for demoing the TFM fraud cockpit to a financial-services
audience. Companion docs: `docs/nexus-ltm-design.md` (NEXUS integration),
`README_DEMO.md` (setup/modes).

## Pre-stage the day before (non-negotiable)

1. **Load TabFormer** from the Data dialog — downloads ~2.4 GB and writes the
   three temporal splits to storage. Minutes to tens of minutes.
2. **Run one full Build artifacts** — the first export spends most of its time
   generating embeddings on the GPU; they're cached per data-target
   afterwards, so the live re-run during the demo is fast.
3. Decide the NEXUS posture in the **Build artifacts dialog's off/stub/live
   selector** (no shell needed; applies immediately, the metrics card appears
   after the next export): off, or stub with the explicit caveat that stub
   scores are placeholders (the UI tags them `stub`), or live if access has
   landed — deploy the endpoint right before, delete right after (~$60+/hr;
   live also needs the `NEXUS_*` env vars on the backend).
4. Verify the header badge says **REAL**.

## Run of show

### 1. Open on the result — metrics strip (60 seconds)

Real held-out test numbers: baseline vs. embeddings vs. combined (vs. NEXUS),
with **+X% AP lift over baseline**. AP lift translates directly to fewer false
positives per caught fraud — less customer friction, smaller analyst queues,
lower loss. Plant the one-liner now:

> "Every one of the first three columns is the same XGBoost — the only thing
> that changes is the features. The middle card replaced hand-crafted features
> with what a foundation model learned from raw transaction sequences, and
> that alone is worth the lift you see."

**Sidebar — what "AP lift" means (know this cold, someone will ask):**

- **AP = Average Precision**, the area under the precision–recall curve. As
  the alert threshold sweeps from strict to loose, precision is *"of what we
  flagged, how much was really fraud"* and recall is *"of all the fraud, how
  much did we catch"* — AP averages precision across every level of recall,
  one number for the whole trade-off curve.
- **Why AP and not just AUC:** fraud is ~1-in-1,000. ROC-AUC flatters
  imbalanced problems because it gets credit for correctly ignoring the ocean
  of legitimate transactions (hence everything scores 0.97+). AP only rewards
  ranking the rare fraud above the noise — that's why the AP numbers are
  lower and why the *differences between models* are much bigger in AP. It's
  the honest metric for this problem.
- **AP lift** = percentage improvement in AP over the raw-features XGBoost
  baseline.
- **The business translation (say this one):** *"higher AP means more real
  fraud caught per alert raised — the same alert budget catches meaningfully
  more fraud, or the same catch rate generates far fewer false positives
  annoying customers and clogging the analyst queue."*

### 2. Data dialog — where the data lives (2 minutes)

Pick the storage backend (S3 object store or Impala/CDW), *Save & test*,
per-split row counts appear. The point: **the data never leaves the governed
platform**, and storage is a pluggable choice, not an architectural
commitment.

### 3. Load TabFormer — mention, don't run (1 minute)

Show the row counts already in place. One point worth making: the splits are
**temporal** — train on the past, test on the future — "because in production
you only ever score tomorrow's fraud with yesterday's training data."

### 4. Build artifacts — live training run (3–5 minutes, cache-warm)

Run it with the dialog open: streaming log + live CPU/RAM/GPU meters are the
show. While it runs, give the full XGBoost-vs-foundation explanation (the log
narrates it — embeddings, PCA, then the heads):

> "Watch the order here. First the decoder foundation model reads every
> transaction the way an LLM reads text — a sequence of tokens — and emits a
> 512-dimensional summary of each one. It was never told what fraud is; it
> just learned the structure of transaction behavior. Then we hand those
> summaries to the exact same XGBoost your team runs today. Your fraud models
> don't get thrown away — they get better inputs."

That last sentence is the FSI hook: additive to the incumbent stack and its
model-risk approvals, not rip-and-replace. When it finishes, the metrics
refresh in place — the lift was just recomputed in front of them.

### 5. Score real transactions — the payoff (5+ minutes, spend the time here)

Click **Real fraud (test set)** → Run inference. Point at the token strip
("that's the transaction as the foundation model sees it — tokens, like words
in a sentence"), then the grouped score bars. Then the move that always
lands: **edit one field** — channel to *Online*, amount up, time to 3 a.m. —
re-run, and the scores and the dot on the embedding map move. Sub-second
single-transaction scoring is the authorization-stream story, not batch.

### 6. Close — embedding map + the two-worlds sweep

The map shows the foundation model organizing transactions into fraud/legit
geometry without ever being told what fraud is. Then do the left-to-right
sweep (below).

**How to read (and narrate) the embedding map:**

- **Fraud examples land inside the tight red cluster** — fraud is
  *distinctive*, so UMAP places it confidently. This is your money shot.
- **Legitimate examples can sit at the rim of the blue cloud.** That's
  expected, not a bug: a generic transaction resembles *everything*, and UMAP
  settles diffuse points at low-density edges. The narration that lands:
  *"what matters isn't where in the normal mass it sits — it's that it's
  nowhere near the fraud cluster."*
- The frame is **pinned to the background extents** — a single outlying live
  point can't rescale the map; it pins to the frame edge and a caption below
  the map says so.
- **Dashed ring = "Expected (batch)"**: when you score an untouched example,
  a hollow ring marks where the export's batch pipeline projected that exact
  row. Live dot on/near its ring = the live scoring path agrees with
  training. A consistently large gap is a diagnostic signal (live-vs-batch
  embedding drift), not something to demo — mention it to engineering, and
  hand-edit any field to hide the ring. Rings appear after the first export
  run with this build.

## The two-worlds framing ("everything XGBoost vs everything foundational")

**Get the subtlety right: three of the four heads ARE XGBoost.** What differs
is what XGBoost gets to see. The comparison isolates the variable — the lift
comes from the representation, not the classifier. The UI now carries this
framing: paradigm chips on the metric cards, grouped score bars (classic ml /
foundation-fed xgboost / foundation model), and a **Compare paradigms** table
(button top-right of the metrics strip) with the qualitative rows + live
numbers side by side.

Choreograph the score bars left/top to right/bottom:

- **Raw head — "everything XGBoost."** Today's world: humans pick 13 columns,
  XGBoost classifies. Knowledge = only this institution's labeled history;
  when fraud mutates, re-engineer features, relabel, retrain.
- **NEXUS — "everything foundational."** No features, no XGBoost: a
  pre-trained Large Tabular Model reads the raw transaction. Arrives already
  knowing tabular structure.
- **Embeddings + Combined — the bridge.** Foundation-model representations
  feeding the XGBoost the customer already governs. This is the adoption
  path: *"same classifier your model-risk team has already approved, better
  inputs — and the fully-foundational column tells you what's coming next."*

Close on the migration story, not a binary bake-off: FSI buyers need a path,
not a rip-and-replace decision.

## Where to call out XGBoost vs foundation (the three moments)

1. **Metrics strip (step 1)** — the one-liner above; plant it and move on.
2. **Build artifacts (step 4)** — the full explanation while the log narrates
   embeddings → PCA → heads.
3. **Scoring (step 5)** — the token strip proves the representation visually;
   if NEXUS is on: "the fourth bar is the only one with no XGBoost at all."

## Honesty footnotes (don't overclaim in the room)

- NEXUS in stub mode = placeholder scores, tagged `stub` in the UI — say so,
  or leave it off.
- All displayed metrics are measured on the held-out temporal test split with
  lift vs. a real XGBoost baseline — this demo's credibility rests on that;
  never substitute slideware numbers.
