# Module 1 — The Paradigm Shift in Fraud Detection (10 min)

**Goal:** by the end of this module the audience can explain, in one sentence, why a
sequence-pretrained foundation model beats point-in-time features for card fraud — and
knows the demo will *prove* it with a live number, not a claim.

---

## Slide 1.1 — Traditional vs. sequence modeling (3 min)

**On the slide**

| Legacy approach | Sequence / foundation-model approach |
| --- | --- |
| Static rule engines ("amount > $500 AND online AND foreign ZIP") | Model pretrained on millions of *transaction sequences* |
| Hand-crafted point-in-time features per transaction | Each transaction scored **in the context of the card's history** |
| Rules decay; analysts chase fraudsters | Behavioral representation transfers across tasks |
| Every new pattern = new feature engineering sprint | New signal is already latent in the embedding |

**Speaker notes**

- Open with the pain: fraud teams maintain thousands of rules and hundreds of
  hand-crafted features, and the models plateau — they miss the *sequential, behavioral*
  signal in a card's history. A $40 online purchase at 3 a.m. is meaningless in
  isolation; against this cardholder's history it can be a screaming anomaly.
- Frame the shift the same way NLP shifted: bag-of-words → language models. Fraud is now
  making the same jump: point features → transaction sequences.
- Tee up the proof: "In 40 minutes you'll click a button and watch the lift get measured
  live on GPU, on Cloudera AI. This is not a slideware number."

---

## Slide 1.2 — What is a Transaction Foundation Model? (4 min)

**On the slide**

- A decoder-only transformer (LLaMA-style architecture) that reads **transactions as
  tokens**, the way a language model reads words.
- Pretrained by NVIDIA on payment sequences; this demo uses the public **IBM TabFormer**
  dataset: **~24 million transactions across 20,000 simulated cardholders** (~2.4 GB).
- Output: a dense **512-dimensional behavioral embedding** per transaction.
- Remarkably small: the checkpoint is **~56 MB** — 8 transformer layers, hidden size
  512, vocab 6,251, context up to 8,192 positions.

**Speaker notes**

- Land the analogy hard: *sentence : words :: card history : transactions*. The
  tokenizer maps fields (amount, merchant, MCC, city, time-of-day, channel…) into a
  6,251-token vocabulary; the decoder predicts what comes next in the sequence, and in
  doing so learns what "normal" looks like for a cardholder.
- The ~56 MB size is a talking point, not a footnote: this is not a 70B-parameter LLM.
  It fits comfortably on a single **NVIDIA L4 (24 GB)** alongside the whole training
  pipeline — foundation-model value at commodity-GPU cost.
- Source anchor if asked: checkpoint fetched by `pipelines/fetch_model.py` from the
  NVIDIA `transaction-foundation-model` blueprint repo; config in
  `models/decoder-foundation-model/config.json`.

---

## Slide 1.3 — Behavioral context: what rules can't see (3 min)

**On the slide**

- Rules and point features see **one row**. The pretrained decoder has internalized
  **multi-transaction history and temporal patterns**.
- The embedding encodes: spend cadence, merchant mix, channel habits, time-of-day
  rhythm — *without anyone engineering those features*.
- The demo's claim, measured live: embeddings deliver a **double-digit lift in Average
  Precision (AP)** over the hand-crafted-feature baseline, with AUC gains alongside.

**Speaker notes**

- Explain why AP is the fraud metric that matters: fraud is heavily imbalanced (~1% of
  rows), and AP measures precision across the recall curve — exactly what an analyst
  queue experiences. AUC alone flatters imbalanced models.
- Preview the benchmark design (Module 2 details it): the demo never asks you to trust
  the foundation model — it trains a **raw-features-only baseline side-by-side** and
  shows the delta as a percentage on screen.
- Transition: "So how does a raw transaction become a 512-number behavioral fingerprint?
  That's Module 2."
