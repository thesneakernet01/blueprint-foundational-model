# AGENTS.md  —  Cloudera + NVIDIA deck builder

A guide for AI agents (and humans) producing customer-facing decks on
the joint **Cloudera + NVIDIA** platform story, built on top of the
official `2025_Cloudera_new_brand_TOOLKIT_PPT.pptx` template.

**This file is the deck-building methodology. The specific domain
content (industry, customer pain, models, schemas, customer logos)
comes from the application sitting in this folder** — read its
`README.md`, architecture docs, source layout, and any business-value
material to derive the topic for each slide. AGENTS.md tells you
*how* to build a deck on-brand; the app tells you *what* to put in it.

---

## Template decks (the starting points)

Two **domain-agnostic templates** under `docs/decks/` are the canonical
starting points for any new deck:

| Template | Purpose |
| --- | --- |
| `Cloudera_NVIDIA_Blueprint_TEMPLATE.pptx` | Technical reference blueprint — 14 slides, fully bracketed placeholders for the workload, modality, and domain |
| `Cloudera_NVIDIA_Business_Value_TEMPLATE.pptx` | Business-value pitch — 21 slides, bracketed placeholders for industry, customer names, statistics, and sources |

Copy one of these into a customer-specific project and fill in the
placeholders.

---

## The concept

When a customer asks "what would this look like in production?",
**answer with the joint Cloudera + NVIDIA platform story, not the
application's internal implementation details.** Specifically:

1. **Customer-facing, not engineering-facing.** The reader is a
   buyer or architect. They care about NIMs, Cloudera AI Inference,
   Cloudera Data Warehouse, SDX, governance, GPU sizing — not the
   HTTP handlers or config files in the repo.
2. **NVIDIA components named explicitly.** Use the same model
   identifiers and product names a customer recognizes (e.g.
   Llama-class NIMs, Riva ASR/TTS, NemoGuard, NeMo Retriever,
   NeMo Guardrails, ace-controller). Derive the *specific* model
   names from the app's source/config — don't make them up.
3. **Cloudera services named explicitly.** Cloudera AI Inference
   (CAI), Cloudera AI Workbench, Cloudera Data Warehouse (CDW) +
   Iceberg, SDX (Ranger, Atlas, Knox). One platform, one identity
   plane, one policy plane.
4. **Lead with the customer's regulated boundary.** Whatever
   matters for the target domain — PHI, PII, financial records,
   classified data, IP — call it out clearly. The story is that
   sensitive data never leaves the customer's CDP environment;
   inference runs on customer-owned GPUs; one auth principal
   authorizes inference and warehouse access.
5. **Brand chrome stays Cloudera's.** Cover slide is the indigo
   double-title with blue mosaic. Body slides have the orange
   bottom accent and ©2025 Cloudera footer. Never replace these
   with custom chrome.

If a draft slide is talking about internal class names, framework
nodes, WebSocket details, or config-file keys — it's at the wrong
level. Hoist it up to platform concepts ("agent orchestrator",
"streaming voice pipeline", "model + endpoint registry").

---

## Deriving content from the app

Before writing any slide, scan the app folder for:

- The top-level `README.md` and any architecture or deployment
  document under `docs/` — the canonical story.
- The application source (agent / service / schema modules) — to
  identify which NVIDIA components and Cloudera services are
  actually in use, and what data model lands in the warehouse.
- Runtime configuration and environment-example files — for the
  list of endpoints and model identifiers (use these verbatim in
  stack tables; don't invent IDs).
- Any business-value or marketing material already in the repo —
  for approved proof points, customer logos, and quotes.

If a fact you need isn't in the app, **don't invent it**. Either
ask, or mark the slide as needing input.

---

## The toolkit

Source: `2025_Cloudera_new_brand_TOOLKIT_PPT.pptx` (kept at the repo
root). 91 slides covering covers, transitions, agendas, one/two/three
column layouts, tables, stats, charts. **Don't edit the toolkit
itself** — treat it as immutable source material.

**Canvas.** 10″ × 5.625″ (Google Slides default — not 16:9 widescreen
at 13.333″ × 7.5″). Anything drawn on top of a template slide must
respect this canvas size.

**Fonts.** Arial primary. Consolas for code / DDL / API paths.

**Brand palette** (extracted from the toolkit theme):

| Name | Hex | Use |
| --- | --- | --- |
| INDIGO | `#100045` | Primary dark — covers, emphasis boxes |
| INDIGO_DEEP | `#110046` | Backgrounds with content overlays |
| PURPLE | `#26177B` | Secondary accent |
| BLUE | `#5555F9` | Connectors, secondary callouts |
| BLUE_LT | `#8789FB` | Light accents, panel strokes |
| ORANGE | `#FF550C` | Primary brand orange — emphasis, accent bars |
| ORANGE_LT | `#FE8756` | Highlight text on indigo |
| GRAY | `#A8AFB9` | Footnotes, dividers, secondary labels |
| WHITE | `#FFFFFF` | Cover text, panel fills |
| NEAR_BLACK | `#1A1A26` | Body text on white |
| SOFT_BG | `#F5F5FA` | Subtle panel fill |

---

## Template slides we reuse

These templates cover ~95% of what a customer deck needs. Refer to
them by 1-based toolkit index:

| Toolkit idx | Layout name | What it gives you |
| --- | --- | --- |
| `2` | Cover (double-title) | Large two-line headline · date line · subhead |
| `7` | Transition | Headline + subhead on a transition background — section breaks |
| `8` | Agenda (6-item, white) | "Agenda" headline + subhead + a 6-row table with orange numerals |
| `9` | Agenda (6-item, indigo) | Same shape as `8`, dark background |
| `10` | Agenda (12-item, white) | Same shape as `8`, two-column table |
| `14` | Headline + One Column | Headline · subhead · single body block |
| `16` | Headline + Two Column | Headline · left column · right column · subhead |
| `26` | Headline + Three Column | Headline · three balanced columns · subhead |
| `70` | Table Design 1 | Headline · subhead — table body drawn on top |

For diagrams (architecture, lifecycle flows, schema with callouts),
the one-column template (`14`) is the workhorse: blank out its body
and lay shapes on top.

### Agenda slides

Toolkit agenda slides keep the orange numerals (`1` – `6` or `1` –
`12`) and the gray rule lines in a real PowerPoint table embedded on
the slide. The numerals live in the **left** column of each row and
are part of the slide's design — leave them alone. Each agenda item
goes into the **right** column of its row.

If you need more than six agenda items, switch to the 12-item
template (`10`) rather than adding rows; new rows won't pick up the
orange-numeral styling.

---

## Pitfalls when working with toolkit text shapes

- **Hanging indents and inherited nested-bullet styles.**
  Three-column and two-column body shapes inherit list-style margin
  and indent values from the slide master. If you write flat content
  into them, lines render at mixed levels (heading → bullet →
  sub-bullet → heading again). Reset paragraph margins and bullets
  to zero/none on every paragraph you write, not just the first.
- **Text overflow.** Each template body shape has a fixed height.
  If content runs past it, text bleeds under the orange accent bar
  / Cloudera logo. Either trim the content or use a smaller body
  size on dense slides.
- **Preserve the captured style.** When replacing text in a
  template shape, keep the existing first-run font/size/color so
  the new text stays on-brand. Don't bypass it unless you're laying
  out per-paragraph styles manually (see "Structured paragraphs"
  below).
- **Overlay coordinates are 10″ × 5.625″.** Anything drawn over a
  template — diagram boxes, callouts, tables — must use coordinates
  inside that canvas. Geometry designed for a 16:9 widescreen
  canvas will run off the right edge and into the Cloudera logo.
- **Big-stat textboxes are narrow.** When swapping a hero stat,
  match the character width of the original (e.g. `XX.X%` is the
  same width as `20.9%`). Adding brackets like `[XX.X%]` makes the
  placeholder wider than the original and forces a wrap.
- **Table cells behave slightly differently from shapes.** When
  walking content, table cells expose a text frame directly even
  though they're not "shapes" in the same sense. Treat both
  uniformly when iterating.

---

## Structured paragraphs in a single text frame

When a column needs more than a flat list — e.g. a column heading,
three labeled sub-sections each with its own label + body — don't
fight the template's bullet inheritance. Clear the text frame and
write the paragraphs one at a time with explicit styling, building
hierarchy from **font size, weight, and color** rather than from
indentation.

A clean pattern for a labeled-section column:

- **Column heading** — 22pt, bold, INDIGO
- *(empty 6pt "gap" paragraph for breathing room)*
- **Sub-label** — 10pt, bold, ORANGE, uppercase
- **Body** — 13pt, regular, NEAR_BLACK
- *(empty 6pt gap)*
- next sub-label / body …
- repeat

Three font sizes plus the orange/indigo contrast carry the
hierarchy. Small (6pt) empty paragraphs give vertical breathing
room without needing more height in the body shape.

---

## Slide-content rubric

For each slide, ask:

1. **Whose vocabulary is this in?** A buyer or architect, not an
   application engineer. Replace internal class names with platform
   concepts ("LangGraph node graph" → "agent orchestrator").
2. **Is there a Cloudera service and an NVIDIA component named?**
   Most slides should name at least one of each. A slide that names
   neither is probably too generic or too app-specific.
3. **Is the regulated boundary visible?** Whatever the customer's
   sensitive-data class is, call out where it sits, who authorizes
   access, and whether weights or data leave the cluster.
4. **Does it cite proof?** Business-value slides should anchor stats
   to a named source (analyst report, customer case study, vendor
   benchmark). Don't invent numbers, and don't drop a stat without
   its source on the same slide.
5. **Does it fit one screen?** If a column needs more than ~8 short
   lines, split it or pick a different template.

---

## Recommended slide flows

These are structural skeletons — the topics each slide covers
should come from the app in this folder.

### Technical blueprint (≈14 slides)

1. **Cover** — joint blueprint title
2. **Agenda** — 5–6 section headers
3. **Why now** — three drivers / pressures relevant to the domain
4. **Joint reference architecture** — three-band diagram (edge ·
   application plane · CAI + CDW + SDX)
5. **NVIDIA AI Enterprise stack** — table mapping the NIMs in use
   to where they run
6. **Cloudera platform services** — three columns
   (inference / app / governed storage)
7. **Data sovereignty & governance** — boundary / identity /
   policy / audit
8. **Workload pipeline** — the customer-visible flow (e.g. voice,
   document, telemetry) using NVIDIA components
9. **Agentic / workload patterns** — patterns this blueprint
   enables, grounded in NeMo Retriever or equivalent
10. **Why Cloudera AI Inference** — platform consolidation argument
11. **Governed data model** — DDL + callouts (write path,
    multi-tenant, query, governance)
12. **Deployment topology** — GPU pool / storage / network /
    resilience
13. **Operating model** — Develop → Register → Deploy → Observe
    + lifecycle columns
14. **Customer deployment journey** — Prove → Relocate → Ground →
    Scale (or equivalent phasing)

### Business value (≈20 slides)

Skeleton, in order:

- **Cover** — "Private AI for *<domain>* — the business case for
  Cloudera + NVIDIA"
- **Agenda** — 4–6 sections
- **Section: the domain reality** — pain stats with named sources
- **Section: why on-prem private AI** — sovereignty / sensitive-data
  exposure / vendor risk
- **Section: the joint platform** — bring AI to your data,
  workload-native on Cloudera, NIM-served stack
- **Section: what it buys you** — analyst proof points, benchmark
  numbers, deployment journey
- **Section: proof** — customers using Cloudera + NVIDIA in this
  domain today (only use customer names from current Cloudera
  marketing material)
- **Thank You**

Stat slides use the toolkit's stat layouts. Cite the source on the
same slide in small gray type.

---

## Customizing a template for a new project

The intended workflow when starting a new customer-specific deck:

1. **Copy the template** that matches your purpose:
   - Technical → `Cloudera_NVIDIA_Blueprint_TEMPLATE.pptx`
   - Business value → `Cloudera_NVIDIA_Business_Value_TEMPLATE.pptx`
2. **Read the application** sitting alongside this AGENTS.md
   (README, architecture docs, source) and pull out the specifics
   that fill each placeholder. See "Deriving content from the app"
   above for the search list.
3. **Replace placeholders, not structure.** Every `[bracketed term]`,
   `XX%`, `XX×`, `n=XX`, `[Month Year]`, `[Customer]`, `[SPEAKER
   NAME]` is a fill-in. Don't add new sections until the templated
   ones are satisfied. Don't remove placeholders without filling
   them — an unfilled placeholder is more honest than a deleted slot.
4. **Render and review.** After every batch of edits, open the deck
   and walk slide-by-slide. Never ship without rendering, because
   the actual layout engine resolves shape sizes slightly
   differently from the editor's reported geometry.
5. **Verify nothing was invented.** Walk the deck once more with
   the "Slide-content rubric" checklist; cross-reference every stat
   and customer name back to the app or the source it cites.

---

## What NOT to do

- Don't write the deck in plain Markdown and convert. The toolkit's
  brand chrome only survives if you build on its slides.
- Don't replace the Cloudera logo or footer. They're load-bearing
  for brand compliance.
- Don't introduce new colors. The eleven swatches above are the
  whole palette.
- Don't use emoji or decorative icons. The toolkit is intentionally
  flat-graphical — emoji read as off-brand.
- Don't paste model names, endpoints, or API paths without
  provenance. Pull them from the app's source/config; don't invent.
- Don't invent customer names, logos, or stats. Customer references
  on proof slides must come from current Cloudera marketing
  material; stats must cite their source on the same slide.
- Don't bake the application's domain (industry, vertical, dataset)
  into AGENTS.md itself. Domain content belongs in the app's
  `README.md` and gets pulled into each deck at build time.
