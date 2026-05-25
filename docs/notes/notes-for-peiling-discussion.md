# Notes for Peiling Discussion

**Date:** 2026-05-20  
**Project:** `google-factcheck-starter` / FakeNews context study

## Overall take

The idea is **feasible**, but it needs **tight scope**.

The strongest version of the project is:

- an empirical study of how **context quantity** and **context quality** affect misinformation detection
- grounded in the datasets you can actually access
- using controlled context settings such as `minimal`, `full`, and `misleading`

The risk is that the current paper structure tries to do too much at once, especially the **adaptive context selection framework**, unless that part is fully implemented and evaluated.

---

## 1) How much contextual information can we realistically obtain?

### Short answer

We can obtain a **useful amount of context**, but not all context types equally well across all datasets.

### Strongest / most realistic context types

These are the context types the current datasets and pipeline can most plausibly support:

- **Textual context**
     - claim text
     - headlines/titles
     - article or post body
     - captions
     - linked article text
     - OCR/subtitles if available or added later

- **Temporal context**
     - publication date
     - upload/share time
     - review date
     - repost timing if exposed by the dataset

- **Source context**
     - publisher / outlet
     - account / author
     - source URL
     - review publisher
     - account credibility or outlet identity when available

- **External knowledge context**
     - fact-check pages
     - linked articles
     - retrieved evidence
     - structured review metadata

- **Some social context**
     - comments
     - repost/discussion signals
     - reactions/community discussion
     - mostly stronger in social-media datasets than in ClaimReview

- **Some cross-modal context**
     - text-image consistency
     - caption-image mismatch
     - post-image mismatch
     - strongest in multimodal datasets such as Fakeddit and MuMiN

### Weak / partial context types

- **Spatial context**
     - sometimes inferable from place names, landmarks, metadata, or article text
     - usually incomplete or inconsistent

- **Reader-response context**
     - possible only when datasets expose engagement / reaction / discussion patterns
     - probably better treated as partial evidence rather than a universal context type in this paper

### Hardest / least reliable context types

- **Cultural / pragmatic context**
     - sarcasm
     - irony
     - memes
     - cultural references
     - very important in theory, but hard to operationalise consistently from the available dataset fields

---

## 2) Dataset-by-dataset reality check

### ClaimReview

**What it supports well:**

- claim text
- claimant
- review publisher
- textual rating
- dates
- review/source URLs
- structured metadata

**Main limitation:**

- the Google/Data Commons feed does **not include the full article body by default**
- weak social context
- limited multimodal richness unless extra enrichment is added

**Best role in the paper:**

- clean structured baseline
- strong for source/time/review-based context
- weaker as a rich multimodal context source

### Fakeddit

**What it supports well:**

- text + image
- platform/post context
- cross-modal signals
- some social context depending on accessible fields

**Best role in the paper:**

- strong multimodal + platform-context dataset

### FakeNewsNet

**What it supports well:**

- article/news text
- source/publisher metadata
- temporal information
- potentially linked media and some social traces depending on available loader fields

**Best role in the paper:**

- good for article-level and source-level context
- useful for testing whether richer article context helps or hurts

### MuMiN

**What it supports well:**

- social context
- source context
- temporal context
- thread-like / claim-network information
- multimodal/social misinformation framing

**Best role in the paper:**

- probably the strongest fit for the broader “context matters” argument

---

## 3) Is the paper idea feasible?

### Yes — if scoped carefully

The core idea is strong:

- more context is **not always better**
- misleading or noisy context can hurt
- there may be a **saturation point**
- this matters for misinformation detection and RAG-style systems

This aligns well with the current repo, which already supports:

- 4 datasets
- a unified experimental pipeline
- context variants: `minimal`, `full`, `misleading`
- evaluation outputs
- heuristic + API-based inference

So the project is definitely feasible as a controlled empirical study.

---

## 4) Main issue with the current paper structure

The draft currently mixes:

- **research plan**
- **expected findings**
- and **already completed findings**

For example, parts of the draft say things like:

- “our results reveal...”
- “we conduct controlled experiments...”
- “further analysis shows...”

If those experiments are not finished yet, that wording sounds too strong.

### Suggested fix

Separate these clearly:

- **motivation**
- **research questions**
- **hypotheses**
- **methodology**
- **results** only after experiments are actually run

---

## 5) Strongest version of the paper

The strongest paper is:

### Core contribution

A controlled study of how:

- **context quantity**
- **context quality**
- and possibly **model choice**

affect misinformation detection performance.

### Why this works

It matches what the pipeline can realistically test now.

The `minimal`, `full`, and `misleading` context setup is already a strong foundation for this.

---

## 6) What is probably too ambitious right now

These parts feel like second-phase work unless they are fully implemented:

- a full **adaptive context selection framework**
- strong learned retrieval filtering
- broad causal claims about why saturation happens
- full reader-response modelling
- equal treatment of all context types across all datasets

The paper becomes much cleaner if the adaptive selection idea is framed as:

- a lighter extension, or
- future work

rather than the central novelty unless there is a full implementation.

---

## 7) Suggested discussion points for tomorrow

### For Question 1

> Based on the datasets, we can realistically obtain strong textual, temporal, source, and external evidence context, plus varying amounts of social and cross-modal context depending on the dataset. Spatial, cultural/pragmatic, and reader-response context are much less consistently available and would likely require additional enrichment or annotation.

### For Question 2

> The idea is feasible if we scope it as a controlled study of context quantity and quality in misinformation detection. The strongest immediate contribution is to show whether performance improves up to a point and then degrades under excessive or misleading context. The adaptive context selection framework is interesting, but it may be better treated as a secondary extension or future direction unless it is fully implemented and evaluated.

---

## 8) Concrete feedback on structure

### Keep

- the “context saturation” concept
- the distinction between relevant / ambiguous / misleading context
- the signal-vs-noise framing

### Revise

- make dataset/context availability explicit
- avoid writing findings as if already confirmed unless experiments are done
- narrow the adaptive retrieval contribution unless it is implemented

### Add

A section like **“Obtainable context by dataset”** would help a lot. For example:

- ClaimReview: structured metadata + source/review context
- Fakeddit: multimodal + platform context
- FakeNewsNet: article/source context
- MuMiN: social/thread/source context

That would ground the paper much better.

---

## 9) Honest verdict

**Good idea, definitely feasible, but it needs tighter scope and more careful wording.**

If framed as:

- a controlled experiment
- context quantity + quality analysis
- dataset-aware context availability
- modest claims

then it is strong.

If it also tries to be:

- a retrieval-method paper
- a full multimodal context theory paper
- a broad benchmark paper

then it risks overreaching.
