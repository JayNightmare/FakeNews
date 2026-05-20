# Short Notes for Peiling Discussion

**Date:** 2026-05-20

## 1) How much contextual information can we realistically obtain?

### Strongest context we can realistically support

- **Textual context**: claim text, headline/title, article/post body, captions
- **Temporal context**: publication date, upload/share time, review date
- **Source context**: publisher, outlet, account, source URL, review publisher
- **External knowledge context**: fact-check pages, linked articles, retrieved evidence
- **Some social context**: comments, repost/discussion signals, reactions when available
- **Some cross-modal context**: text-image consistency or mismatch in multimodal datasets

### Weaker / less consistent context

- **Spatial context**: sometimes inferable, but usually incomplete
- **Reader-response context**: only strong when datasets expose engagement/interpretation signals
- **Cultural / pragmatic context**: sarcasm, irony, memes, cultural references are important but hard to operationalise reliably

## 2) Dataset-level summary

### ClaimReview

- Strong for structured metadata, source context, review context, and time context
- Weak for rich multimodal or social context
- Important limitation: the feed does **not include full article body by default**

### Fakeddit

- Strong for multimodal and platform/post context
- Useful for cross-modal consistency experiments

### FakeNewsNet

- Strong for article text, source/publisher context, and temporal information
- Good for testing whether richer article context helps or hurts

### MuMiN

- Strong for social context, thread/network context, source context, and temporal context
- Probably the best fit for the broader “context matters” framing

## 3) Is the idea feasible?

**Yes, but only if it is scoped carefully.**

The strongest version of the paper is:

- a controlled study of **context quantity** and **context quality**
- across the datasets that are actually available
- using settings like `minimal`, `full`, and `misleading`

This already fits the current repo well.

## 4) Main concern with the draft

The draft sometimes sounds like the results are already confirmed.

It should more clearly separate:

- motivation
- research questions
- hypotheses
- methodology
- actual findings

If the experiments are not complete yet, wording such as “our results reveal” is too strong.

## 5) Best recommendation

Keep the paper focused on:

- whether **more context helps up to a point**
- when **misleading/noisy context starts to hurt**
- whether different models react differently

Be more careful about presenting the **adaptive context selection framework** as a major contribution unless it is fully implemented and evaluated.

## 6) Short verdict

**The idea is feasible and interesting, but it will be much stronger if it is framed as an empirical study of context quantity/quality rather than trying to also be a full retrieval-method paper.**
