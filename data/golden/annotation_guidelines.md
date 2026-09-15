# Golden Set Annotation Guidelines

## Purpose
This golden set is the only source of real, human-verified ground truth in
this project. Every other number in this repo — retrieval similarity,
clustering frequency, TF-IDF classifier accuracy on pseudo-labels — is
either unsupervised or self-referential. This file is what makes the
evaluation results actually mean something.

## Target size
150–250 examples. Below ~150, per-intent breakdowns become too noisy to
trust; above ~250, annotation time stops being worth the marginal
statistical benefit for a personal project.

## Where messages come from
Sample real customer_message values from the brand's processed
conversations file (data/processed/conversations_<brand>.csv) — do NOT
write synthetic messages. Use `src/data/sample.py`'s output as a starting
pool, or sample directly, stratified by cluster (see below).

## Required stratification
Pull examples so the set includes:
- **Multiple examples per intent** — every intent in taxonomy.json should
  have at least a few golden examples, roughly proportional to (but not
  strictly matching) its real-world frequency.
- **A mix of difficulty levels** — don't only pick obvious cases.
- **Ambiguous/edge cases** — messages that plausibly fit two intents, or
  are borderline between AUTO_HANDLE and ESCALATE.
- **Genuine escalation cases** — angry customers, account-specific issues,
  legal/fraud language, anything a real support policy would kick to a
  human regardless of how "confident" a model sounds.

## Column definitions
- **id**: any unique string, e.g. `G001`, `G002`.
- **customer_message**: copied verbatim from the real dataset.
- **intent**: your human judgment of the correct intent — use taxonomy.json's
  names where they clearly fit; write a new label only if truly nothing
  fits (and note it).
- **expected_action**: `AUTO_HANDLE` or `ESCALATE` — your judgment of what a
  good support system should do here, independent of what our system
  predicts.
- **reference_resolution**: a short sentence describing what a genuinely
  correct response would say/do. Doesn't need to be word-for-word.
- **difficulty**: `easy`, `medium`, or `hard` — your subjective read of how
  hard this case is to get right.
- **notes**: anything worth flagging — ambiguity, missing context, a reason
  you second-guessed your own label.

## What NOT to do
- Don't label based on what you think the model WILL predict — label based
  on what's actually correct.
- Don't skip hard/ambiguous cases — they're the most informative ones.
- Don't invent customer messages.

## After annotating
Delete the two `EXAMPLE-*` rows. Run:
```bash
python -m src.evaluation.evaluate
```
Evaluation automatically detects whether `intent` values are populated; if
the file still only contains the example rows, it reports:
```
STATUS: PENDING
REASON: Golden set has not been manually annotated.
```