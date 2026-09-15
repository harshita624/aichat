# Report

## Problem Framing

I built the agent for **Tesco** because it has enough resolvable Twitter support conversations for retrieval and evaluation without making local development painfully slow: 38,468 customer-agent pairs from 15,593 distinct customers.

For this brand, "good" means classifying the message into a useful Tesco-specific intent, drafting a short reply grounded in historical Tesco support patterns, avoiding unsupported promises or private-data leakage, and escalating safety, account-specific, angry, ambiguous, or low-confidence messages.

I did not build a production ticketing workflow, multi-turn memory, live account lookup, refund execution, or a fully fine-tuned model. Those would require permissions and labelled operational data not present in the Kaggle dataset.

## System

The final agent combines TF-IDF/KMeans intent discovery, human-readable taxonomy naming, lexical correction, historical response retrieval, LLM reply generation when configured, deterministic privacy-safe fallback generation when no LLM is available, and an escalation policy using intent confidence, retrieval strength, grounding checks, and safety/anger keywords.

The system is intentionally conservative. It is allowed to over-escalate. It should not silently auto-handle risky messages.

## Golden Set

The golden set has 200 hand-labelled Tesco customer messages in `data/golden/golden_set.csv`.

Sampling was stratified from real Tesco conversation data so the set includes frequent intents, rare clusters, easy cases, ambiguous cases, and escalation-heavy cases. Labels include `intent`, `expected_action`, a short `reference_resolution`, difficulty, and notes. The annotation policy is documented in `data/golden/annotation_guidelines.md`.

Intent distribution: product quality 46, customer service complaint 34, oversized mixed cluster 34, delivery issue 24, stock availability 21, DM/status follow-up 12, technical problem 10, meal-deal cluster 7, free-from cluster 7, Tesco-specific catch-all 5.

Escalation labels: 121 escalate, 79 auto-handle.

## Results

| Metric | Trivial baseline | TF-IDF baseline | Final agent |
|---|---:|---:|---:|
| Intent accuracy | 0.170 | 0.480 | **0.785** |
| Intent macro F1 | 0.029 | 0.439 | **0.789** |
| Escalation precision | 0.605 | - | 0.738 |
| Escalation recall | 1.000 | - | 0.909 |
| Escalation F1 | 0.754 | - | **0.815** |
| Missed escalation rate | 0.000 | - | 0.091 |
| False escalation rate | 1.000 | - | 0.494 |

The trivial baseline predicts the most common intent and escalates everything. That gives perfect escalation recall but no useful automation. The simple baseline is a TF-IDF logistic regression trained on pseudo-labels from the unsupervised clusters. The final agent beats both on intent quality and beats the trivial baseline on escalation F1 while still keeping missed escalations below 10%.

## Reply Quality and Judge Agreement

The LLM judge scores generated replies from 1 to 5 on correctness, relevance, grounding, completeness, and helpfulness. It is not treated as ground truth.

The judge successfully scored 115 examples. Mean scores were correctness 4.27, relevance 4.53, grounding 4.09, completeness 3.47, and helpfulness 3.82.

I manually rated the same 115 examples and computed agreement in `results/judge_human_agreement.json`. Spearman correlations were 0.913 to 0.982. Cohen's kappa was 0.748 to 0.952. This is encouraging, but the sample is still small and likely easier than a production audit because it was drawn from the same completed evaluation run.

## Failure Analysis

The automated failure analysis found 74 failures out of 200 examples. The top five are in `results/failure_analysis.md`. The main failure modes are:

1. **Mixed-intent messages.** A tweet can complain about product quality and store equipment in the same message. The current classifier is single-label.
2. **False escalations on benign follow-ups.** Replies like "thanks" or "that's fine" are often escalated because the policy sees weak evidence or a generic cluster.
3. **Complaint subtypes blur together.** Staff complaints, faulty goods, refunds, and pricing complaints often share vocabulary and can cross intent boundaries.
4. **Catch-all cluster is too large.** `OVERSIZED_MIXED_CLUSTER__5` contains 60% of the training messages and is not a satisfying business intent.
5. **Reply fallback is safe but generic.** When LLM calls are skipped or unavailable, the deterministic fallback avoids privacy leakage but is less complete than a good human reply.

## Overfit / Underfit Check

The model does not look classically overfit to the golden labels because the intent model was not trained on those labels. The golden set is used only for evaluation. The final agent beats the TF-IDF baseline on the same held-out human labels, which suggests the lexical corrections and escalation policy add signal rather than memorizing labels.

The bigger risk is underfitting the business taxonomy. The oversized cluster and several `uncategorized__*` labels show that unsupervised clustering has not fully separated real support intents. The system handles this conservatively by escalating low-trust clusters instead of pretending they are solved.

## What Is Misleading About My Headline Number?

Intent accuracy of 0.785 sounds stronger than the system really is. It hides that 34 golden examples belong to an oversized mixed cluster whose label is not operationally clean. It also says little about reply quality or escalation safety.

Escalation F1 of 0.815 is also incomplete. The system still misses 9.09% of messages that humans labelled for escalation. In support automation, those misses matter more than false escalations. The false-escalation rate is high at 49.37%, which means this version is safer than it is efficient.

The leakage story also needs care. The data has many duplicate and near-duplicate tweets. Evaluation excludes the current conversation id and exact message text during retrieval, and the recorded overlap is 0 after that guard. A temporal check still flags leakage because the train/test split is not strictly chronological. I report that rather than hiding it.

Finally, judge results cover 115 of 200 examples, not all 200. They are useful evidence but not a complete reply-quality measurement.

## What I Would Do With One More Week

I would split the oversized cluster into a cleaner taxonomy, move to multi-label intent classification for mixed complaints, tune escalation thresholds on a separate validation slice, add a repeat-complaint signal, finish judge scoring for all 200 examples, and run a second human audit with another annotator to estimate label noise.