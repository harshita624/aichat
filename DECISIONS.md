# Decision Log

1. **Picked Tesco, not the largest raw brand.** Tesco had 38,468 resolvable customer-agent pairs and high customer diversity while still staying practical for local development.

2. **Used a brand-specific taxonomy.** Generic banking or ecommerce labels would have been cleaner, but the assignment asks for intents defined from this brand's data.

3. **Used unsupervised clustering before naming intents.** This kept the taxonomy grounded in actual Tesco messages instead of invented categories.

4. **Kept ugly cluster names when the data stayed ugly.** Labels like `OVERSIZED_MIXED_CLUSTER__5` are not pretty, but renaming them into confident business terms would misrepresent what the cluster contains.

5. **Did not fine-tune a model.** The labelled set has 200 examples, which is enough for evaluation but not enough to justify fine-tuning without overfitting.

6. **Chose retrieval-augmented replies over direct generation.** Retrieved examples make the reply auditable and show how similar Tesco issues were historically handled.

7. **Never copy historical replies verbatim in fallback mode.** Real support tweets can contain names, addresses, account details, and stale promises, so the fallback response is generic and privacy-safe.

8. **Made LLM usage optional.** The repo should run for a reviewer without API keys; LLM calls improve reply generation and judging but are not required for the headline metric reproduction.

9. **Added TF-IDF retrieval fallback.** If the sentence-transformer model is not cached, the agent still runs locally instead of failing during a demo.

10. **Evaluated against hand labels, not pseudo-labels.** The intent model is built from clusters, but final metrics use the manually labelled golden set.

11. **Used two baselines.** The trivial baseline shows the value of doing anything beyond majority-class prediction; the TF-IDF baseline shows whether the final system beats a simple practical model.

12. **Reported missed escalation separately.** Escalation accuracy or F1 hides the cost difference between over-escalating and wrongly auto-handling a risky case.

13. **Excluded self-retrieval during evaluation.** Golden examples came from the same corpus used for retrieval, so evaluation removes the source conversation id and exact message text at query time.

14. **Kept judge and human ratings separate.** The LLM judge is a useful scalable signal, but the report only treats human labels as ground truth.

15. **Cached expensive evaluation outputs.** This lets reviewers reproduce headline metrics quickly and prevents rate limits from forcing a full rerun.