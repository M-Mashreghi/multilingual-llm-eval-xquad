# Multilingual Reliability of Small LLMs on XQuAD: Accuracy Is Not Enough

Most QA benchmarks report a single number: is the final answer right? This project evaluates
a small instruction-tuned LLM along several reliability axes at once, in several languages:

- **Correctness:** Exact Match, token F1 and fuzzy match.
- **Faithfulness:** is the answer a span of the context? Is the cited evidence really in the
  context? Is the explanation grounded in the context, or does it bring in outside facts?
- **Language fairness:** does the model do much better in English than in other languages?
- **Cross-lingual consistency:** XQuAD has the *same* question in every language. Does the
  model get it right (or wrong) consistently across languages?
- **Self-verification:** does a modular *Generator → Verifier → Reviser* pipeline, run by the
  same model, improve on answering directly?

The default model is `Qwen/Qwen2.5-0.5B-Instruct`; any Hugging Face chat model can be used.

## Data

[XQuAD](https://huggingface.co/datasets/google/xquad) is 240 paragraphs and 1,190 question–answer
pairs from the SQuAD v1.1 dev set, professionally translated into 11 languages. Because the
rows are parallel, differences between languages are differences in the *model*, not in the
questions. By default the evaluation samples 100 aligned questions in `en, es, de, ar, ru, zh`.

## Methods

**Direct baseline.** One call: context + question → JSON `{answer, evidence, explanation}`.

**Modular pipeline.**

```
            ┌────────────┐   answer, evidence   ┌────────────┐  Supported   ┌──────────────┐
context ──► │ Generator  │ ───────────────────► │  Verifier  │ ───────────► │ final answer │
question    └────────────┘                      └────────────┘              └──────────────┘
                                                      │ Unsupported / Uncertain   ▲
                                                      ▼                           │
                                                ┌────────────┐  corrected answer  │
                                                │  Reviser   │ ───────────────────┘
                                                └────────────┘
```

The Verifier labels each answer **Supported / Unsupported / Uncertain** with a reason, and the
Reviser sees that feedback. All three roles use the same model, so any gain comes from the
structure of the pipeline rather than extra capacity. The idea of checking a model's output
against itself without external resources comes from SelfCheckGPT.

## Metrics

| Metric | Definition |
|---|---|
| EM / F1 | SQuAD-style, after Unicode-aware normalisation; character-level F1 for `zh`/`th` |
| Fuzzy match | `difflib` similarity ratio to the closest gold answer |
| Answer-in-context | Share of answers that are verbatim spans of the context |
| Evidence faithfulness | 1 if the evidence is a verbatim span of the context, otherwise the coverage of its longest matching block |
| Explanation faithfulness | Share of the explanation's content tokens found in the context, question or answer |
| Language bias | Best/worst language, max–min gap, std across languages, English − non-English mean |
| Cross-lingual consistency | For each parallel question, agreement on correctness (F1 ≥ 0.5) across language pairs; share right in all languages / wrong in all |
| Verifier diagnostics | Verdict distribution, revision rate, F1 before vs after revision, initial F1 when the verifier said Supported vs Unsupported |

## How to run

Everything (code, explanations, results and charts) is in a single notebook: **[`multilingual_llm_eval_xquad.ipynb`](multilingual_llm_eval_xquad.ipynb)**.

1. Open it in Jupyter, VS Code or Google Colab (*Runtime → Change runtime type → GPU*).
2. Run the first cell to install dependencies.
3. Adjust the configuration cell if needed, then *Run all*.

Outputs (tables, JSON, charts) are written to `results/`.

## Results

`results/summary.md` holds the full tables. Per-question predictions (including verifier
verdicts and revisions) are in `results/predictions/` for error analysis.

| Method | EM | F1 | Evidence faithf. | Explanation faithf. | EN − non-EN (F1) | Cross-lingual agreement |
|---|---:|---:|---:|---:|---:|---:|
| direct | | | | | | |
| modular | | | | | | |

![Scores by language](results/scores_by_language.png)

## Discussion points

- **Why accuracy isn't enough:** two systems with the same F1 can differ widely in whether
  their evidence and explanations are real. That matters for any user who trusts the
  explanation.
- **Does verification help?** Compare F1 before and after revision on revised rows, and check
  whether "Supported" answers really are more accurate than "Unsupported" ones. If they aren't,
  the verifier is not discriminating.
- **Same model as generator and verifier:** a model tends to share the blind spots of the answer
  it is checking (correlated errors), and a 0.5B model has limited ability to judge faithfulness.
  Self-verification also costs 2–3× the compute of a direct answer.
- **Language bias as a fairness issue:** if non-English users systematically get worse and
  less faithful answers from the same product, that is an equity problem, not just a
  benchmark artefact.

## References

- Liang et al., *Holistic Evaluation of Language Models (HELM)*, 2022. [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
- Wang et al., *DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models*, 2023. [arXiv:2306.11698](https://arxiv.org/abs/2306.11698)
- Manakul et al., *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs*, 2023. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- Artetxe et al., *On the Cross-lingual Transferability of Monolingual Representations* (XQuAD), 2020. [arXiv:1910.11856](https://arxiv.org/abs/1910.11856)

---
*Developed as part of the Large Language Models course, University of Tehran (Spring 2026).*
