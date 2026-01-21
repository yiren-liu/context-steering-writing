# Project Memory: context-steering-writing

## Overview
- **Purpose**: Research code for *CoS: Enhancing Personalization and Mitigating Bias with Context Steering* (`cos/`), plus notebooks that demonstrate usage.
- **Core idea**: Context Steering (CoS) compares next-token distributions with/without context and scales the difference by a tunable \(\lambda\).

## Repo layout
- **Library code (migrated)**: `prototype/backend/cos/`
  - **Key implementation**: `prototype/backend/cos/core.py`
  - **Utilities / prompt+context formatting**: `prototype/backend/cos/utils.py`
  - **Model path configuration**: `prototype/backend/cos/model_paths.py`
- **Notebooks (primary demos)**: `notebooks/` (e.g., `apply_cos.ipynb`, `context_steering.ipynb`)
- **Tests**: `prototype/backend/tests/` (note: may require large local HF model weights)

## Setup / install
- **Python**: 3.10 (per README)
- **Install**:
  - `pip install -e .`
- **Dependencies**: `requirements.txt` (notably `torch`, `transformers==4.44.2`, `accelerate==0.34.2`, `pytest`)

## Model weights / paths
- The code expects **local model directories** under `<repo>/models/` (see `cos/model_paths.py`).
- Recommended workflow is to create a **symlink** from `<repo>/models` to wherever weights actually live.

## Common entrypoints
- **Steered generation (HF)**: `cos/core.py:contextual_steering_hf`
- **Response logprob under CoS (HF)**: `cos/core.py:get_cos_logprob_hf`
- **Multi-context steering (HF)**: `cos/core.py:multi_contextual_steering_hf`
- **Pure math core**: `cos/core.py:apply_cos` / `cos/core.py:apply_multi_cos`

## Project conventions / preferences
- **Absolute paths**: Prefer absolute paths when using Cursor tools in this workspace (per user environment note).
- **Frontend stack for POC**: React + Tailwind + ShadCN UI.
- **Logits extraction**: Avoid `model.generate(output_scores=True)` when you need raw next-token logits; use a forward pass (`prepare_inputs_for_generation` + `model(...)`) since `generate` scores are post-processed and may contain many `-inf`.
- **LLM judge**: Use OpenAI models for judge evaluations.
- **Secrets/config**: It’s OK to load `OPENAI_API_KEY` from a local `./.env` file (best-effort), in addition to standard environment variables.
- **Edit dimensions (current)**: `empathy`, `vividness`, `certainty`

## Paper (parsed in `misc/paper.md`) → code grounding: what CoS is doing

### Core mathematical object (per-token contextual influence)
- **Paper idea**: contextual influence for a token is the difference between next-token likelihoods with vs without context:
  - \(F_{C,P}(x_i) = \log p(x_i \mid C,P, x_{<i}) - \log p(x_i \mid \varnothing,P, x_{<i})\)
- **Code grounding**: `cos/core.py:apply_cos` computes log-probs for `logits` (with context) and `logits_nc` (no-context), subtracts to form `influence`, and then applies the linear scaling by `lambdas` before renormalizing.

### CoS forward model (controllable decoding)
- **Paper idea**: steer by modifying next-token distribution:
  - \(\log p_{\text{CoS},\lambda}(x_i) \propto \log p(x_i \mid C,P) + \lambda \cdot F_{C,P}(x_i)\)
- **Code grounding**:
  - `cos/core.py:contextual_steering_hf` does *two forward passes per token step* (with context vs no context), calls `apply_cos(..., return_probs=True)`, then top-p samples from the steered distribution.

### CoS as a scoring model (logprob of an existing response)
- **Paper idea**: use \(p_{\text{CoS},\lambda}(X \mid C,P)\) for inference tasks (e.g., infer \(\lambda\), infer context).
- **Code grounding**:
  - `cos/core.py:get_cos_logprob_hf` computes tokenwise CoS probabilities across the *response span* (masking prompt tokens), then sums log-probs to produce a scalar log-likelihood.

## Capability Q1: can it mix / add multiple steering distributions?
- **Yes (additive mixing is supported in code).**
- **Code grounding**: `cos/core.py:apply_multi_cos` implements multi-context steering as
  - start from the no-context log-prob `next_lp_nc`
  - for each context direction \(k\): add \(\lambda_k \cdot (\log p_k - \log p_{nc})\)
  - then renormalize with `log_softmax`.
- **Practical API**: `cos/core.py:multi_contextual_steering_hf` + notebook `notebooks/multi.ipynb` demonstrate generating with two contexts and a grid of \((\lambda_a, \lambda_b)\) values.

## Capability Q2: can it support interpretability (e.g., extract keywords from a steering distribution)?
- **Yes, at the “token-level attribution” level, because CoS explicitly computes an influence tensor over the vocabulary.**
- **Code grounding**:
  - In `cos/core.py:apply_cos`, `influence = next_lp - next_lp_nc` is a full `(batch, seq, vocab)` tensor; this is exactly a per-token “how much context increases/decreases log-prob” attribution.
  - In `cos/core.py:apply_multi_cos`, the same influence is computed per context direction (`all_influences` is built, but not currently returned).
- **What’s missing in this repo**: there is no packaged helper that *decodes* “top influenced vocab items” into human-readable “keywords/phrases”.
- **How to do it (conceptually)**:
  - For a given decoding step (or averaged across steps), take `influence[..., vocab]`, select top‑k positive entries, and map token IDs → strings via the tokenizer; optionally merge subword pieces to get keyword-like strings.
