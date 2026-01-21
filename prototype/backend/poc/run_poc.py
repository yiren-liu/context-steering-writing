from __future__ import annotations

import argparse
from typing import Dict, List

import torch

from cos.core import multi_contextual_steering_hf
from cos.utils import load_hf_model_and_tokenizer

from .contexts import DEFAULT_CONTEXTS, contexts_as_strings, lambdas_from_slider
from .data import EditExample, load_jsonl, split_by_user, group_by_user
from .method_a import BayesContrastiveModel
from .prompt_baseline import build_style_profile
from .eval import evaluate_method_a, evaluate_prompt_baseline


def _build_profiles(train_by_user: Dict[str, List[EditExample]]):
    return {user_id: build_style_profile(user_id, edits) for user_id, edits in train_by_user.items()}


def _avg_user_mean(model: BayesContrastiveModel) -> torch.Tensor:
    if not model.user_models:
        return model.prior_mean
    means = [u.posterior.mean for u in model.user_models.values()]
    return torch.stack(means, dim=0).mean(dim=0)


def _run_demo(
    model,
    tokenizer,
    bayes_model: BayesContrastiveModel,
    user_id: str | None,
    prompts: List[str],
    slider_values: List[int],
    is_chat: bool,
):
    contexts = contexts_as_strings()
    if user_id and user_id in bayes_model.user_models:
        mean = bayes_model.user_models[user_id].posterior.mean.tolist()
    else:
        mean = _avg_user_mean(bayes_model).tolist()

    all_lambdas = []
    for slider in slider_values:
        all_lambdas.append(lambdas_from_slider(slider, mean))
    all_lambdas = list(map(list, zip(*all_lambdas)))

    all_contexts = [ [c for _ in prompts] for c in contexts ]
    output = multi_contextual_steering_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        all_contexts=all_contexts,
        all_lambdas=all_lambdas,
        is_chat=is_chat,
        show_progress=False,
        max_gen_len=128,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POC for Bayesian contrastive vs prompting baseline.")
    parser.add_argument("--data", required=True, help="Path to JSONL edit traces.")
    parser.add_argument("--model", default="llama-2-7b-chat", help="Model key in cos.utils.")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--prior_var", type=float, default=4.0)
    parser.add_argument("--no_chat", dest="is_chat", action="store_false", default=True)
    parser.add_argument("--demo_prompts", default="", help="Comma-separated prompts for slider demo.")
    parser.add_argument("--demo_user_id", default="", help="User id to use for demo lambdas.")
    parser.add_argument("--demo_slider", default="1,4,7", help="Comma-separated slider values.")
    args = parser.parse_args()

    examples = load_jsonl(args.data)
    train, val = split_by_user(examples, val_fraction=args.val_fraction)
    train_by_user = group_by_user(train)

    model, tokenizer = load_hf_model_and_tokenizer(args.model)
    contexts = contexts_as_strings(DEFAULT_CONTEXTS)

    k = len(contexts)
    prior_mean = torch.zeros(k, device=model.device)
    prior_cov = torch.eye(k, device=model.device) * args.prior_var

    bayes_model = BayesContrastiveModel(
        contexts=contexts,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        beta=args.beta,
        is_chat=args.is_chat,
    )

    for ex in train:
        bayes_model.update_user(model, tokenizer, ex)

    profiles = _build_profiles(train_by_user)

    metrics_a = evaluate_method_a(model, tokenizer, val, bayes_model)
    metrics_b = evaluate_prompt_baseline(model, tokenizer, val, profiles, is_chat=args.is_chat)

    print("=== Bayesian Contrastive CoS ===")
    for k, v in metrics_a.items():
        print(f"{k}: {v:.4f}")
    print("=== Prompt Baseline ===")
    for k, v in metrics_b.items():
        print(f"{k}: {v:.4f}")

    if args.demo_prompts:
        prompts = [p.strip() for p in args.demo_prompts.split(",") if p.strip()]
        sliders = [int(s) for s in args.demo_slider.split(",") if s.strip()]
        output = _run_demo(
            model=model,
            tokenizer=tokenizer,
            bayes_model=bayes_model,
            user_id=args.demo_user_id or None,
            prompts=prompts,
            slider_values=sliders,
            is_chat=args.is_chat,
        )
        for prompt, gen in zip(output["prompts"], output["generation"]):
            print(f"\nPrompt: {prompt}\nGeneration: {gen['content']}")


if __name__ == "__main__":
    main()
