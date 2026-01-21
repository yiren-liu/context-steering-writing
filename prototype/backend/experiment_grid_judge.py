from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Sequence, Tuple

import torch
import tqdm

from cos.core import multi_contextual_steering_hf
from cos.utils import load_hf_model_and_tokenizer
from poc.contexts import DEFAULT_CONTEXTS, slider_to_lambda
from poc.data import load_jsonl


def _select_dimensions(dim_names: Sequence[str]) -> Tuple[List[str], List[str]]:
    context_map = {c.name: c.instruction for c in DEFAULT_CONTEXTS}
    missing = [d for d in dim_names if d not in context_map]
    if missing:
        raise ValueError(f"Unknown dimension names: {missing}. Available: {list(context_map.keys())}")
    instructions = [context_map[d] for d in dim_names]
    return list(dim_names), instructions


def _parse_slider_values(values: str) -> List[int]:
    out = []
    for v in values.split(","):
        v = v.strip()
        if not v:
            continue
        out.append(int(v))
    return out


def _build_baseline_prompt(base_prompt: str, dim_names: List[str], dim_values: List[int]) -> str:
    lines = [
        "Apply the following writing style settings:",
    ]
    for name, value in zip(dim_names, dim_values):
        lines.append(f"- {name}: {value}/7")
    if base_prompt:
        lines.append("")
        lines.append(base_prompt)
    return "\n".join(lines)


def _generate_baseline(
    model,
    tokenizer,
    prompts: List[str],
    dim_names: List[str],
    dim_values: List[int],
    samples_per_setting: int,
    max_new_tokens: int,
) -> List[str]:
    outputs = []
    for prompt in prompts:
        style_prompt = _build_baseline_prompt(prompt, dim_names, dim_values)
        for _ in range(samples_per_setting):
            dialog = [{"role": "user", "content": style_prompt}]
            inputs = tokenizer.apply_chat_template(
                dialog,
                tokenize=True,
                return_tensors="pt",
                padding=False,
                return_dict=True,
            ).to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            prompt_len = inputs.input_ids.shape[1]
            gen_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
            outputs.append(gen_text.strip())
    return outputs


def _generate_cos(
    model,
    tokenizer,
    prompts: List[str],
    dim_indices: List[int],
    dim_values: List[int],
    samples_per_setting: int,
    max_new_tokens: int,
) -> List[str]:
    k = len(DEFAULT_CONTEXTS)
    base_lambda = [0.0] * k
    for idx, val in zip(dim_indices, dim_values):
        base_lambda[idx] = slider_to_lambda(val)
    all_contexts = [[c.instruction for _ in prompts] for c in DEFAULT_CONTEXTS]
    all_lambdas = [[l for _ in prompts] for l in base_lambda]

    outputs = []
    for _ in range(samples_per_setting):
        result = multi_contextual_steering_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            all_contexts=all_contexts,
            all_lambdas=all_lambdas,
            is_chat=True,
            show_progress=False,
            max_gen_len=max_new_tokens,
        )
        outputs.extend([g["content"].strip() for g in result["generation"]])
    return outputs


def _judge_prompt(
    prompt: str,
    dimension_name: str,
    dimension_instruction: str,
    outputs: List[str],
) -> str:
    lines = [
        "You are a judge. Score each candidate ONLY on the specified dimension.",
        "Ignore all other dimensions, styles, or quality factors.",
        "Return ONLY valid JSON with this schema:",
        '{"scores":[{"id":0,"score":1},...],"dimension":"...","notes":"..."}',
        "Scores must be integers from 1 (low) to 7 (high).",
        "",
        f"Dimension name: {dimension_name}",
        f"Dimension definition: {dimension_instruction}",
        f"Original prompt: {prompt}",
        "Candidates:",
    ]
    for i, text in enumerate(outputs):
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_dimension(row: Dict[str, object]) -> Tuple[str, str]:
    context_map = {c.name: c.instruction for c in DEFAULT_CONTEXTS}
    name = row.get("dimension")
    instruction = row.get("dimension_instruction")
    if instruction is None:
        if isinstance(name, str) and name in context_map:
            instruction = context_map[name]
        else:
            instruction = str(name) if name is not None else ""
    if not isinstance(name, str) or name not in context_map:
        maybe = next((k for k, v in context_map.items() if v == instruction), None)
        name = name if isinstance(name, str) else (maybe or "unknown")
    return name, instruction


def _extract_json(text: str) -> Dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _run_judge_local(
    model,
    tokenizer,
    prompt: str,
    dimension_name: str,
    dimension_instruction: str,
    outputs: List[str],
    max_new_tokens: int,
) -> Dict[str, object]:
    dialog = [{
        "role": "user",
        "content": _judge_prompt(prompt, dimension_name, dimension_instruction, outputs),
    }]
    inputs = tokenizer.apply_chat_template(
        dialog,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        return_dict=True,
    ).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    prompt_len = inputs.input_ids.shape[1]
    raw = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    parsed = _extract_json(raw)
    return {"raw": raw, "parsed": parsed}


def _run_judge_openai(
    client,
    model_name: str,
    prompt: str,
    dimension_name: str,
    dimension_instruction: str,
    outputs: List[str],
) -> Dict[str, object]:
    message = _judge_prompt(prompt, dimension_name, dimension_instruction, outputs)
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": message}],
                # temperature=0,
                # max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json(raw)
            return {"raw": raw, "parsed": parsed, "error": None, "attempts": attempt + 1}
        except Exception as exc:
            last_error = str(exc)
            sleep_s = min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.2
            time.sleep(sleep_s)
    return {"raw": "", "parsed": None, "error": last_error, "attempts": 5}


def _judge_pair(
    args,
    judge_client,
    judge_model,
    judge_tokenizer,
    prompt: str,
    dim_name: str,
    dim_instruction: str,
    baseline_outputs: List[str],
    cos_outputs: List[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if args.judge_backend == "local":
        judge_baseline = _run_judge_local(
            judge_model,
            judge_tokenizer,
            prompt,
            dim_name,
            dim_instruction,
            baseline_outputs,
            max_new_tokens=256,
        )
        judge_cos = _run_judge_local(
            judge_model,
            judge_tokenizer,
            prompt,
            dim_name,
            dim_instruction,
            cos_outputs,
            max_new_tokens=256,
        )
    else:
        judge_baseline = _run_judge_openai(
            judge_client,
            args.judge_model,
            prompt,
            dim_name,
            dim_instruction,
            baseline_outputs,
        )
        judge_cos = _run_judge_openai(
            judge_client,
            args.judge_model,
            prompt,
            dim_name,
            dim_instruction,
            cos_outputs,
        )
    return judge_baseline, judge_cos


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-based multi-dim experiment with LLM judge.")
    parser.add_argument("--data", help="JSONL edit traces; used for prompt sampling.")
    parser.add_argument("--model", default="llama-2-7b-chat")
    parser.add_argument(
        "--judge_backend",
        choices=["openai", "local"],
        default="openai",
        help="Judge backend: openai (default) or local HF model.",
    )
    parser.add_argument(
        "--judge_model",
        default="gpt-4o-mini",
        help="OpenAI model name for judging or HF model key if --judge_backend=local.",
    )
    parser.add_argument("--dims", default="concise,formal,structured")
    parser.add_argument("--slider_values", default="1,3,5,7")
    parser.add_argument("--n_prompts", type=int, default=3)
    parser.add_argument("--samples_per_setting", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--judge_only",
        action="store_true",
        help="Only run the judge on an existing JSONL (skips generation).",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run only the top 3 instances of inference/judging.",
    )
    parser.add_argument(
        "--judge_workers",
        type=int,
        default=4,
        help="Number of concurrent judge calls.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input JSONL with baseline_outputs and cos_outputs for judge-only mode.",
    )
    parser.add_argument(
        "--output",
        default="/scratch/yirenl2/projects/context-steering-writing/outputs/grid_judge.jsonl",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    judge_model = None
    judge_tokenizer = None
    judge_client = None
    if args.judge_backend == "local":
        judge_model, judge_tokenizer = load_hf_model_and_tokenizer(args.judge_model)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for --judge_backend=openai. "
                "Install with `pip install openai` and set OPENAI_API_KEY."
            ) from exc
        judge_client = OpenAI()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.judge_only:
        if not args.input:
            raise ValueError("--input is required when --judge_only is set.")
        input_path = Path(args.input)
        rows = _load_jsonl(input_path)
        if args.test_run:
            rows = rows[:3]
        with output_path.open("w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
                futures = []
                for row in rows:
                    prompt = str(row.get("prompt", ""))
                    baseline_outputs = row.get("baseline_outputs", [])
                    cos_outputs = row.get("cos_outputs", [])
                    dim_name, dim_instruction = _resolve_dimension(row)
                    futures.append((
                        executor.submit(
                            _judge_pair,
                            args,
                            judge_client,
                            judge_model,
                            judge_tokenizer,
                            prompt,
                            dim_name,
                            dim_instruction,
                            baseline_outputs,
                            cos_outputs,
                        ),
                        row,
                        dim_name,
                        dim_instruction,
                    ))
                for future, row, dim_name, dim_instruction in tqdm.tqdm(
                    futures, desc="Judge-only"
                ):
                    try:
                        judge_baseline, judge_cos = future.result()
                    except Exception as exc:
                        err = str(exc)
                        judge_baseline = {"raw": "", "parsed": None, "error": err, "attempts": 0}
                        judge_cos = {"raw": "", "parsed": None, "error": err, "attempts": 0}
                    row["dimension"] = dim_name
                    row["dimension_instruction"] = dim_instruction
                    row["judge_baseline"] = judge_baseline
                    row["judge_cos"] = judge_cos
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        if not args.data:
            raise ValueError("--data is required unless --judge_only is set.")
        random.seed(args.seed)
        examples = load_jsonl(args.data)
        prompts = [ex.prompt for ex in examples if ex.prompt]
        if not prompts:
            raise ValueError("No prompts found in data. Provide non-empty prompt fields.")
        random.shuffle(prompts)
        prompts = prompts[: args.n_prompts]
        if args.test_run:
            prompts = prompts[:1]

        dim_names = [d.strip() for d in args.dims.split(",") if d.strip()]
        dim_names, dim_instructions = _select_dimensions(dim_names)
        dim_indices = [i for i, c in enumerate(DEFAULT_CONTEXTS) if c.name in dim_names]
        slider_values = _parse_slider_values(args.slider_values)
        if args.test_run:
            slider_values = slider_values[:1]

        model, tokenizer = load_hf_model_and_tokenizer(args.model)

        with output_path.open("w", encoding="utf-8") as f:
            for prompt in tqdm.tqdm(prompts, desc="Prompts"):
                settings_iter = itertools.product(slider_values, repeat=len(dim_names))
                grid_iter = settings_iter
                if args.test_run:
                    grid_iter = itertools.islice(settings_iter, 3)
                for dim_values in tqdm.tqdm(
                    grid_iter, total=len(slider_values) ** len(dim_names), desc="Grid", leave=False
                ):
                    dim_values = list(dim_values)

                    baseline_outputs = _generate_baseline(
                        model,
                        tokenizer,
                        [prompt],
                        dim_names,
                        dim_values,
                        samples_per_setting=args.samples_per_setting,
                        max_new_tokens=args.max_new_tokens,
                    )
                    cos_outputs = _generate_cos(
                        model,
                        tokenizer,
                        [prompt],
                        dim_indices,
                        dim_values,
                        samples_per_setting=args.samples_per_setting,
                        max_new_tokens=args.max_new_tokens,
                    )

                    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
                        futures = []
                        for dim_name, dim_instruction in zip(dim_names, dim_instructions):
                            futures.append((
                                executor.submit(
                                    _judge_pair,
                                    args,
                                    judge_client,
                                    judge_model,
                                    judge_tokenizer,
                                    prompt,
                                    dim_name,
                                    dim_instruction,
                                    baseline_outputs,
                                    cos_outputs,
                                ),
                                dim_name,
                                dim_instruction,
                            ))
                        for future, dim_name, dim_instruction in tqdm.tqdm(
                            futures, desc="Judge", leave=False
                        ):
                            try:
                                judge_baseline, judge_cos = future.result()
                            except Exception as exc:
                                err = str(exc)
                                judge_baseline = {"raw": "", "parsed": None, "error": err, "attempts": 0}
                                judge_cos = {"raw": "", "parsed": None, "error": err, "attempts": 0}
                            row = {
                                "prompt": prompt,
                                "dim_names": dim_names,
                                "dim_values": dim_values,
                                "dimension": dim_name,
                                "dimension_instruction": dim_instruction,
                                "baseline_outputs": baseline_outputs,
                                "cos_outputs": cos_outputs,
                                "judge_baseline": judge_baseline,
                                "judge_cos": judge_cos,
                            }
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
