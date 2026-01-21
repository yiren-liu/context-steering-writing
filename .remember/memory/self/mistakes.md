# Mistakes and Fixes

### Mistake: Calling `apply_cos` with the wrong keyword argument name
**Wrong**:
```
apply_cos(logits=..., logits_no_context=..., ...)
```

**Correct**:
```
apply_cos(logits=..., logits_nc=..., ...)
```

Rationale: `cos/core.py:apply_cos` defines the second argument as `logits_nc`; using a different kwarg name raises a `TypeError`.

### Mistake: Applying a patch without re-reading the current file context
**Wrong**:
```
# Attempt `apply_patch` using stale or slightly mismatched context.
# Result: "Failed to find context" and no changes applied.
```

**Correct**:
```
# Re-read the target file first, then craft the patch with exact matching context
# (including commas/whitespace), and re-apply.
```

### Mistake: Not reading memory files before taking the first action
**Wrong**:
```
# Ran commands/edits before reading .remember/memory/self.md and .remember/memory/project.md.
```

**Correct**:
```
# Always read .remember/memory/self.md and .remember/memory/project.md
# as the first action in each request, before any other tool call.
```

### Mistake: Using `model.generate(output_scores=True)` to extract raw next-token logits
**Wrong**:
```
output = model.generate(..., output_scores=True, return_dict_in_generate=True)
logits = output.scores[0]  # may be full of -inf
```

**Correct**:
```
model_inputs = model.prepare_inputs_for_generation(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=cache,
    use_cache=True,
)
outputs = model(**model_inputs, return_dict=True, use_cache=True)
logits = outputs.logits[:, -1, :]
cache = outputs.past_key_values
```

Rationale: `generate(..., output_scores=True)` returns *processed* generation scores (after logits processors/warpers),
which can legitimately set most/all tokens to `-inf` (e.g., forced tokens, suppression). For CoS/scoring we usually want
the model’s raw distribution from a forward pass.

### Mistake: 422 Unprocessable Entity caused by overly strict Pydantic constraints
**Wrong**:
```
class GenerateRequest(BaseModel):
    lambda_a: Optional[float] = Field(None, ge=0.0)  # rejects negative lambdas
```

**Correct**:
```
class GenerateRequest(BaseModel):
    lambda_a: Optional[float] = Field(None, ge=-10.0, le=10.0)  # align with UI/model expectations
```

Rationale: FastAPI returns 422 when request-body validation fails; ensure backend schema constraints match the range your UI
and inference logic actually produce (e.g., negative lambdas for context steering).
