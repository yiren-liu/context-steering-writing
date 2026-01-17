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

