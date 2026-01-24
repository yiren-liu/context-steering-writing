import json
import math
from collections import defaultdict

try:
    from scipy import stats
except Exception:
    stats = None


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to grid_judge.jsonl')
args = parser.parse_args()

path = args.input
rows = []
with open(path,'r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

pairs = { 'baseline': defaultdict(list), 'cos': defaultdict(list)}

for r in rows:
    dim = r.get('dimension')
    dim_names = r.get('dim_names') or []
    dim_values = r.get('dim_values') or []
    if dim not in dim_names:
        continue
    idx = dim_names.index(dim)
    target = dim_values[idx]

    for method, key in [('baseline','judge_baseline'), ('cos','judge_cos')]:
        judge = r.get(key, {})
        parsed = judge.get('parsed') if isinstance(judge, dict) else None
        if not parsed:
            continue
        scores = parsed.get('scores')
        if not scores or not isinstance(scores, list):
            continue
        score = scores[0].get('score')
        if score is None:
            continue
        pairs[method][dim].append((target, score))


def pearson_and_p(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan'), float('nan')
    mean_x = sum(xs)/n
    mean_y = sum(ys)/n
    cov = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))
    var_x = sum((x-mean_x)**2 for x in xs)
    var_y = sum((y-mean_y)**2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float('nan'), float('nan')
    r = cov / math.sqrt(var_x*var_y)
    if stats is None:
        return r, float('nan')
    p = stats.pearsonr(xs, ys).pvalue
    return r, p

def _fmt_nan(x, fmt):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return format(x, fmt)


def _fmt_p(p):
    # Use 3 significant digits; switch to scientific when very small.
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "nan"
    return f"{p:.3g}"


table_rows = []
for method in ["baseline", "cos"]:
    for dim in sorted(pairs[method].keys()):
        ts = pairs[method][dim]
        xs = [t for t, _ in ts]
        ys = [s for _, s in ts]
        r, p = pearson_and_p(xs, ys)
        table_rows.append(
            {
                "Method": method,
                "Dimension": dim,
                "N": str(len(ts)),
                "Pearson": _fmt_nan(r, ".3f"),
                "p_value": _fmt_p(p),
            }
        )

    all_ts = []
    all_scores = []
    for dim, ts in pairs[method].items():
        for t, s in ts:
            all_ts.append(t)
            all_scores.append(s)
    r, p = pearson_and_p(all_ts, all_scores)
    table_rows.append(
        {
            "Method": method,
            "Dimension": "ALL",
            "N": str(len(all_ts)),
            "Pearson": _fmt_nan(r, ".3f"),
            "p_value": _fmt_p(p),
        }
    )

cols = ["Method", "Dimension", "N", "Pearson", "p_value"]
widths = {c: len(c) for c in cols}
for row in table_rows:
    for c in cols:
        widths[c] = max(widths[c], len(str(row.get(c, ""))))

# Header
header = (
    f"{cols[0]:<{widths[cols[0]]}}  "
    f"{cols[1]:<{widths[cols[1]]}}  "
    f"{cols[2]:>{widths[cols[2]]}}  "
    f"{cols[3]:>{widths[cols[3]]}}  "
    f"{cols[4]:>{widths[cols[4]]}}"
)
print(header)
print(
    f"{'-' * widths[cols[0]]}  "
    f"{'-' * widths[cols[1]]}  "
    f"{'-' * widths[cols[2]]}  "
    f"{'-' * widths[cols[3]]}  "
    f"{'-' * widths[cols[4]]}"
)

prev_method = None
for row in table_rows:
    method = row["Method"]
    if prev_method is not None and method != prev_method:
        print("")  # visual break between methods
    prev_method = method
    print(
        f"{row['Method']:<{widths['Method']}}  "
        f"{row['Dimension']:<{widths['Dimension']}}  "
        f"{row['N']:>{widths['N']}}  "
        f"{row['Pearson']:>{widths['Pearson']}}  "
        f"{row['p_value']:>{widths['p_value']}}"
    )

if stats is None:
    print('Note: scipy not available; p-values not computed.')