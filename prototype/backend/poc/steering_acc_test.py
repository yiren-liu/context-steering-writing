import json
import math
from collections import defaultdict

try:
    from scipy import stats
except Exception:
    stats = None

path = '/scratch/yirenl2/projects/context-steering-writing/outputs/grid_judge.jsonl'
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

print('Method\tDimension\tN\tPearson\tp_value')
for method in ['baseline','cos']:
    for dim, ts in pairs[method].items():
        xs = [t for t,_ in ts]
        ys = [s for _,s in ts]
        r, p = pearson_and_p(xs, ys)
        p_str = f'{p:.3g}' if not math.isnan(p) else 'nan'
        print(f'{method}\t{dim}\t{len(ts)}\t{r:.3f}\t{p_str}')

for method in ['baseline','cos']:
    all_ts = []
    all_scores = []
    for dim, ts in pairs[method].items():
        for t,s in ts:
            all_ts.append(t)
            all_scores.append(s)
    r, p = pearson_and_p(all_ts, all_scores)
    p_str = f'{p:.3g}' if not math.isnan(p) else 'nan'
    print(f'{method}\tALL\t{len(all_ts)}\t{r:.3f}\t{p_str}')

if stats is None:
    print('Note: scipy not available; p-values not computed.')