#!/usr/bin/env python3
"""Create deterministic, category-stratified Dolly splits. Python >= 3.9."""
import argparse, collections, hashlib, json, pathlib, random

FIELDS = ('id', 'category', 'instruction', 'context', 'response', 'source')
CATEGORIES = {'brainstorming','open_qa','closed_qa','summarization','information_extraction','classification','creative_writing','general_qa'}

def read(path):
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{n}: {e}') from e

def allocate(groups, size):
    total = sum(len(v) for v in groups.values())
    quotas = {k: size * len(v) / total for k, v in groups.items()}
    counts = {k: int(v) for k, v in quotas.items()}
    for k in sorted(groups, key=lambda k: (-(quotas[k] - counts[k]), k))[:size - sum(counts.values())]: counts[k] += 1
    return counts

def dump(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--output-dir', default='data')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--train-seed', type=int, default=43)
    p.add_argument('--limit', type=int, help='Smoke test: first N rows; proportional 80/7/13 split')
    p.add_argument('--force', action='store_true', help='Overwrite frozen outputs intentionally')
    a = p.parse_args(); out = pathlib.Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    files = [out / (s + '.jsonl') for s in ('train_pool','val','test','train_real')]
    if not a.force and any(x.exists() for x in files): p.error('Output exists; frozen splits are never overwritten without --force')
    raw = list(read(a.input)); raw = raw[:a.limit] if a.limit is not None else raw
    if a.limit is not None and a.limit < 8: p.error('--limit must be >= 8')
    if a.limit is None and len(raw) < 15000: p.error('Need at least 15000 rows')
    rows = []
    for i, x in enumerate(raw, 1):
        if any(not isinstance(x.get(k), str) for k in ('category','instruction','context','response')): raise ValueError(f'row {i}: missing or non-string fields')
        if x['category'] not in CATEGORIES: raise ValueError(f'row {i}: unknown category {x["category"]}')
        rows.append(dict(id=f'real-{i:06d}', category=x['category'], instruction=x['instruction'], context=x['context'], response=x['response'], source='real'))
    groups = collections.defaultdict(list)
    for row in rows: groups[row['category']].append(row)
    rng = random.Random(a.seed)
    for k in sorted(groups): rng.shuffle(groups[k])
    sizes = [12000,1000,2000] if a.limit is None else [round(len(rows)*.8), round(len(rows)/15), round(len(rows)*2/15)]
    if a.limit is not None: sizes[0] = len(rows)-sizes[1]-sizes[2]
    split = {}
    # Allocate held-out sets first, then training from the remainder; any surplus is unused.
    for name, size in [('test',sizes[2]),('val',sizes[1]),('train_pool',sizes[0])]:
        counts = allocate(groups, size); chosen = []
        for k in sorted(groups): chosen.extend(groups[k][:counts[k]]); del groups[k][:counts[k]]
        rng.shuffle(chosen); split[name] = chosen
    trgroups = collections.defaultdict(list)
    for x in split['train_pool']: trgroups[x['category']].append(x)
    n = 2000 if a.limit is None else min(len(split['train_pool']), max(1, round(len(split['train_pool'])/6)))
    counts = allocate(trgroups, n); rng2 = random.Random(a.train_seed); selected = []
    for k in sorted(trgroups): rng2.shuffle(trgroups[k]); selected.extend(trgroups[k][:counts[k]])
    rng2.shuffle(selected); split['train_real'] = selected
    for name, data in split.items(): dump(out / (name+'.jsonl'), data)
    hashes = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    metadata = {'input': str(a.input), 'input_sha256': hashlib.sha256(pathlib.Path(a.input).read_bytes()).hexdigest(), 'seed':a.seed,'train_seed':a.train_seed,'limit':a.limit,'source_rows':len(raw),'unused_rows':sum(map(len,groups.values())),'counts':{k:len(v) for k,v in split.items()},'sha256':hashes}
    (out/'split_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))
if __name__ == '__main__': main()
