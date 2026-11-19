#!/usr/bin/env python3
"""Independently audit seed provenance and ROUGE-L F1 leakage. Requires rouge-score."""
import argparse, collections, hashlib, json, pathlib, sys

def read(p, limit=None):
    with open(p, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit is not None else rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train-pool','val','test','seeds','synth'): p.add_argument('--'+name, required=True)
    p.add_argument('--threshold',type=float,default=.6); p.add_argument('--limit',type=int,help='Audit only first N synthetic examples (not a full sign-off)')
    p.add_argument('--output',default='results/leakage_audit.json'); a=p.parse_args()
    if not 0 < a.threshold <= 1 or (a.limit is not None and a.limit < 1): p.error('Invalid threshold or limit')
    try: from rouge_score import rouge_scorer, tokenize
    except ImportError: p.error('Install rouge-score: python -m pip install rouge-score==0.1.2')
    paths = {k:getattr(a,k) for k in ('train_pool','val','test','seeds','synth')}
    data={k:read(v, a.limit if k=='synth' else None) for k,v in paths.items()}
    if any(not v for v in data.values()): p.error('Input files must be nonempty')
    pool={x['id']:x for x in data['train_pool']}; seed_errors=[]
    for s in data['seeds']:
        original=pool.get(s.get('id'))
        if original is None or any(s.get(k)!=original.get(k) for k in ('category','instruction','context','response','source')): seed_errors.append(s.get('id'))
    held=[(part,x) for part in ('val','test') for x in data[part]]
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False)
    # Non-overlapping normalized tokens have ROUGE-L zero, so indexing preserves the exact maximum.
    index=collections.defaultdict(set)
    for i,(_,row) in enumerate(held):
        for t in set(tokenize.tokenize(row['instruction'],None)): index[t].add(i)
    worst=[]; maximum=0.0
    for row in data['synth']:
        candidates=set()
        for t in set(tokenize.tokenize(row['instruction'],None)): candidates.update(index.get(t,()))
        best=(0.0,None,None)
        for i in candidates:
            part, ref=held[i]; score=scorer.score(ref['instruction'],row['instruction'])['rougeL'].fmeasure
            if score>best[0]: best=(score,part,ref['id'])
        maximum=max(maximum,best[0])
        if best[0]>=a.threshold: worst.append({'synth_id':row.get('id'),'score':best[0],'split':best[1],'real_id':best[2]})
    result={'full_audit':a.limit is None,'threshold':a.threshold,'metric':'rouge-score 0.1.2 ROUGE-L F1, use_stemmer=False, instruction only','seed_errors':seed_errors,'synth_count':len(data['synth']),'expected_2000':len(data['synth'])==2000,'max_rougeL_f1':maximum,'violations':sorted(worst,key=lambda x:-x['score']),'sha256':{k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in paths.items()}}
    result['passed'] = a.limit is None and len(data['synth'])==2000 and not seed_errors and not worst
    out=pathlib.Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='violations'},ensure_ascii=False,indent=2)); print(f'Violations: {len(worst)}; details: {out}')
    return 0 if result['passed'] else 1
if __name__=='__main__': sys.exit(main())
