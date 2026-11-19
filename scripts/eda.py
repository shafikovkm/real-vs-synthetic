#!/usr/bin/env python3
"""Generate Dolly EDA summary and two vector plots."""
import argparse, collections, json, pathlib, statistics

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--output-dir',default='figures'); p.add_argument('--limit',type=int); a=p.parse_args()
    try: import matplotlib.pyplot as plt
    except ImportError: p.error('Install matplotlib: python -m pip install -r requirements.txt')
    with open(a.input,encoding='utf-8') as f: rows=[json.loads(x) for x in f if x.strip()]
    if a.limit is not None:
        if a.limit<1: p.error('--limit must be positive')
        rows=rows[:a.limit]
    if not rows: p.error('Empty input')
    out=pathlib.Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter(x['category'] for x in rows); ins=[len(x['instruction'].split()) for x in rows]; ans=[len(x['response'].split()) for x in rows]
    summary={'n':len(rows),'categories':dict(sorted(counts.items())),'context_count':sum(bool(x.get('context','').strip()) for x in rows),'context_fraction':sum(bool(x.get('context','').strip()) for x in rows)/len(rows),'length_unit':'whitespace-separated words','instruction_words':{'mean':statistics.mean(ins),'median':statistics.median(ins),'max':max(ins)},'response_words':{'mean':statistics.mean(ans),'median':statistics.median(ans),'max':max(ans)}}
    (out/'eda_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plt.style.use('ggplot'); fig,ax=plt.subplots(figsize=(9,4.8)); keys=sorted(counts,key=counts.get); ax.barh(keys,[counts[k] for k in keys],color='#347e91'); ax.set_xlabel('Number of examples'); ax.set_title('Dolly: category distribution'); fig.tight_layout(); fig.savefig(out/'categories.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));
    for ax,values,title in zip(axes,(ins,ans),('Instruction length','Response length')):
        ax.hist(values,bins=50,color='#347e91'); ax.set_xscale('symlog',linthresh=1); ax.set_xlabel('Words (log scale)'); ax.set_ylabel('Examples'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out/'lengths.pdf'); plt.close(fig); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
