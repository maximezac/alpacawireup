#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ART = Path('artifacts')

def main(exp_id: str):
    out = ART / f"experiment_summary_{exp_id}.md"
    md = []
    # include top-level summary if exists
    js = ART / f"experiment_summary_{exp_id}.json"
    if js.exists():
        s = json.loads(js.read_text())
        md.append(f"# Experiment {exp_id}\n")
        md.append(f"Start: {s.get('date_range',{}).get('start')} - End: {s.get('date_range',{}).get('end')}\n")
        md.append("## Portfolios\n")
        # include per-portfolio CSV
        pcsv = ART / f"experiment_summary_{exp_id}_per_portfolio.csv"
        if pcsv.exists():
            md.append('See per-portfolio CSV for full metrics.\n')
    # collect portfolio-level summaries
    for d in ART.iterdir():
        if d.is_dir():
            sm = d / 'summary.md'
            if sm.exists():
                md.append(f"## {d.name}\n")
                md.append(sm.read_text())
    out.write_text('\n'.join(md))
    print(f"Wrote {out}")

if __name__ == '__main__':
    import sys
    eid = sys.argv[1] if len(sys.argv)>1 else 'exp_v1'
    main(eid)
