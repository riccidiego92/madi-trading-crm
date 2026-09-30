from __future__ import annotations
import argparse, json
from pathlib import Path
from wfl_public.p43_final_train_sharded import aggregate_final

ap=argparse.ArgumentParser()
ap.add_argument("--candidate-index",type=int,required=True)
ap.add_argument("--calibration",required=True)
ap.add_argument("--shard-dir",required=True)
ap.add_argument("--out",required=True)
args=ap.parse_args()
cal=json.loads(Path(args.calibration).read_text())
paths=sorted(Path(args.shard_dir).glob("*.json"))
shards=[json.loads(p.read_text()) for p in paths]
row=aggregate_final(args.candidate_index,cal,shards)
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
print(json.dumps({"candidate_index":args.candidate_index,"shards":len(shards),"p_max_stat":row["test"]["p_max_stat"],"out":str(out)}))
