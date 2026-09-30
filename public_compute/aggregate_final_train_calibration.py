from __future__ import annotations
import argparse, json
from pathlib import Path
from wfl_public.p43_final_train_sharded import aggregate_calibration

ap=argparse.ArgumentParser()
ap.add_argument("--candidate-index",type=int,required=True)
ap.add_argument("--shard-dir",required=True)
ap.add_argument("--data-dir",default="public_compute/data")
ap.add_argument("--out",required=True)
args=ap.parse_args()
paths=sorted(Path(args.shard_dir).glob("*.json"))
shards=[json.loads(p.read_text()) for p in paths]
row=aggregate_calibration(args.candidate_index,shards,data_dir=args.data_dir)
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
print(json.dumps({"candidate_index":args.candidate_index,"shards":len(shards),"freeze_sha256":row["calibration_freeze_sha256"],"out":str(out)}))
