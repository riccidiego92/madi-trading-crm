from __future__ import annotations
import argparse, json
from pathlib import Path
from wfl_public.p43_final_train_sharded import null_shard

ap=argparse.ArgumentParser()
ap.add_argument("--candidate-index",type=int,required=True)
ap.add_argument("--rep-start",type=int,required=True)
ap.add_argument("--rep-count",type=int,required=True)
ap.add_argument("--calibration",required=True)
ap.add_argument("--data-dir",default="public_compute/data")
ap.add_argument("--out",required=True)
args=ap.parse_args()
cal=json.loads(Path(args.calibration).read_text())
row=null_shard(args.candidate_index,args.rep_start,args.rep_count,cal,data_dir=args.data_dir)
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
print(json.dumps({"candidate_index":args.candidate_index,"rep_start":args.rep_start,"rep_count":args.rep_count,"out":str(out)}))
