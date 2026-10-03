from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_serial_state_fingerprint_v2 import bitcount_table, evaluate, state_table
from wfl_public.p26_public_drbg import DrbgCandidateConfig, simulate_drbg

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "serial_state_sensitivity_v2_config.json").read_text())
V2 = json.loads((ROOT / "results/serial_state_fingerprint_v2.json").read_text())

N=20
K=10
CONTESTS_PER_DAY=17
LCG_A=1664525
LCG_C=1013904223
MASK32=(1<<32)-1


def derive_seed(root, scope):
    b=f"{int(root)}|{scope}".encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:4], "big")


class LCG:
    def __init__(self, seed):
        self.state=int(seed)&MASK32
    def raw(self):
        self.state=(LCG_A*self.state+LCG_C)&MASK32
        return self.state
    def randbelow(self,bound):
        return self.raw()%int(bound)


def fy(rng):
    a=list(range(1,N+1))
    for i in range(N-1,0,-1):
        j=rng.randbelow(i+1)
        a[i],a[j]=a[j],a[i]
    return tuple(sorted(a[:K]))


def lcg_stream(n, seed_root, lifecycle):
    masks=np.zeros(int(n),dtype=np.uint32)
    nums=np.zeros(int(n),dtype=np.int16)
    rng=None
    scope=None
    for i in range(int(n)):
        if lifecycle=="continuous":
            wanted="continuous"
        elif lifecycle=="per_day":
            wanted=f"day:{i//CONTESTS_PER_DAY}"
        elif lifecycle=="per_contest":
            wanted=f"contest:{i}"
        else:
            raise ValueError(lifecycle)
        if rng is None or wanted!=scope:
            scope=wanted
            rng=LCG(derive_seed(seed_root,wanted))
        main=fy(rng)
        num=1+rng.randbelow(N)
        m=0
        for x in main:
            m|=1<<(x-1)
        masks[i]=m
        nums[i]=num
    return masks,nums


def drbg_stream(n, seed_root, lifecycle):
    cfg=DrbgCandidateConfig(
        mapper="fisher_yates",
        stream_mode="single",
        reseed_mode=lifecycle,
        experiment_seed=int(seed_root),
        contests_per_day=CONTESTS_PER_DAY,
    )
    df=simulate_drbg(int(n),cfg)
    masks=np.zeros(len(df),dtype=np.uint32)
    for j in range(1,11):
        vals=df[f"n{j}"].to_numpy(dtype=np.int64)
        masks |= np.left_shift(np.uint32(1),(vals-1).astype(np.uint32))
    return masks,df["numerone"].to_numpy(dtype=np.int16)


def main(out_path):
    if not CFG["guardrails"]["synthetic_only"]:
        raise SystemExit("synthetic-only guardrail changed")
    states=state_table()
    bits=bitcount_table()
    lags=[int(x) for x in V2["config"]["contest_lags"]]
    moduli=[int(x) for x in V2["config"]["partition_rank_moduli"]]
    q99={k:float(v["q99"]) for k,v in V2["null_quantiles"].items()}

    rows=[]
    for engine in CFG["engines"]:
        for lifecycle in CFG["lifecycles"]:
            for seed in CFG["replicate_seeds"]:
                if engine=="LCG32":
                    masks,nums=lcg_stream(CFG["sample_contests"],seed,lifecycle)
                elif engine=="HMAC_DRBG_SHA256":
                    masks,nums=drbg_stream(CFG["sample_contests"],seed,lifecycle)
                else:
                    raise ValueError(engine)
                _,mx,tops=evaluate(masks,nums,states,bits,lags,moduli)
                exceeds={k:bool(float(mx[k])>q99[k]) for k in q99}
                rows.append({
                    "engine":engine,
                    "lifecycle":lifecycle,
                    "seed":int(seed),
                    "max_stats":mx,
                    "top_locations":tops,
                    "exceeds_real_v2_q99":exceeds,
                    "any_family_exceeds_q99":bool(any(exceeds.values()))
                })

    summary={}
    for engine in CFG["engines"]:
        summary[engine]={}
        for lifecycle in CFG["lifecycles"]:
            cell=[r for r in rows if r["engine"]==engine and r["lifecycle"]==lifecycle]
            summary[engine][lifecycle]={
                "replicates":len(cell),
                "any_family_detection_rate":float(np.mean([r["any_family_exceeds_q99"] for r in cell])),
                "family_detection_rate":{
                    k:float(np.mean([r["exceeds_real_v2_q99"][k] for r in cell])) for k in q99
                }
            }

    payload={
        "schema":"wfl-serial-state-sensitivity-v2-result",
        "status":"SYNTHETIC_SENSITIVITY_ONLY",
        "config":CFG,
        "real_v2_q99_thresholds":q99,
        "summary":summary,
        "replicate_rows":rows,
        "interpretation":"This benchmark asks whether the frozen V2 endpoints can detect known synthetic generator/lifecycle structure after the public 10-of-20+Numerone mapping. It does not identify the real Classico engine and does not search operational state."
    }
    p=Path(out_path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    main(a.out)
