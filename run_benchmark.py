import argparse, json
from v416 import benchmark, GATES

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=30)
    p.add_argument("--raw-out")
    args=p.parse_args()
    summary,rows=benchmark(args.seeds)
    print(json.dumps({"gates":GATES,"summary":summary},indent=2))
    if args.raw_out:
        with open(args.raw_out,"w") as f: json.dump(rows,f,indent=2)
