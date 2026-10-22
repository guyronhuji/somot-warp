
import argparse, joblib, pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Apply saved SOM-OT model to new data.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--which", required=True, choices=["A","B"])
    ap.add_argument("--out", default="corrected_new.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.batch) if args.batch.endswith(".csv") else pd.read_table(args.batch)
    y = np.array([args.which]*len(df))
    X_corr = model.transform(df.values, y=y)
    pd.DataFrame(X_corr, columns=list(df.columns)).to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
