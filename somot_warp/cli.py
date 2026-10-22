
import argparse, json, time
import numpy as np
import pandas as pd
from somot_warp.sklearn_api import SOMOTWarpSKExact

def load_matrix(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_table(path)
    return df.values, list(df.columns)

def main():
    ap = argparse.ArgumentParser(description="SOM-OT residual warp (sklearn API, anchorable).")
    ap.add_argument("--batchA", required=True)
    ap.add_argument("--batchB", required=True)
    ap.add_argument("--outA", default="corrected_A.csv")
    ap.add_argument("--outB", default="corrected_B.csv")
    ap.add_argument("--report", default="report.json")
    ap.add_argument("--save_model", default=None)
    ap.add_argument("--cofactor", type=float, default=5.0)
    ap.add_argument("--transform_kind", default="arcsinh", choices=["arcsinh","log1p","none"])
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--som", default="10x10")
    ap.add_argument("--som_iters", type=int, default=8000)
    ap.add_argument("--landmarks", type=int, default=1200)
    ap.add_argument("--min_node", type=int, default=80)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--unbalanced", action="store_true")
    ap.add_argument("--reg_m", type=float, default=5.0)
    ap.add_argument("--interp", default="soft", choices=["soft","hard"])
    ap.add_argument("--soft_k", type=int, default=5)
    ap.add_argument("--soft_tau", type=float, default=0.5)
    ap.add_argument("--anchor", default="barycentric", choices=["barycentric","A","B"])
    ap.add_argument("--alpha_ot", type=float, default=None)
    ap.add_argument("--alpha_warp", type=float, default=None)
    ap.add_argument("--output_scale", default="batch", choices=["batch","A","B"])
    args = ap.parse_args()

    Xa, markersA = load_matrix(args.batchA)
    Xb, markersB = load_matrix(args.batchB)
    mx, my = map(int, args.som.lower().split("x"))

    model = SOMOTWarpSKExact(
        transform_kind=("none" if args.transform_kind=="none" else args.transform_kind),
        cofactor=args.cofactor,
        pca_components=None if args.pca<=0 else args.pca,
        som_grid=(mx,my), som_sigma=1.5, som_learning_rate=0.5,
        som_iters=args.som_iters, som_random_state=0, som_verbose=True,
        n_landmarks_total=args.landmarks, reg=args.reg, unbalanced=args.unbalanced, reg_m=args.reg_m,
        interp=args.interp, soft_k=args.soft_k, soft_tau=args.soft_tau,
        anchor=args.anchor, alpha_ot=args.alpha_ot, alpha_warp=args.alpha_warp,
        output_scale=args.output_scale, min_node_size=args.min_node
    )

    X = np.vstack([Xa, Xb])
    y = np.r_[np.array(['A']*len(Xa)), np.array(['B']*len(Xb))]

    t0 = time.perf_counter()
    Xcorr = model.fit_transform(X, y)
    elapsed = time.perf_counter() - t0

    A_corr = Xcorr[:len(Xa)]; B_corr = Xcorr[len(Xa):]
    pd.DataFrame(A_corr, columns=markersA).to_csv(args.outA, index=False)
    pd.DataFrame(B_corr, columns=markersA).to_csv(args.outB, index=False)

    report = {"elapsed_sec": elapsed, "params": vars(args)}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    if args.save_model:
        import joblib
        joblib.dump(model, args.save_model)
        print("Saved model:", args.save_model)

    print(f"Saved: {args.outA}, {args.outB}, {args.report}")
