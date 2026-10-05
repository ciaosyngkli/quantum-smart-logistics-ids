"""
AI + Quantum Hybrid IDS - stronger experiment (v2)
Dataset: UNSW-NB15 (binary: normal vs attack)

What this adds over v1:
  1. Repeated over several random seeds -> results reported as mean +/- std
  2. Learning curve: performance vs training size (shows where quantum helps)
  3. Entanglement ablation: ZFeatureMap (no entanglement) vs ZZFeatureMap (entangled)
  4. Same PCA features for all models = fair classical-vs-quantum comparison
  5. Full-feature Random Forest as an upper-bound reference
  6. Kernel matrix computed once per seed and sliced -> much faster

SETUP (Linux)
  python -m venv venv && source venv/bin/activate
  pip install numpy pandas scikit-learn matplotlib qiskit qiskit-machine-learning

DATA
  UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in this folder.

RUN
  QUICK=True first (a few minutes) to check everything works,
  then set QUICK=False for the real results.

OUTPUT
  results_all.csv, results_summary.csv, learning_curve.png
"""
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------
QUICK = True                 # True = fast sanity run, False = full experiment
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV = "UNSW_NB15_testing-set.csv"
N_QUBITS = 4                 # PCA features = qubits (4-6 is laptop-friendly)

if QUICK:
    SEEDS, SIZES, N_TEST = [0], [50, 100], 60
else:
    SEEDS, SIZES, N_TEST = [0, 1, 2], [50, 100, 200, 300], 150
MAX_TRAIN = max(SIZES)
N_FULL_RF = 20000            # data for the full-feature RF reference
# -----------------------------------------------------------------------


def get_feature_maps(n):
    """Works on both older and newer Qiskit versions."""
    try:
        from qiskit.circuit.library import zz_feature_map, z_feature_map
        return {
            "Quantum kernel (ZFeatureMap, no entanglement)": z_feature_map(n, reps=2),
            "Quantum kernel (ZZFeatureMap, entangled)": zz_feature_map(n, reps=2, entanglement="linear"),
        }
    except ImportError:
        from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
        return {
            "Quantum kernel (ZFeatureMap, no entanglement)": ZFeatureMap(n, reps=2),
            "Quantum kernel (ZZFeatureMap, entangled)": ZZFeatureMap(n, reps=2, entanglement="linear"),
        }


def load():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    for df in (tr, te):
        df.drop(columns=[c for c in ("id", "attack_cat") if c in df.columns], inplace=True)
    cat = tr.select_dtypes(include="object").columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([tr[cat], te[cat]]))
    for df in (tr, te):
        df[cat] = enc.transform(df[cat])
    return tr, te


def balanced_sample(df, n, seed):
    half = n // 2
    parts = [g.sample(min(half, len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def metrics(y, p):
    return dict(
        Accuracy=accuracy_score(y, p),
        Precision=precision_score(y, p, zero_division=0),
        Recall=recall_score(y, p, zero_division=0),
        F1=f1_score(y, p, zero_division=0),
    )


def main():
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    print("Loading data...")
    train_df, test_df = load()
    rows = []
    t0 = time.time()

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        pool = balanced_sample(train_df, MAX_TRAIN, seed)
        test = balanced_sample(test_df, N_TEST, seed)
        X_pool, y_pool = pool.drop(columns="label").values, pool["label"].values
        X_te, y_te = test.drop(columns="label").values, test["label"].values

        # --- upper-bound reference: Random Forest, all features, lots of data ---
        big = balanced_sample(train_df, N_FULL_RF, seed)
        sc_full = MinMaxScaler().fit(big.drop(columns="label").values)
        rf_full = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf_full.fit(sc_full.transform(big.drop(columns="label").values), big["label"].values)
        m = metrics(y_te, rf_full.predict(sc_full.transform(X_te)))
        rows.append(dict(Seed=seed, Size=N_FULL_RF, Model="RF (all features, 20k rows) [reference]", **m))

        # --- shared preprocessing: scale -> PCA(n_qubits) -> [0, pi] ---
        sc = MinMaxScaler().fit(X_pool)
        pca = PCA(n_components=N_QUBITS, random_state=seed).fit(sc.transform(X_pool))
        Z_pool = pca.transform(sc.transform(X_pool))
        Z_te = pca.transform(sc.transform(X_te))
        ang = MinMaxScaler(feature_range=(0, np.pi)).fit(Z_pool)
        Z_pool, Z_te = ang.transform(Z_pool), ang.transform(Z_te)

        # --- classical models on the SAME PCA features, at each size ---
        for n in SIZES:
            Xn, yn = Z_pool[:n], y_pool[:n]
            for name, model in [
                (f"Classical SVM (RBF, {N_QUBITS} PCA feats)", SVC(kernel="rbf")),
                (f"Random Forest ({N_QUBITS} PCA feats)", RandomForestClassifier(n_estimators=100, random_state=seed)),
            ]:
                model.fit(Xn, yn)
                rows.append(dict(Seed=seed, Size=n, Model=name, **metrics(y_te, model.predict(Z_te))))

        # --- quantum kernels: compute matrix ONCE for max size, slice for smaller ---
        for qname, fmap in get_feature_maps(N_QUBITS).items():
            t = time.time()
            print(f"  computing {qname} ...")
            kernel = FidelityQuantumKernel(feature_map=fmap)
            K_train = kernel.evaluate(x_vec=Z_pool)               # (MAX, MAX)
            K_test = kernel.evaluate(x_vec=Z_te, y_vec=Z_pool)    # (N_TEST, MAX)
            print(f"  kernel time: {time.time() - t:.0f}s")
            for n in SIZES:
                clf = SVC(kernel="precomputed")
                clf.fit(K_train[:n, :n], y_pool[:n])
                pred = clf.predict(K_test[:, :n])
                rows.append(dict(Seed=seed, Size=n, Model=qname, **metrics(y_te, pred)))

        print(f"  seed {seed} done, elapsed {time.time() - t0:.0f}s")

    # ----------------------------- Results -----------------------------
    res = pd.DataFrame(rows)
    res.to_csv("results_all.csv", index=False)

    mets = ["Accuracy", "Precision", "Recall", "F1"]
    g = res.groupby(["Model", "Size"])[mets].agg(["mean", "std"]).round(4)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g.to_csv("results_summary.csv", index=False)
    print("\n=== SUMMARY (mean over seeds) ===")
    print(g[["Model", "Size", "Accuracy_mean", "F1_mean", "F1_std"]].to_string(index=False))

    # Learning curve plot (F1 vs training size)
    fig, ax = plt.subplots(figsize=(8, 5))
    curve = res[~res["Model"].str.contains("reference")]
    for model, d in curve.groupby("Model"):
        s = d.groupby("Size")["F1"].agg(["mean", "std"]).fillna(0)
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=model)
    ref = res[res["Model"].str.contains("reference")]["F1"].mean()
    ax.axhline(ref, ls="--", color="gray", label="RF all features, 20k rows (reference)")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Classical vs Quantum IDS (UNSW-NB15, {N_QUBITS} qubits, {len(SEEDS)} seed(s))")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    print("\nSaved: results_all.csv, results_summary.csv, learning_curve.png")


if __name__ == "__main__":
    main()
