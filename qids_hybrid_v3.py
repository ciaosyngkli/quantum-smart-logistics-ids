"""
AI + Quantum Hybrid Intrusion Detection System for Smart Logistics
Prototype experiment (v3) - mapped to the project objectives

Objective 1  Collect/preprocess IoT traffic data   -> UNSW-NB15 (same dataset as ref. #6, so results are comparable)
Objective 2  Hybrid quantum-classical IDS model    -> PROPOSED: classical autoencoder (trained on NORMAL traffic only,
                                                      as in ref. #12) compresses features -> quantum kernel SVM
                                                      classifies in the quantum feature space
Objective 3  Compare with existing methods         -> classical SVM / RF (refs #1, #6), QSVC on PCA features (ref. #6),
                                                      autoencoder-only anomaly detector (ref. #12)
Gap addressed  "real-time detection / light-weight for logistics edge" (ref. #20) -> inference latency is reported

Models compared (same labelled training sizes, same test set):
  A. Classical SVM, all features                      (existing method baseline)
  B. Random Forest, all features                      (existing method baseline)
  C. Quantum kernel SVM on PCA features               (QSVC as in ref. #6)
  D. Classical SVM on autoencoder features            (ablation: is the quantum part needed?)
  E. PROPOSED hybrid: Autoencoder + quantum kernel SVM
  F. Autoencoder-only anomaly detector (unsupervised) (ref. #12 style, AUROC only)

SETUP (Linux)
  python -m venv venv && source venv/bin/activate
  pip install numpy pandas scikit-learn matplotlib qiskit qiskit-machine-learning

DATA
  UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in this folder.

RUN
  Keep QUICK=True first (few minutes) to check it works, then QUICK=False.

OUTPUT
  results_all.csv, results_summary.csv, learning_curve.png, latency.csv
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
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------
QUICK = True                  # True = fast check, False = full experiment
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV = "UNSW_NB15_testing-set.csv"
N_QUBITS = 4                  # autoencoder bottleneck size = number of qubits

if QUICK:
    SEEDS, SIZES, N_TEST = [0], [50, 100], 60
    N_NORMAL_AE, AE_ITER = 2000, 60
else:
    SEEDS, SIZES, N_TEST = [0, 1, 2], [50, 100, 200, 300], 150
    N_NORMAL_AE, AE_ITER = 8000, 300
MAX_TRAIN = max(SIZES)
# -----------------------------------------------------------------------


def zz_map(n):
    try:
        from qiskit.circuit.library import zz_feature_map
        return zz_feature_map(n, reps=2, entanglement="linear")
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap
        return ZZFeatureMap(n, reps=2, entanglement="linear")


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


def metrics(y, pred, score=None):
    m = dict(Accuracy=accuracy_score(y, pred),
             Precision=precision_score(y, pred, zero_division=0),
             Recall=recall_score(y, pred, zero_division=0),
             F1=f1_score(y, pred, zero_division=0),
             AUROC=np.nan)
    if score is not None:
        m["AUROC"] = roc_auc_score(y, score)
    return m


def ae_latent(ae, X):
    """Bottleneck activations of a (32, N_QUBITS, 32) tanh autoencoder."""
    h1 = np.tanh(X @ ae.coefs_[0] + ae.intercepts_[0])
    return np.tanh(h1 @ ae.coefs_[1] + ae.intercepts_[1])


def main():
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    print("Loading data...")
    train_df, test_df = load()
    rows, lat_rows = [], []
    t0 = time.time()

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        pool = balanced_sample(train_df, MAX_TRAIN, seed)
        test = balanced_sample(test_df, N_TEST, seed)
        X_pool, y_pool = pool.drop(columns="label").values, pool["label"].values
        X_te, y_te = test.drop(columns="label").values, test["label"].values

        # scaler fitted on NORMAL traffic only (the AE never sees attacks)
        normal = train_df[train_df["label"] == 0].drop(columns="label")
        normal = normal.sample(min(N_NORMAL_AE, len(normal)), random_state=seed).values
        sc = MinMaxScaler().fit(normal)
        Xn = sc.transform(normal)
        Xp, Xt = sc.transform(X_pool), sc.transform(X_te)

        # ---- Classical autoencoder, trained unsupervised on normal traffic ----
        print("  training autoencoder on normal traffic...")
        ae = MLPRegressor(hidden_layer_sizes=(32, N_QUBITS, 32), activation="tanh",
                          max_iter=AE_ITER, random_state=seed)
        ae.fit(Xn, Xn)

        # F. Autoencoder-only anomaly detector (reconstruction error, AUROC)
        err = ((Xt - ae.predict(Xt)) ** 2).mean(axis=1)
        thr = np.percentile(((Xn - ae.predict(Xn)) ** 2).mean(axis=1), 95)
        rows.append(dict(Seed=seed, Size=0,
                         Model="F. Autoencoder-only anomaly detector (unsupervised)",
                         **metrics(y_te, (err > thr).astype(int), err)))

        # latent features -> angles in [0, pi] for quantum encoding
        L_pool, L_te = ae_latent(ae, Xp), ae_latent(ae, Xt)
        ang = MinMaxScaler(feature_range=(0, np.pi)).fit(L_pool)
        L_pool, L_te = ang.transform(L_pool), ang.transform(L_te)

        # PCA features (ref. #6 style QSVC input)
        pca = PCA(n_components=N_QUBITS, random_state=seed).fit(Xp)
        P_pool, P_te = pca.transform(Xp), pca.transform(Xt)
        ang2 = MinMaxScaler(feature_range=(0, np.pi)).fit(P_pool)
        P_pool, P_te = ang2.transform(P_pool), ang2.transform(P_te)

        # ---- Classical models at each labelled-data size ----
        for n in SIZES:
            for name, model, Ftr, Fte in [
                ("A. Classical SVM (all features)", SVC(kernel="rbf"), Xp, Xt),
                ("B. Random Forest (all features)", RandomForestClassifier(n_estimators=100, random_state=seed), Xp, Xt),
                ("D. Classical SVM on autoencoder features", SVC(kernel="rbf"), L_pool, L_te),
            ]:
                model.fit(Ftr[:n], y_pool[:n])
                pred = model.predict(Fte)
                if hasattr(model, "decision_function"):
                    score = model.decision_function(Fte)
                else:
                    score = model.predict_proba(Fte)[:, 1]
                rows.append(dict(Seed=seed, Size=n, Model=name, **metrics(y_te, pred, score)))
                if n == MAX_TRAIN:
                    t = time.time(); model.predict(Fte)
                    lat_rows.append(dict(Seed=seed, Model=name,
                                         Latency_ms_per_sample=(time.time() - t) / len(Fte) * 1000))

        # ---- Quantum kernel models (kernel matrix computed once, then sliced) ----
        for qname, Qpool, Qte in [
            ("C. Quantum kernel SVM on PCA features (QSVC)", P_pool, P_te),
            ("E. PROPOSED hybrid: Autoencoder + quantum kernel SVM", L_pool, L_te),
        ]:
            print(f"  computing quantum kernel: {qname}")
            kernel = FidelityQuantumKernel(feature_map=zz_map(N_QUBITS))
            t = time.time()
            K_train = kernel.evaluate(x_vec=Qpool)
            print(f"    train kernel: {time.time() - t:.0f}s")
            t = time.time()
            K_test = kernel.evaluate(x_vec=Qte, y_vec=Qpool)
            lat = (time.time() - t) / len(Qte) * 1000
            lat_rows.append(dict(Seed=seed, Model=qname, Latency_ms_per_sample=lat))
            for n in SIZES:
                clf = SVC(kernel="precomputed").fit(K_train[:n, :n], y_pool[:n])
                pred = clf.predict(K_test[:, :n])
                score = clf.decision_function(K_test[:, :n])
                rows.append(dict(Seed=seed, Size=n, Model=qname, **metrics(y_te, pred, score)))

        print(f"  seed {seed} done, elapsed {time.time() - t0:.0f}s")

    # ----------------------------- Results -----------------------------
    res = pd.DataFrame(rows)
    res.to_csv("results_all.csv", index=False)
    pd.DataFrame(lat_rows).groupby("Model")["Latency_ms_per_sample"].mean().round(3).to_csv("latency.csv")

    mets = ["Accuracy", "Precision", "Recall", "F1", "AUROC"]
    summ = res.groupby(["Model", "Size"])[mets].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv("results_summary.csv", index=False)

    print(f"\n=== RESULTS at {MAX_TRAIN} labelled training samples (mean over seeds) ===")
    final = summ[summ["Size"].isin([MAX_TRAIN, 0])]
    print(final[["Model", "Accuracy_mean", "Precision_mean", "Recall_mean", "F1_mean", "F1_std", "AUROC_mean"]]
          .to_string(index=False))
    print("\n=== Inference latency (ms per sample) ===")
    print(pd.read_csv("latency.csv").to_string(index=False))

    # Learning curve
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for model, d in res[res["Size"] > 0].groupby("Model"):
        s = d.groupby("Size")["F1"].agg(["mean", "std"]).fillna(0)
        style = dict(lw=3) if model.startswith("E.") else {}
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=model, **style)
    ax.set_xlabel("Labelled training samples")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Hybrid AI+Quantum IDS vs existing methods (UNSW-NB15, {N_QUBITS} qubits)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    print("\nSaved: results_all.csv, results_summary.csv, latency.csv, learning_curve.png")


if __name__ == "__main__":
    main()
