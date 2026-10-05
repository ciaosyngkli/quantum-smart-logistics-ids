"""
Classical vs Quantum (QSVC) intrusion detection on UNSW-NB15 (binary).
Runs on a normal laptop CPU. Produces results.csv and results.png.

SETUP
  pip install numpy pandas scikit-learn matplotlib qiskit qiskit-machine-learning

DATA
  Download UNSW-NB15 "training-set" and "testing-set" CSVs
  (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv) and put them
  in the same folder as this script.

RUN
  python qids_compare.py
  (first run with N_QUANTUM_TRAIN = 100 to check it works, then raise it)
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------- CONFIG ---------------------------------
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV = "UNSW_NB15_testing-set.csv"
N_QUBITS = 4               # = number of PCA features (4-6 is laptop-friendly)
N_QUANTUM_TRAIN = 400      # start small (100), raise to 400-1000 if time allows
N_QUANTUM_TEST = 200
N_CLASSICAL_TRAIN = 20000  # classical models can use far more data
SEED = 42
# -----------------------------------------------------------------------

rng = np.random.RandomState(SEED)


def load():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    for df in (tr, te):
        df.drop(columns=[c for c in ("id", "attack_cat") if c in df.columns],
                inplace=True)
    return tr, te


def balanced_sample(df, n):
    """Equal normal/attack samples so accuracy is not misleading."""
    half = n // 2
    parts = [g.sample(min(half, len(g)), random_state=SEED)
             for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def evaluate(name, y_true, y_pred, train_s, n_train, n_feat):
    return {
        "Model": name,
        "Train rows": n_train,
        "Features": n_feat,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Train time (s)": round(train_s, 1),
    }


def main():
    print("Loading data...")
    train_df, test_df = load()

    # Encode text columns (proto, service, state) as numbers
    cat_cols = train_df.select_dtypes(include="object").columns.tolist()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(pd.concat([train_df[cat_cols], test_df[cat_cols]]))
    for df in (train_df, test_df):
        df[cat_cols] = enc.transform(df[cat_cols])

    results = []

    # ---------- Classical baselines (bigger sample, full feature set) ----------
    ctr = balanced_sample(train_df, N_CLASSICAL_TRAIN)
    cte = balanced_sample(test_df, 5000)
    Xc_tr, yc_tr = ctr.drop(columns="label").values, ctr["label"].values
    Xc_te, yc_te = cte.drop(columns="label").values, cte["label"].values
    sc = MinMaxScaler().fit(Xc_tr)
    Xc_tr, Xc_te = sc.transform(Xc_tr), sc.transform(Xc_te)
    n_feat_full = Xc_tr.shape[1]

    for name, model in [
        ("Random Forest (full features)", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
        ("kNN (full features)", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]:
        t = time.time()
        model.fit(Xc_tr, yc_tr)
        el = time.time() - t
        results.append(evaluate(name, yc_te, model.predict(Xc_te), el, len(Xc_tr), n_feat_full))
        print("done:", name)

    # ---------- Fair comparison: SAME small data + SAME PCA features ----------
    qtr = balanced_sample(train_df, N_QUANTUM_TRAIN)
    qte = balanced_sample(test_df, N_QUANTUM_TEST)
    Xq_tr, yq_tr = qtr.drop(columns="label").values, qtr["label"].values
    Xq_te, yq_te = qte.drop(columns="label").values, qte["label"].values

    sc2 = MinMaxScaler().fit(Xq_tr)
    pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(sc2.transform(Xq_tr))
    Xq_tr = pca.transform(sc2.transform(Xq_tr))
    Xq_te = pca.transform(sc2.transform(Xq_te))
    # scale to [0, pi] for quantum angle encoding
    sc3 = MinMaxScaler(feature_range=(0, np.pi)).fit(Xq_tr)
    Xq_tr, Xq_te = sc3.transform(Xq_tr), sc3.transform(Xq_te)

    for name, model in [
        (f"Classical SVM (RBF, {N_QUBITS} PCA feats)", SVC(kernel="rbf")),
        (f"Random Forest ({N_QUBITS} PCA feats)", RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ]:
        t = time.time()
        model.fit(Xq_tr, yq_tr)
        el = time.time() - t
        results.append(evaluate(name, yq_te, model.predict(Xq_te), el, len(Xq_tr), N_QUBITS))
        print("done:", name)

    # ---------- Quantum SVM (simulated on CPU) ----------
    print("Training QSVC (this is the slow part)...")
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC

    feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2, entanglement="linear")
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)

    t = time.time()
    qsvc.fit(Xq_tr, yq_tr)
    el = time.time() - t
    results.append(evaluate(f"Quantum SVM (QSVC, ZZFeatureMap, {N_QUBITS} qubits)",
                            yq_te, qsvc.predict(Xq_te), el, len(Xq_tr), N_QUBITS))
    print("done: QSVC")

    # ---------- Save + plot ----------
    res = pd.DataFrame(results)
    res.to_csv("results.csv", index=False)
    print("\n", res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(res))
    w = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, res[m], w, label=m)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(res["Model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Classical vs Quantum intrusion detection (UNSW-NB15, binary)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved results.csv and results.png")


if __name__ == "__main__":
    main()
