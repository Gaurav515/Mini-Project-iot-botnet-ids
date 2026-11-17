#!/usr/bin/env python3
"""
IoT Network Anomaly Detection using Machine Learning
Exp 7 - Intrusion Detection System (IDS) for IoT botnet traffic

Trains and compares several ML algorithms to separate Benign vs Malicious
(botnet) network flows, using Zeek-style flow features from the IoT-23 dataset.

Two data modes:
  1. Real IoT-23 data  : python iot_ids.py --data path/to/conn.log.labeled [more files...]
  2. Synthetic data    : python iot_ids.py            (IoT-23-like flows, default)

Outputs (in ./results): metrics.csv, plots (PNG) and the best trained model.
"""
import argparse
import os
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

NUMERIC = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
           "orig_ip_bytes", "resp_ip_bytes"]
CATEGORICAL = ["proto", "service", "conn_state", "resp_port"]


# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------
def load_iot23(paths):
    """Load one or more Zeek `conn.log.labeled` files from IoT-23."""
    frames = []
    for path in paths:
        fields, rows = None, []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#fields"):
                    fields = line.split("\t")[1:]
                    continue
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                # last column holds "tunnel_parents   label   detailed-label"
                tail = parts[-1].split()
                parts = parts[:-1] + (tail + ["-", "-", "-"])[:3]
                rows.append(parts)
        cols = fields[:-1] + ["tunnel_parents", "label", "detailed_label"]
        frames.append(pd.DataFrame(rows, columns=cols))
    df = pd.concat(frames, ignore_index=True).replace("-", np.nan)
    df = df.rename(columns={"id.resp_p": "resp_port"})
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["resp_port"] = df["resp_port"].astype(str)
    df["target"] = (df["label"].str.lower() != "benign").astype(int)
    return df[NUMERIC + CATEGORICAL + ["target"]]


def make_synthetic(n=30000, malicious_ratio=0.6, label_noise=0.01, seed=42):
    """Generate IoT-23-like flows (benign IoT traffic vs Mirai/Okiru-style botnet)."""
    rng = np.random.default_rng(seed)

    def pick(options, m, p=None):
        return rng.choice(options, size=m, p=p)

    def lognorm(mu, sigma, m):
        return rng.lognormal(mu, sigma, m)

    def gen(kind, m):
        d = {}
        if kind == "web":
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["http", "ssl"], m)
            d["conn_state"] = pick(["SF", "S1", "RSTO", "REJ"], m, [.93, .03, .02, .02])
            d["duration"] = lognorm(0.0, 1.2, m)
            d["orig_pkts"] = rng.poisson(12, m) + 1
            d["resp_pkts"] = rng.poisson(15, m) + 1
            d["orig_bytes"] = lognorm(6.0, 1.0, m)
            d["resp_bytes"] = lognorm(8.0, 1.5, m)
            d["resp_port"] = pick(["80", "443", "8080"], m, [.4, .5, .1])
        elif kind == "dns":
            d["proto"] = pick(["udp"], m)
            d["service"] = pick(["dns"], m)
            d["conn_state"] = pick(["SF", "S0"], m, [.96, .04])
            d["duration"] = lognorm(-3.0, 0.8, m)
            d["orig_pkts"] = rng.integers(1, 3, m)
            d["resp_pkts"] = rng.integers(1, 3, m)
            d["orig_bytes"] = lognorm(3.7, 0.3, m)
            d["resp_bytes"] = lognorm(4.8, 0.4, m)
            d["resp_port"] = pick(["53"], m)
        elif kind == "mqtt":
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["mqtt", "-"], m, [.8, .2])
            d["conn_state"] = pick(["SF", "S1"], m, [.95, .05])
            d["duration"] = lognorm(0.5, 0.5, m)
            d["orig_pkts"] = rng.poisson(6, m) + 1
            d["resp_pkts"] = rng.poisson(5, m) + 1
            d["orig_bytes"] = lognorm(4.8, 0.4, m)
            d["resp_bytes"] = lognorm(4.0, 0.5, m)
            d["resp_port"] = pick(["1883", "8883"], m, [.7, .3])
        elif kind == "ntp":
            d["proto"] = pick(["udp"], m)
            d["service"] = pick(["ntp"], m)
            d["conn_state"] = pick(["SF"], m)
            d["duration"] = lognorm(-2.5, 0.4, m)
            d["orig_pkts"] = np.ones(m)
            d["resp_pkts"] = np.ones(m)
            d["orig_bytes"] = np.full(m, 48.0)
            d["resp_bytes"] = np.full(m, 48.0)
            d["resp_port"] = pick(["123"], m)
        elif kind == "flaky":  # benign devices with failed connections
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["-"], m)
            d["conn_state"] = pick(["S0", "REJ", "RSTOS0"], m, [.6, .3, .1])
            d["duration"] = np.where(rng.random(m) < .7, 0.0, lognorm(0.5, 0.8, m))
            d["orig_pkts"] = rng.integers(1, 4, m)
            d["resp_pkts"] = np.zeros(m)
            d["orig_bytes"] = np.zeros(m)
            d["resp_bytes"] = np.zeros(m)
            d["resp_port"] = pick(["80", "443", "8080", "8883", "5683"], m)
        elif kind == "scan":  # Mirai-style horizontal port scan
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["-"], m)
            d["conn_state"] = pick(["S0", "REJ", "RSTOS0"], m, [.8, .12, .08])
            d["duration"] = np.where(rng.random(m) < .8, 0.0, lognorm(1.0, 0.5, m))
            d["orig_pkts"] = rng.integers(1, 4, m)
            d["resp_pkts"] = np.zeros(m)
            d["orig_bytes"] = np.zeros(m)
            d["resp_bytes"] = np.zeros(m)
            d["resp_port"] = pick(["23", "2323", "22", "80", "8080", "37215", "5555", "81"],
                                  m, [.4, .2, .1, .1, .05, .05, .05, .05])
        elif kind == "brute":  # telnet brute-force / infection
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["-"], m)
            d["conn_state"] = pick(["SF", "RSTO", "S1"], m, [.5, .35, .15])
            d["duration"] = lognorm(1.6, 0.6, m)
            d["orig_pkts"] = rng.poisson(12, m) + 4
            d["resp_pkts"] = rng.poisson(9, m) + 2
            d["orig_bytes"] = lognorm(6.0, 0.5, m)
            d["resp_bytes"] = lognorm(5.3, 0.6, m)
            d["resp_port"] = pick(["23", "2323", "22"], m, [.6, .25, .15])
        elif kind == "c2":  # command & control beaconing
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["-", "irc", "http"], m, [.6, .2, .2])
            d["conn_state"] = pick(["SF", "S1", "RSTO"], m, [.8, .1, .1])
            d["duration"] = lognorm(0.9, 0.8, m)
            d["orig_pkts"] = rng.poisson(6, m) + 2
            d["resp_pkts"] = rng.poisson(6, m) + 2
            d["orig_bytes"] = lognorm(5.0, 0.7, m)
            d["resp_bytes"] = lognorm(5.4, 0.8, m)
            d["resp_port"] = pick(["6667", "8080", "443", "4444", "9999", "1337"], m)
        elif kind == "ddos":  # flood
            d["proto"] = pick(["udp", "tcp"], m, [.5, .5])
            d["service"] = pick(["-"], m)
            d["conn_state"] = pick(["S0", "SF", "OTH"], m, [.6, .3, .1])
            d["duration"] = lognorm(1.5, 0.9, m)
            d["orig_pkts"] = lognorm(5.0, 0.8, m)
            d["resp_pkts"] = np.where(rng.random(m) < .8, 0, rng.poisson(2, m))
            d["orig_bytes"] = d["orig_pkts"] * rng.uniform(0, 60, m)
            d["resp_bytes"] = np.zeros(m)
            d["resp_port"] = pick(["80", "53", "443", "8080"], m)
        elif kind == "dl":  # malware download over HTTP (looks like web)
            d["proto"] = pick(["tcp"], m)
            d["service"] = pick(["http"], m)
            d["conn_state"] = pick(["SF"], m)
            d["duration"] = lognorm(0.4, 0.9, m)
            d["orig_pkts"] = rng.poisson(10, m) + 2
            d["resp_pkts"] = rng.poisson(30, m) + 3
            d["orig_bytes"] = lognorm(5.8, 0.8, m)
            d["resp_bytes"] = lognorm(9.5, 1.0, m)
            d["resp_port"] = pick(["80", "8080"], m, [.6, .4])
        d["orig_ip_bytes"] = d["orig_bytes"] + np.asarray(d["orig_pkts"]) * 28
        d["resp_ip_bytes"] = d["resp_bytes"] + np.asarray(d["resp_pkts"]) * 28
        return pd.DataFrame(d)

    benign_mix = {"web": .30, "dns": .25, "mqtt": .25, "ntp": .08, "flaky": .12}
    mal_mix = {"scan": .40, "brute": .15, "c2": .15, "ddos": .20, "dl": .10}
    n_mal = int(n * malicious_ratio)
    n_ben = n - n_mal
    parts = []
    for mix, total, label in [(benign_mix, n_ben, 0), (mal_mix, n_mal, 1)]:
        for kind, frac in mix.items():
            df = gen(kind, int(total * frac))
            df["target"] = label
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    flip = rng.random(len(df)) < label_noise  # a little label noise, as in real data
    df.loc[flip, "target"] = 1 - df.loc[flip, "target"]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[NUMERIC + CATEGORICAL + ["target"]]


# --------------------------------------------------------------------------
# 2. PREPROCESSING + MODELS
# --------------------------------------------------------------------------
def build_preprocessor():
    num = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # tame heavy-tailed byte/packet counts
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=0.01, sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def get_models(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Linear SVM": LinearSVC(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
        "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                          random_state=seed),
    }


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics_row(name, y_true, y_pred, y_score, t_train, t_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "Train_s": t_train,
        "Predict_s": t_pred,
    }


# --------------------------------------------------------------------------
# 3. PLOTS
# --------------------------------------------------------------------------
def plot_class_distribution(y, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = pd.Series(y).map({0: "Benign", 1: "Malicious"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette=["#2a9d8f", "#e76f51"], ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Class distribution")
    ax.set_ylabel("Flows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(res, path):
    m = res.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
                 var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=m, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(max(0, m["Score"].min() - 0.05), 1.01)
    ax.set_title("Algorithm comparison - IoT botnet detection")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusions(cms, path):
    n = len(cms)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.array(axes).ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(names, importances, path, top=15):
    idx = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(np.array(names)[idx], importances[idx], color="#264653")
    ax.set_title("Random Forest - top feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="IoT botnet IDS - ML algorithm comparison")
    ap.add_argument("--data", nargs="*", help="IoT-23 conn.log.labeled file(s). "
                                              "If omitted, synthetic IoT-23-like data is used.")
    ap.add_argument("--samples", type=int, default=30000, help="synthetic sample count")
    ap.add_argument("--max-rows", type=int, default=200000,
                    help="subsample real data to this many rows (speed)")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- data
    if args.data:
        df = load_iot23(args.data)
        if len(df) > args.max_rows:
            df = df.sample(args.max_rows, random_state=args.seed)
        source = f"IoT-23 ({', '.join(os.path.basename(p) for p in args.data)})"
    else:
        df = make_synthetic(n=args.samples, seed=args.seed)
        source = "Synthetic IoT-23-like flows"
    X, y = df[NUMERIC + CATEGORICAL], df["target"].values
    print("=" * 70)
    print("IoT NETWORK ANOMALY DETECTION - INTRUSION DETECTION SYSTEM")
    print("=" * 70)
    print(f"Data source : {source}")
    print(f"Total flows : {len(df):,}   (Benign={np.sum(y == 0):,}, Malicious={np.sum(y == 1):,})")
    print(f"Features    : {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          stratify=y, random_state=args.seed)
    print(f"Train / Test: {len(Xtr):,} / {len(Xte):,}\n")
    plot_class_distribution(y, f"{args.out}/class_distribution.png")

    # ---- supervised models
    rows, curves, cms, fitted = [], {}, {}, {}
    print(f"{'Model':<26}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'AUC':>8}{'Train(s)':>10}")
    print("-" * 76)
    for name, clf in get_models(args.seed).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = pipe.predict(Xte)
        t_pred = time.perf_counter() - t0
        score = scores_of(pipe, Xte)
        r = metrics_row(name, yte, pred, score, t_train, t_pred)
        rows.append(r)
        fpr, tpr, _ = roc_curve(yte, score)
        curves[name] = (fpr, tpr, r["ROC_AUC"])
        cms[name] = confusion_matrix(yte, pred)
        fitted[name] = pipe
        print(f"{name:<26}{r['Accuracy']:>8.4f}{r['Precision']:>8.4f}{r['Recall']:>8.4f}"
              f"{r['F1']:>8.4f}{r['ROC_AUC']:>8.4f}{t_train:>10.2f}")

    # ---- unsupervised anomaly detector: trained on BENIGN traffic only
    prep = build_preprocessor()
    Xb = prep.fit_transform(Xtr[ytr == 0])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=args.seed)
    t0 = time.perf_counter()
    iso.fit(Xb)
    t_train = time.perf_counter() - t0
    Xt = prep.transform(Xte)
    t0 = time.perf_counter()
    pred = (iso.predict(Xt) == -1).astype(int)
    t_pred = time.perf_counter() - t0
    score = -iso.decision_function(Xt)
    name = "Isolation Forest (unsup.)"
    r = metrics_row(name, yte, pred, score, t_train, t_pred)
    rows.append(r)
    fpr, tpr, _ = roc_curve(yte, score)
    curves[name] = (fpr, tpr, r["ROC_AUC"])
    cms[name] = confusion_matrix(yte, pred)
    print(f"{name:<26}{r['Accuracy']:>8.4f}{r['Precision']:>8.4f}{r['Recall']:>8.4f}"
          f"{r['F1']:>8.4f}{r['ROC_AUC']:>8.4f}{t_train:>10.2f}")

    # ---- results
    res = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    res.round(4).to_csv(f"{args.out}/metrics.csv", index=False)
    best = res.iloc[0]["Model"]
    print("\n" + "=" * 70)
    print("RANKING (by F1-score)")
    print("=" * 70)
    print(res.round(4).to_string(index=False))
    print(f"\nBest model: {best}  (F1={res.iloc[0]['F1']:.4f}, AUC={res.iloc[0]['ROC_AUC']:.4f})")

    sup = res[~res["Model"].str.contains("unsup")]
    print(f"Fastest to train : {sup.sort_values('Train_s').iloc[0]['Model']}")
    print("\nClassification detail for best supervised model:")
    bsup = sup.iloc[0]["Model"]
    tn, fp, fn, tp = cms[bsup].ravel()
    print(f"  {bsup}: TN={tn} FP={fp} FN={fn} TP={tp}  "
          f"(False-alarm rate={fp / (fp + tn):.4f}, Miss rate={fn / (fn + tp):.4f})")

    # ---- plots
    plot_comparison(res, f"{args.out}/model_comparison.png")
    plot_roc(curves, f"{args.out}/roc_curves.png")
    plot_confusions(cms, f"{args.out}/confusion_matrices.png")
    rf = fitted["Random Forest"]
    plot_importance(rf.named_steps["prep"].get_feature_names_out(),
                    rf.named_steps["clf"].feature_importances_,
                    f"{args.out}/feature_importance.png")
    joblib.dump(fitted[bsup], f"{args.out}/best_model.joblib")
    print(f"\nSaved metrics.csv, 5 plots and best_model.joblib in ./{args.out}/")


if __name__ == "__main__":
    main()
