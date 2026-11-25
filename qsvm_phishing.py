"""QSVM phishing detector (PennyLane simulator). URL in -> phishing / suspicious / likely safe out.

Pipeline: URL string -> ~28 lexical features (no network lookups) -> log + scale -> pick n_qubits best
features (mutual information) -> angle-encode in an n-qubit circuit -> fidelity kernel -> SVM
-> two tiers (warning / critical) + calibrated confidence.

Train (URL CSV with one URL column and one label column):
  python qsvm_phishing.py train --csv urls.csv --url-col url --label-col label
Train (dataset that already has numeric features, e.g. UCI "Phishing Websites"; there -1 = phishing):
  python qsvm_phishing.py train --tabular phishing.csv --label-col Result --phish-label=-1
Scan:
  python qsvm_phishing.py scan "http://paypa1-secure-login.xyz/verify?id=1"
  python qsvm_phishing.py scan --file urls.txt
"""
import argparse, math, re, time
from urllib.parse import urlparse
import numpy as np, pandas as pd, joblib
import pennylane as qml
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import functools

SHORT = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly", "shorturl.at"}
RISKY_TLD = {"xyz", "top", "click", "icu", "tk", "ml", "ga", "cf", "gq", "buzz", "rest", "work", "zip", "mov", "cam"}
KEYWORDS = re.compile(r"login|signin|verify|secure|account|update|bank|wallet|password|confirm|otp|kyc|"
                      r"support|billing|invoice|recover|unlock|suspend|alert", re.I)
BRANDS = ("paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook", "instagram", "whatsapp",
          "sbi", "hdfc", "icici", "paytm", "dhl", "fedex", "dropbox", "linkedin", "binance", "coinbase")
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}
SEVERITY = {0: "likely_safe", 1: "suspicious", 2: "phishing"}


# ---------------------------------------------------------------- features
def _entropy(s):
    if not s: return 0.0
    return -sum(c / len(s) * math.log2(c / len(s)) for c in (s.count(ch) for ch in set(s)))


def url_features(url):
    u = str(url).strip(); p = urlparse(u if "://" in u else "http://" + u)
    host, path, query = (p.hostname or "").lower(), p.path or "", p.query or ""
    labels = host.split(".") if host else []
    k = 3 if len(labels) >= 3 and labels[-2] in SECOND_LEVEL and len(labels[-1]) == 2 else 2
    registered = ".".join(labels[-k:]); sub = ".".join(labels[:-k]) if len(labels) > k else ""
    try: port = p.port
    except ValueError: port = None
    nd = lambda s: sum(ch.isdigit() for ch in s)
    return {
        "url_len": len(u), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        "n_dots_host": host.count("."), "n_subdomains": len(sub.split(".")) if sub else 0,
        "n_digits": nd(u), "digit_ratio": nd(u) / max(1, len(u)), "digits_in_host": nd(host),
        "n_hyphen_host": host.count("-"), "n_at": u.count("@"), "n_qmark": u.count("?"), "n_amp": u.count("&"),
        "n_eq": u.count("="), "n_pct": u.count("%"), "path_depth": path.count("/"),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))), "https": int(p.scheme == "https"),
        "has_port": int(port is not None), "shortener": int(host in SHORT), "punycode": int("xn--" in host),
        "risky_tld": int(bool(labels) and labels[-1] in RISKY_TLD), "kw_count": len(KEYWORDS.findall(u)),
        "brand_in_sub": int(any(b in sub for b in BRANDS) and not any(b in registered.split(".")[0] for b in BRANDS)),
        "brand_in_path": int(any(b in (path + query).lower() for b in BRANDS)),
        "host_entropy": _entropy(host), "longest_token": max((len(t) for t in re.split(r"[^A-Za-z0-9]+", u)), default=0),
        "double_slash_path": int("//" in path),
    }


FEATURES = list(url_features("http://a.example.com/").keys())


def features_df(urls):
    return pd.DataFrame([url_features(u) for u in urls], columns=FEATURES)


def evidence(f):
    ev = []
    if f["has_ip"]: ev.append("host is a raw IP address")
    if f["shortener"]: ev.append("link shortener hides the destination")
    if f["punycode"]: ev.append("punycode host (possible look-alike domain)")
    if f["n_at"]: ev.append("contains '@' (can disguise the real host)")
    if f["brand_in_sub"]: ev.append("brand name used inside an unrelated domain")
    if f["risky_tld"]: ev.append("top-level domain often abused for scams")
    if f["kw_count"] >= 2: ev.append("several credential/urgency words in the URL")
    if f["n_subdomains"] >= 3: ev.append("unusually many subdomains")
    if not f["https"]: ev.append("not using HTTPS")
    if f["url_len"] > 90: ev.append("very long URL")
    return ev


# ---------------------------------------------------------------- quantum kernel
class QuantumKernel:
    """Fidelity kernel K_ij = |<psi_i|psi_j>|^2; each sample's statevector is simulated once."""

    def __init__(self, n_qubits=8, reps=2, fmap="zz", bw=1.0, device="default.qubit", chunk=128):
        self.n, self.reps, self.fmap, self.bw = n_qubits, reps, fmap, bw
        self.device, self.chunk, self._circuit = device, chunk, None

    def __getstate__(self):
        d = self.__dict__.copy(); d["_circuit"] = None; return d

    def _build(self):
        n, reps, fmap = self.n, self.reps, self.fmap
        dev = qml.device(self.device, wires=n)

        @qml.qnode(dev)
        def circuit(x):
            for _ in range(reps):
                for i in range(n):
                    qml.Hadamard(wires=i); qml.RZ(x[..., i], wires=i)
                if fmap == "product": continue                 # control: no entanglement
                pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if fmap == "ring" and n > 2 else [])
                for i, j in pairs:
                    qml.CNOT(wires=[i, j]); qml.RZ((np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j); qml.CNOT(wires=[i, j])
            return qml.state()
        return circuit

    def states(self, Z):
        if self._circuit is None: self._circuit = self._build()
        Z = np.asarray(Z, dtype=float) * self.bw
        return np.vstack([np.asarray(self._circuit(Z[s:s + self.chunk])) for s in range(0, len(Z), self.chunk)]).astype(np.complex64)

    @staticmethod
    def kernel(A, B):
        return (np.abs(A @ B.conj().T) ** 2).astype(np.float64)


# ---------------------------------------------------------------- thresholds
def pick_threshold(y, scores, beta=2.0):
    """WARN tier: maximise F-beta (beta>1 favours catching phishing over false alarms)."""
    p, r, t = precision_recall_curve(y, scores)
    fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-12)
    i = int(np.argmax(fb[:-1]))
    # precision_recall_curve flags score >= t[i], but callers test score > threshold. Use the midpoint to the next
    # lower distinct score: ties land on the flagged side and the cut sits in the middle of the gap, not on its edge.
    return float((t[i] + t[i - 1]) / 2) if i > 0 else float(np.nextafter(t[0], -np.inf))


def pick_precision_threshold(y, scores, target=0.95, floor=-np.inf, min_pos=10):
    """CRITICAL tier: lowest threshold with precision >= target (inf = unreachable -> tier disabled)."""
    p, r, t = precision_recall_curve(y, scores)
    ss = np.sort(scores); n_pred = len(ss) - np.searchsorted(ss, t, side="left")
    ok = (p[:-1] >= target) & (t >= floor) & (n_pred >= min_pos)
    return float(t[ok][0]) if ok.any() else float("inf")


# ---------------------------------------------------------------- the QSVM
class PhishQSVM:
    def __init__(self, n_qubits=8, reps=2, fmap="zz", bw_grid=(0.5, 1.0, 2.0), C_grid=(0.5, 2.0, 8.0),
                 beta=2.0, cv=3, device="default.qubit", target_precision=0.95, mode="url"):
        self.n_qubits, self.reps, self.fmap, self.bw_grid, self.C_grid = n_qubits, reps, fmap, bw_grid, C_grid
        self.beta, self.cv, self.device, self.target_precision, self.mode = beta, cv, device, target_precision, mode

    def _front(self):
        steps = [("log", FunctionTransformer(np.log1p))] if self.mode == "url" else []   # counts/lengths are heavy-tailed
        steps += [("sc", StandardScaler()),
                  ("sel", SelectKBest(functools.partial(mutual_info_classif, random_state=0), k=self.n_qubits)),
                  ("mm", MinMaxScaler((0, np.pi), clip=True))]
        return Pipeline(steps)

    def _oof(self, K, y, C):
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(self.cv, shuffle=True, random_state=0).split(K, y):
            svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(tr, tr)], y[tr])
            oof[va] = svc.decision_function(K[np.ix_(va, tr)])
        return oof

    def fit(self, X, y, verbose=True):
        y = np.asarray(y); self.columns = list(X.columns)
        self.front = self._front(); Z = self.front.fit_transform(X, y)
        self.selected = [c for c, keep in zip(self.columns, self.front.named_steps["sel"].get_support()) if keep]
        best, self.cv_table = None, []
        for bw in self.bw_grid:
            qk = QuantumKernel(self.n_qubits, self.reps, self.fmap, bw, self.device)
            t = time.time(); S = qk.states(Z); K = qk.kernel(S, S)
            if verbose: print(f"  bandwidth {bw}: simulated {len(Z)} states in {time.time() - t:.1f}s")
            for C in self.C_grid:
                oof = self._oof(K, y, C); ap = average_precision_score(y, oof); self.cv_table.append((bw, C, ap))
                if best is None or ap > best[0]: best = (ap, bw, C, qk, S, K, oof)
        self.cv_ap, self.bw, self.C, self.qk, S, K, oof = best
        self.threshold = pick_threshold(y, oof, self.beta)
        self.act_threshold = pick_precision_threshold(y, oof, self.target_precision, floor=self.threshold)
        self.platt = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)           # score -> P(phishing)
        pw, pa = oof > self.threshold, oof >= self.act_threshold
        self.tiers = {"warn": (precision_score(y, pw, zero_division=0), recall_score(y, pw, zero_division=0)),
                      "act": (precision_score(y, pa, zero_division=0), recall_score(y, pa, zero_division=0))
                      if np.isfinite(self.act_threshold) else None}
        svc = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(K, y)
        self.sv_states = S[svc.support_]                       # inference needs only the support vectors
        self.dual, self.b = svc.dual_coef_[0].astype(float), float(svc.intercept_[0])
        return self

    def assess(self, X):
        """-> raw score, calibrated confidence (0-1), severity 0 likely_safe / 1 suspicious / 2 phishing."""
        Z = self.front.transform(X[self.columns]); S = self.qk.states(Z)
        raw = self.qk.kernel(S, self.sv_states) @ self.dual + self.b
        sev = (raw > self.threshold).astype(int); sev[(raw >= self.act_threshold) & (raw > self.threshold)] = 2
        return {"raw": raw, "score": raw - self.threshold,
                "confidence": self.platt.predict_proba(raw.reshape(-1, 1))[:, 1], "severity": sev}

    def assess_urls(self, urls):
        if self.mode != "url": raise ValueError("this model was trained on tabular features, not raw URLs")
        return self.assess(features_df(urls))

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


def finding(url, model_path="phish_qsvm.joblib"):
    """Result in the same shape the SentinelPi dashboard uses."""
    m = PhishQSVM.load(model_path); A = m.assess_urls([url]); sev = int(A["severity"][0])
    return {"pipeline": "phishing_url", "target": url, "score": round(float(A["confidence"][0]), 3), "severity": sev,
            "verdict": {0: "likely_safe", 1: "suspicious", 2: "malicious"}[sev], "evidence": evidence(url_features(url))}


# ---------------------------------------------------------------- data + CLI
POS = {"1", "phishing", "phish", "bad", "malicious", "malware", "unsafe", "spam"}


def parse_labels(col, phish="auto"):
    s = col.astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    return s.isin(POS if phish == "auto" else {str(phish).strip().lower()}).astype(int).values


def subsample_idx(y, n, seed):
    rng = np.random.default_rng(seed); idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]; idx.extend(rng.choice(ci, size=min(int(round(n * len(ci) / len(y))), len(ci)), replace=False))
    return np.array(rng.permutation(idx))


def metrics(y, scores, pred):
    return {"AP": average_precision_score(y, scores), "AUC": roc_auc_score(y, scores),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0)}


def rbf_baseline(det, Xtr, ytr, Xte):
    Ztr, Zte = det.front.transform(Xtr), det.front.transform(Xte)
    gs = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [0.5, 2, 8], "gamma": [0.05, 0.2, 1.0]},
                      scoring="average_precision", cv=det.cv).fit(Ztr, ytr)
    oof = cross_val_predict(gs.best_estimator_, Ztr, ytr, method="decision_function",
                            cv=StratifiedKFold(det.cv, shuffle=True, random_state=0))
    s = gs.decision_function(Zte) - pick_threshold(ytr, oof, det.beta)
    return s, (s > 0).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "scan"]); ap.add_argument("url", nargs="?")
    ap.add_argument("--csv"); ap.add_argument("--url-col", default="url"); ap.add_argument("--label-col", default="label")
    ap.add_argument("--tabular", help="CSV that already holds numeric feature columns"); ap.add_argument("--drop", default="")
    ap.add_argument("--phish-label", default="auto", help="value meaning phishing (UCI Phishing Websites: -1)")
    ap.add_argument("--qubits", type=int, default=8); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--map", default="zz", choices=["zz", "ring", "product"]); ap.add_argument("--compare", action="store_true")
    ap.add_argument("--bw", default="0.5,1,2"); ap.add_argument("--C", default="0.5,2,8")
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--n-train", type=int, default=1000); ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--model", default="phish_qsvm.joblib"); ap.add_argument("--device", default="default.qubit")
    ap.add_argument("--file", help="scan: text file with one URL per line")
    a = ap.parse_args(); fl = lambda s: tuple(float(v) for v in s.split(","))

    if a.cmd == "scan":
        urls = [l.strip() for l in open(a.file) if l.strip()] if a.file else [a.url]
        if not urls or urls == [None]: ap.error("give a URL or --file")
        m = PhishQSVM.load(a.model); A = m.assess_urls(urls)
        for u, sev, conf in zip(urls, A["severity"], A["confidence"]):
            print(f"{SEVERITY[int(sev)].upper():<11} confidence {conf:.2f}  {u}")
            for e in evidence(url_features(u)): print(f"    - {e}")
        return

    if a.tabular:
        df = pd.read_csv(a.tabular); y = parse_labels(df[a.label_col], a.phish_label)
        X = df.drop(columns=[a.label_col] + [c for c in a.drop.split(",") if c]).select_dtypes("number").reset_index(drop=True)
        mode, src = "tabular", None
    elif a.csv:
        df = pd.read_csv(a.csv)[[a.url_col, a.label_col]].dropna()
        df[a.url_col] = df[a.url_col].astype(str).str.strip(); df = df.drop_duplicates(subset=a.url_col).reset_index(drop=True)
        y = parse_labels(df[a.label_col], a.phish_label); src, mode = df[a.url_col], "url"
    else:
        ap.error("train needs --csv or --tabular")
    print(f"{len(y)} samples: {int(y.sum())} phishing / {int((1 - y).sum())} legitimate  <- check this is the right way round")
    if y.min() == y.max(): raise SystemExit("only one class found; check --label-col / --phish-label")
    tr, te = train_test_split(np.arange(len(y)), test_size=min(a.n_test, len(y) // 4), stratify=y, random_state=0)
    tr = tr[subsample_idx(y[tr], a.n_train, 1)]
    feat = (lambda ix: features_df(src.iloc[ix])) if mode == "url" else (lambda ix: X.iloc[ix].reset_index(drop=True))
    Xtr, Xte, ytr, yte = feat(tr), feat(te), y[tr], y[te]

    rows, last = [], None
    for fm in (["product", "zz", "ring"] if a.compare else [a.map]):
        print(f"== feature map: {fm} ==")
        det = PhishQSVM(a.qubits, a.reps, fm, fl(a.bw), fl(a.C), a.beta, a.cv, a.device, a.target_precision, mode)
        t = time.time(); det.fit(Xtr, ytr); fit_s = time.time() - t
        t = time.time(); A = det.assess(Xte); ms = 1000 * (time.time() - t) / len(Xte)
        print(f"best bandwidth={det.bw}, C={det.C}, cv-AP={det.cv_ap:.3f}, SVs={len(det.dual)}/{len(ytr)}, fit {fit_s:.0f}s, {ms:.1f} ms/URL")
        print(f"  features used by the quantum circuit: {', '.join(det.selected)}")
        tw, ta = det.tiers["warn"], det.tiers["act"]
        print(f"  warning tier : out-of-fold precision {tw[0]:.2f}, recall {tw[1]:.2f}")
        print("  critical tier: " + (f"out-of-fold precision {ta[0]:.2f}, recall {ta[1]:.2f}" if ta else
              f"DISABLED (precision {a.target_precision:.2f} not reachable)"))
        print(f"  calibration (test Brier score, lower is better): {brier_score_loss(yte, A['confidence']):.3f}")
        pw = (A["severity"] >= 1).astype(int)
        rows += [(f"QSVM-{fm} warn", metrics(yte, A["score"], pw)),
                 (f"QSVM-{fm} crit", metrics(yte, A["score"], (A["severity"] == 2).astype(int)))]; last = (det, pw)
    bs, bp = rbf_baseline(last[0], Xtr, ytr, Xte); rows.append(("RBF-SVM (tuned)", metrics(yte, bs, bp)))
    print("\nTest results:"); print(f"{'model':<20}{'AP':>7}{'AUC':>7}{'prec':>7}{'recall':>8}")
    for n, m in rows: print(f"{n:<20}{m['AP']:>7.3f}{m['AUC']:>7.3f}{m['precision']:>7.3f}{m['recall']:>8.3f}")
    if not a.compare:
        print("\nConfusion matrix at the warning tier [legit, phishing]:\n", confusion_matrix(yte, last[1]))
        last[0].save(a.model); print("saved", a.model)


if __name__ == "__main__":
    main()
