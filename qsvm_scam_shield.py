"""
QSVM Scam Shield
Pipeline: WARN -> DETECT -> ERADICATE

Run:
    pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn numpy
    python qsvm_scam_shield.py

Every change to the standard QSVM is tagged [MOD-1] ... [MOD-6] in the comments
and listed in QSVM_MODIFICATIONS.txt (generated on each run).
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# =====================================================================
# CONFIG
# =====================================================================
N_QUBITS = 4            # PCA components = qubits
FEATURE_MAP_REPS = 2    # [MOD-3]
C = 2.0                 # [MOD-4]
WARN_RECALL = 0.95      # [MOD-5] warn zone catches ~95% of known scams
MAX_WEIGHT = 1.5        # [MOD-2] maximum feature weight

URGENCY_WORDS = ["urgent", "immediately", "act now", "expires", "final notice",
                 "last chance", "within 24 hours", "suspended", "verify now"]
MONEY_WORDS = ["prize", "winner", "lottery", "fee", "gift card", "bitcoin",
               "wire transfer", "cash", "claim", "inheritance", "refund"]
SUSPICIOUS_TLDS = (".biz", ".xyz", ".top", ".click", ".loan", ".win", ".icu")

# =====================================================================
# FEATURE EXTRACTION (supporting code, not a QSVM change)
# =====================================================================
URL_RE = re.compile(r"https?://[^\s]+", re.I)


def extract_features(text, sender, sender_age_days):
    """Convert a raw message into the 6-feature vector used by the QSVM.
    Columns: n_urls, urgency_words, money_words, log_sender_age,
             spoofed_sender, link_mismatch."""
    lower = text.lower()
    urls = URL_RE.findall(text)

    urgency = sum(lower.count(w) for w in URGENCY_WORDS)
    money = sum(lower.count(w) for w in MONEY_WORDS)

    sender_dom = sender.split("@")[-1].lower()
    spoofed = float(
        sender_dom.endswith(SUSPICIOUS_TLDS)
        or bool(re.search(r"\d", sender_dom.split(".")[0]))
    )

    mismatch = 0.0
    for u in urls:
        d = urlparse(u).netloc.lower()
        if d and not (d == sender_dom or d.endswith("." + sender_dom)):
            mismatch = 1.0
            break

    vec = np.array([len(urls), urgency, money,
                    np.log1p(max(sender_age_days, 0)), spoofed, mismatch],
                   dtype=float)
    return vec, (urls[0] if urls else None)


def make_synthetic_data(n_legit=120, n_scam=40, seed=1):
    """Synthetic data in the same 6-feature format.
    Replace with a real labelled dataset for real use."""
    rng = np.random.default_rng(seed)

    def block(n, url, urg, mon, age, spoof, mism):
        return np.column_stack([
            rng.poisson(url, n),
            rng.poisson(urg, n),
            rng.poisson(mon, n),
            np.clip(rng.normal(age, 1.0, n), 0, None),
            (rng.random(n) < spoof).astype(float),
            (rng.random(n) < mism).astype(float),
        ])

    legit = block(n_legit, 1.0, 0.3, 0.3, 5.5, 0.10, 0.15)
    scam = block(n_scam, 1.5, 3.0, 2.5, 1.2, 0.85, 0.70)
    X = np.vstack([legit, scam])
    y = np.r_[np.zeros(n_legit), np.ones(n_scam)]
    return X, y


# =====================================================================
# QSVM MODEL (modified for scam detection)
# =====================================================================
class ScamQSVM:
    def __init__(self):
        n = N_QUBITS
        # [MOD-1] PCA + scaling; range leaves headroom for weights up to MAX_WEIGHT
        self.pca = PCA(n_components=n)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi / MAX_WEIGHT))
        self.feat_w = None                                          # [MOD-2]

        fmap = ZZFeatureMap(feature_dimension=n, reps=FEATURE_MAP_REPS,
                            entanglement="linear")                  # [MOD-3]
        kernel = FidelityQuantumKernel(feature_map=fmap)
        self.model = QSVC(quantum_kernel=kernel,
                          class_weight="balanced", C=C)             # [MOD-4]

        self.detect_thr = 0.0                                       # [MOD-5]
        self.warn_thr = -0.05                                       # [MOD-5]

    def _prep(self, X, fit=False, y=None):
        if fit:
            Xp = self.pca.fit_transform(X)
            mi = mutual_info_classif(Xp, y, random_state=0)         # [MOD-2]
            self.feat_w = 0.5 + (MAX_WEIGHT - 0.5) * mi / (mi.max() + 1e-9)
            Xp = self.scaler.fit_transform(Xp)
        else:
            Xp = self.scaler.transform(self.pca.transform(X))
        return np.clip(Xp * self.feat_w, 0, np.pi)                  # [MOD-2]

    def fit(self, X, y):
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0)
        self.model.fit(self._prep(Xtr, fit=True, y=ytr), ytr)

        # [MOD-5] dual thresholds calibrated on validation data
        s = self.model.decision_function(self._prep(Xval))
        scam_scores = s[yval == 1]
        self.detect_thr = 0.0
        warn = float(np.quantile(scam_scores, 1 - WARN_RECALL))
        self.warn_thr = min(warn, -0.05)   # warn zone always below detect
        return self

    def score(self, X):
        return self.model.decision_function(self._prep(np.atleast_2d(X)))


# =====================================================================
# WARN -> DETECT -> ERADICATE  [MOD-6]
# =====================================================================
class ScamShield:
    def __init__(self, model, state_dir="shield_state"):
        self.model = model
        self.dir = Path(state_dir)
        self.quarantine_dir = self.dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.blocklist_file = self.dir / "blocklist.json"
        self.log_file = self.dir / "incidents.jsonl"
        self.blocklist = set()
        if self.blocklist_file.exists():
            self.blocklist = set(json.loads(self.blocklist_file.read_text()))

    def handle(self, msg_id, text, sender, sender_age_days):
        # Known bad sender: stop immediately, before scoring
        if sender in self.blocklist:
            print(f"[BLOCKED]    {msg_id}: {sender} is already blocklisted")
            return self._log({"id": msg_id, "sender": sender,
                              "status": "blocked_known_sender"})

        vec, url = extract_features(text, sender, sender_age_days)
        s = float(self.model.score(vec)[0])
        result = {"id": msg_id, "sender": sender, "score": round(s, 3),
                  "status": "clean"}

        # STAGE 1: WARN
        if s >= self.model.warn_thr:
            print(f"[WARNING]    {msg_id}: suspicious message from {sender} "
                  f"(score={s:.2f})")
            result["status"] = "warned"
            result["action"] = "held_for_review"

        # STAGE 2: DETECT
        if s >= self.model.detect_thr:
            print(f"[DETECTED]   {msg_id}: scam confirmed (score={s:.2f})")
            result["status"] = "detected"
            # STAGE 3: ERADICATE
            self._eradicate(msg_id, text, sender, url, result)

        return self._log(result)

    def _eradicate(self, msg_id, text, sender, url, result):
        # 1. isolate the message (kept in quarantine for evidence/review)
        (self.quarantine_dir / f"{msg_id}.txt").write_text(text, encoding="utf-8")
        # 2. block sender and link, persist the blocklist
        self.blocklist.add(sender)
        if url:
            self.blocklist.add(url)
        self.blocklist_file.write_text(json.dumps(sorted(self.blocklist), indent=2))
        result["action"] = "quarantined+blocked"
        print(f"[ERADICATED] {msg_id}: quarantined; blocked {sender}"
              + (f" and {url}" if url else ""))

    def _log(self, result):
        result["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        return result


# =====================================================================
# MODIFICATIONS FILE (written automatically on each run)
# =====================================================================
MODS_TEXT = """MODIFICATIONS MADE TO THE STANDARD QSVM (SCAM DETECTION ONLY)
File: qsvm_scam_shield.py
================================================================

BASELINE QSVM: raw features -> ZZFeatureMap (reps=1) -> quantum kernel
-> SVC -> predict 0/1.

[MOD-1] Preprocessing            WHERE: ScamQSVM.__init__, ScamQSVM._prep
  - PCA reduces the 6 scam features to N_QUBITS (4) dimensions.
  - MinMaxScaler maps values to [0, pi/1.5] so that after weighting
    (max 1.5x) angles stay within [0, pi].

[MOD-2] Scam-feature weighting   WHERE: ScamQSVM._prep
  - Mutual information measures how strongly each component predicts scams.
  - Components are multiplied by weights 0.5..1.5, so scam-indicative
    features get larger rotation angles and more kernel influence.

[MOD-3] Feature map              WHERE: ScamQSVM.__init__, FEATURE_MAP_REPS
  - ZZFeatureMap reps 1 -> 2, linear entanglement, to capture feature
    interactions (urgency + money request + spoofed sender).

[MOD-4] Imbalance handling       WHERE: ScamQSVM.__init__ (QSVC arguments)
  - class_weight="balanced", C=2.0, because scams are rare and a missed
    scam costs more than a false alarm.

[MOD-5] Dual-threshold decision  WHERE: ScamQSVM.fit
  - Standard QSVM has one cutoff. Now:
      warn_thr   (calibrated for ~95% scam recall on validation data)
      detect_thr (= 0, the SVM boundary)
  - warn_thr <= score < detect_thr : WARNING, held for review
  - score >= detect_thr            : DETECTED
  - score < warn_thr               : clean

[MOD-6] Warn-Detect-Eradicate    WHERE: ScamShield.handle, ScamShield._eradicate
  - Stage 1 WARN: alert on suspicious score.
  - Stage 2 DETECT: confirm using the QSVM boundary.
  - Stage 3 ERADICATE: quarantine message, block sender and URL, persist
    blocklist, write timestamped incident log.
  - Known blocklisted senders are stopped before scoring.

SUPPORTING CODE (not QSVM changes)
  - extract_features()      : raw message text -> 6 scam features
  - make_synthetic_data()   : demo training data
  - main()                  : training, evaluation and demo

UNCHANGED: quantum kernel computation (FidelityQuantumKernel) and SVM
optimisation.
"""


# =====================================================================
# MAIN
# =====================================================================
def main():
    Path("QSVM_MODIFICATIONS.txt").write_text(MODS_TEXT, encoding="utf-8")
    print("Wrote QSVM_MODIFICATIONS.txt\n")

    X, y = make_synthetic_data()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2)

    print("Training QSVM (quantum kernel on a simulator, may take a few minutes)...")
    model = ScamQSVM().fit(Xtr, ytr)
    print(f"Thresholds: warn={model.warn_thr:.3f}  detect={model.detect_thr:.3f}\n")

    pred = (model.score(Xte) >= model.detect_thr).astype(int)
    print(classification_report(yte, pred, target_names=["legit", "scam"]))

    shield = ScamShield(model)

    print("\n--- Legitimate message ---")
    shield.handle("MSG-001",
                  "Your monthly statement is ready: https://www.mybank.com/statement",
                  "alerts@mybank.com", sender_age_days=2000)

    print("\n--- Scam message ---")
    shield.handle("MSG-002",
                  "URGENT! You are a winner of our lottery prize. Act now, your "
                  "claim expires within 24 hours. Pay the fee: http://claim-prize.xyz/x",
                  "promo@win-cash.biz", sender_age_days=2)

    print("\n--- Same sender again ---")
    shield.handle("MSG-003", "Final notice: claim your cash prize!",
                  "promo@win-cash.biz", sender_age_days=2)


if __name__ == "__main__":
    main()