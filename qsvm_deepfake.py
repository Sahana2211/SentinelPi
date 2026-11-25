"""QSVM deepfake screening (PennyLane simulator): voice clip or face video in -> verdict out.

  audio : 45 signal features (MFCC timbre, spectrum shape, high-frequency content, noise floor, pitch movement)
  video : 16 face features (sharpness vs background, frequency content, noise residual, frame-to-frame flicker)
  -> pick the n_qubits most informative -> angle-encode in an n-qubit circuit -> fidelity kernel -> SVM
  -> warning / critical tiers + calibrated confidence + "which features look unlike genuine media".
Features use only NumPy/SciPy (+ OpenCV for video): no librosa/numba to install on the board.

Data (one model per kind):
  folders : data_audio/real/*.wav  data_audio/fake/*.wav            (or .flac)   |  data_video/real/*.mp4  data_video/fake/*.mp4
  manifest: CSV with path,label[,group]   (label FAKE/SPOOF/DEEPFAKE/1 = fake; REAL/BONAFIDE/GENUINE/0 = real)
  python qsvm_deepfake.py train --kind audio --data data_audio --compare
  python qsvm_deepfake.py train --kind video --manifest videos.csv --path-col path --label-col label --group-col source
  python qsvm_deepfake.py scan  --kind audio call.wav
Give a `group` column (speaker or source-video id) whenever you can: otherwise the same speaker lands in train and
test and the scores are inflated.
"""
import argparse, functools, glob, os, time
from math import gcd
import numpy as np, pandas as pd, joblib
from scipy import signal as ss
from scipy.fft import dct
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

SR = 16000
AUDIO_EXT, VIDEO_EXT = ("wav", "flac"), ("mp4", "avi", "mov", "mkv")
VERDICT = {0: "no_strong_signal", 1: "inconclusive", 2: "suspected_synthetic"}
FAKE_TOKENS = {"1", "fake", "spoof", "spoofed", "deepfake", "synthetic", "generated", "bogus", "ai"}
CAVEAT = "Screening signal only; confirm through a call-back or another channel."


# ================================================================ audio features (NumPy/SciPy only)
def load_audio(path, sr=SR, max_s=10.0):
    """Mono float64 at 16 kHz, peak-normalised. WAV always; FLAC/OGG if `soundfile` is installed."""
    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            fs = f.samplerate; y = f.read(int(max_s * fs), dtype="float32", always_2d=True)
    except Exception:
        from scipy.io import wavfile
        fs, y = wavfile.read(path)                       # raises for non-WAV files: the caller reports it
        y = (y.astype(np.float32) / np.iinfo(y.dtype).max if y.dtype.kind == "i" else
             (y.astype(np.float32) - 128) / 128 if y.dtype.kind == "u" else y.astype(np.float32))
        y = y.reshape(len(y), -1)[: int(max_s * fs)]
    y = y.mean(axis=1)
    if fs != sr:
        g = gcd(sr, fs); y = ss.resample_poly(y, sr // g, fs // g)
    y = y[: int(max_s * sr)].astype(np.float64)
    if len(y) < sr: raise ValueError("audio shorter than 1 second")
    return y / (np.abs(y).max() + 1e-9)


def _mel_fb(sr, n_fft, n_mels=26, fmin=50.0):
    mel, inv = (lambda f: 2595 * np.log10(1 + f / 700)), (lambda m: 700 * (10 ** (m / 2595) - 1))
    bins = np.floor((n_fft + 1) * inv(np.linspace(mel(fmin), mel(sr / 2), n_mels + 2)) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i:i + 3]
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def audio_features(y, sr=SR):
    f, _, Z = ss.stft(y, fs=sr, nperseg=512, noverlap=256, boundary=None, padded=False)
    P = np.abs(Z) ** 2 + 1e-12; mag = np.sqrt(P); eps = 1e-12
    mfcc = dct(np.log(_mel_fb(sr, 512) @ P + 1e-10), axis=0, norm="ortho")[1:14]      # c1..c13 (c0 = loudness, skipped)
    d = {}
    for i in range(13): d[f"mfcc{i + 1}_mean"], d[f"mfcc{i + 1}_std"] = mfcc[i].mean(), mfcc[i].std()
    cent = (f[:, None] * mag).sum(0) / (mag.sum(0) + eps)
    d["centroid_mean"], d["centroid_std"] = cent.mean(), cent.std()
    d["bandwidth_mean"] = np.sqrt((((f[:, None] - cent) ** 2) * mag).sum(0) / (mag.sum(0) + eps)).mean()
    cs = np.cumsum(P, 0); d["rolloff85_mean"] = f[(cs >= 0.85 * cs[-1]).argmax(0)].mean()
    flat = np.exp(np.log(P).mean(0)) / P.mean(0); d["flatness_mean"], d["flatness_std"] = flat.mean(), flat.std()
    flux = np.sqrt((np.diff(mag, axis=1) ** 2).sum(0)) / (mag.sum(0)[1:] + eps); d["flux_mean"], d["flux_std"] = flux.mean(), flux.std()
    hf = P[f >= 4000].sum(0) / P.sum(0); d["hf4k_mean"], d["hf4k_std"] = hf.mean(), hf.std()
    d["top7k_ratio"] = P[f >= 7000].sum() / P.sum()                                   # codec/vocoder band-limits show up here
    fr = np.lib.stride_tricks.sliding_window_view(y, 512)[::256]
    z = (np.diff(np.signbit(fr).astype(np.int8), axis=1) != 0).mean(1); d["zcr_mean"], d["zcr_std"] = z.mean(), z.std()
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-8)
    d["noise_floor_db"] = np.percentile(db, 10); d["dynamic_range_db"] = np.percentile(db, 95) - np.percentile(db, 10)
    d["silence_ratio"] = (db < db.max() - 40).mean()
    fp = np.lib.stride_tricks.sliding_window_view(y, 1024)[::512]; rms = np.sqrt((fp ** 2).mean(1))
    win, lo, hi, f0 = np.hanning(1024), sr // 400, sr // 70, []
    for x in fp[rms > 0.03 * rms.max()]:                                              # crude autocorrelation pitch tracker
        ac = np.fft.irfft(np.abs(np.fft.rfft((x - x.mean()) * win, 2048)) ** 2)[:1024]
        if ac[0] <= 0: continue
        ac = ac / ac[0]; k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.4: f0.append(sr / k)
    if len(f0) >= 5:
        st = 12 * np.log2(np.array(f0) / np.median(f0)); d["f0_std_st"], d["f0_jump_st"] = st.std(), np.abs(np.diff(st)).mean()
    else: d["f0_std_st"] = d["f0_jump_st"] = 0.0
    d["voiced_ratio"] = len(f0) / max(1, len(fp))
    return d


AUDIO_FEATURES = list(audio_features(np.random.default_rng(0).standard_normal(SR * 2) * 0.1).keys())


# ================================================================ video features (OpenCV + NumPy)
VIDEO_FEATURES = ["face_sharp_mean", "face_sharp_std", "sharp_ratio_mean", "sharp_ratio_std", "hf_ratio_mean", "hf_ratio_std",
                  "noise_mean", "noise_std", "face_diff_mean", "bg_diff_mean", "flicker_ratio_mean", "flicker_ratio_std",
                  "move_x", "move_y", "size_var", "detect_rate"]


def _haar():
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("this OpenCV build has no CascadeClassifier (OpenCV 5 removed it): pip install 'opencv-python-headless<5'")
    c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if c.empty(): raise RuntimeError("OpenCV face model file not found")
    return lambda g: c.detectMultiScale(g, 1.1, 5, minSize=(60, 60))


def video_features(path, n_pos=12, detect=None):
    """Returns None if a face is found in fewer than 3 of the sampled positions."""
    import cv2
    detect = detect or _haar(); cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError("cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); hann = np.outer(np.hanning(128), np.hanning(128))
    lap = lambda im: cv2.Laplacian(im, cv2.CV_64F).var(); rows, boxes = [], []
    for i in np.linspace(0, max(total - 2, 0), n_pos).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok1, f1 = cap.read(); ok2, f2 = cap.read()   # two ADJACENT frames
        if not (ok1 and ok2): continue
        g1, g2 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY); fs = detect(g1)
        if len(fs) == 0: continue
        x, y, w, h = max(fs, key=lambda b: b[2] * b[3])
        c1, c2 = (cv2.resize(g[y:y + h, x:x + w], (128, 128)) for g in (g1, g2))
        F = np.abs(np.fft.fftshift(np.fft.fft2(c1.astype(np.float64) * hann))) ** 2
        hf = 1 - F[48:80, 48:80].sum() / (F.sum() + 1e-9)                       # energy outside the central low-frequency block
        noise = (c1.astype(np.float64) - cv2.GaussianBlur(c1, (5, 5), 0).astype(np.float64)).std()
        fd = np.abs(c1.astype(np.float64) - c2).mean() / 255; bd = np.abs(g1.astype(np.float64) - g2).mean() / 255
        rows.append((lap(c1), lap(g1), hf, noise, fd, bd, fd / (bd + 1e-3)))
        boxes.append(((x + w / 2) / g1.shape[1], (y + h / 2) / g1.shape[0], np.log(w * h)))
    cap.release()
    if len(rows) < 3: return None
    R, B = np.array(rows), np.array(boxes); sr_ = R[:, 0] / (R[:, 1] + 1e-9)
    return {"face_sharp_mean": R[:, 0].mean(), "face_sharp_std": R[:, 0].std(), "sharp_ratio_mean": sr_.mean(), "sharp_ratio_std": sr_.std(),
            "hf_ratio_mean": R[:, 2].mean(), "hf_ratio_std": R[:, 2].std(), "noise_mean": R[:, 3].mean(), "noise_std": R[:, 3].std(),
            "face_diff_mean": R[:, 4].mean(), "bg_diff_mean": R[:, 5].mean(), "flicker_ratio_mean": R[:, 6].mean(),
            "flicker_ratio_std": R[:, 6].std(), "move_x": B[:, 0].std(), "move_y": B[:, 1].std(), "size_var": B[:, 2].std(),
            "detect_rate": len(rows) / n_pos}


def features_for(path, kind):
    """Feature vector (fixed order) or None when the clip has nothing to analyse (e.g. no face)."""
    if kind == "audio": d, names = audio_features(load_audio(path)), AUDIO_FEATURES
    else:
        d, names = video_features(path), VIDEO_FEATURES
        if d is None: return None
    return np.nan_to_num(np.array([d[k] for k in names], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def extract_all(paths, kind, cache_path=None):
    """-> (feature DataFrame for usable clips, boolean mask of which inputs were usable). Results are cached."""
    cache = joblib.load(cache_path) if cache_path and os.path.exists(cache_path) else {}
    names = AUDIO_FEATURES if kind == "audio" else VIDEO_FEATURES; vecs, ok = [], []
    for i, p in enumerate(paths):
        key = (p, os.path.getsize(p))
        if key not in cache:
            try: cache[key] = features_for(p, kind)
            except Exception as e:
                print(f"  skipped {os.path.basename(p)}: {str(e)[:70]}"); cache[key] = None
        v = cache[key]; ok.append(v is not None)
        if v is not None: vecs.append(v)
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(paths)}")
            if cache_path: joblib.dump(cache, cache_path)
    if cache_path: joblib.dump(cache, cache_path)
    return pd.DataFrame(vecs, columns=names), np.array(ok)


DESC = {"top7k_ratio": "energy above 7 kHz", "hf4k_mean": "average energy above 4 kHz", "noise_floor_db": "background noise floor",
        "dynamic_range_db": "loudness range", "f0_std_st": "pitch variation", "f0_jump_st": "pitch jitter between frames",
        "voiced_ratio": "share of voiced speech", "flatness_mean": "spectral flatness (noisiness)", "flux_mean": "spectral change rate",
        "silence_ratio": "share of near-silence", "zcr_mean": "zero-crossing rate", "centroid_mean": "spectral brightness",
        "face_sharp_mean": "face sharpness", "sharp_ratio_mean": "face-versus-scene sharpness", "hf_ratio_mean": "fine facial detail",
        "noise_mean": "skin/texture noise", "face_diff_mean": "face change between adjacent frames", "bg_diff_mean": "scene change between adjacent frames",
        "flicker_ratio_mean": "face flicker relative to the scene", "move_x": "horizontal face movement", "move_y": "vertical face movement",
        "size_var": "face size variation", "detect_rate": "share of frames with a detectable face"}


def describe(name):
    return DESC.get(name, f"voice timbre ({name.replace('_', ' ')})" if name.startswith("mfcc") else name.replace("_", " "))


# ================================================================ quantum kernel + thresholds (same design as the other QSVMs)
class QuantumKernel:
    """Fidelity kernel K_ij = |<psi_i|psi_j>|^2; each sample's statevector is simulated once."""

    def __init__(self, n_qubits=8, reps=2, fmap="zz", bw=1.0, device="default.qubit", chunk=128):
        self.n, self.reps, self.fmap, self.bw = n_qubits, reps, fmap, bw
        self.device, self.chunk, self._circuit = device, chunk, None

    def __getstate__(self):
        d = self.__dict__.copy(); d["_circuit"] = None; return d

    def _build(self):
        import pennylane as qml
        n, reps, fmap = self.n, self.reps, self.fmap
        dev = qml.device(self.device, wires=n)

        @qml.qnode(dev)
        def circuit(x):
            for _ in range(reps):
                for i in range(n):
                    qml.Hadamard(wires=i); qml.RZ(x[..., i], wires=i)
                if fmap == "product": continue                     # control: no entanglement
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


def pick_threshold(y, scores, beta=2.0):
    """WARNING tier: maximise F-beta. Midpoint to the next lower score so ties and the class gap are handled."""
    p, r, t = precision_recall_curve(y, scores)
    fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-12); i = int(np.argmax(fb[:-1]))
    return float((t[i] + t[i - 1]) / 2) if i > 0 else float(np.nextafter(t[0], -np.inf))


def pick_precision_threshold(y, scores, target=0.95, floor=-np.inf, min_pos=10):
    """CRITICAL tier: lowest threshold with precision >= target (inf = unreachable -> tier disabled)."""
    p, r, t = precision_recall_curve(y, scores)
    s = np.sort(scores); n_pred = len(s) - np.searchsorted(s, t, side="left")
    ok = (p[:-1] >= target) & (t >= floor) & (n_pred >= min_pos)
    return float(t[ok][0]) if ok.any() else float("inf")


# ================================================================ the QSVM
class DeepfakeQSVM:
    def __init__(self, kind="audio", n_qubits=8, reps=2, fmap="zz", bw_grid=(0.5, 1.0, 2.0), C_grid=(0.5, 2.0, 8.0),
                 beta=2.0, cv=3, device="default.qubit", target_precision=0.95):
        self.kind, self.n_qubits, self.reps, self.fmap, self.bw_grid, self.C_grid = kind, n_qubits, reps, fmap, bw_grid, C_grid
        self.beta, self.cv, self.device, self.target_precision = beta, cv, device, target_precision

    def _front(self):
        return Pipeline([("sc", StandardScaler()),
                         ("sel", SelectKBest(functools.partial(mutual_info_classif, random_state=0), k=self.n_qubits)),
                         ("mm", MinMaxScaler((0, np.pi), clip=True))])

    def _oof(self, K, y, C):
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(self.cv, shuffle=True, random_state=0).split(K, y):
            svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(tr, tr)], y[tr])
            oof[va] = svc.decision_function(K[np.ix_(va, tr)])
        return oof

    def fit(self, X, y, verbose=True):
        y = np.asarray(y); self.columns = list(X.columns)
        self.real_mean, self.real_std = X[y == 0].mean(), X[y == 0].std()          # reference for "unlike genuine" evidence
        self.front = self._front(); Z = self.front.fit_transform(X, y)
        self.selected = [c for c, k in zip(self.columns, self.front.named_steps["sel"].get_support()) if k]
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
        self.platt = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)           # score -> P(fake)
        pw, pa = oof > self.threshold, oof >= self.act_threshold
        self.tiers = {"warn": (precision_score(y, pw, zero_division=0), recall_score(y, pw, zero_division=0)),
                      "act": (precision_score(y, pa, zero_division=0), recall_score(y, pa, zero_division=0))
                      if np.isfinite(self.act_threshold) else None}
        svc = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(K, y)
        self.sv_states = S[svc.support_]
        self.dual, self.b = svc.dual_coef_[0].astype(float), float(svc.intercept_[0])
        return self

    def assess(self, X):
        """-> raw score, calibrated confidence that the clip is fake, severity 0 / 1 (warning) / 2 (critical)."""
        S = self.qk.states(self.front.transform(X[self.columns]))
        raw = self.qk.kernel(S, self.sv_states) @ self.dual + self.b
        sev = (raw > self.threshold).astype(int); sev[(raw >= self.act_threshold) & (raw > self.threshold)] = 2
        return {"raw": raw, "score": raw - self.threshold,
                "confidence": self.platt.predict_proba(raw.reshape(-1, 1))[:, 1], "severity": sev}

    def explain(self, row, top=3, min_z=2.0):
        """Which of the features the circuit uses look least like the genuine training clips."""
        z = ((row[self.selected] - self.real_mean[self.selected]) / (self.real_std[self.selected] + 1e-9))
        amount = lambda v: f"{abs(v):.1f}" if abs(v) <= 10 else "more than 10"
        return [f"{describe(n)} is {amount(z[n])} standard deviations {'above' if z[n] > 0 else 'below'} typical genuine {self.kind}"
                for n in z.abs().sort_values(ascending=False).index[:top] if abs(z[n]) >= min_z]

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)


def finding(path, kind, model_path=None):
    """Result in the shape the SentinelPi dashboard uses (same verdict words as deepfake_screen)."""
    m = DeepfakeQSVM.load(model_path or f"{kind}_qsvm.joblib"); name = os.path.basename(path)
    v = features_for(path, kind)
    if v is None:
        return {"pipeline": f"deepfake_{kind}", "target": name, "score": None, "severity": 0, "verdict": "inconclusive",
                "evidence": ["no usable face found in the video"]}
    X = pd.DataFrame([v], columns=m.columns); A = m.assess(X); sev = int(A["severity"][0])
    return {"pipeline": f"deepfake_{kind}", "target": name, "score": round(float(A["confidence"][0]), 3), "severity": sev,
            "verdict": VERDICT[sev],
            "evidence": (m.explain(X.iloc[0]) if sev >= 1 else []) + [CAVEAT]}      # reasons only when something is flagged


# ================================================================ data + CLI
def parse_labels(col, fake="auto"):
    s = col.astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
    return s.isin(FAKE_TOKENS if fake == "auto" else {str(fake).strip().lower()}).astype(int).values


def subsample_idx(y, n, seed):
    rng = np.random.default_rng(seed); idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]; idx.extend(rng.choice(ci, size=min(int(round(n * len(ci) / len(y))), len(ci)), replace=False))
    return np.array(rng.permutation(idx), dtype=int)


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


def collect(a):
    ext = AUDIO_EXT if a.kind == "audio" else VIDEO_EXT
    if a.manifest:
        df = pd.read_csv(a.manifest); paths = [os.path.join(a.base_dir, str(p)) for p in df[a.path_col]]
        y = parse_labels(df[a.label_col], a.fake_label); groups = df[a.group_col].astype(str).values if a.group_col else None
    else:
        paths, y, groups = [], [], None
        for lab, name in enumerate(["real", "fake"]):
            found = [p for e in ext for p in glob.glob(os.path.join(a.data, name, "**", f"*.{e}"), recursive=True)]
            paths += sorted(found); y += [lab] * len(found)
        y = np.array(y)
    keep = np.array([os.path.exists(p) for p in paths]); paths = [p for p, k in zip(paths, keep) if k]
    return paths, y[keep], (groups[keep] if groups is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "scan"]); ap.add_argument("paths", nargs="*"); ap.add_argument("--kind", choices=["audio", "video"], required=True)
    ap.add_argument("--data", default=None, help="folder containing real/ and fake/")
    ap.add_argument("--manifest"); ap.add_argument("--path-col", default="path"); ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col"); ap.add_argument("--base-dir", default=""); ap.add_argument("--fake-label", default="auto")
    ap.add_argument("--qubits", type=int, default=8); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--map", default="zz", choices=["zz", "ring", "product"]); ap.add_argument("--compare", action="store_true")
    ap.add_argument("--bw", default="0.5,1,2"); ap.add_argument("--C", default="0.5,2,8")
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--n-train", type=int, default=600); ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--cache", default=None, help="feature cache file (default: <kind>_features.joblib)")
    ap.add_argument("--model", default=None); ap.add_argument("--device", default="default.qubit")
    a = ap.parse_intermixed_args(); fl = lambda s: tuple(float(v) for v in s.split(",")); model_path = a.model or f"{a.kind}_qsvm.joblib"

    if a.cmd == "scan":
        if not a.paths: ap.error("give one or more files to scan")
        for p in a.paths:
            f = finding(p, a.kind, model_path)
            conf = "n/a" if f["score"] is None else f"{f['score']:.2f}"
            print(f"{f['verdict'].upper():<20} confidence {conf}  {p}")
            for e in f["evidence"]: print(f"    - {e}")
        return

    if not (a.data or a.manifest): ap.error("train needs --data folder or --manifest file")
    paths, y, groups = collect(a)
    print(f"{len(y)} clips: {int(y.sum())} fake / {int((1 - y).sum())} real  <- check this is the right way round")
    if len(y) < 40 or y.min() == y.max(): raise SystemExit("need at least 40 clips with both classes")
    frac = min(a.n_test / len(y), 0.25); idx = np.arange(len(y))
    if groups is not None: tr, te = next(GroupShuffleSplit(1, test_size=frac, random_state=0).split(idx, y, groups))
    else:
        tr, te = train_test_split(idx, test_size=frac, stratify=y, random_state=0)
        print("  note: no --group-col, so the same speaker/source may appear in both train and test (scores optimistic)")
    tr, te = tr[subsample_idx(y[tr], a.n_train, 1)], te[subsample_idx(y[te], a.n_test, 2)]
    cache = a.cache or f"{a.kind}_features.joblib"
    print("extracting training features..."); Xtr, oktr = extract_all([paths[i] for i in tr], a.kind, cache)
    print("extracting test features..."); Xte, okte = extract_all([paths[i] for i in te], a.kind, cache)
    ytr, yte = y[tr][oktr], y[te][okte]
    print(f"usable clips: {len(ytr)} train, {len(yte)} test")
    if len(set(ytr)) < 2 or len(set(yte)) < 2: raise SystemExit("a split lost one class (many unreadable clips?)")

    rows, last = [], None
    for fm in (["product", "zz", "ring"] if a.compare else [a.map]):
        print(f"== feature map: {fm} ==")
        det = DeepfakeQSVM(a.kind, a.qubits, a.reps, fm, fl(a.bw), fl(a.C), a.beta, a.cv, a.device, a.target_precision)
        t = time.time(); det.fit(Xtr, ytr); fit_s = time.time() - t
        t = time.time(); A = det.assess(Xte); ms = 1000 * (time.time() - t) / len(Xte)
        print(f"best bandwidth={det.bw}, C={det.C}, cv-AP={det.cv_ap:.3f}, SVs={len(det.dual)}/{len(ytr)}, fit {fit_s:.0f}s, {ms:.1f} ms/clip (excl. feature extraction)")
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
        print("\nConfusion matrix at the warning tier [real, fake]:\n", confusion_matrix(yte, last[1]))
        last[0].save(model_path); print("saved", model_path)


if __name__ == "__main__":
    main()
