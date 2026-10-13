# -*- coding: utf-8 -*-
import os, re, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


BASE = Path(r"D:\path\to\data")


LOW_TRAIN  = BASE / r"cA2\train"
MID_TRAIN  = BASE / r"cD1\train"
HIGH_TRAIN = BASE / r"cD2\train"

LOW_VAL    = BASE / r"cA2\val"
MID_VAL    = BASE / r"cD1\val"
HIGH_VAL   = BASE / r"cD2\val"

LOW_TEST   = BASE / r"cA2\test"
MID_TEST   = BASE / r"cD1\test"
HIGH_TEST  = BASE / r"cD2\test"



LABEL_MAP_JSON = "label2index_3band.json"
MODEL_WEIGHTS  = "convnext_cosface.pth"


# ============================================================


BACKBONE        = "convnext_tiny"     # convnext_tiny/small/base
IMG_SIZE        = 224
BATCH_SIZE      = 128
MAX_EPOCHS      = 50
WARMUP_EPOCHS   = 3
BASE_LR         = 1e-3
MIN_LR          = 1e-5
WEIGHT_DECAY    = 0.05
EARLY_PATIENCE  = 8
SEED            = 42
USE_BF16        = True
CHANNELS_LAST   = True
NUM_WORKERS     = 6
PIN_MEMORY      = True
PERSISTENT_WKR  = True
PREFETCH_FACTOR = 2


ALPHA_ENERGY_INIT = 0.6                  
AUTO_ALPHA        = True                 
ALPHA_GRID        = [i/10 for i in range(0, 11)]

CALIB_ACCEPT_P    = 0.90                 
MARGIN_Q          = 0.20                 
RHO               = 0.00                 # Zero threshold offset: matches the manuscript gate rule                 

K_CENTROIDS       = 3                   
SPEC_TMASK_P      = 0.3                  
SPEC_FMASK_P      = 0.3                  
# =====================================================

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _build_label_map_from_triple(mid_root: Path, save_path: str) -> Dict[str,int]:
    names = sorted([d.name for d in mid_root.iterdir() if d.is_dir()], key=natural_key)
    mp = {c:i for i,c in enumerate(names)}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mp, f, indent=2, ensure_ascii=False)
    return mp


class ThreeBandDataset(Dataset):

    def __init__(self, root_low: Path, root_mid: Path, root_high: Path,
                 label_map: Dict[str,int], train=True):
        self.root_low  = Path(root_low)
        self.root_mid  = Path(root_mid)
        self.root_high = Path(root_high)
        self.train = train
        self.label_map = label_map
        self.samples: List[Tuple[str,str,str,int]] = []

        if not (self.root_low.exists() and self.root_mid.exists() and self.root_high.exists()):
            raise FileNotFoundError("One of the three-band roots does not exist.")

        for cname in sorted([d.name for d in self.root_mid.iterdir() if d.is_dir()], key=natural_key):
            y = self.label_map.get(cname, -1)
            low_dir  = self.root_low  / cname
            mid_dir  = self.root_mid  / cname
            high_dir = self.root_high / cname
            if not (low_dir.exists() and mid_dir.exists() and high_dir.exists()):
                continue
            for p_mid in sorted(mid_dir.glob("*.png"), key=lambda x: natural_key(x.name)):
                p_low  = low_dir  / p_mid.name
                p_high = high_dir / p_mid.name
                if p_low.exists() and p_high.exists():
                    self.samples.append((str(p_low), str(p_mid), str(p_high), y))

    def __len__(self): return len(self.samples)

    @staticmethod
    def _read_gray(path: str) -> torch.Tensor:
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)  # (H,W)
        t = (t - t.mean()) / (t.std() + 1e-6)  # per-image z-score
        return t

    @staticmethod
    def _pad_to_center(x: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        H, W = x.shape
        out = torch.zeros((Ht, Wt), dtype=x.dtype)
        sh = (Ht - H) // 2
        sw = (Wt - W) // 2
        out[sh:sh+H, sw:sw+W] = x
        return out

    def _stack3(self, pL: str, pM: str, pH: str) -> torch.Tensor:
        L = self._read_gray(pL); M = self._read_gray(pM); Ht = self._read_gray(pH)
        Hmax = int(max(L.shape[0], M.shape[0], Ht.shape[0]))
        Wmax = int(max(L.shape[1], M.shape[1], Ht.shape[1]))
        Lp = self._pad_to_center(L,  Hmax, Wmax)
        Mp = self._pad_to_center(M,  Hmax, Wmax)
        Hp = self._pad_to_center(Ht, Hmax, Wmax)
        return torch.stack([Lp, Mp, Hp], dim=0)  # [3,Hmax,Wmax]

    def _weak_geo(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train: return x
        H, W = x.shape[-2:]
        angle = float((torch.rand(1)*2-1) * 3.0)
        tx = int(W * 0.05 * (torch.rand(1)*2-1))
        ty = int(H * 0.05 * (torch.rand(1)*2-1))
        scale = float(1.0 + (torch.rand(1)*2-1) * 0.05)
        return TF.affine(x, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0,0.0])

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train: return x
        if torch.rand(1).item() < SPEC_FMASK_P:
            Fm = max(1, int(x.shape[-2] * 0.06))
            f0 = torch.randint(0, max(1, x.shape[-2]-Fm), (1,)).item()
            x[:, f0:f0+Fm, :] = 0
        if torch.rand(1).item() < SPEC_TMASK_P:
            Tm = max(1, int(x.shape[-1] * 0.08))
            t0 = torch.randint(0, max(1, x.shape[-1]-Tm), (1,)).item()
            x[:, :, t0:t0+Tm] = 0
        return x

    def __getitem__(self, idx):
        pL, pM, pH, y = self.samples[idx]
        x = self._stack3(pL, pM, pH)
        if self.train:
            x = self._weak_geo(x)
            x = self._spec_augment(x)
        x = TF.resize(x, [IMG_SIZE, IMG_SIZE], antialias=True)
        return x, y


class CosFace(nn.Module):
    def __init__(self, in_feats, n_classes, s=30.0, m=0.25):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, in_feats))
        nn.init.xavier_uniform_(self.W)
        self.s = s; self.m = m

    def forward(self, feats, labels=None):
        x = F.normalize(feats); w = F.normalize(self.W)
        cos = F.linear(x, w)                  # (B,C)
        if labels is None:
            return self.s * cos
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        logits = self.s * (cos - one_hot * self.m)
        return logits

# ---------------- Backbone + BNNeck + CosFace ----------------
class ConvNeXtCosFace(nn.Module):
    def __init__(self, n_classes, backbone="convnext_tiny", embed_dim=256, pretrained_try=True):
        super().__init__()
        ok=False
        if pretrained_try:
            try:
                self.backbone = timm.create_model(backbone, pretrained=True, in_chans=3, num_classes=0)
                ok=True
            except Exception as e:
                print(f"[WARN] timm pretrained failed: {e}")
        if not ok:
            self.backbone = timm.create_model(backbone, pretrained=False, in_chans=3, num_classes=0)

        feat_dim = int(getattr(self.backbone, "num_features", 768))
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.head   = CosFace(feat_dim, n_classes, s=30.0, m=0.25)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, x, labels=None, return_feats=False):
        if CHANNELS_LAST and x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        f = self.backbone(x)              # (B,feat_dim)
        f = self.bnneck(f)
        logits = self.head(f, labels)     # (B,C)
        if return_feats:
            return logits, f
        return logits


def cosine_lr(optimizer, epoch, base_lr=1e-3, min_lr=1e-5, warmup=3, max_epochs=50):
    if epoch < warmup:
        lr = base_lr * float(epoch+1)/float(max(1, warmup))
    else:
        t = (epoch - warmup) / float(max(1, max_epochs - warmup))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups: pg["lr"] = lr
    return lr

@torch.no_grad()
def extract_logits_feats(model, loader, device):
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits, feats = model(x, labels=None, return_feats=True)
        all_logits.append(logits.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return np.vstack(all_logits), np.vstack(all_feats), np.array(all_labels)

def energy_score(logits, T=1.0):
    # E(x) = -T * logsumexp(logits/T)  
    x = logits / T
    m = x.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))
    return (-T * lse).ravel()

def get_margin(logits):
    part = np.partition(logits, -2, axis=1)
    top2 = part[:, -2]
    top1 = logits.max(axis=1)
    return top1 - top2


def train():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = _build_label_map_from_triple(MID_TRAIN, LABEL_MAP_JSON)
    n_classes = len(label_map)
    print(f"Train classes: {n_classes}")

    train_ds = ThreeBandDataset(LOW_TRAIN, MID_TRAIN, HIGH_TRAIN, label_map, train=True)
    val_ds   = ThreeBandDataset(LOW_VAL,   MID_VAL,   HIGH_VAL,   label_map, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WKR, prefetch_factor=PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WKR, prefetch_factor=PREFETCH_FACTOR
    )

    model = ConvNeXtCosFace(n_classes=n_classes, backbone=BACKBONE, pretrained_try=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and not USE_BF16)
    autocast_dtype = torch.bfloat16 if (USE_BF16 and torch.cuda.is_available()) else torch.float16
    best_val = 1e9; bad = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        lr = cosine_lr(optimizer, epoch, BASE_LR, MIN_LR, WARMUP_EPOCHS, MAX_EPOCHS)
        running = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)

            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                logits = model(x, labels=y)
                loss = F.cross_entropy(logits, y)

            if autocast_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())


        model.eval()
        val_loss, tot = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.as_tensor(y, device=device)
                logits = model(x, labels=y)
                val_loss += float(F.cross_entropy(logits, y).item()) * x.size(0)
                tot += x.size(0)
        val_loss = val_loss / max(1, tot)
        print(f"[{epoch+1:03d}] train_ce={running/len(train_loader):.4f}  val_ce={val_loss:.4f}  lr={lr:.2e}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            print(f"  ↳ saved new best weights: {MODEL_WEIGHTS}")
        else:
            bad += 1
            if bad >= EARLY_PATIENCE:
                print(f"Early stopping. best val_ce={best_val:.4f}")
                break

    if not os.path.exists(MODEL_WEIGHTS):
        torch.save(model.state_dict(), MODEL_WEIGHTS)
    print(f"Saved weights to {MODEL_WEIGHTS}")


def evaluate():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(LABEL_MAP_JSON, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    n_classes = len(label_map)

    val_ds  = ThreeBandDataset(LOW_VAL,  MID_VAL,  HIGH_VAL,  label_map, train=False)
    test_ds = ThreeBandDataset(LOW_TEST, MID_TEST, HIGH_TEST, label_map, train=False)

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WKR, prefetch_factor=PREFETCH_FACTOR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WKR, prefetch_factor=PREFETCH_FACTOR)

    model = ConvNeXtCosFace(n_classes=n_classes, backbone=BACKBONE, pretrained_try=False).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device), strict=True)
    model.eval()


    log_val, feat_val, y_val = extract_logits_feats(model, val_loader, device)
    log_te,  feat_te,  y_te  = extract_logits_feats(model, test_loader, device)


    en_val = energy_score(log_val)
    en_te  = energy_score(log_te)


    class_stats = {}
    for c in sorted(set(y_val)):
        Xc = feat_val[y_val == c]
        k = min(K_CENTROIDS, max(1, len(Xc)))
        km = KMeans(n_clusters=k, n_init="auto", random_state=SEED).fit(Xc)
        mus = km.cluster_centers_
        lwc = LedoitWolf().fit(Xc)
        Sig_inv_c = np.linalg.pinv(lwc.covariance_)
        class_stats[c] = (mus, Sig_inv_c)

    def mahal_min_classwise(X):
        out = np.empty((len(X),), dtype=np.float32)
        for i, x in enumerate(X):
            best = 1e9
            for (mus, Sig_inv_c) in class_stats.values():
                for mu in mus:
                    d = float(np.sqrt((x - mu) @ Sig_inv_c @ (x - mu).T))
                    if d < best: best = d
            out[i] = best
        return out

    md_val = mahal_min_classwise(feat_val)
    md_te  = mahal_min_classwise(feat_te)


    ez_val = (en_val - en_val.mean()) / (en_val.std() + 1e-6)
    dz_val = (md_val - md_val.mean()) / (md_val.std() + 1e-6)
    ez_te  = (en_te  - en_val.mean()) / (en_val.std() + 1e-6)
    dz_te  = (md_te  - md_val.mean()) / (md_val.std() + 1e-6)


    def objective(alpha):
        fused_v = alpha * ez_val + (1.0 - alpha) * dz_val
        pred_v  = log_val.argmax(axis=1)  
        tau = {}
        for c in set(pred_v):
            idx = (pred_v == c)
            tau[c] = np.quantile(fused_v[idx], CALIB_ACCEPT_P) if idx.sum() > 0 else 1e9

        rej = fused_v > np.array([tau.get(int(p), 1e9) for p in pred_v])
        rej_rate = rej.mean()
        miscls_rate = (pred_v != y_val).mean()
        return rej_rate + miscls_rate, tau

    alpha_opt = ALPHA_ENERGY_INIT
    tau_opt = {}
    if AUTO_ALPHA:
        best_obj = 1e9
        for a in ALPHA_GRID:
            obj, tau = objective(a)
            if obj < best_obj:
                best_obj = obj
                alpha_opt, tau_opt = a, tau
    else:
        _, tau_opt = objective(ALPHA_ENERGY_INIT)


    margin_val = get_margin(log_val)
    margin_te  = get_margin(log_te)
    mz_val = (margin_val - margin_val.mean()) / (margin_val.std() + 1e-6)
    mz_te  = (margin_te  - margin_val.mean()) / (margin_val.std() + 1e-6)

    pred_v = log_val.argmax(axis=1)
    tau_margin = {}
    for c in set(pred_v):
        idx = (pred_v == c)
        tau_margin[c] = np.quantile(mz_val[idx], MARGIN_Q) if idx.sum() > 0 else -1e9

    print(f"[Calib] alpha={alpha_opt:.2f}  (per-class accept ≥ {int(CALIB_ACCEPT_P*100)}%, "
          f"margin@{MARGIN_Q:.2f}, RHO={RHO:+.2f})")

    fused_te = alpha_opt * ez_te + (1.0 - alpha_opt) * dz_te
    y_known = np.array([1 if y != -1 else 0 for y in y_te])
    fpr_f, tpr_f, thr_f = roc_curve(y_known, fused_te, pos_label=0)
    auc_f = auc(fpr_f, tpr_f)
    fnr_f = 1 - tpr_f
    idx_f = np.argmin(np.abs(fpr_f - fnr_f))
    eer_f = (fpr_f[idx_f] + fnr_f[idx_f]) / 2
    print(f"[Diag] AUC_fused={auc_f:.4f} | EER_fused={eer_f:.4f}")


    pred_closed = log_te.argmax(axis=1)
    thr_vec_fused  = np.array([tau_opt.get(int(c), 1e9) for c in pred_closed]) + RHO
    thr_vec_margin = np.array([tau_margin.get(int(c), -1e9) for c in pred_closed])

    d1 = fused_te - thr_vec_fused
    d2 = thr_vec_margin - mz_te
    accept_known = (d1 <= 0) & (d2 <= 0)
    pred_open = np.where(accept_known, pred_closed, -1)


    gate_score = np.maximum(d1, d2) 
    fpr_g, tpr_g, thr_g = roc_curve(y_known, gate_score, pos_label=0)
    auc_g = auc(fpr_g, tpr_g)
    fnr_g = 1 - tpr_g
    idx_g = np.argmin(np.abs(fpr_g - fnr_g))
    eer_g = (fpr_g[idx_g] + fnr_g[idx_g]) / 2
    print(f"[Gate] AUC_gate={auc_g:.4f} | EER_gate={eer_g:.4f}")


    labels_all = sorted(list(set(y_te) | set(pred_open)))
    cm = confusion_matrix(y_te, pred_open, labels=labels_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_all)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Open-set Confusion Matrix (Two-stage gating + classwise-Mahalanobis)")
    plt.tight_layout(); plt.show()

    total = len(y_te)
    rejected = np.sum(pred_open == -1)
    rej_rate = 100.0 * rejected / total
    valid = [i for i,(p,y) in enumerate(zip(pred_open, y_te)) if p!=-1 and y!=-1]
    if valid:
        acc = accuracy_score(y_te[valid], pred_open[valid])
        print(f"Closed-set Accuracy (after gating): {acc*100:.2f}%")
    print(f"Rejected: {rejected}/{total} ({rej_rate:.2f}%)")


    plt.figure()
    plt.plot(fpr_f, tpr_f, label=f"fused AUC={auc_f:.3f}")
    plt.plot(fpr_g, tpr_g, label=f"gate  AUC={auc_g:.3f}")
    plt.plot([0,1],[0,1],"k--", alpha=0.4)
    plt.xlabel("FPR (unknown accepted)"); plt.ylabel("TPR (unknown rejected)")
    plt.title("ROC comparison: fused vs. two-stage gate")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

    train()
    evaluate()
