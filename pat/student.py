"""Stage B -- the amortized STUDENT that learns to reproduce the teacher's optimized splats in one
forward pass (no per-mesh optimization at inference).

Two networks, exactly as the user asked:

* **GroupNet** -- "how many points to group per one output supertori point".  Per-point (NOT a global
  slot competition, which mode-collapses): the verified :class:`pat.model.CoeffNet` trunk gives each
  point a feature; a **seed head** scores seed-ness and a **group head** gives a metric embedding.  At
  inference, non-max-suppressed seeds become groups and every point joins its nearest seed in embedding
  space, so the number of groups ``K`` EMERGES per mesh (no count regression).
* **FitNet** -- a *separate* permutation-invariant set encoder that best-fits one point-group into a
  single supertoroid+box splat (a ``ROW_W``-vector consumed by :meth:`SuperToroidSplats.from_rows`).

Supervision comes from the cached teacher artifacts (``pat.teacher``): ``owner`` (per-point hard owner
splat) trains GroupNet; the per-splat ``params`` + soft ``resp`` groups train FitNet via a geometry-first
loss on the induced single-splat SDF.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core
from .neighbors import neighborhood_features
from .splat import SuperToroidSplats, ROW_W, _ROW_SLICES

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except Exception:                                   # pragma: no cover
    _HAVE_TQDM = False

EPS = 1e-9


# --------------------------------------------------------------------------- #
#  Differentiable single-splat solid SDF (functional form of SuperToroidSplats._g_splat)
# --------------------------------------------------------------------------- #
def single_splat_sdf(row, q, p_max=6.0):
    """Signed solid value of ONE splat described by a ``(..., ROW_W)`` param row, at query points
    ``q (..., Q, 3)`` -> ``(..., Q)``.  Differentiable in ``row`` (the FitNet geometry loss flows
    through this) -- mirrors :meth:`SuperToroidSplats._g_splat` (supertoroid clipped by the box)."""
    c = row[..., _ROW_SLICES["center"]]
    u = F.normalize(row[..., _ROW_SLICES["raw_u"]], dim=-1, eps=EPS)
    ea = row[..., _ROW_SLICES["raw_ea"]]
    ea = F.normalize(ea - (ea * u).sum(-1, keepdim=True) * u, dim=-1, eps=EPS)
    eb = torch.cross(u, ea, dim=-1)
    R = row[..., _ROW_SLICES["log_R"]].squeeze(-1).clamp(-5, 2).exp()
    r = row[..., _ROW_SLICES["log_r"]].squeeze(-1).clamp(-5, 2).exp()
    pt = core.raw_to_p(row[..., _ROW_SLICES["raw_pt"]].squeeze(-1), p_max=p_max)
    pr = core.raw_to_p(row[..., _ROW_SLICES["raw_pr"]].squeeze(-1), p_max=p_max)
    b = row[..., _ROW_SLICES["log_b"]].clamp(-4, 1.5).exp()
    boxc = c + row[..., _ROW_SLICES["box_offset"]].clamp(-1, 1)
    sign = torch.sign(row[..., _ROW_SLICES["sign"]].squeeze(-1)).clamp_min(-1.0)
    cE = c.unsqueeze(-2); uE = u.unsqueeze(-2); eaE = ea.unsqueeze(-2); ebE = eb.unsqueeze(-2)
    g_s = sign.unsqueeze(-1) * core.supertoroid_sdf(
        q, cE, uE, eaE, R.unsqueeze(-1), r.unsqueeze(-1), pt.unsqueeze(-1), pr.unsqueeze(-1))
    relb = q - boxc.unsqueeze(-2)
    lx = (relb * uE).sum(-1); ly = (relb * eaE).sum(-1); lz = (relb * ebE).sum(-1)
    qb = torch.stack([lx, ly, lz], -1).abs() - b.unsqueeze(-2)
    g_box = qb.clamp_min(0.0).norm(dim=-1) + qb.amax(-1).clamp_max(0.0)
    return torch.maximum(g_s, g_box)


# --------------------------------------------------------------------------- #
#  Neighborhood construction (the CoeffNet trunk input)
# --------------------------------------------------------------------------- #
def build_neighborhoods(P, N, k=24, device="cpu"):
    """``(P,N)`` cloud -> per-point kNN neighborhoods ``(Npts, k+1, 3)`` pos + nrm (col 0 = the point)."""
    P = torch.as_tensor(np.asarray(P), dtype=torch.float32, device=device)
    N = torch.as_tensor(np.asarray(N), dtype=torch.float32, device=device)
    idx = torch.cdist(P, P).topk(k + 1, dim=1, largest=False).indices       # (Npts, k+1), self at 0
    return P[idx], N[idx]


# --------------------------------------------------------------------------- #
#  GroupNet
# --------------------------------------------------------------------------- #
class CoeffNetTrunk(nn.Module):
    """The verified :class:`pat.model.CoeffNet` encoder WITHOUT the coeff head: maps each point's
    neighborhood to a ``d_embed`` feature (the central token ``enc[:,0,:]``).  Per-point by
    construction -- the batch axis is the points, attention is over each point's own neighbors."""

    def __init__(self, d_embed=128, n_layers=8, n_heads=8, d_ff=512, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(6, d_embed)
        layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, nbr_pos, nbr_nrm):
        feats, _, _, _ = neighborhood_features(nbr_pos, nbr_nrm)            # (B, k+1, 6)
        return self.encoder(self.embed(feats))[:, 0, :]                    # (B, d_embed)


class GroupNet(nn.Module):
    """Per-point seed-ness + metric embedding for grouping the cloud into "supertori points"."""

    def __init__(self, d_embed=128, n_layers=8, n_heads=8, d_ff=512, d_g=32):
        super().__init__()
        self.trunk = CoeffNetTrunk(d_embed, n_layers, n_heads, d_ff)
        # heads see the trunk feature PLUS the point's absolute position: the local-frame neighborhood
        # features are translation-invariant, so without position two identical neighborhoods at
        # different places are indistinguishable and grouping (a spatial task) is ill-posed.
        self.seed_head = nn.Linear(d_embed + 3, 1)
        self.group_head = nn.Linear(d_embed + 3, d_g)

    def forward(self, nbr_pos, nbr_nrm, chunk=4096):
        seeds, embs = [], []
        for a in range(0, nbr_pos.shape[0], chunk):                        # chunk the points (memory)
            blk = nbr_pos[a:a + chunk]
            h = self.trunk(blk, nbr_nrm[a:a + chunk])
            h = torch.cat([h, blk[:, 0, :]], dim=-1)                       # + absolute position
            seeds.append(self.seed_head(h).squeeze(-1))
            embs.append(F.normalize(self.group_head(h), dim=-1))
        return torch.cat(seeds), torch.cat(embs)                           # (Npts,), (Npts, d_g)


def groupnet_loss(seed, emb, owner, *, margin=1.0, w_seed=1.0, w_emb=1.0):
    """Seed BCE (positives = each splat's max-ownership point) + supervised-contrastive embedding loss
    (same-owner attract, cross-owner repel past ``margin``) -- permutation-invariant to splat ids."""
    owner = torch.as_tensor(owner, dtype=torch.long, device=emb.device)
    # seed labels: the argmax-ownership point of each present splat is a positive
    is_seed = torch.zeros_like(seed)
    for o in torch.unique(owner):
        members = (owner == o).nonzero(as_tuple=True)[0]
        is_seed[members[seed[members].argmax()]] = 1.0                     # 1 seed per active splat
    pos_w = (is_seed.numel() - is_seed.sum()) / is_seed.sum().clamp_min(1)
    l_seed = F.binary_cross_entropy_with_logits(seed, is_seed, pos_weight=pos_w)
    # contrastive on a random subsample of point pairs (full N^2 is too big)
    n = emb.shape[0]; m = min(n, 512)
    sel = torch.randperm(n, device=emb.device)[:m]
    e = emb[sel]; o = owner[sel]
    d2 = (e[:, None, :] - e[None]).pow(2).sum(-1)                          # (m,m)
    same = (o[:, None] == o[None]).float()
    eye = torch.eye(m, device=e.device)
    l_attract = (same * (1 - eye) * d2).sum() / (same * (1 - eye)).sum().clamp_min(1)
    l_repel = ((1 - same) * (margin - d2).clamp_min(0)).sum() / (1 - same).sum().clamp_min(1)
    return w_seed * l_seed + w_emb * (l_attract + l_repel), dict(
        seed=float(l_seed.detach()), attract=float(l_attract.detach()), repel=float(l_repel.detach()))


@torch.no_grad()
def group_points(seed, emb, positions, *, nms_radius=0.12, seed_thresh=0.0, max_groups=160, w_emb=0.25):
    """Turn per-point (seed, emb) into groups: NMS the seeds by 3D radius, then assign every point to
    its nearest seed by 3D POSITION (spatially coherent by construction), with the embedding as a light
    tie-breaker for touching parts.  Returns ``(seed_idx (K,), assign (Npts,))``; K emerges per mesh."""
    pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=emb.device)
    cand = (seed > seed_thresh).nonzero(as_tuple=True)[0]
    if cand.numel() == 0:
        cand = seed.topk(min(8, seed.numel())).indices
    order = cand[seed[cand].argsort(descending=True)]
    chosen = []
    for i in order.tolist():
        if not chosen or (pos[i] - pos[chosen]).norm(dim=-1).min() > nms_radius:
            chosen.append(i)
        if len(chosen) >= max_groups:
            break
    chosen = torch.as_tensor(chosen, device=emb.device)
    d_pos = (pos[:, None, :] - pos[chosen][None]).pow(2).sum(-1)            # (N,K) 3D distance (primary)
    d_emb = (emb[:, None, :] - emb[chosen][None]).pow(2).sum(-1)            # (N,K) embedding (tie-break)
    return chosen, (d_pos + w_emb * d_emb).argmin(1)                       # spatially-coherent assignment


# --------------------------------------------------------------------------- #
#  FitNet
# --------------------------------------------------------------------------- #
class FitNet(nn.Module):
    """A point-group -> ONE supertoroid+box splat (a ``ROW_W`` row).  Permutation-invariant set encoder
    (CLS token over the group members); the splat center is anchored to the group's weighted centroid for
    pose stability, the rest is predicted in that local frame."""

    def __init__(self, d_embed=128, n_layers=6, n_heads=8, d_ff=512, p_max=6.0):
        super().__init__()
        self.p_max = p_max
        self.in_proj = nn.Linear(6, d_embed)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_embed))
        layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_embed, ROW_W)
        self._init_head()

    def _init_head(self):
        nn.init.zeros_(self.head.weight)
        b = torch.zeros(ROW_W)
        b[_ROW_SLICES["raw_u"]] = torch.tensor([0, 0, 1.0])
        b[_ROW_SLICES["raw_ea"]] = torch.tensor([1.0, 0, 0])
        b[_ROW_SLICES["log_R"]] = np.log(0.3); b[_ROW_SLICES["log_r"]] = np.log(0.15)
        b[_ROW_SLICES["raw_pt"]] = core.P2_RAW; b[_ROW_SLICES["raw_pr"]] = core.P2_RAW
        b[_ROW_SLICES["log_sigma"]] = np.log(0.18); b[_ROW_SLICES["log_b"]] = np.log(2.0)
        b[_ROW_SLICES["sign"]] = 1.0
        with torch.no_grad():
            self.head.bias.copy_(b)

    def forward(self, grp_pos, grp_nrm, grp_w, mask):
        """``grp_pos/grp_nrm (B,G,3)``, ``grp_w (B,G)`` membership, ``mask (B,G)`` bool valid.
        Returns ``(B, ROW_W)`` splat rows (center = weighted centroid + predicted offset)."""
        w = (grp_w * mask.float()); wn = w / w.sum(-1, keepdim=True).clamp_min(EPS)
        centroid = (wn[..., None] * grp_pos).sum(1)                         # (B,3) weighted center
        scale = ((grp_pos - centroid[:, None]).norm(dim=-1) * w).sum(-1) / w.sum(-1).clamp_min(EPS)
        scale = scale.clamp_min(0.05)
        rel = (grp_pos - centroid[:, None]) / scale[:, None, None]          # canonical local coords
        x = torch.cat([rel, grp_nrm], dim=-1)                              # (B,G,6)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), self.in_proj(x)], dim=1)
        pad = torch.cat([torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device), ~mask], 1)
        z = self.encoder(tok, src_key_padding_mask=pad)[:, 0, :]
        out = self.head(z)
        # un-canonicalize the geometric columns: center, radii, box back to world scale
        row = out.clone()
        row[:, _ROW_SLICES["center"]] = centroid + out[:, _ROW_SLICES["center"]] * scale[:, None]
        for key in ("log_R", "log_r", "log_sigma", "log_b"):
            row[:, _ROW_SLICES[key]] = out[:, _ROW_SLICES[key]] + torch.log(scale)[:, None]
        return row


def fitnet_loss(row_pred, row_true, grp_pos, mask, *, p_max=6.0, w_geo=1.0, w_par=0.2, w_sign=0.5,
                n_band=256):
    """Geometry-first FitNet loss: match the induced single-splat SDF (gauge-invariant) on band points
    around the group, plus light param regression on unambiguous scalars + a sign term."""
    B = row_pred.shape[0]
    dev = row_pred.device
    # band query points: group members + small jitter (where g should differ most informatively)
    base = grp_pos + torch.randn_like(grp_pos) * 0.05
    q = base                                                              # (B,G,3)
    g_pred = single_splat_sdf(row_pred, q, p_max=p_max)
    g_true = single_splat_sdf(row_true.to(dev), q, p_max=p_max)
    mexp = mask.float()
    l_geo = ((g_pred - g_true).abs() * mexp).sum() / mexp.sum().clamp_min(1)
    par = torch.cat([row_pred[:, _ROW_SLICES[k]].reshape(B, -1)
                     for k in ("log_R", "log_r", "raw_pt", "raw_pr", "log_b")], 1)
    par_t = torch.cat([row_true[:, _ROW_SLICES[k]].reshape(B, -1).to(dev)
                       for k in ("log_R", "log_r", "raw_pt", "raw_pr", "log_b")], 1)
    l_par = F.smooth_l1_loss(par, par_t)
    sgn_t = (torch.sign(row_true[:, _ROW_SLICES["sign"]].squeeze(-1).to(dev)) > 0).float()
    l_sign = F.binary_cross_entropy_with_logits(row_pred[:, _ROW_SLICES["sign"]].squeeze(-1), sgn_t)
    loss = w_geo * l_geo + w_par * l_par + w_sign * l_sign
    return loss, dict(geo=float(l_geo.detach()), par=float(l_par.detach()), sign=float(l_sign.detach()))


# --------------------------------------------------------------------------- #
#  Inference composition (cloud -> splat field -> mesh)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_amortized(P, N, groupnet, fitnet, *, k=24, min_group=8, nms_radius=0.12,
                          device="cuda", p_max=6.0):
    """Cloud -> GroupNet groups -> FitNet per group -> assembled :class:`SuperToroidSplats` (no per-mesh
    optimization).  Returns ``(splat, n_groups)``."""
    groupnet.eval(); fitnet.eval()
    nbr_pos, nbr_nrm = build_neighborhoods(P, N, k=k, device=device)
    Pt = nbr_pos[:, 0, :]; Nt = nbr_nrm[:, 0, :]
    seed, emb = groupnet(nbr_pos, nbr_nrm)
    chosen, assign = group_points(seed, emb, Pt, nms_radius=nms_radius)
    rows = []
    for gi in range(len(chosen)):
        m = assign == gi
        if int(m.sum()) < min_group:
            continue
        gp = Pt[m][None]; gn = Nt[m][None]
        gw = torch.ones(1, gp.shape[1], device=device)
        mk = torch.ones(1, gp.shape[1], dtype=torch.bool, device=device)
        rows.append(fitnet(gp, gn, gw, mk))
    if not rows:
        return None, 0
    return SuperToroidSplats.from_rows(torch.cat(rows, 0), p_max=p_max), len(rows)


# --------------------------------------------------------------------------- #
#  Training over the cached teacher shards
# --------------------------------------------------------------------------- #
def iter_shards(teacher_dir, iou_min=0.0, status_ok_only=False):
    """Yield loaded teacher artifacts.  Gate on reconstruction quality ``iou >= iou_min`` (the robust,
    scale-free filter -- a low IoU means a bad teacher example, whether from a poor fit or an unreliable
    GT on a non-watertight mesh; either way it should not train the student)."""
    for path in sorted(glob.glob(os.path.join(teacher_dir, "shard_*", "mesh_*.pt"))):
        a = torch.load(path, weights_only=False, map_location="cpu")
        if status_ok_only and a.get("status") != "ok":
            continue
        if float(a.get("iou", 0.0)) < iou_min:
            continue
        yield a


def train_groupnet(teacher_dir, *, epochs=4, lr=1e-3, k=24, d_g=32, device="cuda", log_every=50,
                   net=None, max_meshes=None, iou_min=0.0):
    """Train GroupNet on cached teacher ``owner`` labels.  One mesh per step (the points are the batch)."""
    net = (net or GroupNet(d_g=d_g)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    arts = list(iter_shards(teacher_dir, iou_min=iou_min))
    if not arts:
        raise ValueError(f"no teacher shards with iou >= {iou_min} in {teacher_dir} -- lower iou_min "
                         f"(the teacher's IoU distribution is in the QA stats).")
    if max_meshes:
        arts = arts[:max_meshes]
    hist = []
    for ep in range(epochs):
        order = np.random.default_rng(ep).permutation(len(arts))
        it = tqdm(order, desc=f"groupnet ep{ep}") if _HAVE_TQDM else order
        for step, idx in enumerate(it):
            a = arts[idx]
            P = a["P"].float().numpy(); N = a["N"].float().numpy()
            nbr_pos, nbr_nrm = build_neighborhoods(P, N, k=k, device=device)
            seed, emb = net(nbr_pos, nbr_nrm)
            loss, parts = groupnet_loss(seed, emb, a["owner"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if step % log_every == 0:
                hist.append(dict(ep=ep, step=step, loss=float(loss.detach()), **parts))
    return net, hist


def _gather_groups(a, device, min_group=6, max_group=256):
    """From a teacher artifact, build per-splat FitNet training tuples (grp_pos, grp_nrm, grp_w, target)."""
    P = a["P"].float().to(device); N = a["N"].float().to(device)
    owner = a["owner"].to(device).long(); resp = a["resp"].float().to(device)
    params = a["params"].float().to(device)
    tuples = []
    for i in range(a["M"]):
        m = (owner == i).nonzero(as_tuple=True)[0]
        if m.numel() < min_group:                                          # back off to top soft members
            m = resp[:, i].topk(min(min_group, resp.shape[0])).indices
        if m.numel() > max_group:
            m = m[torch.randperm(m.numel(), device=device)[:max_group]]
        tuples.append((P[m], N[m], resp[m, i], params[i]))
    return tuples


def train_fitnet(teacher_dir, *, epochs=4, lr=1e-3, batch=32, device="cuda", log_every=50,
                 net=None, max_meshes=None, p_max=6.0, iou_min=0.0):
    """Train FitNet to map each teacher group -> its splat row (geometry-first loss)."""
    net = (net or FitNet(p_max=p_max)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    arts = list(iter_shards(teacher_dir, iou_min=iou_min))
    if not arts:
        raise ValueError(f"no teacher shards with iou >= {iou_min} in {teacher_dir} -- lower iou_min.")
    if max_meshes:
        arts = arts[:max_meshes]
    hist = []
    for ep in range(epochs):
        order = np.random.default_rng(ep).permutation(len(arts))
        it = tqdm(order, desc=f"fitnet ep{ep}") if _HAVE_TQDM else order
        for step, idx in enumerate(it):
            tuples = _gather_groups(arts[idx], device)
            for b0 in range(0, len(tuples), batch):
                chunk = tuples[b0:b0 + batch]
                G = max(t[0].shape[0] for t in chunk)
                B = len(chunk)
                gp = torch.zeros(B, G, 3, device=device); gn = torch.zeros(B, G, 3, device=device)
                gw = torch.zeros(B, G, device=device); mk = torch.zeros(B, G, dtype=torch.bool, device=device)
                tgt = torch.stack([t[3] for t in chunk])
                for j, (p, n, w, _) in enumerate(chunk):
                    g = p.shape[0]; gp[j, :g] = p; gn[j, :g] = n; gw[j, :g] = w; mk[j, :g] = True
                row = net(gp, gn, gw, mk)
                loss, parts = fitnet_loss(row, tgt, gp, mk, p_max=p_max)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if step % log_every == 0:
                hist.append(dict(ep=ep, step=step, loss=float(loss.detach()), **parts))
    return net, hist
