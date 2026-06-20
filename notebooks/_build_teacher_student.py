"""Generate notebooks/train_teacher_student_colab.ipynb (generator -> no hand-edited JSON).

A copy of train_pat_colab_processed.ipynb's setup (Drive, repo, Objaverse++ mesh cache) followed by the
TEACHER -> STUDENT amortized-splat pipeline, every stage cached to Drive, presence-checked, regenerated
ONLY if its artifact is missing.  Built for a Colab **G4 (Tesla T4, 16 GB)**.

Stages:
  1  mesh cache (Objaverse++ subset)        -> DRIVE/mesh_cache.pt          (reused if present)
  2  TEACHER per-mesh optimize (sharded)    -> DRIVE/teacher/shard_*/*.pt   (per-mesh skip, resumable)
  2b teacher QA stats                        -> DRIVE/teacher/stats.png
  3  GroupNet train                          -> DRIVE/student/groupnet.pt   (skip if present)
  4  FitNet train                            -> DRIVE/student/fitnet.pt      (skip if present)
  5  amortized eval (held-out)               -> DRIVE/eval/metrics.json + recon_*.png
"""
import json, os

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _src(lines)}
def _src(lines):
    flat = []
    for l in lines:
        flat.extend(l.split("\n"))
    return [s + "\n" for s in flat[:-1]] + [flat[-1]] if flat else []

cells = []

cells.append(md(
"# Points as **Supertoroids** — teacher→student amortized splat optimizer (Colab **G4 / T4 16 GB**)",
"",
"**Stage A (teacher, slow, per-mesh):** for each Objaverse++ mesh, optimize the MINIMAL set of",
"supertoroid + cut-out-box splats whose **Minkowski filled-volume distance** to the mesh is ≤ `MD_TARGET`",
"(default 0.001), **respecting holes**.  Each mesh's optimized splats + the point→splat grouping are",
"cached to Drive.",
"",
"**Stage B (student, amortized, fast):** after caching the teacher targets, train two networks — a",
"**GroupNet** that decides *how many input points to group per output supertori point*, and a separate",
"**FitNet** that best-fits each group into a single splat.  At inference they reconstruct a mesh in one",
"forward pass (no per-mesh optimization).",
"",
"Every stage's result is **cached to Drive, checked for presence, and regenerated only if missing** —",
"so an interrupted Colab session resumes where it left off.",
))

# ----------------------------------------------------------------------------- 1. setup
cells.append(md("## 1 · Setup — Google Drive, repo, deps"))
cells.append(code(
"import os, sys, subprocess",
"from google.colab import drive",
"drive.mount('/content/drive')",
"DRIVE_DIR = '/content/drive/MyDrive/points_as_supertoroids'",
"os.makedirs(DRIVE_DIR, exist_ok=True)",
"print('outputs will be saved under:', DRIVE_DIR)",
"",
"REPO_URL    = 'https://github.com/OlegJakushkin/Points_as_supertoroids.git'",
"REPO_BRANCH = 'main'   # branch holding pat/teacher.py + pat/student.py",
"REPO_DIR    = 'Points_as_supertoroids'",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'trimesh', 'scikit-image', 'scipy'], check=False)",
"if not any(os.path.isdir(os.path.join(c, 'pat')) for c in [REPO_DIR, '.', '..']):",
"    subprocess.run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, REPO_URL, REPO_DIR], check=True)",
"for cand in [REPO_DIR, '.', '..']:",
"    if os.path.isdir(os.path.join(cand, 'pat')):",
"        os.chdir(cand); break",
"sys.path.insert(0, os.getcwd())",
"import torch, pat",
"assert torch.cuda.is_available(), 'Set the Colab runtime to a GPU (T4 / G4).'",
"print('cwd', os.getcwd(), '| pat ready | GPU', torch.cuda.get_device_name(0))",
))

# ----------------------------------------------------------------------------- 2. config
cells.append(md("## 2 · Config — every knob commented"))
cells.append(code(
"# ---- dataset (Stage 1) ----",
"MESHES_TARGET = 6000   # Objaverse++ high-quality meshes to cache (heavy download; resumable on Drive).",
"MIN_SCORE     = 2      # Objaverse++ quality: 2 = High, 3 = Superior.",
"DENSE         = 1536   # dense surface points cached per mesh (the teacher densifies these to 50k).",
"NQUERY        = 160    # GT query points per mesh (only used by the legacy trainer; teacher ignores).",
"MAXFACES      = 200000 # skip pathologically heavy meshes.",
"",
"# ---- teacher (Stage A) — THE EXPENSIVE STAGE; sharded + resumable; BATCHED across meshes ----",
"TEACHER_SUBSET = 800   # how many cached meshes to run the teacher on. START SMALL — it is sharded and",
"                       #   resumable, so raise it later and re-run to fill in more (the student needs",
"                       #   DIVERSE teachers, not all of them).",
"BATCH_MESHES = 'auto'  # meshes optimized IN PARALLEL on the GPU per batch — the key throughput knob.",
"                       #   'auto' probes VRAM (peak at B=1,2) and picks the largest SAFE batch, cleaning",
"                       #   up after; or set an int (raise = faster until VRAM-bound; lower on OOM).",
"MD_TARGET   = 0.001    # target Minkowski filled-volume distance = FRACTION of the cube volume that the",
"                       #   reconstruction and GT solids disagree on, vol(A xor B)/vol(cube). 0.001 = 0.1%",
"                       #   disagreement (a near-perfect fit; e.g. a clean torus fit scores ~0.0009).",
"IOU_OK      = 0.6      # QUALITY GATE: a mesh counts as a usable teacher example if IoU >= this (or MD <=",
"                       #   MD_TARGET). Detailed meshes rarely hit the tight MD bar; IoU is the scale-free",
"                       #   gate. The student trains ONLY on meshes >= this IoU (lower for more/rougher data).",
"RES         = 64       # MD grid resolution (64^3, 4 antithetic offsets). 64 is ~8x faster than 128 and",
"                       #   adequate for the prune decisions; raise to 96/128 for a finer MD (much slower).",
"M_INIT      = 40       # splats alive at the START (the warm-fit fits these).",
"M_MAX       = 128      # capacity/mesh; GROW activates dormant splats up to this where a mesh misses",
"                       #   MD_TARGET. Meshes still short at M_MAX are marked 'hard'. (Also bounds VRAM.)",
"GROW_ADD    = 16       # dormant splats activated per grow round, at the worst-fit regions.",
"MAX_GROW    = 4        # max grow rounds before the speculative prune.",
"STEPS_WARM  = 300      # batched warm-fit steps (all meshes in the batch optimized together, on-GPU).",
"STEPS_REFIT = 70       # batched refit steps per grow/prune round.",
"MIN_KEEP    = 8        # never prune below this many splats.",
"",
"# ---- student (Stage B) ----",
"K_NBR        = 24      # kNN neighborhood size fed to the GroupNet/FitNet transformer trunk.",
"GROUP_EPOCHS = 4       # GroupNet training epochs over the cached teacher shards.",
"FIT_EPOCHS   = 4       # FitNet training epochs.",
"D_EMBED      = 128     # transformer width (paper CoeffNet = 128).",
"N_LAYERS     = 8       # transformer depth.",
"N_EVAL       = 8       # held-out meshes to reconstruct + score at the end.",
"",
"FORCE_TEACHER = False  # set True to re-run the teacher even if shards exist.",
"FORCE_STUDENT = False  # set True to re-train the student even if weights exist.",
"",
"MESH_CACHE  = os.path.join(DRIVE_DIR, 'mesh_cache.pt')",
"TEACHER_DIR = os.path.join(DRIVE_DIR, 'teacher')",
"STUDENT_DIR = os.path.join(DRIVE_DIR, 'student'); os.makedirs(STUDENT_DIR, exist_ok=True)",
"EVAL_DIR    = os.path.join(DRIVE_DIR, 'eval');    os.makedirs(EVAL_DIR, exist_ok=True)",
"print('config set | teacher subset', TEACHER_SUBSET, '| MD target', MD_TARGET)",
))

# ----------------------------------------------------------------------------- 3. mesh cache
cells.append(md(
"## 3 · Mesh cache — Objaverse++ high-quality subset (reused if present, else built & resumed)",
))
cells.append(code(
"import json, shutil, urllib.request",
"if os.path.exists(MESH_CACHE):",
"    cache = torch.load(MESH_CACHE, weights_only=False)",
"    print(f'reusing {cache[\"P\"].shape[0]} cached meshes (dense={cache[\"P\"].shape[1]})')",
"else:",
"    from pat.datasets import (stratified_sample, build_mesh_cache, objaverse_object_paths, fetch_objaverse_glbs)",
"    ANN='https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus/resolve/main/annotated_800k.json'",
"    os.makedirs('data', exist_ok=True)",
"    if not os.path.exists('data/ann.json'):",
"        urllib.request.urlretrieve(ANN, 'data/ann.json')",
"    items=json.load(open('data/ann.json')); items=list(items.items()) if isinstance(items,dict) else [(r.get('UID'),r) for r in items]",
"    _t=lambda v: str(v).strip().lower() in ('true','1','yes')",
"    hq=[(u,a) for u,a in items if u and int(a.get('score',0))>=MIN_SCORE and not _t(a.get('is_scene',False))]",
"    style={u:str(a.get('style','?')) for u,a in hq}",
"    sel=stratified_sample([u for u,_ in hq], total=int(MESHES_TARGET*1.25), min_per_class=3, seed=0, class_of=lambda u:style[u])",
"    PROG=os.path.join(DRIVE_DIR,'objxx_progress.json'); GLB='data/glb_batch'; BATCH=500",
"    parts=[torch.load(MESH_CACHE,weights_only=False)] if os.path.exists(MESH_CACHE) else []",
"    cached=parts[0]['P'].shape[0] if parts else 0",
"    i=json.load(open(PROG))['idx'] if os.path.exists(PROG) else 0",
"    opaths=objaverse_object_paths()",
"    while cached<MESHES_TARGET and i<len(sel):",
"        batch=sel[i:i+BATCH]; i+=BATCH",
"        objs=fetch_objaverse_glbs(batch, GLB, paths=opaths, workers=32)",
"        d=build_mesh_cache(list(objs.values()), DENSE, NQUERY, max_faces=MAXFACES, seed=i, shuffle=False)",
"        shutil.rmtree(GLB, ignore_errors=True)",
"        if d is not None:",
"            parts.append(d); cached+=d['P'].shape[0]",
"            cache={k:torch.cat([p[k] for p in parts],0) for k in ('P','N','Q','PHI')}",
"            torch.save(cache, MESH_CACHE); json.dump({'idx':i,'target':MESHES_TARGET}, open(PROG,'w'))",
"        print(f'  cached {cached}/{MESHES_TARGET}', flush=True)",
"print('dataset ready:', cache['P'].shape[0], 'meshes')",
))

# ----------------------------------------------------------------------------- 4. teacher
cells.append(md(
"## 4 · Stage A — TEACHER: minimal-splat optimize, **BATCHED across meshes** (sharded, resumable)",
"",
"`fit_and_cache_batch` optimizes **`BATCH_MESHES` meshes in parallel on the GPU**: it over-provisions",
"`M_INIT` splats/mesh, batched warm-fits, then runs a **speculative prune** — for a descending keep-",
"schedule it keeps the top-ownership splats per mesh, refits ALL meshes together, scores MD per mesh, and",
"remembers each mesh's smallest field that still met `MD_TARGET` (independently per mesh).  Ground-truth",
"occupancy is built **hole-respecting from the cached cloud P+N** (no mesh re-download).  Each mesh writes",
"one shard atomically; already-cached meshes are **skipped**, so it resumes after a Colab disconnect.",
))
cells.append(code(
"from pat import teacher_batch as TB",
"import glob, time, torch",
"from tqdm.auto import tqdm",
"Pall, Nall = cache['P'], cache['N']",
"TOT = min(TEACHER_SUBSET, Pall.shape[0])",
"os.makedirs(TEACHER_DIR, exist_ok=True)",
"if BATCH_MESHES == 'auto':                       # probe VRAM -> largest safe batch (cleans up after)",
"    BATCH_MESHES = TB.auto_batch_size(Pall[0].numpy(), Nall[0].numpy(), m_max=M_MAX, res=RES, device='cuda')",
"    print('auto-detected safe BATCH_MESHES =', BATCH_MESHES)",
"have = len(glob.glob(os.path.join(TEACHER_DIR, 'shard_*', 'mesh_*.pt')))",
"print(f'teacher: {have} shards already cached; target {TOT} | batch {BATCH_MESHES}')",
"ok = hard = ran = 0; t0 = time.time()",
"for s in tqdm(range(0, TOT, BATCH_MESHES), desc='teacher (batched)'):",
"    chunk = list(range(s, min(s + BATCH_MESHES, TOT)))",
"    rows = TB.fit_and_cache_batch(",
"        [Pall[g].numpy() for g in chunk], [Nall[g].numpy() for g in chunk], chunk, TEACHER_DIR,",
"        force=FORCE_TEACHER, m_init=M_INIT, m_max=M_MAX, grow_add=GROW_ADD, max_grow=MAX_GROW,",
"        md_target=MD_TARGET, iou_ok=IOU_OK, res=RES, steps_warm=STEPS_WARM, steps_refit=STEPS_REFIT,",
"        min_keep=MIN_KEEP, device='cuda')",
"    for g, status, *rest in rows:",
"        ran += status != 'cached'; ok += status == 'ok'; hard += status == 'hard'",
"    if torch.cuda.is_available(): torch.cuda.empty_cache()   # clean up between batches",
"    if ran:",
"        print(f'  {min(s+BATCH_MESHES,TOT)}/{TOT} | ran {ran} | ok {ok} hard {hard} | {(time.time()-t0)/max(ran,1):.1f}s/mesh', flush=True)",
"n_shards = len(glob.glob(os.path.join(TEACHER_DIR, 'shard_*', 'mesh_*.pt')))",
"print(f'teacher cache now holds {n_shards} meshes ({ok} ok / {hard} hard this session)')",
))

cells.append(md("## 4b · Teacher QA — splat-count & MD/IoU distribution, USABLE (IoU≥IOU_OK) fraction"))
cells.append(code(
"import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
"from pat import student as ST",
"rows=[{'M':a['M'],'md':a['md'],'iou':a['iou'],'status':a['status']} for a in ST.iter_shards(TEACHER_DIR)]",
"Ms=[r['M'] for r in rows]; mds=[r['md'] for r in rows]; ious=[r['iou'] for r in rows]",
"usable=int(np.sum([i>=IOU_OK for i in ious])); frac=usable/max(len(rows),1)",
"json.dump(rows, open(os.path.join(TEACHER_DIR,'manifest.json'),'w'))",
"fig,ax=plt.subplots(1,3,figsize=(13,3.4))",
"ax[0].hist(Ms,bins=30,color='C0'); ax[0].set_title('# splats per mesh'); ax[0].set_xlabel('M')",
"ax[1].hist(mds,bins=30,color='C2'); ax[1].axvline(MD_TARGET,ls=':',c='r'); ax[1].set_title('MD = vol(A xor B)/vol(cube)'); ax[1].set_xlabel('MD')",
"ax[2].hist(ious,bins=30,color='C3'); ax[2].axvline(IOU_OK,ls=':',c='b'); ax[2].set_title('IoU'); ax[2].set_xlabel('IoU')",
"fig.suptitle(f'teacher QA — {len(rows)} meshes | median M={int(np.median(Ms)) if Ms else 0} | usable IoU>={IOU_OK}: {usable} ({frac:.0%})')",
"fig.tight_layout(); fig.savefig(os.path.join(TEACHER_DIR,'stats.png'),dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(TEACHER_DIR,'stats.png')))",
"print(f'median splats/mesh {int(np.median(Ms)) if Ms else 0} | median IoU {np.median(ious):.2f} | USABLE (IoU>={IOU_OK}): {usable}/{len(rows)} ({frac:.0%}) -> these train the student')",
"if usable<32: print('WARNING: very few usable meshes. Lower IOU_OK, or run the DIAGNOSTIC cell to see if it is the GT (non-watertight) or the fit.')",
))

cells.append(md(
"## 4c · DIAGNOSTIC — is a low IoU the GT or the fit?",
"",
"For a few meshes: **surface error** = mean `|blend SDF|` on the input cloud (≈0 means the reconstruction",
"surface sits ON the points → a GOOD fit; if this is small but IoU is low, the *ground-truth occupancy* is",
"unreliable — typically a non-watertight mesh whose inside/outside is ill-defined from points+normals).  A",
"large surface error means the *fit* itself didn't converge (raise STEPS_WARM / M_MAX).  Renders show GT vs",
"reconstruction side by side.",
))
cells.append(code(
"from pat import teacher as TCH, splat as SP",
"import numpy as np, torch, glob as _g",
"paths=sorted(_g.glob(os.path.join(TEACHER_DIR,'shard_*','mesh_*.pt')))",
"pick=paths[:: max(1,len(paths)//6)][:6]",
"print('%-8s %6s %8s %8s %8s %9s'%('gid','M','surf_err','MD','IoU','GT_vol%'))",
"for p in pick:",
"    a=TCH.load_teacher(p); sp=a['splat'].cuda(); P=a['P'].float().numpy(); N=a['N'].float().numpy()",
"    surf=float(sp.sdf_torch(torch.as_tensor(P,dtype=torch.float32,device='cuda')).abs().mean())",
"    cs=TCH.CloudShape(P,N); occ=TCH.gt_occupancy(cs,res=RES); volpct=100*float(occ.mean())",
"    md,iou=TCH.md_filled_volume(sp,occ,res=RES,device='cuda',return_iou=True)   # fresh (new MD scale)",
"    print('%-8d %6d %8.4f %8.4f %8.3f %9.1f'%(a['gid'],a['M'],surf,md,iou,volpct))",
"    try: SP.render_comparison(cs, sp, os.path.join(TEACHER_DIR,f'diag_{a[\"gid\"]:06d}.png'), title=f'gid {a[\"gid\"]} IoU {a[\"iou\"]:.2f}')",
"    except Exception as e: print('  render skip', e)",
"from IPython.display import Image, display",
"for p in sorted(_g.glob(os.path.join(TEACHER_DIR,'diag_*.png')))[:4]: display(Image(p))",
"print('READ: small surf_err + low IoU => GT/occupancy problem (non-watertight). large surf_err => fit problem.')",
))

# ----------------------------------------------------------------------------- 5. GroupNet
cells.append(md(
"## 5 · Stage B1 — GroupNet: learn *how many points to group per supertori point*",
"",
"Per-point (position-aware) seed-ness + metric embedding on the cached teacher `owner` labels, trained ONLY",
"on **usable meshes (IoU≥IOU_OK)** — bad teacher examples are gated out, so they can't poison the student.",
"At inference, NMS over the seeds gives K groups and every point joins its nearest seed (spatially coherent).",
))
cells.append(code(
"GN_PATH=os.path.join(STUDENT_DIR,'groupnet.pt')",
"if os.path.exists(GN_PATH) and not FORCE_STUDENT:",
"    ck=torch.load(GN_PATH, weights_only=False)",
"    gnet=ST.GroupNet(d_embed=D_EMBED, n_layers=N_LAYERS, d_g=ck.get('d_g',32)).cuda()",
"    gnet.load_state_dict(ck['state']); print('reused groupnet.pt')",
"else:",
"    gnet, gh = ST.train_groupnet(TEACHER_DIR, epochs=GROUP_EPOCHS, k=K_NBR, device='cuda', iou_min=IOU_OK,",
"        net=ST.GroupNet(d_embed=D_EMBED, n_layers=N_LAYERS, d_g=32).cuda())",
"    torch.save({'state':gnet.state_dict(),'d_g':32}, GN_PATH)",
"    json.dump(gh, open(os.path.join(STUDENT_DIR,'groupnet_log.json'),'w'), indent=1)",
"    if gh: print('GroupNet trained; loss', round(gh[0]['loss'],3), '->', round(gh[-1]['loss'],3))",
))

# ----------------------------------------------------------------------------- 6. FitNet
cells.append(md(
"## 6 · Stage B2 — FitNet: best-fit each group into a single supertoroid splat",
"",
"A separate permutation-invariant set encoder maps one point-group → one splat's parameters, supervised",
"by the teacher's per-splat params via a **geometry-first** loss on the induced single-splat SDF.",
))
cells.append(code(
"FN_PATH=os.path.join(STUDENT_DIR,'fitnet.pt')",
"if os.path.exists(FN_PATH) and not FORCE_STUDENT:",
"    ck=torch.load(FN_PATH, weights_only=False)",
"    fnet=ST.FitNet(d_embed=D_EMBED, n_layers=max(6,N_LAYERS-2)).cuda(); fnet.load_state_dict(ck['state'])",
"    print('reused fitnet.pt')",
"else:",
"    fnet, fh = ST.train_fitnet(TEACHER_DIR, epochs=FIT_EPOCHS, device='cuda', iou_min=IOU_OK,",
"        net=ST.FitNet(d_embed=D_EMBED, n_layers=max(6,N_LAYERS-2)).cuda())",
"    torch.save({'state':fnet.state_dict()}, FN_PATH)",
"    json.dump(fh, open(os.path.join(STUDENT_DIR,'fitnet_log.json'),'w'), indent=1)",
"    if fh: print('FitNet trained; loss', round(fh[0]['loss'],3), '->', round(fh[-1]['loss'],3))",
))

# ----------------------------------------------------------------------------- 7. eval
cells.append(md(
"## 7 · Amortized eval — reconstruct held-out meshes in one forward pass + score MD",
"",
"For meshes the teacher never saw, GroupNet+FitNet reconstruct directly (no per-mesh optimization); we",
"score the **Minkowski filled-volume distance** of the amortized reconstruction against the hole-aware GT.",
))
cells.append(code(
"from pat import splat as SP",
"metrics=[]; start=min(TEACHER_SUBSET, cache['P'].shape[0]); end=min(start+N_EVAL, cache['P'].shape[0])",
"holdout=list(range(start,end)) if end>start else list(range(max(0,cache['P'].shape[0]-N_EVAL), cache['P'].shape[0]))",
"for j,gid in enumerate(holdout):",
"    P=cache['P'][gid].numpy(); N=cache['N'][gid].numpy()",
"    sp,K=ST.reconstruct_amortized(P,N,gnet,fnet,k=K_NBR,device='cuda')",
"    if sp is None: continue",
"    cs=TCH.CloudShape(P,N); occ=TCH.gt_occupancy(cs,res=RES)",
"    md,iou=TCH.md_filled_volume(sp,occ,res=RES,device='cuda',return_iou=True)",
"    metrics.append({'gid':int(gid),'K':int(K),'md':float(md),'iou':float(iou)})",
"    try:",
"        SP.render_comparison(cs, sp, os.path.join(EVAL_DIR,f'recon_{gid:06d}.png'), title=f'mesh {gid} (K={K})')",
"    except Exception as e: print('render skip', gid, e)",
"json.dump(metrics, open(os.path.join(EVAL_DIR,'metrics.json'),'w'), indent=1)",
"import numpy as np",
"if metrics:",
"    print('amortized held-out: mean MD %.4f | mean IoU %.3f | mean K %.1f'%(",
"        np.mean([m['md'] for m in metrics]), np.mean([m['iou'] for m in metrics]), np.mean([m['K'] for m in metrics])))",
"    from IPython.display import Image, display",
"    import glob as _g",
"    for p in sorted(_g.glob(os.path.join(EVAL_DIR,'recon_*.png')))[:4]: display(Image(p))",
"print('done — teacher shards, student weights, and eval all cached under', DRIVE_DIR)",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "train_teacher_student_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "(%d cells)" % len(cells))
