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
"# ---- teacher (Stage A) — THE EXPENSIVE STAGE; sharded + resumable ----",
"TEACHER_SUBSET = 2000  # how many cached meshes to run the teacher on (~30-60 s/mesh on a T4).",
"                       #   The student needs DIVERSE teachers, not all of them; raise opportunistically.",
"MD_TARGET   = 0.001    # target Minkowski filled-volume distance vol(A xor B). The teacher minimizes the",
"                       #   splat count subject to MD <= this (holes respected). See the QA stats for the",
"                       #   achievable floor: thin/sharp shapes may not reach 0.001 -> status='hard' (gated).",
"RES         = 128      # occupancy grid resolution for the MD metric (128^3, 4 antithetic offsets).",
"N_INIT      = 64       # splats the teacher over-provisions before pruning down to the minimum.",
"STEPS_WARM  = 400      # warm-fit optimizer steps before pruning.",
"STEPS_REFIT = 150      # refit steps after each grow/prune (survivors close the gap).",
"TIME_BUDGET = 60       # hard per-mesh wall-clock cap (s) so one hard mesh can't stall the whole run.",
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
"## 4 · Stage A — TEACHER: per-mesh minimal-splat optimize (sharded, presence-checked, resumable)",
"",
"For each mesh, `fit_teacher` over-provisions splats, warm-fits, grows at the worst-fit regions if it",
"missed `MD_TARGET`, then **greedily prunes to the minimum** (drop the splat owning the fewest surface",
"points, refit, accept iff MD still ≤ target).  Ground-truth occupancy is built **hole-respecting from",
"the cached cloud P+N** (no mesh re-download).  Each mesh writes one shard file atomically; a re-run",
"**skips** meshes already cached, so it resumes after a Colab disconnect.",
))
cells.append(code(
"from pat import teacher as TCH",
"import glob, time",
"from tqdm.auto import tqdm",
"Pall, Nall = cache['P'], cache['N']",
"TOT = min(TEACHER_SUBSET, Pall.shape[0])",
"os.makedirs(TEACHER_DIR, exist_ok=True)",
"have = len(glob.glob(os.path.join(TEACHER_DIR, 'shard_*', 'mesh_*.pt')))",
"print(f'teacher: {have} shards already cached; target {TOT}')",
"ok = hard = ran = 0; t0 = time.time()",
"for gid in tqdm(range(TOT), desc='teacher'):",
"    status, M, md = TCH.fit_and_cache(",
"        Pall[gid].numpy(), Nall[gid].numpy(), gid, TEACHER_DIR, force=FORCE_TEACHER,",
"        md_target=MD_TARGET, res=RES, n_init=N_INIT, steps_warm=STEPS_WARM, steps_refit=STEPS_REFIT,",
"        time_budget_s=TIME_BUDGET, min_keep=MIN_KEEP, device='cuda')",
"    ran += status != 'cached'; ok += status == 'ok'; hard += status == 'hard'",
"    if ran and ran % 50 == 0:",
"        print(f'  {gid+1}/{TOT} | ran {ran} this session | ok {ok} hard {hard} | {(time.time()-t0)/max(ran,1):.1f}s/mesh', flush=True)",
"n_shards = len(glob.glob(os.path.join(TEACHER_DIR, 'shard_*', 'mesh_*.pt')))",
"print(f'teacher cache now holds {n_shards} meshes ({ok} ok / {hard} hard this session)')",
))

cells.append(md("## 4b · Teacher QA — splat-count & MD/IoU distribution, hard-mesh fraction"))
cells.append(code(
"import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt",
"from pat import student as ST",
"rows=[{'M':a['M'],'md':a['md'],'iou':a['iou'],'status':a['status']} for a in ST.iter_shards(TEACHER_DIR, status_ok_only=False)]",
"Ms=[r['M'] for r in rows]; mds=[r['md'] for r in rows]; ious=[r['iou'] for r in rows]",
"hard_frac=float(np.mean([r['status']!='ok' for r in rows])) if rows else 0.0",
"json.dump(rows, open(os.path.join(TEACHER_DIR,'manifest.json'),'w'))",
"fig,ax=plt.subplots(1,3,figsize=(13,3.4))",
"ax[0].hist(Ms,bins=30,color='C0'); ax[0].set_title('# splats per mesh'); ax[0].set_xlabel('M')",
"ax[1].hist(mds,bins=30,color='C2'); ax[1].axvline(MD_TARGET,ls=':',c='r'); ax[1].set_title('MD (vol distance)'); ax[1].set_xlabel('MD')",
"ax[2].hist(ious,bins=30,color='C3'); ax[2].set_title('IoU'); ax[2].set_xlabel('IoU')",
"fig.suptitle(f'teacher QA — {len(rows)} meshes | median M={int(np.median(Ms)) if Ms else 0} | hard(MD>{MD_TARGET})={hard_frac:.0%}')",
"fig.tight_layout(); fig.savefig(os.path.join(TEACHER_DIR,'stats.png'),dpi=130)",
"from IPython.display import Image, display; display(Image(os.path.join(TEACHER_DIR,'stats.png')))",
"print(f'median splats/mesh: {int(np.median(Ms)) if Ms else 0} | hard fraction (could not reach MD_TARGET): {hard_frac:.1%}')",
"if hard_frac>0.5: print('NOTE: most meshes did not reach MD_TARGET — raise TIME_BUDGET/N_INIT or relax MD_TARGET (see memory note on the achievable floor).')",
))

# ----------------------------------------------------------------------------- 5. GroupNet
cells.append(md(
"## 5 · Stage B1 — GroupNet: learn *how many points to group per supertori point*",
"",
"Per-point (position-aware) seed-ness + metric embedding on the cached teacher `owner` labels (status=ok",
"meshes only — hard meshes are gated out, so a few unfittable shapes can never stall training).  At",
"inference, NMS over the seeds gives K groups and every point joins its nearest seed (spatially coherent).",
))
cells.append(code(
"GN_PATH=os.path.join(STUDENT_DIR,'groupnet.pt')",
"if os.path.exists(GN_PATH) and not FORCE_STUDENT:",
"    ck=torch.load(GN_PATH, weights_only=False)",
"    gnet=ST.GroupNet(d_embed=D_EMBED, n_layers=N_LAYERS, d_g=ck.get('d_g',32)).cuda()",
"    gnet.load_state_dict(ck['state']); print('reused groupnet.pt')",
"else:",
"    gnet, gh = ST.train_groupnet(TEACHER_DIR, epochs=GROUP_EPOCHS, k=K_NBR, device='cuda',",
"        net=ST.GroupNet(d_embed=D_EMBED, n_layers=N_LAYERS, d_g=32).cuda())",
"    torch.save({'state':gnet.state_dict(),'d_g':32}, GN_PATH)",
"    json.dump(gh, open(os.path.join(STUDENT_DIR,'groupnet_log.json'),'w'), indent=1)",
"    print('GroupNet trained; loss', round(gh[0]['loss'],3), '->', round(gh[-1]['loss'],3))",
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
"    fnet, fh = ST.train_fitnet(TEACHER_DIR, epochs=FIT_EPOCHS, device='cuda',",
"        net=ST.FitNet(d_embed=D_EMBED, n_layers=max(6,N_LAYERS-2)).cuda())",
"    torch.save({'state':fnet.state_dict()}, FN_PATH)",
"    json.dump(fh, open(os.path.join(STUDENT_DIR,'fitnet_log.json'),'w'), indent=1)",
"    print('FitNet trained; loss', round(fh[0]['loss'],3), '->', round(fh[-1]['loss'],3))",
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
