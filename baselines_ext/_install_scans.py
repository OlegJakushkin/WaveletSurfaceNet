"""Unpack the real-scans workflow result: write the authored scripts + paper LaTeX to disk."""
import json, os, sys
src = json.load(open(sys.argv[1]))
r = src.get("result", src)
p, g = r["pipeline"], r["rigor"]
os.makedirs("baselines_ext/scans", exist_ok=True)
open("compare/scan_recon.py", "w", newline="\n").write(p["scan_recon_py"])
open("compare/fig_scans.py", "w", newline="\n").write(p["fig_scans_py"])
open("compare/gate_sweep2.py", "w", newline="\n").write(g["gate_sweep2_py"])
open("baselines_ext/scans/paper_scans.tex", "w", newline="\n").write(p.get("paper_latex", ""))
open("baselines_ext/scans/paper_gate.tex", "w", newline="\n").write(g.get("paper_latex", ""))
print("wrote compare/{scan_recon,fig_scans,gate_sweep2}.py + 2 paper_*.tex")
for s in r["scans"]:
    print(f"  scan {s.get('kind','?')[:40]:40s} downloaded={s.get('downloaded')} real={s.get('real_scan')} -> {os.path.basename(s.get('path','?'))}")
