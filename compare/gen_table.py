"""Emit a LaTeX quantitative table from a benchmark run (results/local/metrics_val.json).  Includes the
hole/validity metrics F-score alone hides: boundary (hole-rim) edges, self-intersections, components,
watertight fraction -- so a holey or self-intersecting 'high-F' mesh can't masquerade as clean."""
import sys, os, json; sys.path.insert(0, "compare")
import numpy as np
from core import agg_ci95

SRC = os.environ.get("SRC", "results/local/metrics_val.json")
rows = json.load(open(SRC))
closed = [r for r in rows if r.get("kind") == "closed"]; opn = [r for r in rows if r.get("kind") == "open"]
n_c, n_o = len(closed), len(opn)
METHODS = ["SPSR", "BPA", "APSS", "RIMLS", "tori", "ours"]
LABEL = {"SPSR": "SPSR~\\cite{kazhdan2013}", "BPA": "BPA~\\cite{bernardini1999}", "APSS": "APSS~\\cite{apss}",
         "RIMLS": "RIMLS~\\cite{rimls}", "tori": "tori~\\cite{pat}", "ours": "\\textbf{ours}"}


def mean(rs, m, k):
    vals = [r["methods"][m][k] for r in rs if m in r.get("methods", {})
            and isinstance(r["methods"][m].get(k), (int, float)) and r["methods"][m][k] == r["methods"][m][k]
            and r["methods"][m][k] >= 0]
    return float(np.mean(vals)) if vals else float("nan")


def wtfrac(rs, m):
    vals = [1 if r["methods"][m].get("watertight_out") else 0 for r in rs if m in r.get("methods", {})]
    return 100 * float(np.mean(vals)) if vals else 0.0


def big(x):
    return "--" if x != x else (f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")


print(r"\begin{table*}[t]\centering\footnotesize")
print(r"\begin{tabular}{lcccccccc}")
print(r"\toprule")
print(r" & \multicolumn{2}{c}{\textbf{F-score} @ $\tau{=}0.05\;\uparrow$} & & "
      r"\multicolumn{4}{c}{\textbf{mesh validity (open shells)}} & \\")
print(r"\cmidrule(lr){2-3}\cmidrule(lr){5-8}")
print(r"\textbf{method} & closed & open & \textbf{Cham}$\downarrow$ & "
      r"\textbf{holes}$\downarrow$ & \textbf{self-X}$\downarrow$ & \textbf{\#comp}$\downarrow$ & "
      r"\textbf{w.t.\%}$\uparrow$ & \textbf{s}$\downarrow$ \\")
print(r"\midrule")
for m in METHODS:
    fc, fch, _ = agg_ci95(closed, m, "fscore"); fo, foh, _ = agg_ci95(opn, m, "fscore")
    row = (f"{LABEL[m]} & ${fc:.0f}$ & ${fo:.1f}\\!\\pm\\!{foh:.1f}$ & ${mean(rows,m,'chamfer'):.1f}$ & "
           f"${big(mean(opn,m,'boundary_edges'))}$ & ${big(mean(opn,m,'self_x'))}$ & "
           f"${mean(opn,m,'parts'):.0f}$ & ${wtfrac(rows,m):.0f}$ & ${mean(rows,m,'time'):.1f}$ \\\\")
    if m == "ours":
        print(r"\midrule")
    print(row)
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{\textbf{Quantitative comparison on " + f"{len(rows)}" + r" ModelNet40 test shapes} "
      r"(" + f"{n_c}" + r" closed, " + f"{n_o}" + r" open by the gate's thin-fraction).  F-score @ $\tau{=}0.05$ "
      r"(\%, open shown mean$\pm95\%$CI), Chamfer-to-GT ($\times100$), and the mesh-validity metrics F-score "
      r"\emph{hides}, averaged on the open shells: \textbf{holes} (boundary / hole-rim edges), "
      r"\textbf{self-X} (self-intersecting faces), connected \textbf{components}, watertight fraction, runtime.  "
      r"\emph{No method is clean.}  APSS and RIMLS reach a \emph{single} component yet are riddled with $\sim\!9$k "
      r"hole edges and $\sim\!55$k self-intersections; BPA's high F-score is a $1.5$k-component fragment soup; "
      r"SPSR over-fills open shells (low open-F).  Ours fragments ($" + f"{mean(opn,'ours','parts'):.0f}" + r"$ "
      r"components) but is the only learned/primitive method with \emph{zero} self-intersections and an order of "
      r"magnitude fewer hole edges than "
      r"APSS/RIMLS, at the highest open-shell F-score among non-soup methods.  Same $4096$ oriented points; ours "
      r"at $R{=}128$.}")
print(r"\label{tab:quant}")
print(r"\end{table*}")
