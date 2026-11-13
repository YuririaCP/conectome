"""Build data/raw/domhof_hcp/schaefer100.npz from the EBRAINS release of Domhof et al. (2021).

Download "Parcellation-based structural and resting-state functional brain connectomes of a
healthy cohort" (EBRAINS, doi:10.25493/81EV-ZVT; requires an EBRAINS account and acceptance of the
HCP data-use terms), and extract 100-Schaefer17Networks.zip into data/raw/domhof_hcp/.
Output: fc (200 subjects, 4 sessions, 100, 100), sc (200, 100, 100) streamline counts, sessions.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "raw" / "domhof_hcp" / "100-Schaefer17Networks"
SESSIONS = ["REST1-LR", "REST1-RL", "REST2-LR", "REST2-RL"]

subs = sorted(p.name for p in (D / "1StructuralConnectivity").iterdir() if p.is_dir())
sc = np.stack([np.loadtxt(D / "1StructuralConnectivity" / s / "Counts.csv") for s in subs])
fc = np.stack([[np.loadtxt(D / "2FunctionalConnectivity" / s / f"EmpCorrFC_{ses}.csv") for ses in SESSIONS] for s in subs])
OUT = D.parent / ("schaefer100_rebuilt.npz" if __import__("sys").argv[1:] == ["--check"] else "schaefer100.npz")
np.savez(OUT, fc=fc, sc=sc, sessions=np.array(SESSIONS))
print(f"{len(subs)} subjects; fc {fc.shape}, sc {sc.shape}")
