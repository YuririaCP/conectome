# The global-mode confound — code and manuscript

Code, figures and LaTeX source for *The global-mode confound: what fitted whole-brain models can and
cannot tell about the distance to a bifurcation* (Y. Cortés Poza, IIMAS-UNAM; submitted to *Chaos*).
The manuscript is in `paper/` (`main.tex`, compiled `main.pdf`).

## Data (not included)
The data are not redistributed here, following the HCP data-use terms.
* **Time series (7 HCP subjects, AAL2):** the `hcp` dataset shipped with
  [neurolib](https://github.com/neurolib-dev/neurolib) (`neurolib/data/datasets/hcp/`). Copy the
  subject folders to `data/raw/neurolib_hcp/<subject>/`.
* **Connectomes (200 HCP subjects, Schaefer-100):** Domhof, Jung, Eickhoff & Popovych, EBRAINS,
  doi:[10.25493/81EV-ZVT](https://doi.org/10.25493/81EV-ZVT) (EBRAINS account and HCP terms required).
  Extract `100-Schaefer17Networks.zip` into `data/raw/domhof_hcp/` and run
  `python src/build_domhof.py`.

## Reproducing the paper
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/fit_group.py            # Sec. 4.2-4.3 (then fit_heterogeneous, fit_ec, fit_ec_spectra[--mou])
.venv/bin/python src/surrogate_fcd.py         # Fig. 3, pooled FCD
.venv/bin/python src/surrogate_stats.py       # Sec. 4.2, per-subject tests
.venv/bin/python src/fc_only.py               # Sec. 4.4 (then improve_fc_fit.py, additive.py)
.venv/bin/python src/prop5_check.py           # numerical check of Proposition 5
.venv/bin/python src/paper_figs.py            # Figs. 1-2
.venv/bin/python src/paper_figs_empirical.py  # Figs. 3-5 (needs results/ from the scripts above)
.venv/bin/python src/theorem_general_check.py # numerical check of Theorem (additive coupling)
.venv/bin/python src/nonlinear_confound.py    # Sec. 4.6, full nonlinear model (~10 min)
.venv/bin/python src/temporal_demo.py         # Sec. 4.7, delta from the global signal
.venv/bin/python src/paper_figs_extra.py      # Figs. 6-7
```
Fitting procedures, bounds and validation splits are described in Appendix A of the paper.
The notebooks in `notebooks/` document the exploratory phases (in Spanish).

---

# Conectoma — modelo de Stuart–Landau de cerebro completo

Osciladores de Stuart–Landau (forma normal de Hopf) acoplados por el conectoma estructural,
ajustados a fMRI en reposo. Objetivo a mediano plazo: medir la **distancia a la criticalidad**
por sujeto y relacionarla con resiliencia, desarrollo o enfermedad (ABIDE, ADNI).

## Estructura
```
src/hopf.py          modelo: Jacobiano, Lyapunov, FC filtrada analítica, simulación Euler–Maruyama
src/metrics.py       FC, FCD, metaestabilidad, frecuencias pico
src/data.py          carga de los 7 sujetos HCP de neurolib (AAL2, 94 regiones, TR 0.72 s)
src/check_linear.py  Fase 0: teoría lineal vs simulación no lineal
src/fit_group.py     Fase 1: ajuste de grupo a HCP
src/fit_heterogeneous.py  Fase 2: a_j por región (L-BFGS analítico, validación cruzada)
src/surrogate_fcd.py Fase 2: diagnóstico de FCD con surrogados de fase
src/fit_ec.py        Fase 3: conectividad efectiva dirigida (validación cruzada, techo de ruido)
src/spectra_check.py Fase 3: controles espectrales de la FCD
src/fit_ec_spectra.py Fase 4: EC + espectros regionales (--mou: nodos Ornstein-Uhlenbeck)
src/fc_only.py       Fase 5: índice adimensional kappa con FC sola (Domhof, 200 sujetos HCP)
src/fcdata.py        utilidades compartidas para datos Domhof (FC, GSR, ICC)
src/improve_fc_fit.py Fase 6: SC transformada y EC simétrica; kappa individual
src/theta_sensitivity.py Fase 6: por qué GSR borra la información de criticalidad
src/additive.py      Fase 7: acoplamiento aditivo vs difusivo (Perron ~ señal global)
notebooks/01..07_*.ipynb   resultados y lectura por fase
figures/, results/   salidas
data/raw/neurolib_hcp/   datos (github.com/neurolib-dev/neurolib)
data/raw/domhof_hcp/     Domhof et al. 2021, EBRAINS DOI 10.25493/81EV-ZVT (CC-BY; aplican términos HCP)
```

## Entorno
El entorno virtual vive fuera de Dropbox para no sincronizar miles de archivos:
```bash
python3 -m venv ~/.venvs/conectoma && ~/.venvs/conectoma/bin/pip install -r requirements.txt jupyter
```
En Jupyter/VS Code, elegir el kernel **Python (conectoma)**.

## Reproducir
```bash
~/.venvs/conectoma/bin/python src/check_linear.py   # ~20 s
~/.venvs/conectoma/bin/python src/fit_group.py      # ~1 min
```
