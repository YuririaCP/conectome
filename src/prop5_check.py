"""Numerical check of Proposition 5 (GSR asymptotics) and of the limiting sensitivity ratio rho_inf."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from fcdata import gsr, laplacian, iu, N, SC_all
C = SC_all.mean(0); C=(C+C.T)/2; np.fill_diagonal(C,0); C=C/C.sum(1).mean()
L = laplacian(C); Lp = np.linalg.pinv(L); lam=np.linalg.eigvalsh(L)
one=np.ones(N); l=np.diag(Lp); Om = l[:,None]+l[None,:]-2*Lp; h=Lp@l
def corr(S): s=np.sqrt(np.diag(S)); return S/np.outer(s,s)
F=lambda th, s2=0: corr(np.linalg.inv(th*np.eye(N)+L)+s2*np.outer(one,one))
G0 = corr(Lp)
Q1 = -(Lp@Lp - 0.5*(np.outer(one,h)+np.outer(h,one)))
# derivative of corr at Lp in direction Q1
def dcorr(S,dS):
    s=np.sqrt(np.diag(S)); ds=np.diag(dS)/(2*s)
    return dS/np.outer(s,s) - S*(np.outer(ds,s)+np.outer(s,ds))/np.outer(s,s)**2
Gp = dcorr(Lp,Q1)
print("lambda2", lam[1], "N", N)
for th in [1e-1,1e-2,1e-3,1e-4]:
    Fa=F(th); errA=np.abs((1-Fa)[iu]-(N*th/2)*Om[iu]).max()
    Ga=gsr(Fa); errB=np.abs((Ga-G0-th*Gp)[iu]).max()
    h_=0.01
    dF=(F(th*np.exp(h_))-F(th*np.exp(-h_)))[iu]/(2*h_); dG=(gsr(F(th*np.exp(h_)))-gsr(F(th*np.exp(-h_))))[iu]/(2*h_)
    print(f"th={th:g}  errA/th^2={errA/th**2:.3g}  errB/th^2={errB/th**2:.3g}  ratio={np.linalg.norm(dG)/np.linalg.norm(dF):.4f}")
print("predicted limit ratio", 2*np.linalg.norm(Gp[iu])/(N*np.linalg.norm(Om[iu])))
# artifact equivalence
for th in [1e-2,1e-3]:
    s2=1.0/(N*th)  # doubles global variance
    the=1/(1/th+N*s2)
    print(f"artifact th={th}: |F(th,s2)-F(th_eff)|max={np.abs((F(th,s2)-F(the))[iu]).max():.2e} vs |F(th)-F(th_eff)|={np.abs((F(th)-F(the))[iu]).max():.2e}; GSR diff={np.abs((gsr(F(th,s2))-gsr(F(th)))[iu]).max():.2e}")

# --- rho_inf across graphs ---
def rho(C):
    N=len(C); iu=np.triu_indices(N,1); L=laplacian(C); Lp=np.linalg.pinv(L); one=np.ones(N)
    l=np.diag(Lp); Om=l[:,None]+l[None,:]-2*Lp; h=Lp@l
    dS=-(Lp@Lp-0.5*(np.outer(one,h)+np.outer(h,one)))
    s=np.sqrt(l); ds=np.diag(dS)/(2*s)
    Gp=dS/np.outer(s,s)-Lp*(np.outer(ds,s)+np.outer(s,ds))/np.outer(s,s)**2
    return 2*np.linalg.norm(Gp[iu])/(N*np.linalg.norm(Om[iu]))
C=SC_all.mean(0); C=(C+C.T)/2; np.fill_diagonal(C,0)
print("group SC", rho(C))
# EC disconnected nodes? use largest component check
print("indiv SC (5 subj)", [round(rho((lambda X:(X+X.T)/2)(SC_all[k])),4) for k in range(5)])
rng=np.random.default_rng(0)
for N in [25,50,100,200,400]:
    A=rng.random((N,N))<0.2; W=np.triu(A*rng.random((N,N)),1); W=W+W.T
    print("ER N",N, round(rho(W),5), "N*rho", round(N*rho(W),3))
