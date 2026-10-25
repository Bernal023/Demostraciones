import numpy as np
from sklearn.linear_model import Lasso, lars_path
rng=np.random.default_rng(0)
# ---- Ridge MSE formula
n,p,lam,sig=30,4,2.5,1.3
X=rng.normal(size=(n,p)); beta=rng.normal(size=p)
A=X.T@X+lam*np.eye(p); Ai=np.linalg.inv(A)
l=np.linalg.eigvalsh(X.T@X)
cov=sig**2*Ai@X.T@X@Ai
bias=-lam*Ai@beta
mse_formula=sig**2*np.sum(l/(l+lam)**2)+lam**2*beta@Ai@Ai@beta
print("ridge trace(cov)+|bias|^2 =",np.trace(cov)+bias@bias,"formula",mse_formula)
# Monte Carlo
est=[];
for _ in range(100000//10):
    y=X@beta+sig*rng.normal(size=n); b=Ai@X.T@y; est.append(((b-beta)**2).sum())
print("MC",np.mean(est))
# ---- Lasso orthogonal case
n,p=40,3
Q,_=np.linalg.qr(rng.normal(size=(n,p))); X=Q*np.sqrt(n)   # X'X = n I, columns sum x^2 = n
X=X-X.mean(0); 
# re-orthogonalize with centering: build via QR on centered
Z=rng.normal(size=(n,p)); Z-=Z.mean(0); Q,_=np.linalg.qr(Z); X=Q*np.sqrt(n)
y=X@np.array([1.5,-0.4,0.1])+0.3*rng.normal(size=n); y-=y.mean()
lam=1.2
b=X.T@y/n; soft=np.sign(b)*np.maximum(np.abs(b)-lam/(2*n),0)
sk=Lasso(alpha=lam/(2*n),fit_intercept=False,tol=1e-12,max_iter=100000).fit(X,y).coef_
print("orth soft",soft,"sk",sk)
# ---- active-set closed form + LARS formulas (general X)
n,p=60,5
X=rng.normal(size=(n,p)); X-=X.mean(0); X/=X.std(0)   # sum x^2 = n
y=X@np.array([3,-2,0,1,0])+rng.normal(size=n); y-=y.mean()
lam=60.0
sk=Lasso(alpha=lam/(2*n),fit_intercept=False,tol=1e-12,max_iter=100000).fit(X,y).coef_
A=np.where(sk!=0)[0]; sA=np.sign(sk[A])
bA=np.linalg.solve(X[:,A].T@X[:,A],X[:,A].T@y-lam/2*sA)
print("active closed form",bA,"sk",sk[A])
# LARS-lasso by my formulas
def lars_lasso(X,y):
    n,p=X.shape; beta=np.zeros(p); r=y.copy(); c=X.T@r; C=np.abs(c).max()
    act=[int(np.argmax(np.abs(c)))]; knots=[2*C]
    while True:
        A=np.array(act); sA=np.sign(c[A]); d=np.zeros(p)
        d[A]=np.linalg.solve(X[:,A].T@X[:,A],sA); u=X@d; a=X.T@u
        cand=[]
        for j in range(p):
            if j in act: continue
            for g in ((C-c[j])/(1-a[j]),(C+c[j])/(1+a[j])):
                if g>1e-12: cand.append((g,j))
        gt,jn=min(cand) if cand else (C,None)
        gt=min(gt,C)
        drop=None
        for k,j in enumerate(act):
            g0=-beta[j]/d[j]
            if g0>1e-12 and g0<gt: gt,drop=g0,j
        beta=beta+gt*d; r=r-gt*u; c=X.T@r; C=C-gt; knots.append(2*C)
        if drop is not None: act.remove(drop)
        elif jn is not None: act.append(jn)
        if C<1e-9 or (jn is None and drop is None): break
    return knots,beta
kn,b=lars_lasso(X,y)
al,_,co=lars_path(X,y,method='lasso')
print("my knots (lambda=2C):",np.round(kn,4))
print("sk knots*2n        :",np.round(al*2*n,4))
print("final beta",np.round(b,4),"OLS",np.round(np.linalg.lstsq(X,y,rcond=None)[0],4))
