"""Independent P/SV wave-amplitude matching; no layer stiffness poles.

Unknowns: four P/SV amplitudes per finite layer and two decaying halfspace
amplitudes. Enforce zero surface traction and displacement/traction continuity.
Exponentials referenced to their decay-side interface to avoid overflow.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import brentq
H=Path(__file__).resolve().parent
a=np.load(H/'comparison.npz');rows=a['model_km']*1000

def matrix(f,c):
    w=2*np.pi*f;k=w/c
    def basis(layer,z):
        h,vp,vs,rho=rows[layer];mu=rho*vs*vs
        cols=[]
        for speed,kind in [(vp,'P'),(vs,'S')]:
            q=np.sqrt(k*k-(w/speed)**2+0j)
            for sign in ([-1,1] if layer<3 else [-1]):
                s=sign*q;factor=np.exp(s*(z-(h if sign==1 else 0)))
                if kind=='P':v=np.array([1j*k,s,2*mu*1j*k*s,2*mu*k*k-rho*w*w],complex)
                else:v=np.array([-s,1j*k,-mu*(s*s+k*k),2*mu*1j*k*s],complex)
                v[2:]/=1e9
                cols.append(v*factor)
        return np.array(cols).T
    m=np.zeros((14,14),complex);m[:2,:4]=basis(0,0)[2:]
    for j in range(3):
        ix=2+4*j;left=4*j;right=4*(j+1)
        m[ix:ix+4,left:left+4]=basis(j,rows[j,0])
        m[ix:ix+4,right:right+(4 if j<2 else 2)]=-basis(j+1,0)
    m/=np.linalg.norm(m,axis=0)
    return m

checks=[]
for i,f in enumerate(a['frequency_hz']):
    q=a['qed'][i]
    for mode in [0,1]:
        c=q[mode];gap=abs(q[1]-q[0]);radius=min(.005,gap*.2)
        lo,hi=c-radius,c+radius
        d=np.linalg.det(matrix(f,lo));phase=d/abs(d)
        fun=lambda x:float((np.linalg.det(matrix(f,x))/phase).real)
        r={'frequency_hz':float(f),'reported_mode':mode,'qed_mps':float(c),'bracket_mps':[float(lo),float(hi)]}
        try:
            root=brentq(fun,lo,hi,xtol=1e-10)
            mat=matrix(f,root);sv=np.linalg.svd(mat,compute_uv=False)
            rel=float(sv[-1]/sv[0]);diff=abs(root-c)
            r.update(root_mps=root,relative_min_singular_value=rel,abs_difference_mps=diff,passed=bool(rel<1e-10 and diff<.001))
        except Exception as ex:r.update(passed=False,error=str(ex))
        checks.append(r)
report={'method':'P/SV wave-amplitude global matching: traction-free surface, displacement/traction continuity, decaying halfspace; no QED/CPS routines',
        'scope':'local verification of 302 candidate roots; not a formal proof of exhaustive modal count',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all_passed':all(x['passed'] for x in checks),'checks':checks}
(H/'matching_checks.json').write_text(json.dumps(report,indent=2))
print('Matching verified',sum(x['passed'] for x in checks),'/',len(checks),'max difference',max(x.get('abs_difference_mps',0) for x in checks))
print('failures',[x for x in checks if not x['passed']])
