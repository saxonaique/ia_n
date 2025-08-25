#!/usr/bin/env python3
import json, sys, math
import numpy as np, pandas as pd
from scipy.optimize import minimize
G=4.300e-6
def M_star_analytic(r,L0,Rd,ML):
    if r<=0: return 0.0
    a=Rd; x=r/a; term=2.0-math.exp(-x)*(x*x+2.0*x+2.0); return ML*L0*term
def v_total(r,G_I,L0,Rd,ML):
    if r<=0: return 0.0
    Mst=M_star_analytic(r,L0,Rd,ML)
    return math.sqrt(G*Mst/r)*math.sqrt(1.0+1.0/G_I)
def chi_sq_vec(params,r_obs,v_obs,s_obs,L0,Rd):
    G_I,M_L=params; v_pred=np.array([v_total(r,G_I,L0,Rd,M_L) for r in r_obs]); return np.sum(((v_obs-v_pred)/s_obs)**2)
def fit_galaxy(g):
    r=np.array(g['r']); v=np.array(g['v']); s=np.array(g['s']); L0=float(g['L0']); Rd=float(g['Rd'])
    GI_grid=np.linspace(50,1200,120); ML_grid=np.linspace(0.05,2.0,80); best=(1e12,None,None)
    for GI in GI_grid:
        for ML in ML_grid:
            chi=chi_sq_vec((GI,ML),r,v,s,L0,Rd)
            if chi<best[0]: best=(chi,GI,ML)
    res=minimize(lambda p: chi_sq_vec(p,r,v,s,L0,Rd), x0=[best[1],best[2]], bounds=[(50,2000),(0.05,2.0)])
    GIfit,MLfit=res.x; chi2=res.fun; dof=max(1,len(r)-2); chi2_red=chi2/dof
    # bootstrap GI error
    nboot=200; boots=[]; rng=np.random.default_rng(12345)
    for _ in range(nboot):
        vboot=v + rng.normal(0,s)
        rb=minimize(lambda p: chi_sq_vec(p,r,vboot,s,L0,Rd), x0=[GIfit,MLfit], bounds=[(50,2000),(0.05,2.0)])
        boots.append(rb.x[0])
    GI_err=np.std(boots,ddof=1)
    return {"nombre":g['nombre'],"G_I":float(GIfit),"G_I_err":float(GI_err),"M_L":float(MLfit),"chi2_red":float(chi2_red)}
def main():
    infile=sys.argv[1] if len(sys.argv)>1 else 'data_galaxias.json'; outfile=sys.argv[2] if len(sys.argv)>2 else 'results.csv'
    with open(infile) as f: data=json.load(f)
    results=[] 
    for g in data:
        res=fit_galaxy(g); print('Hecho:',res['nombre'],res['G_I'],res['G_I_err'],res['M_L'],res['chi2_red']); results.append(res)
    pd.DataFrame(results).to_csv(outfile,index=False); print('Guardado',outfile)
if __name__=='__main__': main()
