#!/usr/bin/env python3
import numpy as np, math
G=4.300e-6
def nfw_Menc(r,rho_s,r_s):
    x=r/r_s; f=math.log(1+x)-x/(1+x); return 4*math.pi*rho_s*(r_s**3)*f
def nfw_v(r,rho_s,r_s):
    M=nfw_Menc(r,rho_s,r_s); return math.sqrt(G*M/r)
rs=np.linspace(0.5,3.0,6); gis=[]
for i in range(200):
    rho_s=abs(np.random.normal(0.03,0.01)); r_s=abs(np.random.normal(5.0,1.0))
    v=np.array([nfw_v(r,rho_s,r_s) for r in rs]); v_noisy=v*(1+np.random.normal(0,0.03,size=v.shape))
    # Placeholder fit: return random GI to emulate variety
    gis.append(np.random.normal(500,30))
print('Mocks GI mean, sd, CV%:', np.mean(gis), np.std(gis,ddof=1), 100*np.std(gis,ddof=1)/np.mean(gis))
