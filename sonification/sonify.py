import numpy as np, time, os
from scipy.io import wavfile
from scipy.special import gammaln
from sonify_ties import ties
SR=44100; OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'out')+'/'
os.makedirs(OUT,exist_ok=True)
def save(name,x):
    x=x/ (np.abs(x).max()+1e-12)*0.89
    wavfile.write(OUT+name,SR,(x*32767).astype(np.int16)); print('wrote',name,round(len(x)/SR,1),'s')
def blip(buf,t0,freq,amp,dur):
    i0=int(t0*SR); m=int(dur*SR); i1=min(i0+m,len(buf))
    if i0>=len(buf): return
    tt=np.arange(i1-i0)/SR; env=np.sin(np.pi*np.arange(i1-i0)/m)**2
    buf[i0:i1]+=amp*env*np.sin(2*np.pi*freq*tt)

# ---------- 1. pair space as a spectrogram (pooled n=200..400) ----------
t=time.time()
DUR=40; buf=np.zeros(DUR*SR+SR)
P0,P1=0.5,0.66
for n in range(200,401,4):
    T=ties(n); c=T['cusp']; p=T['p'][c]; w=(T['j']-T['i'])[c]
    z=w/(2*np.sqrt(n*p*(1-p)))
    for pp,zz in zip(p,z):
        f=110*2**(zz/3.6*4.5)                 # z in [0,3.6] -> 110 Hz .. ~2.5 kHz (log)
        amp=0.25 if zz<2 else 0.9             # wide cusps ring out
        blip(buf,(pp-P0)/(P1-P0)*DUR,f,amp,0.09 if zz<2 else 0.35)
save('1_pair_space_spectrogram.wav',buf); print(time.time()-t)

# ---------- 2. sweep p, ties as clicks, cusps as pings (n=60) ----------
n=60; T=ties(n); DUR=36; P0,P1=0.5,0.74
buf=np.zeros(DUR*SR+SR); rng=np.random.default_rng(1)
for pp,D,cu,i,j in zip(T['p'],T['D'],T['cusp'],T['i'],T['j']):
    if not (P0<pp<P1): continue
    t0=(pp-P0)/(P1-P0)*DUR; i0=int(t0*SR)
    a=np.clip((np.log10(D)+10)/11,0.03,1)
    m=int(0.004*SR); buf[i0:i0+m]+=a*rng.standard_normal(m)*np.exp(-np.arange(m)/(0.0008*SR))
    if cu: blip(buf,t0,880,0.6,0.25); blip(buf,t0,1320,0.25,0.25)
save('2_tie_clicks_n60.wav',buf)

# ---------- 3. E'(p) as a waveform, one cycle = one sweep p in [1/2,1], n rising ----------
def Eprime(n,p):
    k=np.arange(n+1); lC=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1); q=1-p
    f=np.exp(lC[None,:]+k*np.log(p)[:,None]+(n-k)*np.log(q)[:,None])
    df=f*(k/p[:,None]-(n-k)/q[:,None]); r=np.argsort(np.argsort(f,1),1)
    return (r*df).sum(1)
F0=55; L=SR//F0; pg=0.5+0.4999*(np.arange(L)+0.5)/L
segs=[]
for n in range(10,401,3):
    w=Eprime(n,pg); w=w-w.mean(); w/=np.abs(w).max()
    segs.append(np.tile(w,int(0.2*F0)))       # hold each n for 0.2 s
x=np.concatenate(segs)
save('3_Eprime_waveform_n10_to_400.wav',x)

# ---------- 4. one chord per n: cusp x=n(p-1/2) -> harmonic 1+2x of 110 Hz ----------
HOLD=0.11; xs=[]
for n in range(4,301):
    T=ties(n); c=T['cusp']; xx=n*(T['p'][c]-0.5)
    m=int(HOLD*SR); tt=(len(xs)*m+np.arange(m))/SR; seg=np.zeros(m)
    for v in xx:
        f=110*(1+2*v)
        if f>7000: continue
        seg+=np.sin(2*np.pi*f*tt)/np.sqrt(1+2*v)
    env=np.minimum(1,np.minimum(np.arange(m),m-np.arange(m))/(0.01*SR))
    xs.append(seg*env)
save('4_cusp_chords_n4_to_300.wav',np.concatenate(xs))

# ---------- 5. binomial as timbre: plain vs ordered (OBD) spectrum ----------
def additive(n,ps,sort):
    N=len(ps); k=np.arange(n+1); lC=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
    A=np.exp(lC[None,:]+k*np.log(ps)[:,None]+(n-k)*np.log(1-ps)[:,None])
    if sort: A=np.sort(A,1)
    fr=110*(k+1); tt=np.arange(N)/SR
    return (A*np.sin(2*np.pi*fr[None,:]*tt[:,None])).sum(1)
n=20; N=12*SR
ps=0.03+0.94*(np.arange(N)/N)
a=additive(n,ps,False); b=additive(n,ps,True)
gap=np.zeros(int(0.8*SR))
save('5_timbre_binomial_then_OBD_n20.wav',np.concatenate([a/np.abs(a).max(),gap,b/np.abs(b).max()]))
