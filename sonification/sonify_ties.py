import numpy as np
from scipy.special import gammaln
def ties(n):
    """All tie points with p*>1/2, with left/right slopes of E. Returns dict of arrays."""
    k=np.arange(n+1); lC=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
    I,J=np.array([(i,j) for i in range(n+1) for j in range(i+1,n+1) if i+j>n]).T
    lr=(lC[I]-lC[J])/(J-I); p=1/(1+np.exp(-lr)); q=1-p
    lf=lC[None,:]+k[None,:]*np.log(p)[:,None]+(n-k)[None,:]*np.log(q)[:,None]
    f=np.exp(lf); f/=f.sum(1,keepdims=True)
    # force exact tie, break left-limit order: j just below i
    f[np.arange(len(I)),J]=f[np.arange(len(I)),I]
    key=f.copy(); key[np.arange(len(I)),J]*= (1-1e-12)
    order=np.argsort(key,axis=1,kind='stable'); rank=np.empty_like(order)
    rows=np.arange(len(I))[:,None]; rank[rows,order]=np.arange(n+1)[None,:]
    df=f*(k[None,:]/p[:,None]-(n-k)[None,:]/q[:,None])
    Sl=(rank*df).sum(1)
    D=(J-I)*f[np.arange(len(I)),I]/(p*q)       # slope jump at the tie
    Sr=Sl+D
    E=(rank*f).sum(1)
    return dict(i=I,j=J,p=p,fi=f[np.arange(len(I)),I],Sl=Sl,Sr=Sr,D=D,cusp=(Sl<0)&(Sr>0),E=E)
if __name__=="__main__":
    import time
    for n in [20,100,200,300,600]:
        t=time.time(); T=ties(n); print(n,len(T['p']),T['cusp'].sum(),round(time.time()-t,2),'s')
