import numpy as np
from scipy.stats import t
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def log_nonnegative(score):
    x=np.asarray(score,np.float64)
    if np.any(x<0) or np.isnan(x).any():raise ValueError('invalid nonnegative score')
    with np.errstate(divide='ignore'):return np.log(x)


def metrics(score, is_ood):
    """ID-high, empirical ID acceptance >= .95, accept all ties; no interpolation."""
    score=np.asarray(score,np.float64);is_ood=np.asarray(is_ood,bool)
    # -inf is the exact log of a true zero product, not numerical underflow.
    if np.isnan(score).any() or np.isposinf(score).any():raise ValueError('invalid issued score')
    id_s,ood_s=score[~is_ood],score[is_ood]
    naccept=int(np.ceil(.95*len(id_s)))
    threshold=np.sort(id_s)[len(id_s)-naccept]
    return {'AUROC':100*roc_auc_score(~is_ood,rankdata(score)) if len(ood_s) else np.nan,
            'FPR95':100*np.mean(ood_s>=threshold) if len(ood_s) else np.nan,
            'actual_ID_acceptance':100*np.mean(id_s>=threshold),'threshold':float(threshold),
            'ID_ties_at_threshold':int(np.sum(id_s==threshold)), 'OOD_ties_at_threshold':int(np.sum(ood_s==threshold)),
            'n_ID':len(id_s),'n_OOD':len(ood_s)}


def ci(values):
    x=np.asarray(values,float);n=len(x);mean=float(np.mean(x))
    h=float(t.ppf(.975,n-1)*np.std(x,ddof=1)/np.sqrt(n)) if n>1 else float('nan')
    return {'mean':mean,'lo':mean-h,'hi':mean+h,'n':n,'unit':'support/calibration draw, after averaging matched orders'}


def paired(frame, a, b, column):
    p=frame.pivot(index=['draw','stream','seed'],columns='method',values=column)
    d=(p[a]-p[b]).dropna().groupby(level='draw').mean()
    return ci(d.values)
