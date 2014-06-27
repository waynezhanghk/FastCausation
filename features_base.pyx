
cimport numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.stats import skew, kurtosis
from collections import Counter, defaultdict
import itertools

cdef int BINARY      = 0 #"Binary"
cdef int CATEGORICAL = 1 #"Categorical"
cdef int NUMERICAL   = 2 #"Numerical"

def count_unique(x):
    return len(set(x))

def count_unique_ratio(x):
    return len(set(x))/float(len(x))

def binary(tp):
    #assert type(tp) is str
    return tp == BINARY

def categorical(tp):
    #assert type(tp) is str
    return tp == CATEGORICAL

def numerical(tp):
    #assert type(tp) is str
    return tp == NUMERICAL

def binary_entropy(p, base):
    assert p <= 1 and p >= 0
    h = -(p*np.log(p) + (1-p)*np.log(1-p)) if (p != 0) and (p != 1) else 0
    cdef float factor = 1.0 / np.log(base)
    return h * factor

def standardize(x):
    cdef float factor = np.std(x)
    if factor > 0:
        x = (x - np.mean(x)) * (1.0 / factor)
    else:
        x = (x - np.mean(x))
    return x

def discrete_seq(x, tx, ffactor=3, maxdev=3):
    cdef int maxthr = ffactor * maxdev
    if numerical(tx) and (len(set(x)) > 2 * (maxthr + 1)):
        x = standardize(x)
        xf = x[np.abs(x) < maxdev]
        x = (x - np.mean(xf)) * (ffactor / np.std(xf))
        x = np.floor(x)
        x[x > maxthr] = maxthr
        x[x < - (maxthr + 1)] = - (maxthr + 1)
    return x

def discrete_probability(x, tx, xd=None, ffactor=3, maxdev=3):
    if xd is None:
        xd = discrete_seq(x, tx, ffactor, maxdev)
    return Counter(xd)

def discretized_sequences(x, tx, y, ty, ffactor=3, maxdev=3):
    x = discrete_seq(x, tx, ffactor, maxdev)
    y = discrete_seq(y, ty, ffactor, maxdev)
    return x, y

def normalized_error_probability(x, tx, y, ty, xd=None, yd=None, cx=None, cy=None, ffactor=3, maxdev=3):
    x = discrete_seq(x, tx, ffactor, maxdev) if xd is None else xd
    y = discrete_seq(y, ty, ffactor, maxdev) if yd is None else yd
    if cx is None:
        cx = Counter(x)
    if cy is None:
        cy = Counter(y)
    pxy = defaultdict(lambda: 0)
    for p in itertools.izip(x, y):
        pxy[p] += 1
    pxy = np.array([[pxy[(a,b)] for b in cy] for a in cx], dtype = float)
    pxy = pxy/min(len(x), len(y))
    perr = 1 - np.sum(pxy.max(axis=1))
    max_perr = 1 - np.max(pxy.sum(axis=0))
    pnorm = perr/max_perr
    assert (pnorm <= 1)
    return pnorm

def discrete_entropy(x, tx, c=None, ffactor=3, maxdev=3, bias_factor=0.7):
    if c is None:
        c = discrete_probability(x, tx, None, ffactor, maxdev)
    pk = np.array(c.values(), dtype=float)
    pk = pk/pk.sum()
    vec = pk*np.log(pk)
    S = -np.sum(vec, axis=0)
    return S + bias_factor*(len(pk) - 1)/float(2*len(x))

def discrete_divergence(cx, cy):
    for a, v in cx.most_common():
        if cy[a] == 0: cy[a] = 1

    nx = float(np.sum(cx.values()))
    ny = float(np.sum(cy.values()))
    sum = 0.
    for a, v in cx.most_common():
        px = v/nx
        py = cy[a]/ny
        sum += px*np.log(px/py)
    return sum

def discrete_joint_entropy(x, tx, y, ty, xd=None, yd=None, ffactor=3, maxdev=3):
    x = discrete_seq(x, tx, ffactor, maxdev) if xd is None else xd
    y = discrete_seq(y, ty, ffactor, maxdev) if yd is None else yd
    return discrete_entropy(zip(x,y), CATEGORICAL)

def normalized_discrete_joint_entropy(x, tx, y, ty, e=None, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, tx, y, ty, ffactor, maxdev)
    if e is None:
        e = discrete_entropy(zip(x,y), CATEGORICAL)
    nx = 2*(ffactor*maxdev+1) if numerical(tx) else count_unique(x)
    ny = 2*(ffactor*maxdev+1) if numerical(ty) else count_unique(y)
    if nx*ny>0: e = e/np.log(nx*ny)
    return e

def discrete_conditional_entropy(x, tx, y, ty, exy=None, ey=None):
    if exy is None:
        exy = discrete_joint_entropy(x, tx, y, ty)
    if ey is None:
        ey = discrete_entropy(y, ty)
    return exy - ey

def adjusted_mutual_information(x, tx, y, ty, xd=None, yd=None, ffactor=3, maxdev=3):
    x = discrete_seq(x, tx, ffactor, maxdev) if xd is None else xd
    y = discrete_seq(y, ty, ffactor, maxdev) if yd is None else yd
    return adjusted_mutual_info_score(x, y)

def discrete_mutual_information(x, tx, y, ty, exy=None, ex=None, ey=None):
    if ex is None:
        ex = discrete_entropy(x, tx)
    if ey is None:
        ey = discrete_entropy(y, ty)
    if exy is None:
        exy = discrete_joint_entropy(x, tx, y, ty)
    mxy = max((ex + ey) - exy, 0) # Mutual information is always positive: max() avoid negative values due to numerical errors
    return mxy

def normalized_discrete_entropy(x, tx, e=None, uni=None, ffactor=3, maxdev=3):
    if e is None:
        e = discrete_entropy(x, tx, None, ffactor, maxdev)

    if uni is None:
        uni = count_unique(x)
    n = 2*(ffactor*maxdev+1) if numerical(tx) else uni
    if n>0:
        e = e*(1.0/np.log(n))
    return e

# Continuous information measures
def to_numerical(x, y):
    dx = defaultdict(lambda: np.zeros(2))
    for i, a in enumerate(x):
        dx[a][0] += y[i]
        dx[a][1] += 1
    for a in dx.keys():
        dx[a][0] /= dx[a][1]
    x = np.array([dx[a][0] for a in x], dtype=float)
    return x

def normalize(x, tx):
    if not numerical(tx): # reassign labels according to its frequency
        cx = Counter(x)
        xmap = dict()
        for i, k in enumerate(cx.most_common()):
            xmap[k[0]] = i
        y = np.array([xmap[a] for a in x], dtype = float)
    else:
        y = x

    y = (y - np.mean(y))
    cdef float factor = np.std(y)
    if factor > 0:
        y = y * (1.0 / factor)
    return y

def normalized_entropy_baseline(x, tx, xd=None):
    if len(set(x)) < 2:
        return 0
    x = normalize(x, tx) if xd is None else xd
    xs = np.sort(x)
    delta = xs[1:] - xs[:-1]
    delta = delta[delta != 0]
    hx = np.mean(np.log(delta))
    hx += psi(len(delta))
    hx -= psi(1)
    return hx


def count_value(x, tx, xd=None):
    x = normalize(x, tx) if xd is None else xd
    cx = Counter(x)
    return cx

def normalized_entropy(x, tx, cx=None, m=2):
    if cx is None:
        cx = count_value(x, tx)
    if len(cx) < 2:
        return 0
    xk = np.array(cx.keys(), dtype=float)
    xk.sort()
    delta = (xk[1:] - xk[:-1])/m
    counter = np.array([cx[i] for i in xk], dtype=float)
    hx = np.sum(counter[1:]*np.log(delta/counter[1:]))/len(x)
    hx += (psi(len(delta)) - np.log(len(delta)))
    hx += np.log(len(x))
    hx -= (psi(m) - np.log(m))
    return hx

def igci(x, tx, y, ty, xd=None, yd=None):
    if len(set(x)) < 2:
        hxy = 0
    else:
        x = normalize(x, tx) if xd is None else xd
        y = normalize(y, ty) if yd is None else yd
        len_x = len(x)
        if len_x != len(set(x)):
            dx = defaultdict(lambda: np.zeros(2))
            for a, b in itertools.izip(x, y):
                dx[a][0] += b
                dx[a][1] += 1
            xy = np.array([[a, dx[a][0]] for a in sorted(dx.keys())], dtype=float)
            counter = np.array([dx[a][1] for a in xy[:,0]], dtype=int)
            xy[:, 1] /= counter
        else:
            xy = pd.DataFrame({'x': x, 'y': y}).sort('x').values
            counter = np.ones(len_x)

        delta = np.diff(xy, axis=0)
        selec = delta[:,1] != 0
        delta = delta[selec]
        counter = np.min([counter[1:], counter[:-1]], axis=0)
        counter = counter[selec]
        hxy = np.sum(counter*np.log(delta[:,0]/np.abs(delta[:,1])))/ len_x
    return hxy

def gaussian_divergence(x, tx, cx=None, m=2):
    if cx is None:
        cx = count_value(x, tx)
    xk = np.array(cx.keys(), dtype=float)
    xk.sort()
    delta = np.zeros(len(xk))
    if len(xk) > 1:
        delta[0] = xk[1] - xk[0]
        delta[1:-1] = (xk[m:] - xk[:-m])/m
        delta[-1] = xk[-1] - xk[-2]
    else:
        delta = np.array(np.sqrt(12))
    counter = np.array([cx[i] for i in xk], dtype=float)
    boundaries = np.zeros(len(xk) + 1)
    boundaries[0] = xk[0] - delta[0]*0.5
    boundaries[1:-1] = (xk[:-1] + xk[1:])*0.5
    boundaries[-1] = xk[-1] + delta[-1]*0.5
    refvalues = np.diff(boundaries**3) * (1.0/(6.0*delta))
    hx = np.sum(counter*(refvalues - np.log(delta/counter)))/len(x) + np.log(np.sqrt(2*np.pi))
    hx -= np.log(len(x))
    hx += (psi(m) - np.log(m))
    return hx

def uniform_divergence(x, tx, cx=None, m=2):
    if cx is None:
        cx = count_value(x, tx)
    xk = np.array(cx.keys(), dtype=float)
    xk.sort()
    delta = np.zeros(len(xk))
    if len(xk) > 1:
        delta[0] = xk[1]-xk[0]
        delta[1:-1] = (xk[m:]-xk[:-m])/m
        delta[-1] = xk[-1]-xk[-2]
    else:
        delta = np.array(np.sqrt(12))
    counter = np.array([cx[i] for i in xk], dtype=float)
    delta = delta/np.sum(delta)
    hx = np.sum(counter*np.log(counter/delta))/len(x)
    hx -= np.log(len(x))
    hx += (psi(m) - np.log(m))
    return hx

def normalized_skewness(x, tx, xd=None):
    y = normalize(x, tx) if xd is None else xd
    return skew(y)

def normalized_kurtosis(x, tx, xd=None):
    y = normalize(x, tx) if xd is None else xd
    return kurtosis(y)

def normalized_moment(x, tx, y, ty, n, m, xd, yd):
    x = normalize(x, tx) if xd is None else xd
    y = normalize(y, ty) if yd is None else yd
    return np.mean((x**n)*(y**m))

def moment21(x, tx, y, ty, xd=None, yd=None):
    return normalized_moment(x, tx, y, ty, 2, 1, xd, yd)

def moment31(x, tx, y, ty, xd=None, yd=None):
    return normalized_moment(x, tx, y, ty, 3, 1, xd, yd)

def fit(x, tx, y, ty):
    if (not numerical(tx)) or (not numerical(ty)):
        return 0
    if (len(set(x)) <= 2) or (len(set(y)) <= 2):
        return 0
    x = (x - np.mean(x))/np.std(x)
    y = (y - np.mean(y))/np.std(y)
    xy1 = np.polyfit(x, y, 1)
    xy2 = np.polyfit(x, y, 2)
    return abs(2*xy2[0]) + abs(xy2[1]-xy1[0])

def fit_error(x, tx, y, ty, m=2):
    if categorical(tx) and categorical(ty):
        x = normalize(x, tx)
        y = normalize(y, ty)
    elif categorical(tx) and numerical(ty):
        x = to_numerical(x, y)
    elif numerical(tx) and categorical(ty):
        y = to_numerical(y, x)
    x = (x - np.mean(x))/np.std(x)
    y = (y - np.mean(y))/np.std(y)
    if (count_unique(x) <= m) or (count_unique(y) <= m):
        xy = np.polyfit(x, y, min(count_unique(x), count_unique(y))-1)
    else:
        xy = np.polyfit(x, y, m)
    return np.std(y - np.polyval(xy, x))

def fit_noise_entropy(x, tx, y, ty, xd=None, yd=None, cx=None, ffactor=3, maxdev=3, minc=10):
    x = discrete_seq(x, tx, ffactor, maxdev) if xd is None else xd
    y = discrete_seq(y, ty, ffactor, maxdev) if yd is None else yd
    if cx is None:
        cx = Counter(xd)
    entyx = []
    for a in cx.iterkeys():
        if cx[a] > minc:
            entyx.append(discrete_entropy(y[x==a], CATEGORICAL))
    if len(entyx) == 0: return 0
    n = 2*(ffactor*maxdev+1) if numerical(ty) else len(set(y))
    return np.std(entyx)/np.log(n)

def fit_noise_skewness(x, tx, y, ty, xd=None, cx=None, ffactor=3, maxdev=3, minc=8):
    if xd is None:
        xd = discrete_seq(x, tx, ffactor, maxdev)
    if cx is None:
        cx = Counter(xd)
    skewyx = []
    for a in cx.iterkeys():
        if cx[a] >= minc:
            skewyx.append(normalized_skewness(y[xd==a], ty))
    return np.std(skewyx) if len(skewyx) > 0 else 0

def fit_noise_kurtosis(x, tx, y, ty, xd=None, cx=None, ffactor=3, maxdev=3, minc=8):
    xd = discrete_seq(x, tx, ffactor, maxdev) if xd is None else xd
    if cx is None:
        cx = Counter(xd)
    kurtyx = []
    for a in cx.iterkeys():
        if cx[a] >= minc:
            kurtyx.append(normalized_kurtosis(y[xd==a], ty))
    return np.std(kurtyx) if len(kurtyx) > 0 else 0

def discrete_seq2(x, tx, ffactor=2, maxdev=3):
    cdef int maxthr = ffactor * maxdev
    if numerical(tx) and (len(set(x)) > 2 * (maxthr + 1)):
        x = standardize(x)
        xf = x[np.abs(x) < maxdev]
        x = (x - np.mean(xf)) * (ffactor / np.std(xf))
        x = np.floor(x)
        x[x > maxthr] = maxthr
        x[x < - (maxthr + 1)] = - (maxthr + 1)
    return x

def discrete_probability2(x, tx, xd=None, ffactor=2, maxdev=3):
    if xd is None:
        xd = discrete_seq(x, tx, ffactor, maxdev)
    return Counter(xd)

def conditional_distribution_similarity(x, tx, y, ty, xd=None, cx=None, cy=None, ffactor=2, maxdev=3, minc=12):
    if xd is None:
        xd = discrete_seq(x, tx, ffactor, maxdev)
    if cx is None:
        cx = Counter(xd)
    if cy is None:
        yd = discrete_seq(y, ty, ffactor, maxdev)
        cy = Counter(yd)
    yrange = sorted(cy.keys())
    ny = len(yrange)
    py = np.array([cy[i] for i in yrange], dtype=float)
    py = py/py.sum()
    pyx = []
    if not numerical(ty):
        for a in cx.iterkeys():
            if cx[a] > minc:
                cyx = Counter(y[xd==a])
                pyxa = np.array([cyx[i] for i in yrange], dtype=float)
                pyxa.sort()
                pyxa = pyxa/cx[a]
                pyx.append(pyxa)
    elif len(set(y)) > 2*(ffactor*maxdev+1):
        for a in cx.iterkeys():
            if cx[a] > minc:
                yx = y[xd==a]
                yx = (yx - np.mean(yx))
                yx = np.floor(yx*ffactor)
                yx[yx > ffactor*maxdev] = ffactor*maxdev
                yx[yx < -(ffactor*maxdev+1)] = -(ffactor*maxdev+1)
                cyx = Counter(yx.astype(int))
                pyxa = np.array([cyx[i] for i in xrange(-(ffactor*maxdev+1), (ffactor*maxdev+1))], dtype=float)
                pyxa = pyxa/cx[a]
                pyx.append(pyxa)
    else:
        for a in cx.iterkeys():
            if cx[a] > minc:
                cyx = Counter(y[xd==a])
                pyxa = [cyx[i] for i in yrange]
                pyxax = np.array([0]*(ny-1) + pyxa + [0]*(ny-1), dtype=float)
                xcorr = [sum(py*pyxax[i:i+ny]) for i in xrange(2*ny-1)]
                imax = xcorr.index(max(xcorr))
                pyxa = np.array([0]*(2*ny-2-imax) + pyxa + [0]*imax, dtype=float)
                pyxa = pyxa/cx[a]
                pyx.append(pyxa)

    if len(pyx)==0: return 0
    pyx = np.array(pyx)
    pyx = pyx - pyx.mean(axis=0)
    return np.std(pyx)

def correlation(x, tx, y, ty, nepxy=None, nepyx=None):
    if categorical(tx) and categorical(ty):
        if nepxy is None:
            nepxy = normalized_error_probability(x, tx, y, ty)
        if nepyx is None:
            nepyx = normalized_error_probability(y, ty, x, tx)
        nperr = min(nepxy, nepyx)
        r = 1 - nperr
    else:
        if categorical(tx) and numerical(ty):
            x = to_numerical(x, y)
        elif numerical(tx) and categorical(ty):
            y = to_numerical(y, x)
        x = (x-np.mean(x))/np.std(x)
        y = (y-np.mean(y))/np.std(y)
        r = pearsonr(x, y)[0]
    return r

################################ HSIC ######################################
def rbf_dot(X, deg):
#Set kernel size to median distance between points, if no kernel specified
    if X.ndim == 1:
        X = X[:, np.newaxis]
    G = np.sum(X*X, axis=1)[:, np.newaxis]
    H = G + G.T - 2.0*np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5*np.median(dists[dists>0]))
    H = np.exp(-H/(2.0*deg**2))

    return H


def FastHsicTestGamma(X, Y, sig=[-1,-1]):
#This function implements the HSIC independence test using a Gamma approximation
#to the test threshold
#Inputs:
#        X contains dx columns, m rows. Each row is an i.i.d sample
#        Y contains dy columns, m rows. Each row is an i.i.d sample
#        sig[0] is kernel size for x (set to median distance if -1)
#        sig[1] is kernel size for y (set to median distance if -1)
#Outputs:
#        testStat: test statistic
#
#Use at most 200 points to save time.
    maxpnt = 200

    m = X.shape[0]
    if m>maxpnt:
        indx = np.floor(np.r_[0:m:float(m-1)/(maxpnt-1)]).astype(int)
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0/m*np.ones((m,m))

    K = rbf_dot(Xm,sig[0])
    L = rbf_dot(Ym,sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0/m)*(Kc.T*Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    return testStat

def normalized_hsic(x, tx, y, ty, h=None):
    if categorical(tx) and categorical(ty):
        if h is None:
            h = correlation(x, tx, y, ty)
    else:
        if categorical(tx) and numerical(ty):
            x = to_numerical(x, y)
        elif numerical(tx) and categorical(ty):
            y = to_numerical(y, x)
        x = (x-np.mean(x))/np.std(x)
        y = (y-np.mean(y))/np.std(y)
        h = FastHsicTestGamma(x, y)
    return h
