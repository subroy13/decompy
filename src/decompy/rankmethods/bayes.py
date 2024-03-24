import numpy as np
from scipy.linalg import null_space
from scipy.stats import mode
from scipy.special import loggamma
from tqdm import tqdm

from ..utils.bayesmethods import rXL, ln2moment, rmf_vector


def _gibbs_sample(E, Nu, Nv, phi, mu, psi, lor, llb=50, lub=750):
    """
    One step of gibbs sampling (restricted integral approximation)
    E: (m, n), Nu: (m, m-r+1), Nv: (n, n-r+1)
    phi, mu, psi: scalar
    lor = log odds ratio
    """

    def lcr(theta, L):
        n = theta.shape[0]
        lc = np.zeros(L + 1)
        ls = np.zeros(L + 1)
        lr = np.zeros(L + 1)
        lr[0] = 0
        lc[0] = 0
        for i in range(1, L + 1):
            tmp = np.sum((theta / theta[0]) ** i)
            ls[i] = i * np.log(theta[0]) + np.log(tmp)
            tmp = np.sum(np.exp(lc + ls[::-1] - ls - np.log(2 * i)))
            lc[i] = ls[i] + np.log(tmp)
            lr[i] = lc[i] + loggamma(i + 1) + loggamma(n / 2) - loggamma(n / 2 + i)
        return lr

    m, n = E.shape

    E0 = Nu.T @ E @ Nv  # (m-r+1, m) x (m, n) x (n, n-r+1) -> (m-r+1, n-r+1)
    m0, n0 = E0.shape  # m0 = m-r+1, n0 = n-r+1
    svdE0 = np.linalg.svd(E0, full_matrices=False)  # (m0, k), (k, ), (k, n0)
    z = svdE0[1] ** 2  # (k, )
    E02 = np.sum(z)  # scalar
    theta = z / E02  # (k, )

    delta = 0
    u = np.zeros(m0)  # (m0, )
    v = np.zeros(n0)  # (n0, )

    # critical value for approximating bessel function
    a = (svdE0[1] ** 2) * (phi**2) * np.max(z)  # (k, )
    lcrit = 0.5 * (
        np.sqrt((n0 / 2 + 1) ** 2 + 4 * (a - n0 / 2)) - (n0 / 2 + 1)
    )  # (k, )
    lmax = 1.5 * np.max(lcrit)
    lmax = int(min(max(lmax, llb), lub))
    lseq = np.arange(lmax + 1)  # (lmax + 1, )

    # log terms
    la = (
        loggamma(m0 / 2)
        - loggamma(m0 / 2 + lseq)
        - loggamma(lseq + 1)
        - 2 * lseq * np.log(2)
        + lcr(theta, lmax)
    )  # (lmax + 1, )
    lb = (
        ln2moment(mu * psi / (phi + psi), 1 / (phi + psi), lmax)
        + 0.5 * np.log(psi / (phi + psi))
        + 0.5 * mu**2 * psi * phi / (phi + psi)
    )  # (lmax + 1, )
    lc = lseq * np.log(E02 * phi**2)  # (lmax + 1, )
    lt = la + lb + lc  # (lmax + 1, )
    lper10 = np.max(lt) + np.log(np.sum(np.exp(lt - np.max(lt))))  # scalar

    # generating the number of ranks
    s = np.random.random() <= 1 / (1 + np.exp(-(lor + lper10)))  # boolean
    if s:
        # sample delta
        lpd = la + lb + lc  # (lmax + 1, )
        pdl = np.exp(lpd - np.max(lpd))  # (lmax + 1, )
        pdl /= np.sum(pdl)  # make it probability
        ell = np.random.choice(lseq, p=pdl)  # index between 1 to lmax + 1
        if ell > 0:
            delta = rXL(mu * psi / (psi + phi), np.sqrt(1 / (phi + psi)), ell)  # scalar
        else:
            delta = np.random.randn(1) / np.sqrt(phi + psi) + mu * psi / (phi + psi)

        # sample u, v
        A = phi * delta * E0  # (m0, n0)
        svdE0D = svdE0[1] * phi * delta  # this is same as svd(A)

        # sample a node
        pvals = np.exp(svdE0D - np.max(svdE0D))
        pvals /= pvals.sum()
        j = np.random.choice(
            np.arange(svdE0D.shape[0]), p=pvals
        )  # random index between 1 to k
        v = (svdE0[2][j, :] * (-1 if np.random.random() < 0.5 else 1)).reshape(
            -1
        )  # (n0, )
        for _ in range(25):
            u = rmf_vector(A @ v)  # (m0, n0) x (n0, ) -> (m0, )
            v = rmf_vector(A.T @ u)  # (n0, m0) x (m0, ) -> (n0, )
    return (delta, u, v)


def _gibbs_fixedrank(U, VT, D, Y, phi, mu, psi):
    """
    ## Gibbs Sample Fixed Rank
    Y: (m, n), U: (m, r), D: (r, ), VT: (r, n)
    phi, mu, psi: scalar
    """
    m, n = Y.shape
    indices = np.arange(D.shape[0])[D != 0]  # (r, )
    for j in indices:
        if len(indices) == 1:
            Nu = np.identity(m)  # r = 1, so (m, m)
            Nv = np.identity(n)  # r = 1, so (n, n)
        else:
            Nu = null_space(np.delete(U, j, 1).T)  # Null((r-1, m)) -> (m, m-r+1)
            Nv = null_space(np.delete(VT, j, 0))  # Null((r-1, n)) -> (n, n-r+1)

        E = Y - np.delete(U, j, 1) @ np.diag(np.delete(D, j, 0)) @ np.delete(
            VT, j, 0
        )  # (m, n) - (m, r-1) x (r-1, r-1) x (r-1, n)
        VT[j, :] = Nv @ rmf_vector(
            D[j] * phi * (Nv.T @ E.T @ U[:, j])
        )  # (n, n-r+1) x rmf( (n-r+1, n) x (n, m) x (m, ) )
        U[:, j] = Nu @ rmf_vector(
            D[j] * phi * (Nu.T @ E @ VT[j, :])
        )  # (m, m-r+1) x rmf( (m-r+1, m) x (m, n) x (n, ) )

        mn = (U[:, j].reshape(1, -1) @ E @ VT[j, :].reshape(-1, 1) * phi + mu * psi) / (
            phi + psi
        )  # (1, m) x (m, n) x (n, 1)
        se = np.sqrt(1 / (phi + psi))
        d = np.random.randn() * se + mn
        D[j] = d
    return (U, VT, D)


def _gibbs_varrank(U, VT, D, Y, phi, mu, psi, nsamp):
    """
    # Gibbs Variable Rank
    Y: (m, n), U: (m, r), D: (r, ), VT: (r, n)
    phi, mu, psi: scalar
    nsamp = effective value of r
    """
    m, n = Y.shape
    indices = np.random.choice(np.arange(n), size=nsamp)  # (nsamp, )
    for j in indices:
        if len(indices) == 1:
            Nu = np.identity(m)
            Nv = np.identity(n)
        else:
            Nu = null_space(np.delete(U, j, 1).T)  # Null((r-1, m)) -> (m, m-r+1)
            Nv = null_space(np.delete(VT, j, 0))  # Null((r-1, n)) -> (n, n-r+1)

        E = Y - np.delete(U, j, 1) @ np.diag(np.delete(D, j, 0)) @ np.delete(VT, j, 0)

        # log odds ratio
        nonzero_d = int(np.sum(D != 0) - (D[j] != 0))
        lor = np.log((nonzero_d + 1) / (n - nonzero_d))
        D[j], tmpu, tmpv = _gibbs_sample(
            E, Nu, Nv, phi, mu, psi, lor
        )  # scalar, (m-r+1, ), (n-r+1,)
        U[:, j] = Nu @ tmpu  # (m, m-r+1) x (m-r+1) -> (m, )
        VT[j, :] = Nv @ tmpv  # (n, n-r+1) x (n-r+1) -> (n, )
    return (U, VT, D)


def rank_hoffbayes(Y, svdfunc, gibbstype="fixed", kmax=20, verbose=False):
    """
    Rank estimation using Bayesian method
    - Reference: https://www.jstor.org/stable/27639896
    """
    assert gibbstype in ["fixed", "var"], "Invalid gibbs type"

    # set up the hyperparameters
    sY = svdfunc(Y)
    m, n = Y.shape
    kmax = min(m, n, kmax)
    s20est, t20est, mu0est = np.zeros(n), np.zeros(n), np.zeros(n)
    s20est[0], t20est[0], mu0est[0] = np.var(Y), 0, 0
    for k in range(1, n):
        s20est[k] = np.var(Y - sY[0][:, :k] @ np.diag(sY[1][:k]) @ sY[2][:k, :])
        t20est[k] = np.var(sY[1][:k])
        mu0est[k] = np.mean(sY[1][:k])

    nu0, s20 = 2, np.mean(s20est)  # prior for phi
    eta0, t20 = 2, np.mean(t20est)  # prior for psi
    mu0, premu0 = np.mean(mu0est), 1 / np.var(mu0est)  # prior for mu

    # MCMC parameters
    nburnin = 100
    nsamp = 100
    nthin = 10
    dvals = np.zeros((nsamp, kmax))
    niter = nburnin + (nsamp + 1) * nthin

    # set up the initial values
    mu = mu0
    phi = 1 / s20
    psi = 1 / t20

    if gibbstype == "fixed":
        U = sY[0][:, :kmax]
        D = sY[1][:kmax]
        VT = sY[2][:kmax, :]
    elif gibbstype == "var":
        U = np.zeros((m, n))
        VT = np.zeros((n, n))
        D = np.zeros(n)
        U[:, :kmax] = sY[0][:, :kmax]
        D[:kmax] = sY[1][:kmax]
        VT[:kmax, :] = sY[2][:kmax, :]

    ncount = 0
    iterrange = tqdm(range(niter)) if verbose else range(niter)
    for it in iterrange:
        if gibbstype == "var":
            U, VT, D = _gibbs_varrank(U, VT, D, Y, phi, mu, psi, kmax)
        U, VT, D = _gibbs_fixedrank(U, VT, D, Y, phi, mu, psi)
        if it > nburnin and it % nthin == 0:
            dvals[ncount, :] = D
            ncount += 1

        currank = np.sum(D != 0)
        phi = np.random.gamma(
            (nu0 + m * n) / 2,
            scale=2 / (nu0 * s20 + np.sum((Y - U @ np.diag(D) @ VT) ** 2)),
        )  # update phi
        mu = np.random.randn() / np.sqrt(premu0 + psi * currank) + (
            premu0 * mu0 + psi * np.sum(D)
        ) / (
            premu0 + psi * currank
        )  # update mu
        psi = np.random.gamma(
            (eta0 + currank) / 2, 2 / (eta0 * t20 + np.sum((D[D != 0]) ** 2))
        )  # update psi

    # finally look at the zeros
    tol = 1e-5
    if gibbstype == "fixed":
        quants = np.quantile(dvals, q=[0.025, 0.975], axis=0)
        rank = np.sum((quants[0, :] < 0) & (quants[1, :] > 0))
    elif gibbstype == "var":
        rank = mode(np.sum((np.abs(dvals) <= tol), axis=1), axis=None)
    return kmax - rank
