import numpy as np
from scipy.linalg import null_space
from scipy.special import iv
from scipy.stats import t

from ..utils.rootmethods import roots_quartic

def rW(kappa, m):
    """
        Wood (1994) simulation of auxilliary W variable 
        - (https://www.tandfonline.com/doi/abs/10.1080/03610919408813161)
    """
    b = (-2.0 * kappa + np.sqrt(4*(kappa**2) + (m-1)**2)) / (m-1)
    x0 = (1-b)/(1+b)
    c = kappa * x0 + (m-1) * np.log(1-x0**2)
    done = False
    while not done:
        Z = np.random.beta((m-1)/2, (m-1)/2)
        W = (1 - (1+b)*Z)/(1 - (1-b)*Z)
        U = np.random.random(1)
        thres = kappa * W + (m-1)*np.log(1 - x0*W) - c
        if thres > np.log(U):
            done = True
    return W


def rmf_vector(kmu: np.array):
    """
        Simulate a random normal vector from the von Mises-Fisher distribution as described in Wood(1994).
    """
    kappa = np.linalg.norm(kmu)  # scalar
    mu = kmu / kappa  # (m,)
    m = kmu.shape[0]
    if kappa == 0:
        u = np.random.randn(m)  
        u /= np.linalg.norm(u)  # (m, )
    else:
        if (m == 1):
            rb = np.random.binomial(1, 1/(1 + np.exp(2 * kappa * mu)), size = 1)
            u = (-1)**rb
        else:
            W = rW(kappa, m)  # scalar
            V = np.random.randn(m-1)
            V /= np.linalg.norm(V)   # (m-1, )
            x = np.hstack([(1-W**2)**0.5 * V, W])  # (m, )
            nullmu = null_space(mu.reshape(1, -1))  # (m, m-1)
            u = np.hstack([nullmu, mu.reshape(-1, 1)]) @ x   # (m, m) @ (m, ) -> (m, )
    return u


def rmf_matrix(M: np.array):
    """
        Simulate a random orthonormal matrix from the von Mises-Fisher distribution.
    """
    if M.shape[1] == 1:
        return rmf_vector(M.reshape(-1)).reshape(-1, 1)  # this is a vector
    else:
        # Assume M.shape = (m, R)
        # Simulate from the matrix mf distribution using the rejection sampler as described in Hoff (2009)
        svd_MU, svd_Ms, svd_MVT = np.linalg.svd(M, full_matrices=False)  # (m, R), (R, ), (R, R col ortho)
        H = svd_MU @ np.diag(svd_Ms)  # (m, R) x (R, R) -> (m, R)
        m, R = H.shape

        # do the rejection iteration
        cmet = False
        rej = 0
        while not cmet:
            U = np.zeros((m, R))
            U[:, 0] = rmf_vector(H[:, 0])  # (m, )
            lr = 0

            for j in range(1, R):
                N = null_space(U[:, :j].T) # (j, m) -> (m, R-j)
                kmu = N.T @ H[:, j] # (R-j, m) x (m,) -> (R-j,)
                x = rmf_vector(kmu) # (R-j,)
                U[:, j] = N @ x  # (m, R-j) x (R-j, ) -> (m,)

                if svd_Ms[j] > 0:
                    xn = np.linalg.norm(kmu)
                    xd = np.linalg.norm(H[:, j])
                    lbr = np.log(iv(xn, 0.5 *(m-j-2) )) - np.log(iv(xd, 0.5 * (m-j-2)))  # besselI with expon.scaled = True returns e^(-x)I_v(x)
                    if np.isnan(lbr):
                        lbr = 0.5 * (np.log(xd) - np.log(xn))
                    lr += lbr + (xn - xd) + 0.5 * (m - j - 2) * (np.log(xd) - np.log(xn))

            cmet = np.log(np.random.uniform()) < lr
            rej += 1 - int(cmet)

        X = U @ svd_MVT
    return X

def ln2moment(mu, s2, lmax):
    mu = np.abs(mu)
    l2mom = np.zeros(lmax + 1)
    lmom = np.zeros(lmax * 2 + 1)
    l2mom[0] = 0
    lmom[0] = 0
    lmom[1] = np.log(mu)
    for i in range(2, 2*lmax+1):
        lmom[i] = lmom[i-1] + np.log((i-1)*s2 * np.exp(lmom[i-2] - lmom[i-1]) + mu)
        if i % 2 == 0:
            l2mom[int(i/2)] = lmom[i]
    return l2mom

def rXL(mu, sigma, l, nu = 1):
    """
        Simulate XL random variable
        - Reference: https://www.jstor.org/stable/27639896
    """

    def ldxl(x, mu, sigma, l, ln2m):
        return l*np.log(x**2) - np.log(sigma) - 0.5*np.log(2*np.pi) - 0.5*((x - mu)/sigma)**2 - ln2m
    

    theta = 0.5 * mu * (1 + np.sqrt(1 + 8*l*sigma**2/mu**2))
    tau = 1/np.sqrt(1 / sigma**2 + 2*l/theta**2)
    a = -2*theta - mu
    b = theta**2 + 2*mu*theta + nu*tau**2 + sigma**2 * (-nu*2*l-1)
    c = (-mu*theta**2) + (nu + 4*l+1)*sigma**2*theta - mu*nu*tau**2
    d = -2*l*sigma**2 * theta**2 - 2*l*nu*sigma**2 * tau**2

    z4 = roots_quartic(a, b, c, d)
    xc = np.real(z4[np.isreal(z4)])
    ln2m = ln2moment(mu, sigma, l)[l]
    lM = np.max( ldxl(xc, mu, sigma, l, ln2m) - (-np.log(tau) + np.log(t.pdf( (xc - theta)/tau, nu )) ) )
    sample = True
    while sample:
        x = theta + np.random.standard_t(nu) * tau
        lrratio = ldxl(x, mu, sigma, l, ln2m) - (-np.log(tau) + np.log(t.pdf( (x - theta)/tau, nu )) ) - lM
        sample = np.log(np.random.random()) > lrratio
    return x
