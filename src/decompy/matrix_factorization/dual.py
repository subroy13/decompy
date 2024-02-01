import numpy as np
from typing import Union

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult

class DualRobustPCA:

    """
    This solves the robust PCA problem using dual formalization

    The Primal Robust PCA relaxation
        min tau ( |A|_* + \lambda |E|_1 ) + 1/2 |(A,E)|_F^2
        subj  A+E = D

    The Dual problem
        max trace(D' * Y)
        subj max( |Y|_2,1 + \lambda |Y|_inf) <= 1

    Notes
    -----
    [1] Robust PCA: Exact Recovery of Corrupted Low-Rank Matrices via Convex Optimization", J. Wright et al., preprint 2009.
    [2] "Fast Convex Optimization Algorithms for Exact Recovery of a Corrupted Low-Rank Matrix", Z. Lin et al.,  preprint 2009.
    """

    def __init__(self, **kwargs) -> None:
        self.maxiter = kwargs.get('maxiter', 1000)
        self.tol = kwargs.get('tol', 2e-5)
        self.eps = kwargs.get('eps', 1e-4)
        self.beta = kwargs.get('beta', 0.6)   # linesearch
        self.eps_proj = kwargs.get('eps_proj', 1e-1)   # projection
        self.linesearch = kwargs.get('linesearch', False)

    def choosvd(self, n: int, d: int):
        if n <= 100:
            return (d / n <= 0.02)
        elif n <= 200:
            return (d / n <= 0.10)
        elif n <= 300:
            return (d / n <= 0.13)
        elif n <= 400:
            return (d / n <= 0.14)
        elif n <= 500:
            return (d / n <= 0.17)
        else:
            return (d / n <= 0.19)


    def decompose(self, M: np.ndarray, lambd: Union[float, None] = None):
        check_real_matrix(M)
        D = M.copy()   # this is copy so that we don't modify the thing
        m, n = D.shape
        if lambd is None:
            lambd = 1 / np.sqrt(m)
        
        # initialize
        Y = np.sign(D)
        norm_two = np.linalg.norm(Y, ord=2)
        norm_inf = np.linalg.norm(Y, ord='inf') / lambd
        dual_norm = max(norm_two, norm_inf)
        Y /= dual_norm
        norm_two /= dual_norm
        norm_inf /= dual_norm

        # projection
        A_dual = np.zeros((m, n))
        E_dual = np.zeros((m, n))
        tolProj = 1e-8 * np.linalg.norm(D, 'fro')

        # linesearch
        t = 1
        K = 7
        delta = 0.1
        memo_step_size = np.ones(K) * 0.1

        niter = 0
        converged = False
        while not converged:
            # compute Z, projection of D onto the normal cone of Y
            # get the search direction D - Z
            niter += 1

            if norm_two < norm_inf - self.eps and niter < self.maxiter:
                threshold = np.linalg.norm(Y, 'inf') * (1 - self.eps_proj)
                Z = np.maximum(D * (Y > threshold), 0) + np.minimum( D * (Y < -threshold), 0)
            else:
                t = max(np.round(t * 1.1), t+1)
                u, s, vt = np.linalg.svd(Y)
                t = np.max(np.where(s >= s[0] * (1 - 1e-2) ))

                if norm_two > norm_inf + self.eps and niter < self.maxiter:
                    D_bar = u[:, :t].T @ D @ vt[:t, :].T
                    J, S = np.linalg.eig((D_bar + D_bar.T)/2)
                    temp = S @ np.diag(np.maximum(J, 0)) @ S.T
                    Z = u[:,:t] @ temp @ vt[:t, :]
                else:
                    convergedProjection = False
                    A_dual = np.zeros((m, n))
                    E_dual = np.zeros((m, n))
                    proj = 0
                    threshold = np.linalg.norm(Y, 'inf') * (1 - self.eps_proj)
                    while not convergedProjection:
                        Z = D - A_dual
                        Z = np.maximum(Z * (Y > threshold), 0) + np.minimum(Z * (Y < -threshold), 0)
                        D_bar = u[:, :t].T @ (D - Z) @ vt[:t, :].T
                        J, S = np.linalg.eig((D_bar + D_bar.T)/2)
                        temp = S @ np.diag(np.maximum(J, 0)) @ S.T
                        X = u[:,:t] @ temp @ vt[:t, :]

                        if np.linalg.norm(Z - E_dual, 'fro') < tolProj and np.linalg.norm(X - A_dual, 'fro') < tolProj:
                            convergedProjection = True
                            E_dual, A_dual = Z, X
                            Z += X
                        else:
                            E_dual, A_dual = Z, X
                
                        proj += 1
                        if proj > 50:
                            # max inner iteration reached
                            convergedProjection = True

                # linesearch to find max trace(D'*(Y+delta*(D-Z)))/J(Y+delta*(D-Z))
                Z = D - Z
                a = np.dot(D.reshape(-1), Y.reshape(-1))
                b = np.dot(D.reshape(-1), Z.reshape(-1))
                c = np.dot(Z.reshape(-1), Z.reshape(-1))

                if not self.linesearch:

                    # non exact linesearch
                    stepsize = max(1.3 * np.median(memo_step_size), 1e-4)
                    converged_line_search_like = False
                    num_trial_point = 0
                    while not converged_line_search_like:
                        X = Y + stepsize * Z
                        norm_two = np.linalg.norm(X, 2)
                        norm_inf = np.linalg.norm(X, 'inf') / lambd
                        dual_norm = max(norm_two, norm_inf)
                        tempv = (a + b * stepsize) / dual_norm
                        diff = tempv - a - stepsize / 2 * c
                        if diff > 0 or num_trial_point >= 50:
                            converged_line_search_like = True
                            obj_v = tempv
                            norm_two /= dual_norm
                            norm_inf /= dual_norm
                            Y = X / dual_norm
                            delta = stepsize
                        else:
                            stepsize *= self.beta
                        num_trial_point += 1
                    memo_step_size[int(niter / K)] = delta

                else:
                    # exact linesearch
                    eps_line_search = 1e-10
                    stepsize = max(np.abs(np.median(memo_step_size)), 1e-4)

                    current_position = 0
                    direction = 1
                    current_value = a
                    position_trace = np.array([0])
                    value_trace = np.array([current_value])
                    num_trial_point = 0
                    converged_line_search = False

                    while not converged_line_search:

                        next_position = current_position + stepsize * direction
                        findnp = (np.abs(position_trace - next_position) < 1e-1 * eps_line_search)
                        if findnp.sum() > 0:
                            next_value = value_trace[findnp]
                        else:
                            X = Y + next_position * Z
                            norm_two = np.linalg.norm(X, 2)
                            norm_inf = np.linalg.norm(X, 'inf') / lambd
                            dual_norm = max(norm_two, norm_inf)
                            next_value = (a + b * next_position) / dual_norm
                            position_trace = np.append(position_trace, next_position)
                            value_trace = np.append(value_trace, next_value)
                            num_trial_point += 1
                        
                        if next_value <= current_value:
                            direction *= (-1)  # reverse the step direction
                            stepsize /= 2  # search using smaller steps now
                        else:
                            current_position = next_position
                            current_value = next_value

                        if (stepsize < eps_line_search) or (num_trial_point >= 50) or (stepsize < 1e-1 * current_position):
                            delta, obj_v = current_position, current_value
                            Y += (delta * Z)
                            norm_two = np.linalg.norm(Y, 2)
                            norm_inf = np.linalg.norm(Y, 'inf') / lambd
                            dual_norm = max(norm_two, norm_inf)
                            Y /= dual_norm
                            norm_two /= dual_norm
                            norm_inf /= dual_norm
                            converged_line_search = True

                    memo_step_size[int(niter / K)] = delta
                
                # stop criterion
                stop_criterion = np.linalg.norm(D - A_dual - E_dual, 'fro')
                if stop_criterion < self.tol or (niter >= self.maxiter):
                    converged = True

        return LSNResult(
            L = A_dual,
            S = E_dual,
            N = None,
            convergence = {
                'niter': niter,
                'converged': (niter < self.maxiter),
                'final_error': stop_criterion,
            }
        )  
                
