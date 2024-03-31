import numpy as np
from itertools import combinations

def orth(v):
    n = len(v)
    # Complete to an orthonormal basis
    v = np.concatenate(([v], np.random.randn(n - 1,n)), axis=0)
    # Gram-Schmidt
    u = np.zeros((n, n))
    for i in range(n):
        u[i] = v[i]
        for j in range(i):
            u[i] -= np.dot(u[j], v[i]) * u[j]
        u[i] /= np.linalg.norm(u[i])
    return u

def solve(F, G, x0, damping=0, max_iter = 1000, tol = 1e-6):
    # Solve
    # la = x^T G x - 2 (x^T F x)^2
    # x  = (G x - 2 (x^T F x) F x) / la
    # x0: initial guess
    # damping: damping factor
    # max_iter: maximum number of iterations
    # tol: tolerance
    last_x = x0
    success = False
    for i in range(max_iter):
        Gx = np.dot(G, last_x)
        Fx = np.dot(F, last_x)
        xFx = np.dot(last_x, Fx)
        la = np.dot(last_x, Gx) - 2 * xFx**2
        x1 = (Gx - 2 * xFx * Fx) / la
        x  = (1 - damping) * x1 + damping * last_x
        err = np.linalg.norm(x - last_x)
        if err < tol:
            success = True
            break
        last_x = x
    return x, success, i, err

def rec(Al, Ul, Sl, L=0):
    # If L = 0, return the original matrix
    H = np.diag(np.diag(Al[L]))
    H[np.ix_(Sl[L], Sl[L])] = Al[L][np.ix_(Sl[L], Sl[L])]
    Arec = H.copy()
    Urec = np.eye(Arec.shape[0])
    for l in range(L-1, -1, -1):
        Arec = np.dot(Ul[l].T, np.dot(Arec, Ul[l]))
        Urec = np.dot(Urec, Ul[l])
    return Arec, Urec, H

def MMF(A, L, k):
    # Implement multiresolution matrix factorization
    # A: input matrix
    # L: number of levels
    # k: width of each level

    n = A.shape[0]
    S = list(range(n))
    U = []
    Al = [A.copy()]
    Sl = [S.copy()]
    removed = []
    kept = list(range(n))
    for l in range(L):
        # El = 2 |[Al]_Jl, Sl|^2
        # Al = Ul Al-1 Ul.T
        # The Sl-1 x Sl-1 corner of this is
        # O [Al-1]_I,I O.T
        # So when Jl has size 1, = {ik}
        # This becomes
        # sum((O [Al-1]_I,I O.T)^2_{ik, i}) for i in SL
        # sum((O [Al-1]_I,I O.T)^2_{ik, I}) + |(O [Al-1]_I, SL\I)_{ik, SL\I}|^2_F
        # = sum((O [Al-1]_I,I O.T)^2_{ik, I}) + [O B O.T]_{ik,ik}
        # where B = [Al-1]_I, SL\I )[Al-1]_I, SL\I).T
        best_El = np.inf
        for j, ik in enumerate(S):
            # Make a version of S without ik            
            Sleft = S[:j] + S[j+1:]            
            # Iterate through all unique k-1 subsets of the Sleft
            for I in combinations(Sleft, k-1):
                Isub = list(I) + [ik]
                Asub = A[Isub][:,Isub]
                Ioth = [i for i in Sleft if i not in Isub]
                Aoth = A[Isub][:,Ioth]

                F = Asub
                G = F @ F.T + Aoth @ Aoth.T
                x0 = F[:,0]/np.linalg.norm(F[:,0])
                x, success, i, err = solve(F, G, x0, damping=0.9)
                if not success:
                    # print(f"Warning: solve failed at level {l+1}")
                    continue
                Usub = orth(x)
                # Find the rotation that diagonalizes Asub
                #Usub = np.linalg.eigh(Asub)[1].T
                #U    = eye(n)
                #U[np.ix_(Isub,Isub)] = Usub
                #El = 2 * sum((U @ A @ U.T)[Isub, ik]**2)
                El_1 = sum((Usub @ Asub @ Usub.T)[-1,:-1]**2)

                B = Aoth @ Aoth.T
                El_2 = (Usub @ B @ Usub.T)[-1,-1]

                El = El_1 + El_2
                if El < best_El:
                    best_El = El
                    best_inds = list(I)+ [ik]
                    best_U = Usub

        best_I, best_ik = best_inds[:-1], best_inds[-1]
        print(f"Level {l+1}: {best_El=:.2f} {best_inds=}")
        Ul = np.eye(n)
#        print(f"{best_U=}")
        Ul[np.ix_(best_inds,best_inds)] = best_U
        U.append(Ul) 
        A = Ul @ A @ Ul.T
        Al.append(A.copy())
        S.remove(best_ik)
        removed.append(best_ik)
        kept.remove(best_ik)
        Sl.append(S.copy())
        
    return U, Sl, Al, removed, kept
  
