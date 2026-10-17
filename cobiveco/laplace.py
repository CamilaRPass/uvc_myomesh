"""
Núcleo numérico do Cobiveco/UVC em Python.

Contém:
  - cotmatrix(P, T): Laplaciano cotangente para malha tetraédrica
                     (equivalente ao cotmatrix() do gptoolbox)
  - solveLaplace(L, ids, val, tol, maxit): resolve L*u = 0 com
                     condições de Dirichlet u(ids) = val
                     (equivalente ao solveLaplace.m de Steffen Schuler)

A ideia física: o Laplaciano cotangente é a versão discreta do operador
de difusão numa malha. Resolver Laplace com bordas fixas é como deixar
calor se espalhar até o equilíbrio — cada coordenada UVC é um desses
"campos de equilíbrio" com bordas escolhidas de forma esperta.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
from scipy.sparse.csgraph import reverse_cuthill_mckee


def cotmatrix(P, T):
    """
    Laplaciano cotangente para malha tetraédrica.

    Replica o cotmatrix() do gptoolbox para tets. O resultado L é
    simétrico, com constantes no núcleo (somas de linha ~ 0). Resolver
    L*u = 0 anula funções harmônicas — exatamente o que as coordenadas
    UVC precisam.

    Parâmetros
    ----------
    P : (n, 3) float    pontos da malha
    T : (m, 4) int      tetraedros (índices base-0)

    Retorna
    -------
    L : (n, n) scipy.sparse.csr_matrix
    """
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    n = P.shape[0]
    m = T.shape[0]
    v = P[T]  # (m, 4, 3)

    # Volume com sinal de cada tetraedro
    e10 = v[:, 1, :] - v[:, 0, :]
    e20 = v[:, 2, :] - v[:, 0, :]
    e30 = v[:, 3, :] - v[:, 0, :]
    vol = np.einsum('ij,ij->i', e10, np.cross(e20, e30)) / 6.0
    absvol = np.abs(vol)
    absvol = np.maximum(absvol, 1e-300)

    # Face f é oposta ao vértice f. Normal area-ponderada APONTANDO PARA FORA.
    # Para P1-FEM, o gradiente da coordenada baricêntrica do vértice i é
    #   grad(lambda_i) = -N_i / (6V)
    # com N_i a normal externa (area-ponderada, |N_i| = 2*area) da face oposta.
    # A rigidez local é K_ij = V * grad_i . grad_j = (N_i . N_j) / (36 V).
    # Isso reproduz funções lineares exatamente (núcleo = constantes) e dá
    # somas de linha nulas, pois sum_i N_i = 0 (teorema da divergência).
    face_verts = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

    N = np.empty((m, 4, 3))
    for f in range(4):
        i, j, k = face_verts[f]
        nf = np.cross(v[:, j, :] - v[:, i, :], v[:, k, :] - v[:, i, :])
        # orienta para fora: deve apontar para longe do vértice oposto (f)
        to_opp = v[:, f, :] - v[:, i, :]
        sign = np.sign(np.einsum('ij,ij->i', nf, to_opp))
        sign[sign == 0] = 1.0
        # se aponta na direção do vértice oposto, inverte
        N[:, f, :] = nf * (-sign)[:, None]

    inv36V = 1.0 / (36.0 * absvol)

    # K_ij = (N_i . N_j) / (36 V). L = -K (convenção gptoolbox:
    # off-diagonais positivas, diagonal negativa).
    I, J, V_ = [], [], []
    for a in range(4):
        for b in range(4):
            kij = np.einsum('ij,ij->i', N[:, a, :], N[:, b, :]) * inv36V
            I.append(T[:, a])
            J.append(T[:, b])
            V_.append(-kij)   # L = -K
    I = np.concatenate(I)
    J = np.concatenate(J)
    V_ = np.concatenate(V_)
    L = sp.csr_matrix((V_, (I, J)), shape=(n, n))
    return L


def solveLaplace(L, boundaryIds, boundaryVal, tol=1e-8, maxit=3000):
    """
    Resolve L*u = 0 com Dirichlet u(boundaryIds) = boundaryVal.

    Réplica de solveLaplace.m: substitui as linhas/colunas das bordas,
    aplica reordenamento reverse Cuthill-McKee, fatoração ILU como
    precondicionador e resolve com BiCGSTAB.

    Parâmetros
    ----------
    L : (n, n) sparse        Laplaciano (de cotmatrix)
    boundaryIds : (k,) int   índices dos nós de borda (base-0)
    boundaryVal : (k,) float valores de Dirichlet
    tol, maxit               parâmetros do solver iterativo

    Retorna
    -------
    u : (n,) float
    """
    L = sp.csr_matrix(L).astype(np.float64).copy()
    n = L.shape[0]
    boundaryIds = np.asarray(boundaryIds, dtype=np.int64).ravel()
    boundaryVal = np.asarray(boundaryVal, dtype=np.float64).ravel()

    if len(boundaryIds) != len(np.unique(boundaryIds)):
        raise ValueError('Entradas duplicadas em boundaryIds.')

    K = len(boundaryIds)

    # lado direito: b = -L[:, bnd] @ val ; depois fixa b[bnd] = val
    b = -(L[:, boundaryIds] @ boundaryVal)
    b = np.asarray(b).ravel()
    b[boundaryIds] = boundaryVal

    # Impõe identidade nas linhas E colunas das bordas (mantém simetria,
    # como no solveLaplace.m). Versão vetorizada via COO:
    mask_bnd = np.zeros(n, dtype=bool)
    mask_bnd[boundaryIds] = True
    Lc = L.tocoo()
    keep = (~mask_bnd[Lc.row]) & (~mask_bnd[Lc.col])
    row = np.concatenate([Lc.row[keep], boundaryIds])
    col = np.concatenate([Lc.col[keep], boundaryIds])
    dat = np.concatenate([Lc.data[keep], np.ones(K)])
    L = sp.csr_matrix((dat, (row, col)), shape=(n, n))
    L.eliminate_zeros()

    # reverse Cuthill-McKee
    p = reverse_cuthill_mckee(sp.csr_matrix((L != 0).astype(np.int8)))
    inv_p = np.empty_like(p)
    inv_p[p] = np.arange(n)
    Lp = L[p][:, p].tocsc()
    bp = b[p]

    # precondicionador ILU
    try:
        ilu = spilu(Lp)
        M = LinearOperator(Lp.shape, ilu.solve)
    except Exception:
        M = None

    x, info = bicgstab(Lp, bp, rtol=tol, maxiter=maxit, M=M)
    if info != 0:
        import warnings
        warnings.warn(f'bicgstab não convergiu (info={info}).')

    u = np.empty(n)
    u[p] = x
    return u
