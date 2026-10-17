"""
Camada de utilidades VTK para o Cobiveco/UVC em Python.

Cada função aqui replica uma função correspondente do vtkToolbox (MATLAB),
mas chamando o mesmo VTK por baixo, via pyvista. Por isso o comportamento
é fiel ao original — não é reimplementação, é a mesma biblioteca.

Mapa MATLAB -> Python:
  vtkRead / vtkWrite              -> read / write
  vtkDataSetSurfaceFilter        -> surface_filter
  vtkMapPointIds(A, B)           -> map_point_ids(A, B)
  vtkEdgeLengths / mean          -> mean_edge_length
  vtkThreshold(m,'points',...)   -> threshold(m, 'points', ...)
  vtkThreshold(m,'cells',...)    -> threshold(m, 'cells', ...)
  vtkConnectivityFilter          -> connectivity
  vtkTetConnectivityFilter       -> connectivity (volume)
  vtkFeatureEdges                -> feature_edges
  vtkDeleteDataArrays            -> delete_data_arrays
  baryInterpMat                  -> bary_interp_mat
"""

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
import scipy.sparse as sp


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read(path):
    return pv.read(path)


def write(mesh, path):
    mesh.save(path)


# ----------------------------------------------------------------------
# Conectividade (células) como array (m, k)
# ----------------------------------------------------------------------
def get_cells(mesh):
    """Retorna a conectividade como array (n_cells, pts_por_celula)."""
    if isinstance(mesh, pv.PolyData):
        faces = mesh.faces
        # assume triângulos homogêneos
        if faces.size and faces[0] == 3:
            return faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        # fallback genérico
        return _cells_from_offsets(mesh)
    # UnstructuredGrid: usa cells_dict (assume tipo único)
    cd = mesh.cells_dict
    key = max(cd.keys(), key=lambda k: cd[k].shape[0])
    return cd[key].astype(np.int64)


def _cells_from_offsets(mesh):
    c = mesh.cells
    out = []
    i = 0
    while i < len(c):
        k = c[i]
        out.append(c[i + 1:i + 1 + k])
        i += 1 + k
    return np.array(out, dtype=np.int64)


# ----------------------------------------------------------------------
# Superfície de uma malha de volume
# ----------------------------------------------------------------------
def surface_filter(vol):
    """
    Equivalente a vtkDataSetSurfaceFilter. Retorna a superfície (PolyData)
    com o array 'vtkOriginalPointIds' preservado (mapeamento p/ o volume).
    """
    surf = vol.extract_surface(pass_pointid=True, pass_cellid=False,
                               algorithm='dataset_surface')
    return surf


def original_point_ids(surf):
    return np.asarray(surf.point_data['vtkOriginalPointIds']).astype(np.int64)


# ----------------------------------------------------------------------
# Mapeamento de pontos por vizinho mais próximo (vtkMapPointIds)
# ----------------------------------------------------------------------
def map_point_ids(source, target):
    """
    Para cada ponto de `target`, encontra o índice do ponto mais próximo
    em `source`. Réplica de vtkMapPointIds(source, target).
    """
    sp_pts = source.points if hasattr(source, 'points') else np.asarray(source)
    tg_pts = target.points if hasattr(target, 'points') else np.asarray(target)
    tree = cKDTree(sp_pts)
    _, idx = tree.query(tg_pts)
    return idx.astype(np.int64)


# ----------------------------------------------------------------------
# Comprimento médio de aresta
# ----------------------------------------------------------------------
def mean_edge_length(vol):
    cells = get_cells(vol)
    P = vol.points
    k = cells.shape[1]
    # todas as arestas únicas dentro de cada célula
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    lengths = []
    for a, b in pairs:
        d = P[cells[:, a]] - P[cells[:, b]]
        lengths.append(np.sqrt(np.sum(d * d, axis=1)))
    return float(np.mean(np.concatenate(lengths)))


# ----------------------------------------------------------------------
# Threshold (pontos ou células) sobre um array escalar
# ----------------------------------------------------------------------
def threshold(mesh, association, array_name, rng):
    """
    Réplica de vtkThreshold(mesh, association, arrayName, [lo hi]).

    association='points': mantém células onde TODOS os pontos têm valor
                          em [lo, hi]  (all_scalars=True)
    association='cells' : mantém células cujo valor de célula está em [lo,hi]
    """
    lo, hi = rng
    lo = -1e30 if lo == -np.inf else lo
    hi = 1e30 if hi == np.inf else hi
    if association == 'points':
        out = mesh.threshold(value=(lo, hi), scalars=array_name,
                             preference='point', all_scalars=True)
    else:
        out = mesh.threshold(value=(lo, hi), scalars=array_name,
                             preference='cell')
    return out


# ----------------------------------------------------------------------
# Connectivity (RegionId)
# ----------------------------------------------------------------------
def connectivity(mesh):
    """Réplica de vtkConnectivityFilter: adiciona pointData 'RegionId'."""
    out = mesh.connectivity(extraction_mode='all')
    # pyvista nomeia 'RegionId' tanto em point_data quanto cell_data
    return out


def region_id_point(mesh):
    return np.asarray(mesh.point_data['RegionId']).astype(np.int64)


def region_id_cell(mesh):
    return np.asarray(mesh.cell_data['RegionId']).astype(np.int64)


# ----------------------------------------------------------------------
# Feature edges
# ----------------------------------------------------------------------
def feature_edges(mesh, boundary=True, feature=False, manifold=False,
                  non_manifold=False):
    """Réplica de vtkFeatureEdges (por padrão, só boundary edges)."""
    out = mesh.extract_feature_edges(
        boundary_edges=boundary,
        feature_edges=feature,
        manifold_edges=manifold,
        non_manifold_edges=non_manifold,
    )
    return out


# ----------------------------------------------------------------------
# Limpeza de arrays
# ----------------------------------------------------------------------
def delete_data_arrays(mesh, keep=None):
    """Remove todos os arrays, exceto os listados em keep (ex: ['class'])."""
    keep = set(keep or [])
    m = mesh.copy()
    for name in list(m.point_data.keys()):
        if name not in keep:
            del m.point_data[name]
    for name in list(m.cell_data.keys()):
        if name not in keep:
            del m.cell_data[name]
    return m


# ----------------------------------------------------------------------
# Matriz de interpolação baricêntrica (baryInterpMat)
# ----------------------------------------------------------------------
def bary_interp_mat(P, C, target_points):
    """
    Matriz M (n_target x n_source) que interpola valores dos nós de uma
    malha tetraédrica de origem para pontos-alvo (interpolação baricêntrica).
    Réplica de baryInterpMat.m.

    Usa o localizador de células do VTK para achar o tet que contém cada
    ponto-alvo; para pontos fora da malha, usa o tet de centróide mais
    próximo (extrapolação), como no original.
    """
    P = np.asarray(P, dtype=np.float64)
    C = np.asarray(C, dtype=np.int64)
    target_points = np.asarray(target_points, dtype=np.float64)
    m = target_points.shape[0]
    n = P.shape[0]

    # malha de origem como UnstructuredGrid de tetraedros
    cells = np.hstack([np.full((C.shape[0], 1), 4, np.int64), C]).ravel()
    ctypes = np.full(C.shape[0], vtk.VTK_TETRA, np.uint8)
    src = pv.UnstructuredGrid(cells, ctypes, P)

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(src)
    locator.BuildLocator()

    cell_ids = np.full(m, -1, np.int64)
    bcoords = np.zeros((m, 4), np.float64)
    pt_ids = np.zeros((m, 4), np.int64)

    gen_cell = vtk.vtkGenericCell()
    closest = [0.0, 0.0, 0.0]
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0]

    centroids = P[C].mean(axis=1)
    ctree = cKDTree(centroids)

    for i in range(m):
        pt = target_points[i].tolist()
        cid = locator.FindCell(pt)
        if cid < 0:
            # fora da malha -> tet de centróide mais próximo
            cid = int(ctree.query(pt)[1])
            verts = C[cid]
            bc = _bary_coords(P[verts], target_points[i])
        else:
            cell = src.GetCell(cid)
            cell.EvaluatePosition(pt, closest, sub_id, pcoords, dist2, weights)
            bc = np.array(weights, np.float64)
            verts = C[cid]
        pt_ids[i] = verts
        bcoords[i] = bc

    rows = np.repeat(np.arange(m), 4)
    cols = pt_ids.ravel()
    vals = bcoords.ravel()
    M = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))
    return M


def _bary_coords(tet_pts, p):
    """Coordenadas baricêntricas de p num tetraedro (4 pontos)."""
    A = np.vstack([tet_pts.T, np.ones(4)])
    b = np.append(p, 1.0)
    try:
        bc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        bc = np.full(4, 0.25)
    return bc


# ----------------------------------------------------------------------
# Normais de vértice (triangulação de superfície)
# ----------------------------------------------------------------------
def vertex_normals(surf):
    s = surf.compute_normals(point_normals=True, cell_normals=False,
                             auto_orient_normals=True, consistent_normals=True)
    return np.asarray(s.point_data['Normals']).astype(np.float64)


# ----------------------------------------------------------------------
# PCA (eixos principais; última coluna = direção de menor variância)
# ----------------------------------------------------------------------
def pca(points):
    """Réplica de pca() do MATLAB: colunas = componentes em ordem
    decrescente de variância. coeff[:, -1] é a direção de menor variância
    (normal a uma nuvem ~planar)."""
    X = np.asarray(points, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T  # colunas são os componentes principais


# ----------------------------------------------------------------------
# Sub-malha de tetraedros a partir de subconjunto de células
# ----------------------------------------------------------------------
def make_tet_submesh(global_points, tets_subset, point_data=None):
    """
    Constrói um UnstructuredGrid de tetraedros a partir de um subconjunto de
    células (índices globais), remapeando para índices locais e carregando
    point data. Equivale ao padrão zeroCross do computeRotational.m
    (incl. o uso de 'changem').
    """
    pointIds = np.unique(tets_subset.ravel())
    remap = -np.ones(global_points.shape[0], np.int64)
    remap[pointIds] = np.arange(len(pointIds))
    local_cells = remap[tets_subset]
    pts = global_points[pointIds]

    cells = np.hstack([np.full((local_cells.shape[0], 1), 4, np.int64),
                       local_cells]).ravel()
    ctypes = np.full(local_cells.shape[0], vtk.VTK_TETRA, np.uint8)
    mesh = pv.UnstructuredGrid(cells, ctypes, pts)
    mesh.point_data['ids'] = pointIds.astype(np.int64)
    if point_data:
        for k, v in point_data.items():
            mesh.point_data[k] = np.asarray(v)[pointIds]
    return mesh, pointIds
