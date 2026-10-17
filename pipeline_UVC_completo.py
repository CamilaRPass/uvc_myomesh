"""
pipeline_UVC_completo.py
------------------------
Pipeline unificado: parte de uma malha biventricular VOLUMÉTRICA (.vtu) e
entrega a superfície (.vtp) com as 6 marcações do padrão UVC / Cobiveco.

Unifica o antigo `extract_surface.py` (Etapa 0) com o `pipeline_UVC.py`
(Etapas 1–5) num único comando.

Etapas:

    0. Extrai a superfície do volume (.vtu → .vtp) com vtkGeometryFilter,
       carregando junto o CellData (region_id) das células de contorno.
       Pulada automaticamente se a entrada já for um .vtp.
    1. Extrai o rótulo de região do CellData (region_id / elemTag / class ...)
       e o propaga para os pontos por VOTO DE MAIORIA (não por média).
    2. Marca o septo dentro do LV Endo (proximidade LV↔RV + orientação
       da normal, com detecção automática do sinal da normal).
    3. (Opcional) Corrige a Base por um plano em Z.
    4. Remapeia os rótulos para o padrão UVC (1–6).
    5. Marca o Apex com exatamente 1 ponto (projeção máxima ao longo da
       normal do plano basal), ou a partir de um arquivo com Point ID(s)
       ou com a coordenada (x, y, z) do apex.

Uso típico (tudo automático, a partir do volume):
    python pipeline_UVC_completo.py Patient_5.vtu P5.vtp

Ainda aceita entrada já em superfície (a Etapa 0 é ignorada):
    python pipeline_UVC_completo.py Patient_5.vtp P5.vtp

Arquivos gravados:
    intermediarios/<base>_etapa0_superficie.vtp   ← superfície crua (.vtu → .vtp)
    intermediarios/<base>_etapa1_regioes.vtp
    intermediarios/<base>_etapa2_septum.vtp
    intermediarios/<base>_etapa3_base.vtp
    intermediarios/<base>_etapa4_remap.vtp
    <saida>.vtp                                   ← final, com as 6 marcações

Opções úteis:
    --region-array region_id   # nome do array de regiões no arquivo de entrada
    --dist 15 --dot -0.8       # parâmetros do septo
    --z-threshold 43.201       # ativa a etapa 3 (desativada por padrão)
    --apex apex_coord.txt      # apex manual: Point ID(s) OU coordenada x,y,z
    --apex-mode id|xyz|auto    # força a interpretação do arquivo do apex
    --triangular               # triangula a superfície logo após a etapa 0
    --so-superficie            # roda apenas a etapa 0 e para
    --enxuto                   # grava só a superfície e a saída final
    --keep-cell-class          # mantém o array 'class' também em CellData
    --keep-original-arrays     # preserva fenotipo/tecido/f_0/s_0/n_0

Dependências:
    pip install vtk numpy scipy
"""

import argparse
import os
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree


# ════════════════════════════════════════════════════════════════════════════
# Constantes
# ════════════════════════════════════════════════════════════════════════════

# IDs de região no arquivo original (convenção do gerador da malha)
REGION_BASE   = 10
REGION_LV     = 20
REGION_RV     = 30
REGION_EPI    = 40
REGION_SEPTUM = 60

LABELS_ORIG = {
    REGION_BASE: "Base", REGION_LV: "LV Endo", REGION_RV: "RV Endo",
    REGION_EPI: "Epi", REGION_SEPTUM: "Septo",
}

# Remapeamento para o padrão UVC (etapa 4).
#   OBS: 10 → 1 é essencial; sem ele a Base nunca vira 1 e a etapa 5 quebra.
#        1 → 1 mantém compatibilidade quando a etapa 3 (Z) é usada.
REMAP = {
    1:  1,   # Base já corrigida pela etapa 3
    10: 1,   # Base (rótulo original)
    40: 2,   # Epi
    30: 3,   # RV Endo
    20: 4,   # LV Endo
    60: 6,   # Septo
}

LABELS_UVC = {
    1: "Base", 2: "Epi", 3: "RV Endo",
    4: "LV Endo", 5: "Apex", 6: "Septo",
}

UVC_BASE, UVC_EPI, UVC_RV, UVC_LV, UVC_APEX, UVC_SEPTUM = 1, 2, 3, 4, 5, 6

ARRAY_NAME = "class"

VERSION = ("v3.2 (vtu→vtp integrado + region_id + despeckle + apex por x,y,z "
           "+ saída .vtp/.vtu)")

# Nomes candidatos para o array de regiões no arquivo de entrada
CANDIDATOS_REGIAO = ["region_id", "regionid", "class", "elemtag", "elemtags",
                     "region", "tag", "label", "material", "matid"]

# Ordem de prioridade para desempate no voto de maioria ponto←células.
# Base primeiro produz um anel basal limpo (importante para o Cobiveco).
PRIORIDADE_DESEMPATE = [REGION_BASE, REGION_SEPTUM, REGION_LV,
                        REGION_RV, REGION_EPI]

EXT_VOLUME     = {".vtu"}
EXT_SUPERFICIE = {".vtp"}

# Quão longe do vértice mais próximo a coordenada do --apex pode cair,
# em fração da diagonal da malha. Acima do aviso, só reclama; acima do
# máximo, a coordenada é de outra malha (ou de outra unidade) e usá-la
# envenena o abLaplace — e o computeRotational quebra lá na frente.
APEX_DIST_AVISO = 0.02
APEX_DIST_MAX   = 0.10


class ApexCoordenadaForaDaMalha(ValueError):
    """A coordenada do --apex não corresponde a nenhum vértice desta malha."""


# ════════════════════════════════════════════════════════════════════════════
# Utilitários de leitura / escrita
# ════════════════════════════════════════════════════════════════════════════

def read_vtu(path: str) -> vtk.vtkUnstructuredGrid:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    mesh = reader.GetOutput()
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Não consegui ler pontos de '{path}'. "
                           f"Arquivo corrompido ou não é um VTU "
                           f"UnstructuredGrid?")
    return mesh


def read_vtp(path: str) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    mesh = reader.GetOutput()
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Não consegui ler pontos de '{path}'. "
                           f"Arquivo corrompido ou não é um VTP PolyData?")
    return mesh


def write_vtp(mesh: vtk.vtkPolyData, path: str, compress: bool = True) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(mesh)
    if compress:
        writer.SetCompressorTypeToZLib()
    else:
        writer.SetCompressorTypeToNone()
    writer.Write()


def write_vtu(mesh: vtk.vtkUnstructuredGrid, path: str,
              compress: bool = True) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(path)
    writer.SetInputData(mesh)
    if compress:
        writer.SetCompressorTypeToZLib()
    else:
        writer.SetCompressorTypeToNone()
    writer.Write()


def polydata_para_unstructured(mesh: vtk.vtkPolyData) -> vtk.vtkUnstructuredGrid:
    """Mesma superfície, embalada como UnstructuredGrid.

    MergePointsOff é essencial: preserva a ORDEM e a CONTAGEM dos pontos, de
    modo que o Point ID do apex continua válido depois da conversão.
    """
    app = vtk.vtkAppendFilter()
    app.SetInputData(mesh)
    app.MergePointsOff()
    app.Update()
    out = app.GetOutput()
    if out.GetNumberOfPoints() != mesh.GetNumberOfPoints():
        raise RuntimeError(
            f"A conversão PolyData→UnstructuredGrid mudou o número de pontos "
            f"({mesh.GetNumberOfPoints()} → {out.GetNumberOfPoints()}); os "
            f"Point IDs (inclusive o do apex) deixariam de bater.")
    return out


def read_mesh(path: str):
    """Lê .vtp (PolyData) ou .vtu (UnstructuredGrid) conforme a extensão."""
    ext = os.path.splitext(path)[1].lower()
    if ext in EXT_SUPERFICIE:
        return read_vtp(path)
    if ext in EXT_VOLUME:
        return read_vtu(path)
    raise ValueError(f"Extensão '{ext}' não suportada para leitura.")


def write_mesh(mesh: vtk.vtkPolyData, path: str, compress: bool = True) -> None:
    """Grava RESPEITANDO a extensão do nome do arquivo.

    O conteúdo tem que casar com a extensão: o ParaView (e o VTK) escolhem o
    leitor pelo sufixo, então um PolyData gravado como '.vtu' faz o ParaView
    tentar um leitor errado e abortar.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in EXT_SUPERFICIE:
        write_vtp(mesh, path, compress)
    elif ext in EXT_VOLUME:
        write_vtu(polydata_para_unstructured(mesh), path, compress)
        print(f"  (superfície gravada como UnstructuredGrid para casar com "
              f"a extensão .vtu)")
    else:
        raise ValueError(
            f"Extensão de saída '{ext}' não suportada. Use .vtp (PolyData) "
            f"ou .vtu (UnstructuredGrid).")


def section(title: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def _nomes_arrays(container) -> list:
    return [container.GetArrayName(i)
            for i in range(container.GetNumberOfArrays())]


def _print_dist(valores: np.ndarray, labels: dict, titulo: str,
                total_ref: int = None) -> None:
    uniq, counts = np.unique(valores, return_counts=True)
    total = total_ref if total_ref else valores.size
    print(titulo)
    for v, c in zip(uniq, counts):
        print(f"    {ARRAY_NAME}={int(v):5d}  ({labels.get(int(v), '?'):10s})"
              f"  {c:8d}  ({c / total * 100:5.1f}%)")


def _set_int_array(container, values: np.ndarray, name: str = ARRAY_NAME):
    """Substitui (ou cria) um array inteiro em PointData/CellData."""
    arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.int32),
                       deep=True, array_type=vtk.VTK_INT)
    arr.SetName(name)
    container.RemoveArray(name)
    container.AddArray(arr)
    container.SetActiveScalars(name)
    return arr


def _limpar_arrays(mesh: vtk.vtkPolyData, manter_pd=(), manter_cd=()) -> None:
    """Remove todos os arrays exceto os nomeados."""
    for container, manter in ((mesh.GetPointData(), manter_pd),
                              (mesh.GetCellData(), manter_cd)):
        nomes = _nomes_arrays(container)
        for nome in nomes:
            if nome is not None and nome not in manter:
                container.RemoveArray(nome)


# ════════════════════════════════════════════════════════════════════════════
# Núcleo: célula → ponto por VOTO DE MAIORIA
# ════════════════════════════════════════════════════════════════════════════

def _conectividade(mesh: vtk.vtkPolyData):
    """Retorna (point_ids_por_entrada, cell_id_por_entrada) das células
    poligonais. Funciona para triângulos e polígonos gerais."""
    polys = mesh.GetPolys() if hasattr(mesh, "GetPolys") else None
    n_cells = mesh.GetNumberOfCells()

    if polys is not None and polys.GetNumberOfCells() == n_cells:
        conn = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
        offs = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
        sizes = np.diff(offs)
        cell_of_entry = np.repeat(np.arange(n_cells, dtype=np.int64), sizes)
        return conn, cell_of_entry

    # Fallback genérico (malhas mistas)
    conn, cell_of_entry = [], []
    for i in range(n_cells):
        cell = mesh.GetCell(i)
        for j in range(cell.GetNumberOfPoints()):
            conn.append(cell.GetPointId(j))
            cell_of_entry.append(i)
    return (np.asarray(conn, dtype=np.int64),
            np.asarray(cell_of_entry, dtype=np.int64))


def cell_labels_to_point_labels(mesh: vtk.vtkPolyData,
                                cell_labels: np.ndarray) -> np.ndarray:
    """Cada ponto recebe o rótulo MAIS FREQUENTE entre as células incidentes.

    Isto substitui o vtkCellDataToPointData, que faz MÉDIA ARITMÉTICA e
    portanto destrói rótulos categóricos (ex.: média de 20 e 40 = 30, que
    por acidente é o rótulo do RV Endo).

    Empates são resolvidos por PRIORIDADE_DESEMPATE.
    """
    n_points = mesh.GetNumberOfPoints()
    cell_labels = np.asarray(cell_labels).astype(np.int64)

    presentes = np.unique(cell_labels)
    ordenados = [l for l in PRIORIDADE_DESEMPATE if l in presentes]
    ordenados += [int(l) for l in presentes if l not in ordenados]
    ordenados = np.asarray(ordenados, dtype=np.int64)
    lab2idx = {int(l): k for k, l in enumerate(ordenados)}

    conn, cell_of_entry = _conectividade(mesh)
    idx_por_entrada = np.array([lab2idx[int(l)] for l in cell_labels],
                               dtype=np.int64)[cell_of_entry]

    counts = np.zeros((n_points, ordenados.size), dtype=np.int32)
    np.add.at(counts, (conn, idx_por_entrada), 1)

    point_labels = ordenados[np.argmax(counts, axis=1)]

    # Pontos órfãos (sem célula incidente) ficam com 0
    orfaos = counts.sum(axis=1) == 0
    if orfaos.any():
        point_labels = point_labels.copy()
        point_labels[orfaos] = 0
        print(f"  AVISO: {int(orfaos.sum())} ponto(s) sem célula incidente "
              f"→ class=0")

    return point_labels.astype(np.int32)


# ════════════════════════════════════════════════════════════════════════════
# Limpeza topológica — remove "pintinhas" (ilhas e pontos isolados)
# ════════════════════════════════════════════════════════════════════════════

def _arestas(mesh: vtk.vtkPolyData):
    """Arestas (p0,p1) de cada célula + o id da célula de cada aresta."""
    conn, cell_of_entry = _conectividade(mesh)
    n_cells = mesh.GetNumberOfCells()

    sizes = np.bincount(cell_of_entry, minlength=n_cells)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    pos = np.arange(len(conn), dtype=np.int64) - starts[cell_of_entry]
    prox = starts[cell_of_entry] + (pos + 1) % sizes[cell_of_entry]

    e = np.sort(np.stack([conn, conn[prox]], axis=1), axis=1)
    return e, cell_of_entry


def _pares_de_celulas_vizinhas(mesh: vtk.vtkPolyData):
    """Pares (cA, cB) de células que compartilham uma aresta."""
    e, cell_of_entry = _arestas(mesh)
    chave = e[:, 0].astype(np.int64) * mesh.GetNumberOfPoints() + e[:, 1]

    ordem = np.argsort(chave, kind="stable")
    k, c = chave[ordem], cell_of_entry[ordem]
    m = k[:-1] == k[1:]
    return c[:-1][m], c[1:][m]


def _pares_de_pontos_vizinhos(mesh: vtk.vtkPolyData):
    e, _ = _arestas(mesh)
    e = np.unique(e, axis=0)
    return e[:, 0], e[:, 1]


def despeckle_celulas(mesh: vtk.vtkPolyData,
                      cell_labels: np.ndarray,
                      min_ilha: int = 16,
                      max_passes: int = 3) -> np.ndarray:
    """Reabsorve ilhas de células menores que `min_ilha`.

    Uma "ilha" é um grupo conexo de células com o mesmo rótulo. Ilhas
    minúsculas são ruído do gerador da malha (region_id não é perfeitamente
    suave) e produzem pontos de classe estranha no meio de outra região —
    as pintinhas vistas no ParaView. Cada ilha pequena recebe o rótulo mais
    comum entre as células que a cercam.
    """
    if min_ilha <= 1:
        return cell_labels

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    labels = cell_labels.copy()
    n = len(labels)
    a, b = _pares_de_celulas_vizinhas(mesh)
    total_reabsorvidas = 0

    for _ in range(max_passes):
        mesma = labels[a] == labels[b]
        ga, gb = a[mesma], b[mesma]
        g = coo_matrix((np.ones(len(ga), dtype=np.int8), (ga, gb)),
                       shape=(n, n))
        _, comp = connected_components(g, directed=False)

        tam = np.bincount(comp)
        pequena = tam[comp] < min_ilha
        if not pequena.any():
            break

        # Rótulo majoritário na vizinhança de cada ilha pequena
        ea = np.concatenate([a, b])
        eb = np.concatenate([b, a])
        sel = pequena[ea] & (comp[ea] != comp[eb])
        if not sel.any():
            break

        comps_ruins = np.unique(comp[pequena])
        idx_comp = {c: i for i, c in enumerate(comps_ruins)}
        valores = np.unique(labels)
        idx_val = {int(v): i for i, v in enumerate(valores)}

        votos = np.zeros((len(comps_ruins), len(valores)), dtype=np.int32)
        linhas = np.array([idx_comp[c] for c in comp[ea[sel]]], dtype=np.int64)
        colunas = np.array([idx_val[int(v)] for v in labels[eb[sel]]],
                           dtype=np.int64)
        np.add.at(votos, (linhas, colunas), 1)

        novo_por_comp = valores[np.argmax(votos, axis=1)]
        sem_vizinho = votos.sum(axis=1) == 0

        alterados = 0
        for i, c in enumerate(comps_ruins):
            if sem_vizinho[i]:
                continue
            membros = comp == c
            if labels[membros][0] != novo_por_comp[i]:
                labels[membros] = novo_por_comp[i]
                alterados += int(membros.sum())

        total_reabsorvidas += alterados
        if alterados == 0:
            break

    if total_reabsorvidas:
        print(f"  Despeckle (células): {total_reabsorvidas} célula(s) em "
              f"ilhas menores que {min_ilha} foram reabsorvidas.")
    else:
        print(f"  Despeckle (células): nenhuma ilha menor que {min_ilha}.")
    return labels


def despeckle_pontos(mesh: vtk.vtkPolyData,
                     point_labels: np.ndarray,
                     proteger=(UVC_APEX, REGION_SEPTUM),
                     max_passes: int = 3) -> np.ndarray:
    """Corrige pontos cujo rótulo não aparece em NENHUM vizinho imediato.

    Mesmo com as células limpas, o desempate do voto de maioria pode deixar
    um ponto solitário com classe diferente de toda a sua vizinhança — é
    isso que o ParaView desenha como uma pintinha clara, porque interpola a
    cor do ponto pelos triângulos ao redor.
    """
    labels = point_labels.copy()
    a, b = _pares_de_pontos_vizinhos(mesh)
    ea = np.concatenate([a, b])
    eb = np.concatenate([b, a])
    total = 0

    for _ in range(max_passes):
        iguais = np.zeros(len(labels), dtype=np.int64)
        np.add.at(iguais, ea, (labels[ea] == labels[eb]).astype(np.int64))

        isolados = iguais == 0
        for p in proteger:
            isolados &= labels != p
        if not isolados.any():
            break

        valores = np.unique(labels)
        idx_val = {int(v): i for i, v in enumerate(valores)}
        votos = np.zeros((len(labels), len(valores)), dtype=np.int32)
        col = np.array([idx_val[int(v)] for v in labels[eb]], dtype=np.int64)
        np.add.at(votos, (ea, col), 1)

        novos = valores[np.argmax(votos, axis=1)]
        labels[isolados] = novos[isolados]
        total += int(isolados.sum())

    if total:
        print(f"  Despeckle (pontos): {total} ponto(s) isolado(s) "
              f"reclassificado(s) pela vizinhança.")
    else:
        print("  Despeckle (pontos): nenhum ponto isolado.")
    return labels


# ════════════════════════════════════════════════════════════════════════════
# Etapa 0 — Extrair a superfície do volume (.vtu → .vtp)
# ════════════════════════════════════════════════════════════════════════════

def step0_extrair_superficie(input_vtu: str,
                             output_vtp: str,
                             triangular: bool = False) -> None:
    """Superfície externa da malha volumétrica, preservando os arrays.

    O vtkGeometryFilter extrai as faces de contorno e COPIA o CellData da
    célula volumétrica-mãe para a face gerada — é assim que o `region_id`
    (10/20/30/40) chega à superfície. Esta era a única coisa que o antigo
    `extract_surface.py` fazia; agora é a etapa 0 do pipeline.
    """
    section("Etapa 0 — Extrair superfície do volume (.vtu → .vtp)")

    vol = read_vtu(input_vtu)
    print(f"Lido (volume): {vol.GetNumberOfPoints()} pontos, "
          f"{vol.GetNumberOfCells()} células")
    print(f"  Arrays em CellData : {_nomes_arrays(vol.GetCellData())}")
    print(f"  Arrays em PointData: {_nomes_arrays(vol.GetPointData())}")

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(vol)
    geometry_filter.Update()
    surf = geometry_filter.GetOutput()

    if surf is None or surf.GetNumberOfCells() == 0:
        raise RuntimeError(
            f"A extração de superfície de '{input_vtu}' não produziu células. "
            f"A malha volumétrica está vazia ou tem apenas células 0D/1D?")

    if triangular:
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(surf)
        tri.PassLinesOff()
        tri.PassVertsOff()
        tri.Update()
        surf = tri.GetOutput()
        print("  Superfície triangulada (--triangular).")

    print(f"\nSuperfície: {surf.GetNumberOfPoints()} pontos, "
          f"{surf.GetNumberOfCells()} células "
          f"({surf.GetNumberOfCells() / max(vol.GetNumberOfCells(), 1) * 100:.1f}% "
          f"das células do volume)")
    print(f"  Arrays em CellData : {_nomes_arrays(surf.GetCellData())}")
    print(f"  Arrays em PointData: {_nomes_arrays(surf.GetPointData())}")

    nomes_cd = [n for n in _nomes_arrays(surf.GetCellData()) if n]
    if not nomes_cd:
        print("\n  AVISO: a superfície saiu SEM CellData. A etapa 1 não vai "
              "encontrar o array de regiões — confira se o region_id existe "
              "mesmo no .vtu de entrada.")

    write_vtp(surf, output_vtp)
    print(f"\nSalvo: {output_vtp}")


# ════════════════════════════════════════════════════════════════════════════
# Etapa 1 — Extrair region_id do CellData → PointData
# ════════════════════════════════════════════════════════════════════════════

def _encontrar_array_regiao(mesh: vtk.vtkPolyData, nome_pedido=None):
    """Localiza o array de regiões. Retorna (valores_por_celula, nome, onde).

    ESTE É O BUG PRINCIPAL DO SCRIPT ORIGINAL: ele usava
    PointData.GetArray(0), que nesta malha é 'fenotipo' (valores 0/1),
    e não o array de regiões 'region_id' (10/20/30/40) do CellData.
    """
    cd, pd = mesh.GetCellData(), mesh.GetPointData()

    nomes_cd = _nomes_arrays(cd)
    nomes_pd = _nomes_arrays(pd)
    print(f"  Arrays em CellData : {nomes_cd}")
    print(f"  Arrays em PointData: {nomes_pd}")

    def _norm(s):
        return s.lower().replace(" ", "").replace("_", "") if s else ""

    if nome_pedido:
        arr = cd.GetArray(nome_pedido)
        if arr is not None:
            return vtk_to_numpy(arr).ravel(), nome_pedido, "CellData"
        arr = pd.GetArray(nome_pedido)
        if arr is not None:
            return vtk_to_numpy(arr).ravel(), nome_pedido, "PointData"
        raise ValueError(
            f"Array '{nome_pedido}' não existe. "
            f"CellData={nomes_cd}  PointData={nomes_pd}")

    # Busca por nome canônico, CellData primeiro (é onde region_id vive)
    for cand in CANDIDATOS_REGIAO:
        for nomes, container, onde in ((nomes_cd, cd, "CellData"),
                                       (nomes_pd, pd, "PointData")):
            for nome in nomes:
                if _norm(nome) == cand:
                    vals = vtk_to_numpy(container.GetArray(nome)).ravel()
                    return vals, nome, onde

    raise ValueError(
        f"Não encontrei um array de regiões. Procurei por "
        f"{CANDIDATOS_REGIAO}.\n  CellData={nomes_cd}\n  PointData={nomes_pd}\n"
        f"  Use --region-array NOME para indicar o correto.")


def step1_extrair_regioes(input_path: str,
                          output_path: str,
                          region_array: str = None,
                          keep_original: bool = False,
                          min_ilha: int = 16) -> None:
    section("Etapa 1 — Extrair rótulos de região → PointData + CellData")

    mesh = read_vtp(input_path)
    print(f"Lido: {mesh.GetNumberOfPoints()} pontos, "
          f"{mesh.GetNumberOfCells()} células")

    valores, nome, onde = _encontrar_array_regiao(mesh, region_array)
    print(f"\n  → Usando '{nome}' de {onde} como array de regiões "
          f"(dtype {valores.dtype})")

    # Os rótulos vêm como Float64 no arquivo; são inteiros por natureza.
    cell_labels = np.rint(valores).astype(np.int32)

    if onde == "PointData":
        # Entrada já rotulada por ponto: converte ponto → célula (maioria)
        print("  Array está em PointData; derivando rótulo por célula "
              "(moda dos vértices)...")
        conn, cell_of_entry = _conectividade(mesh)
        point_labels = cell_labels
        n_cells = mesh.GetNumberOfCells()
        presentes = np.unique(point_labels)
        lab2idx = {int(l): k for k, l in enumerate(presentes)}
        counts = np.zeros((n_cells, presentes.size), dtype=np.int32)
        idx = np.array([lab2idx[int(l)] for l in point_labels],
                       dtype=np.int64)[conn]
        np.add.at(counts, (cell_of_entry, idx), 1)
        cell_labels = presentes[np.argmax(counts, axis=1)].astype(np.int32)
    else:
        print()
        cell_labels = despeckle_celulas(mesh, cell_labels, min_ilha=min_ilha)
        point_labels = cell_labels_to_point_labels(mesh, cell_labels)
        point_labels = despeckle_pontos(mesh, point_labels)

    _print_dist(cell_labels, LABELS_ORIG, "\n  Distribuição por CÉLULA:")
    _print_dist(point_labels, LABELS_ORIG, "  Distribuição por PONTO:")

    esperados = {REGION_LV, REGION_RV, REGION_EPI}
    faltando = esperados - set(int(v) for v in np.unique(cell_labels))
    if faltando:
        print(f"\n  AVISO: rótulos esperados ausentes: "
              f"{[LABELS_ORIG.get(f, f) for f in sorted(faltando)]}")

    if not keep_original:
        _limpar_arrays(mesh)
    _set_int_array(mesh.GetPointData(), point_labels)
    _set_int_array(mesh.GetCellData(), cell_labels)

    write_vtp(mesh, output_path)
    print(f"\nSalvo: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# Etapa 2 — Marcar Septo
# ════════════════════════════════════════════════════════════════════════════

def _cell_centroids(mesh: vtk.vtkPolyData) -> np.ndarray:
    """Centróides de todas as células (vetorizado)."""
    pts = vtk_to_numpy(mesh.GetPoints().GetData())
    conn, cell_of_entry = _conectividade(mesh)
    n_cells = mesh.GetNumberOfCells()

    soma = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(soma, cell_of_entry, pts[conn])
    n_por_celula = np.bincount(cell_of_entry, minlength=n_cells)
    return soma / np.maximum(n_por_celula, 1)[:, None]


def _cell_normals(mesh: vtk.vtkPolyData) -> np.ndarray:
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(mesh)
    nf.ComputeCellNormalsOn()
    nf.ComputePointNormalsOff()
    nf.SplittingOff()       # preserva a ordem/contagem das células
    nf.ConsistencyOn()
    nf.AutoOrientNormalsOn()
    nf.Update()
    out = nf.GetOutput()
    normals = out.GetCellData().GetNormals()
    if normals is None or out.GetNumberOfCells() != mesh.GetNumberOfCells():
        raise RuntimeError("Falha ao calcular normais por célula.")
    return vtk_to_numpy(normals).copy()


def step2_label_septum(input_path: str,
                       output_path: str,
                       dist_thresh: float = 15.0,
                       dot_thresh: float = -0.8,
                       min_ilha: int = 16) -> None:
    section("Etapa 2 — Marcar Septo (parede septal do LV Endo)")

    mesh = read_vtp(input_path)
    cell_arr = mesh.GetCellData().GetArray(ARRAY_NAME)
    if cell_arr is None:
        raise RuntimeError("Array 'class' ausente em CellData "
                           "(a etapa 1 deve criá-lo).")
    cell_labels = vtk_to_numpy(cell_arr).astype(np.int32).copy()

    print(f"Lido: {mesh.GetNumberOfCells()} células, "
          f"{mesh.GetNumberOfPoints()} pontos")
    _print_dist(cell_labels, LABELS_ORIG, "  Distribuição inicial por CÉLULA:")

    lv_mask = cell_labels == REGION_LV
    rv_mask = cell_labels == REGION_RV
    print(f"\n  LV Endo: {int(lv_mask.sum())} células  |  "
          f"RV Endo: {int(rv_mask.sum())} células")

    if lv_mask.sum() == 0 or rv_mask.sum() == 0:
        encontrados = {int(v): LABELS_ORIG.get(int(v), '?')
                       for v in np.unique(cell_labels)}
        raise RuntimeError(
            f"LV Endo (={REGION_LV}) ou RV Endo (={REGION_RV}) ausente — "
            f"impossível detectar o septo.\n"
            f"  Rótulos encontrados: {encontrados}\n"
            f"  Verifique se a etapa 1 leu o array de regiões correto "
            f"(use --region-array).")

    print("  Computando centróides e normais...")
    centroids = _cell_centroids(mesh)
    normals = _cell_normals(mesh)

    lv_centroids, rv_centroids = centroids[lv_mask], centroids[rv_mask]
    lv_normals = normals[lv_mask]

    dists, nn_idx = cKDTree(rv_centroids).query(lv_centroids, k=1)

    local_dir = rv_centroids[nn_idx] - lv_centroids
    local_dir /= np.linalg.norm(local_dir, axis=1, keepdims=True) + 1e-12
    dots = (lv_normals * local_dir).sum(axis=1)

    perto = dists < dist_thresh
    print(f"\n  Parâmetros: dist_thresh = {dist_thresh} mm, "
          f"dot_thresh = {dot_thresh}")
    print(f"  Células do LV a menos de {dist_thresh} mm do RV: "
          f"{int(perto.sum())}")

    if perto.sum() == 0:
        d_min, d_med = float(dists.min()), float(np.median(dists))
        raise RuntimeError(
            f"Nenhuma célula do LV está a menos de {dist_thresh} mm do RV "
            f"(mínimo = {d_min:.2f} mm, mediana = {d_med:.2f} mm).\n"
            f"  Aumente --dist (ex.: --dist {max(d_min * 1.5, 5):.0f}) ou "
            f"verifique as unidades da malha.")

    # --- Robustez: o sinal da normal depende da orientação da superfície. ---
    # Numa superfície fechada bem orientada, a normal do LV Endo aponta para
    # DENTRO da cavidade do LV, logo dot(normal, dir_LV→RV) ≈ -1.
    # Se a malha estiver orientada ao contrário, o teste inverte. Detectamos
    # isso comparando os dois sinais e escolhendo o que produz um septo
    # plausível.
    n_neg = int((perto & (dots < dot_thresh)).sum())
    n_pos = int((perto & (dots > -dot_thresh)).sum())
    print(f"  Candidatos com dot < {dot_thresh}: {n_neg}  |  "
          f"com dot > {-dot_thresh}: {n_pos}")

    if n_pos > n_neg:
        print("  → Normais aparentemente invertidas; usando -dot "
              "(auto-correção de orientação).")
        dots = -dots

    septo_lv = perto & (dots < dot_thresh)

    if septo_lv.sum() == 0:
        print("  AVISO: o critério de normal não selecionou nenhuma célula. "
              "Usando apenas o critério de proximidade (fallback).")
        septo_lv = perto

    frac = septo_lv.sum() / max(int(lv_mask.sum()), 1) * 100
    print(f"\n  Células do LV marcadas como septo: {int(septo_lv.sum())} "
          f"({frac:.1f}% do LV Endo)")
    if not (5.0 <= frac <= 70.0):
        print(f"  AVISO: fração septal atípica ({frac:.1f}%). "
              f"Esperado ~20–45%. Ajuste --dist / --dot e confira no ParaView.")

    cell_labels[np.where(lv_mask)[0][septo_lv]] = REGION_SEPTUM

    # Limpa ilhas septais soltas (o critério de normal é local e pode deixar
    # células isoladas espalhadas pela parede livre).
    print()
    cell_labels = despeckle_celulas(mesh, cell_labels, min_ilha=min_ilha)

    # Ponto ← célula por maioria (evita o "vazamento" do script original,
    # que marcava TODOS os vértices de cada célula septal, alargando o septo).
    point_labels = cell_labels_to_point_labels(mesh, cell_labels)
    point_labels = despeckle_pontos(mesh, point_labels)

    _set_int_array(mesh.GetCellData(), cell_labels)
    _set_int_array(mesh.GetPointData(), point_labels)
    write_vtp(mesh, output_path)

    _print_dist(cell_labels, LABELS_ORIG, "\n  Distribuição final por CÉLULA:")
    _print_dist(point_labels, LABELS_ORIG, "  Distribuição final por PONTO:")
    print(f"Salvo: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# Etapa 3 — (Opcional) Corrigir Base por plano em Z
# ════════════════════════════════════════════════════════════════════════════

def step3_corrigir_class(input_path: str,
                         output_path: str,
                         z_threshold=None,
                         modo: str = "below",
                         valor_corrigido: int = REGION_BASE,
                         min_ilha: int = 16) -> None:
    if z_threshold is None:
        section("Etapa 3 — Corrigir Base por Z (DESATIVADA)")
        print("  --z-threshold não informado: a Base já vem rotulada no "
              "region_id. Etapa ignorada (cópia direta).")
        write_vtp(read_vtp(input_path), output_path)
        print(f"Salvo: {output_path}")
        return

    op = "≤" if modo == "below" else "≥"
    section(f"Etapa 3 — Corrigir Base (Z {op} {z_threshold} → "
            f"class = {valor_corrigido})")

    mesh = read_vtp(input_path)
    pts = vtk_to_numpy(mesh.GetPoints().GetData())
    z = pts[:, 2]
    print(f"  Faixa de Z na malha: [{z.min():.3f}, {z.max():.3f}]")

    point_labels = vtk_to_numpy(
        mesh.GetPointData().GetArray(ARRAY_NAME)).astype(np.int32).copy()
    cell_labels = vtk_to_numpy(
        mesh.GetCellData().GetArray(ARRAY_NAME)).astype(np.int32).copy()

    sel_pt = z <= z_threshold if modo == "below" else z >= z_threshold
    point_labels[sel_pt] = valor_corrigido

    zc = _cell_centroids(mesh)[:, 2]
    sel_cl = zc <= z_threshold if modo == "below" else zc >= z_threshold
    cell_labels[sel_cl] = valor_corrigido

    print(f"  Pontos alterados : {int(sel_pt.sum())}")
    print(f"  Células alteradas: {int(sel_cl.sum())}")
    if sel_pt.sum() == 0:
        print("  AVISO: nenhum ponto selecionado — o threshold está fora da "
              "faixa de Z desta malha.")

    print()
    cell_labels = despeckle_celulas(mesh, cell_labels, min_ilha=min_ilha)
    point_labels = cell_labels_to_point_labels(mesh, cell_labels)
    point_labels = despeckle_pontos(mesh, point_labels)

    _set_int_array(mesh.GetPointData(), point_labels)
    _set_int_array(mesh.GetCellData(), cell_labels)
    write_vtp(mesh, output_path)
    print(f"Salvo: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# Etapa 4 — Remapear para o padrão UVC
# ════════════════════════════════════════════════════════════════════════════

def _remap_container(container, location: str) -> None:
    vtk_arr = container.GetArray(ARRAY_NAME)
    if vtk_arr is None:
        return
    region = vtk_to_numpy(vtk_arr).astype(np.int32).copy()

    print(f"\n  ── Remapeando '{ARRAY_NAME}' em {location} ──")
    uniq, counts = np.unique(region, return_counts=True)
    nao_mapeados = []
    for v, c in zip(uniq, counts):
        dest = REMAP.get(int(v))
        if dest is None:
            nao_mapeados.append(int(v))
            txt = "→ SEM MAPEAMENTO (mantido)"
        else:
            txt = f"→ UVC {dest} ({LABELS_UVC.get(dest, '?')})"
        print(f"     {int(v):5d}  {c:8d}   {txt}")

    if nao_mapeados:
        print(f"     AVISO: rótulos sem mapeamento: {nao_mapeados}. "
              f"O Cobiveco espera apenas 1–6.")

    region_new = region.copy()
    for old, new in REMAP.items():
        region_new[region == old] = new

    _set_int_array(container, region_new)
    _print_dist(region_new, LABELS_UVC, "  Distribuição DEPOIS (UVC):")


def step4_remap_class(input_path: str, output_path: str) -> None:
    section("Etapa 4 — Remapear Class para o padrão UVC (1–6)")

    mesh = read_vtp(input_path)
    print(f"Lido: {mesh.GetNumberOfPoints()} pontos, "
          f"{mesh.GetNumberOfCells()} células")

    _remap_container(mesh.GetPointData(), "PointData")
    _remap_container(mesh.GetCellData(), "CellData")

    write_vtp(mesh, output_path)
    print(f"\nSalvo: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# Etapa 5 — Marcar Apex (1 único ponto)
# ════════════════════════════════════════════════════════════════════════════

def _eixo_longo_da_base(pts: np.ndarray, base_mask: np.ndarray):
    """Centróide e normal do plano basal (PCA: direção de menor variância).
    A normal é orientada apontando para o corpo do ventrículo."""
    base_pts = pts[base_mask]
    centroide = base_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(base_pts - centroide, full_matrices=False)
    normal = vt[-1]
    # Orienta a normal na direção da maior parte da malha (apex)
    if np.median((pts - centroide) @ normal) < 0:
        normal = -normal
    return centroide, normal / np.linalg.norm(normal)


def _detectar_apex(pts: np.ndarray,
                   classes: np.ndarray,
                   base_value: int = UVC_BASE,
                   restringir_a=(UVC_EPI,)) -> int:
    """Apex = ponto com MAIOR projeção ao longo da normal do plano basal.

    É mais robusto que 'ponto mais distante do centróide da base', que pode
    escolher um ponto lateral do epicárdio em ventrículos largos/oblíquos.
    """
    base_mask = classes == base_value
    if base_mask.sum() < 3:
        raise RuntimeError(
            f"Menos de 3 pontos com class={base_value} (Base) — impossível "
            f"ajustar o plano basal. Rótulos presentes: "
            f"{np.unique(classes).tolist()}")

    centroide, normal = _eixo_longo_da_base(pts, base_mask)
    proj = (pts - centroide) @ normal

    candidatos = np.zeros(len(pts), dtype=bool)
    for c in restringir_a:
        candidatos |= (classes == c)
    alvo = "Epi"
    if candidatos.sum() == 0:
        candidatos = ~base_mask
        alvo = "qualquer ponto não-basal"
        print(f"  AVISO: classe de restrição ausente; buscando o apex em "
              f"{alvo}.")

    proj_val = np.where(candidatos, proj, -np.inf)
    apex_id = int(np.argmax(proj_val))

    dist_centro = float(np.linalg.norm(pts[apex_id] - centroide))
    print(f"  Plano basal: {int(base_mask.sum())} pontos, centróide "
          f"({centroide[0]:.2f}, {centroide[1]:.2f}, {centroide[2]:.2f})")
    print(f"  Eixo longo (normal): "
          f"({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
    print(f"  Buscando o apex em: {alvo} ({int(candidatos.sum())} candidatos)")
    print(f"  Apex: altura ao longo do eixo = {proj[apex_id]:.2f} mm, "
          f"distância ao centróide = {dist_centro:.2f} mm")
    return apex_id


def _localizar_coluna_point_id(csv_apex: str) -> int:
    with open(csv_apex, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip()

    colunas = [c.strip().strip('"') for c in header.split(",")]
    normalizadas = [c.lower().replace(" ", "").replace("_", "")
                    for c in colunas]

    for cand in ["pointid", "id", "pointids", "index", "idx", "vtkoriginalpointids"]:
        if cand in normalizadas:
            idx = normalizadas.index(cand)
            print(f"  Coluna de Point ID: '{colunas[idx]}' (índice {idx})")
            return idx

    raise ValueError(
        f"Nenhuma coluna de Point ID no cabeçalho de '{csv_apex}'. "
        f"Colunas: {colunas}")


# ── Leitura do arquivo do --apex ────────────────────────────────────────────
# Dois formatos são aceitos:
#   (a) Point ID(s) — índices de pontos da SUPERFÍCIE;
#   (b) coordenada(s) x, y, z — o ponto da malha mais próximo vira o apex.
# (b) é o formato do `apex_coord.txt`: os três números em linhas separadas.

_COLUNAS_XYZ = (("points0", "points1", "points2"),
                ("x", "y", "z"),
                ("coordinatesx", "coordinatesy", "coordinatesz"))

_COLUNAS_ID = ("pointid", "id", "pointids", "index", "idx",
               "vtkoriginalpointids")


def _tokens_numericos(linha: str) -> list:
    """Separa a linha por vírgula, ponto-e-vírgula ou espaço."""
    return [t for t in linha.replace(",", " ").replace(";", " ").split() if t]


def _linha_e_numerica(linha: str) -> bool:
    toks = _tokens_numericos(linha)
    if not toks:
        return False
    try:
        for t in toks:
            float(t)
    except ValueError:
        return False
    return True


def ler_arquivo_apex(path: str, modo: str = "auto"):
    """Lê o arquivo do `--apex`. Retorna ('ids', (N,)) ou ('xyz', (N,3)).

    Formatos reconhecidos:

      • CSV com cabeçalho (exportação do ParaView):
            "Point ID","Points_0","Points_1","Points_2"
            15784,12.482,-3.201,40.117
        → usa a coluna de Point ID; com --apex-mode xyz usa Points_0/1/2.

      • Três números em linhas separadas (apex_coord.txt):
            7.247119544760797e+01
            -7.614557053832483e+01
            -1.020838342354730e+02
        → uma coordenada (x, y, z).

      • Um ponto por linha, "x y z" ou "x,y,z" → N coordenadas.

      • Um inteiro por linha → N Point IDs.

    A ambiguidade real é uma coluna única com exatamente 3 números: se forem
    inteiros não-negativos, `auto` assume Point IDs; caso contrário, uma
    coordenada. Use --apex-mode id|xyz para forçar.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        linhas = [l.strip() for l in f if l.strip()]

    if not linhas:
        raise ValueError(f"'{path}' está vazio.")

    # ── Com cabeçalho: CSV no estilo do ParaView ────────────────────────
    if not _linha_e_numerica(linhas[0]):
        colunas = [c.strip().strip('"') for c in linhas[0].split(",")]
        norm = [c.lower().replace(" ", "").replace("_", "") for c in colunas]
        corpo = [[c.strip().strip('"') for c in l.split(",")]
                 for l in linhas[1:]]
        if not corpo:
            raise ValueError(f"'{path}' tem cabeçalho mas nenhuma linha de "
                             f"dados.")

        col_id = next((norm.index(c) for c in _COLUNAS_ID if c in norm), None)
        cols_xyz = next(([norm.index(c) for c in trio]
                         for trio in _COLUNAS_XYZ
                         if all(c in norm for c in trio)), None)

        if modo == "xyz" or (modo == "auto" and col_id is None):
            if cols_xyz is None:
                raise ValueError(
                    f"Nenhuma coluna de coordenada (Points_0/1/2 ou x/y/z) "
                    f"no cabeçalho de '{path}'. Colunas: {colunas}")
            print(f"  Colunas de coordenada: {[colunas[c] for c in cols_xyz]}")
            return "xyz", np.array([[float(r[c]) for c in cols_xyz]
                                    for r in corpo], dtype=float)

        if col_id is None:
            raise ValueError(
                f"Nenhuma coluna de Point ID no cabeçalho de '{path}'. "
                f"Colunas: {colunas}")
        print(f"  Coluna de Point ID: '{colunas[col_id]}' (índice {col_id})")
        return "ids", np.array([int(round(float(r[col_id]))) for r in corpo],
                               dtype=np.int64)

    # ── Sem cabeçalho: só números ──────────────────────────────────────
    vals = [[float(t) for t in _tokens_numericos(l)] for l in linhas]
    larguras = {len(v) for v in vals}

    if larguras == {3}:
        if modo == "id":
            raise ValueError(
                f"--apex-mode id pedido, mas '{path}' tem 3 números por "
                f"linha (formato de coordenada).")
        return "xyz", np.asarray(vals, dtype=float)

    if larguras == {1}:
        flat = np.array([v[0] for v in vals], dtype=float)
        parecem_ids = bool(np.all(flat >= 0) and
                           np.allclose(flat, np.rint(flat)))
        if modo == "id" or (modo == "auto" and parecem_ids):
            return "ids", np.rint(flat).astype(np.int64)
        if flat.size % 3 != 0:
            raise ValueError(
                f"'{path}' tem {flat.size} número(s) em coluna única — não "
                f"formam coordenadas (x, y, z) completas. Use --apex-mode id "
                f"se forem Point IDs.")
        return "xyz", flat.reshape(-1, 3)

    raise ValueError(
        f"Formato de '{path}' não reconhecido: linhas com "
        f"{sorted(larguras)} número(s). Esperado 1 número por linha "
        f"(Point IDs, ou x/y/z em três linhas) ou 3 por linha (coordenadas).")


def _apex_por_coordenada(pts: np.ndarray,
                         coords: np.ndarray,
                         classes: np.ndarray,
                         base_value: int = UVC_BASE,
                         dist_max: float = APEX_DIST_MAX) -> int:
    """Point ID do ponto da malha mais próximo da coordenada informada.

    A coordenada raramente coincide com um vértice: ela vem de um clique no
    ParaView, de outra malha ou do centróide de uma região. O apex do UVC é
    um VÉRTICE, então a busca é um vizinho mais próximo (KD-tree).

    Pontos da Base são excluídos: o apex nunca está no anel basal, e essa
    exclusão evita que uma coordenada mal posicionada grude no plano da base.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.shape[1] != 3:
        raise ValueError(f"Coordenadas do apex devem ter 3 componentes; "
                         f"recebi shape {coords.shape}.")

    nao_base = classes != base_value
    elegiveis = np.where(nao_base)[0] if nao_base.any() else np.arange(len(pts))
    if not nao_base.any():
        print("  AVISO: todos os pontos estão marcados como Base; buscando "
              "o apex na malha inteira.")

    dists, idx = cKDTree(pts[elegiveis]).query(coords, k=1)
    ids = np.atleast_1d(elegiveis[idx])
    dists = np.atleast_1d(dists)

    for i, (c, d, pid) in enumerate(zip(coords, dists, ids)):
        cl = int(classes[pid])
        print(f"  Coordenada {i}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) "
              f"→ Point ID {int(pid)} a {d:.4f} mm "
              f"(class atual = {cl}, {LABELS_UVC.get(cl, '?')})")

    if len(ids) > 1:
        centroide, normal = _eixo_longo_da_base(pts, classes == base_value)
        k = int(np.argmax((pts[ids] - centroide) @ normal))
        print(f"  AVISO: {len(ids)} coordenadas no arquivo — mantendo apenas "
              f"a mais apical (Point ID {int(ids[k])}).")
    else:
        k = 0

    apex_id, d = int(ids[k]), float(dists[k])

    # Se o vértice mais próximo está longe, a coordenada provavelmente veio
    # de outra malha ou está em outra unidade (cm vs mm).
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if d > dist_max * diag:
        raise ApexCoordenadaForaDaMalha(
            f"o vértice mais próximo da coordenada do apex está a {d:.3f} mm "
            f"({d / diag * 100:.1f}% da diagonal da malha, {diag:.1f} mm). "
            f"Essa coordenada não é desta malha, ou está em outra unidade / "
            f"outro referencial. O ponto escolhido "
            f"({pts[apex_id].round(2).tolist()}) não é o ápice, e o "
            f"abLaplace calculado a partir dele faz o computeRotational "
            f"falhar ao dividir a junção VE-VD.")
    if d > APEX_DIST_AVISO * diag:
        print(f"  AVISO: o vértice mais próximo está a {d:.3f} mm "
              f"({d / diag * 100:.1f}% da diagonal da malha, {diag:.1f} mm). "
              f"Confira se a coordenada é desta malha e na mesma unidade.")
    return apex_id


def step5_define_apex(input_path: str,
                      output_path: str,
                      csv_apex: str = None,
                      classe_apex: int = UVC_APEX,
                      base_value: int = UVC_BASE,
                      keep_cell_class: bool = False,
                      apex_mode: str = "auto",
                      apex_dist_max: float = APEX_DIST_MAX,
                      apex_estrito: bool = False) -> None:
    modo = (f"arquivo '{csv_apex}'" if csv_apex
            else "automático (eixo longo da base)")
    section(f"Etapa 5 — Marcar Apex (class = {classe_apex}) — modo: {modo}")

    mesh = read_vtp(input_path)
    class_array = mesh.GetPointData().GetArray(ARRAY_NAME)
    if class_array is None:
        raise ValueError("Array 'class' não encontrado em PointData.")

    pts = vtk_to_numpy(mesh.GetPoints().GetData())
    classes = vtk_to_numpy(class_array).astype(np.int32).copy()
    n_points = mesh.GetNumberOfPoints()

    if csv_apex:
        tipo, dados = ler_arquivo_apex(csv_apex, modo=apex_mode)
        rotulo = ("coordenadas (x, y, z)" if tipo == "xyz"
                  else "Point ID(s)")
        print(f"  Formato lido: {rotulo} — {len(dados)} entrada(s)")

        if tipo == "xyz":
            try:
                apex_id = _apex_por_coordenada(pts, dados, classes,
                                               base_value=base_value,
                                               dist_max=apex_dist_max)
            except ApexCoordenadaForaDaMalha as erro:
                if apex_estrito:
                    raise
                print(f"\n  ✗ Coordenada do apex REJEITADA: {erro}")
                print(f"  → Ignorando '{csv_apex}' e detectando o ápice "
                      f"automaticamente pelo eixo longo da base.\n")
                apex_id = _detectar_apex(pts, classes, base_value=base_value)
        else:
            cand = dados.astype(int)
            fora = cand[(cand < 0) | (cand >= n_points)]
            cand = cand[(cand >= 0) & (cand < n_points)]
            if len(fora):
                print(f"  AVISO: {len(fora)} Point ID(s) fora da faixa "
                      f"[0, {n_points}) descartado(s): {fora[:5].tolist()}")
            print(f"  Candidatos válidos: {len(cand)}")
            if len(cand) == 0:
                raise ValueError(
                    f"Nenhum Point ID válido em '{csv_apex}'. Os IDs devem "
                    f"ser da SUPERFÍCIE ({n_points} pontos), não do volume.")
            if len(cand) > 1:
                base_mask = classes == base_value
                centroide, normal = _eixo_longo_da_base(pts, base_mask)
                proj = (pts[cand] - centroide) @ normal
                apex_id = int(cand[np.argmax(proj)])
                print(f"  AVISO: {len(cand)} pontos no arquivo — mantendo "
                      f"apenas o mais apical ({apex_id}).")
            else:
                apex_id = int(cand[0])
    else:
        apex_id = _detectar_apex(pts, classes, base_value=base_value)

    classe_anterior = int(classes[apex_id])
    classes[apex_id] = classe_apex
    _set_int_array(mesh.GetPointData(), classes)

    print(f"\n  Point ID marcado como Apex: {apex_id} "
          f"(coordenadas {pts[apex_id].round(3).tolist()})")
    print(f"  Classe anterior desse ponto: {classe_anterior} "
          f"({LABELS_UVC.get(classe_anterior, '?')})")
    print(f"  Pontos alterados: 1")

    if not keep_cell_class:
        mesh.GetCellData().RemoveArray(ARRAY_NAME)
        print("  CellData 'class' removido da saída final "
              "(use --keep-cell-class para mantê-lo).")

    write_mesh(mesh, output_path)
    print(f"Salvo: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(input_mesh, output_vtp, apex_csv, outdir, z_threshold,
                 z_mode, dist, dot, classe_apex, base_value,
                 region_array=None, keep_cell_class=False,
                 keep_original_arrays=False, min_ilha=16,
                 triangular=False, enxuto=False,
                 so_superficie=False, apex_mode="auto",
                 apex_dist_max=APEX_DIST_MAX, apex_estrito=False) -> None:

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_mesh))[0]
    ext = os.path.splitext(input_mesh)[1].lower()

    # A superfície sempre vai para o disco: é o intermediário que você abre no
    # ParaView para conferir antes de confiar nas marcações.
    p0 = os.path.join(outdir, f"{base}_etapa0_superficie.vtp")

    # Os demais intermediários podem ir para um diretório temporário quando
    # --enxuto está ligado (aí só sobram a superfície e a saída final).
    dir_mid = outdir if not enxuto else os.path.join(outdir, "_tmp")
    os.makedirs(dir_mid, exist_ok=True)

    p1 = os.path.join(dir_mid, f"{base}_etapa1_regioes.vtp")
    p2 = os.path.join(dir_mid, f"{base}_etapa2_septum.vtp")
    p3 = os.path.join(dir_mid, f"{base}_etapa3_base.vtp")
    p4 = os.path.join(dir_mid, f"{base}_etapa4_remap.vtp")

    section(f"PIPELINE INICIADO — {VERSION}")
    print(f"Script        : {os.path.abspath(__file__)}")
    print(f"Entrada       : {input_mesh}  ({ext})")
    print(f"Saída final   : {output_vtp}")
    print(f"Array regiões : {region_array or 'auto-detecção'}")
    print(f"Arquivo apex  : {apex_csv or 'nenhum (detecção automática)'}"
          f"{'' if not apex_csv else f'  (--apex-mode {apex_mode})'}")
    print(f"Etapa 3 (Z)   : {'z ' + ('<=' if z_mode == 'below' else '>=') + f' {z_threshold}' if z_threshold is not None else 'desativada'}")
    print(f"Intermediários: {outdir}/")

    # ── Etapa 0 ──────────────────────────────────────────────────────────
    if ext in EXT_VOLUME:
        step0_extrair_superficie(input_mesh, p0, triangular=triangular)
        entrada_etapa1 = p0
    elif ext in EXT_SUPERFICIE:
        section("Etapa 0 — Extrair superfície (IGNORADA)")
        print(f"  A entrada já é uma superfície ({ext}); nada a extrair.")
        if triangular:
            mesh = read_vtp(input_mesh)
            tri = vtk.vtkTriangleFilter()
            tri.SetInputData(mesh)
            tri.PassLinesOff()
            tri.PassVertsOff()
            tri.Update()
            write_vtp(tri.GetOutput(), p0)
            print(f"  Superfície triangulada (--triangular).")
            print(f"Salvo: {p0}")
            entrada_etapa1 = p0
        else:
            entrada_etapa1 = input_mesh
            p0 = None
    else:
        raise ValueError(
            f"Extensão '{ext}' não suportada. Use um .vtu (volume) ou "
            f"um .vtp (superfície).")

    if so_superficie:
        section("PARADA SOLICITADA (--so-superficie)")
        print(f"  • {p0 or input_mesh}")
        return

    # ── Etapas 1–5 ───────────────────────────────────────────────────────
    step1_extrair_regioes(entrada_etapa1, p1, region_array=region_array,
                          keep_original=keep_original_arrays,
                          min_ilha=min_ilha)
    step2_label_septum(p1, p2, dist_thresh=dist, dot_thresh=dot,
                       min_ilha=min_ilha)
    step3_corrigir_class(p2, p3, z_threshold=z_threshold, modo=z_mode,
                         min_ilha=min_ilha)
    step4_remap_class(p3, p4)
    step5_define_apex(p4, output_vtp, csv_apex=apex_csv,
                      classe_apex=classe_apex, base_value=base_value,
                      keep_cell_class=keep_cell_class, apex_mode=apex_mode,
                      apex_dist_max=apex_dist_max,
                      apex_estrito=apex_estrito)

    # ── Verificação final ────────────────────────────────────────────────
    section("Verificação final")
    mesh = read_mesh(output_vtp)
    arr = vtk_to_numpy(mesh.GetPointData().GetArray(ARRAY_NAME))
    _print_dist(arr, LABELS_UVC, "  Distribuição final por PONTO:")

    ok = True
    n_apex = int((arr == classe_apex).sum())
    marca_apex = "✓" if n_apex == 1 else "✗ (esperado exatamente 1)"
    print(f"\n  {'Apex':8s} (class={classe_apex}): {n_apex:6d} pontos  "
          f"{marca_apex}")
    if n_apex != 1:
        ok = False

    for val, nome in ((UVC_BASE, "Base"), (UVC_EPI, "Epi"),
                      (UVC_RV, "RV Endo"), (UVC_LV, "LV Endo"),
                      (UVC_SEPTUM, "Septo")):
        n = int((arr == val).sum())
        marca = "✓" if n > 0 else "✗ (ausente)"
        print(f"  {nome:8s} (class={val}): {n:6d} pontos  {marca}")
        if n == 0:
            ok = False

    fora = sorted(set(int(v) for v in np.unique(arr)) - set(LABELS_UVC))
    if fora:
        print(f"\n  ✗ Rótulos fora do padrão UVC (1–6): {fora}")
        ok = False

    # Pintinhas: grupos conexos minúsculos de pontos com a mesma classe.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a, b = _pares_de_pontos_vizinhos(mesh)
    mesma = arr[a] == arr[b]
    g = coo_matrix((np.ones(int(mesma.sum()), dtype=np.int8),
                    (a[mesma], b[mesma])), shape=(len(arr), len(arr)))
    _, comp = connected_components(g, directed=False)
    tam = np.bincount(comp)
    manchas = (tam[comp] < 4) & (arr != classe_apex)
    n_manchas = len(np.unique(comp[manchas])) if manchas.any() else 0
    print(f"\n  Manchas isoladas (pintinhas): {n_manchas}  "
          f"{'✓' if n_manchas == 0 else '✗ aumente --min-ilha'}")
    if n_manchas:
        u, c = np.unique(arr[manchas], return_counts=True)
        print(f"    classes afetadas: {dict(zip(u.tolist(), c.tolist()))}")
        ok = False

    # ── Limpeza opcional dos intermediários de meio de caminho ───────────
    if enxuto:
        for p in (p1, p2, p3, p4):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(dir_mid)
        except OSError:
            pass

    section("PIPELINE CONCLUÍDO" if ok else "PIPELINE CONCLUÍDO COM AVISOS")
    if p0:
        print(f"  • {p0}  (superfície extraída do volume)")
    if not enxuto:
        for p in (p1, p2, p3, p4):
            print(f"  • {p}")
    print(f"  • {output_vtp}  (saída final, 6 marcações UVC)")
    if not ok:
        print("\n  Revise os avisos acima antes de rodar o Cobiveco.")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline unificado: .vtu → superfície .vtp → "
                    "region_id→class, septo, base, remap UVC e apex.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"pipeline_UVC_completo.py {VERSION}")
    parser.add_argument("input",
                        help="Malha de entrada: .vtu (volume, roda a etapa 0) "
                             "ou .vtp (superfície, pula a etapa 0)")
    parser.add_argument("output",
                        help="Arquivo de saída: .vtp (PolyData, padrão) ou "
                             ".vtu (a mesma superfície como UnstructuredGrid)")
    parser.add_argument("--region-array", default=None,
                        help="Nome do array de regiões na entrada "
                             "(padrão: auto-detecta region_id, elemTag, ...)")
    parser.add_argument("--apex", default=None,
                        help="Arquivo com o apex manual: Point ID(s) ou "
                             "coordenada(s) x,y,z (formato auto-detectado)")
    parser.add_argument("--apex-mode", choices=["auto", "id", "xyz"],
                        default="auto",
                        help="Como interpretar o arquivo do --apex: 'id' = "
                             "Point IDs, 'xyz' = coordenadas. 'auto' decide "
                             "pelo conteúdo")
    parser.add_argument("--apex-dist-max", type=float, default=APEX_DIST_MAX,
                        help="Distância máxima (fração da diagonal da malha) "
                             "entre a coordenada do --apex e o vértice mais "
                             "próximo. Acima disso a coordenada é rejeitada "
                             "e o ápice é detectado automaticamente")
    parser.add_argument("--apex-estrito", action="store_true",
                        help="Aborta (em vez de cair para a detecção "
                             "automática) quando a coordenada do --apex está "
                             "fora da malha")
    parser.add_argument("--base-value", type=int, default=UVC_BASE,
                        help="Valor UVC da Base (referência do apex)")
    parser.add_argument("--outdir", default="intermediarios",
                        help="Diretório dos VTP intermediários")
    parser.add_argument("--triangular", action="store_true",
                        help="Triangula a superfície logo após a etapa 0")
    parser.add_argument("--so-superficie", action="store_true",
                        help="Roda apenas a etapa 0 (.vtu → .vtp) e para")
    parser.add_argument("--enxuto", action="store_true",
                        help="Grava apenas a superfície (etapa 0) e a saída "
                             "final; apaga os intermediários das etapas 1–4")
    parser.add_argument("--z-threshold", type=float, default=None,
                        help="Ativa a etapa 3: reclassifica como Base os "
                             "pontos de um lado deste plano em Z. "
                             "Padrão: desativado")
    parser.add_argument("--z-mode", choices=["below", "above"], default="below",
                        help="Lado do plano que vira Base na etapa 3")
    parser.add_argument("--dist", type=float, default=15.0,
                        help="Distância máxima (mm) LV→RV para o septo")
    parser.add_argument("--dot", type=float, default=-0.8,
                        help="Limiar do produto interno normal·dir_LV→RV")
    parser.add_argument("--classe-apex", type=int, default=UVC_APEX,
                        help="Valor de 'class' do ponto do apex")
    parser.add_argument("--min-ilha", type=int, default=16,
                        help="Tamanho mínimo (em células) de uma região "
                             "conexa. Ilhas menores são reabsorvidas pela "
                             "vizinhança — é isto que elimina as 'pintinhas' "
                             "no ParaView. Use 0 ou 1 para desativar.")
    parser.add_argument("--keep-cell-class", action="store_true",
                        help="Mantém 'class' também em CellData na saída")
    parser.add_argument("--keep-original-arrays", action="store_true",
                        help="Preserva fenotipo/tecido/f_0/s_0/n_0")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Erro: arquivo de entrada não encontrado: {args.input}")
    ext = os.path.splitext(args.input)[1].lower()
    if ext not in EXT_VOLUME | EXT_SUPERFICIE:
        sys.exit(f"Erro: extensão '{ext}' não suportada. Use .vtu ou .vtp.")
    ext_out = os.path.splitext(args.output)[1].lower()
    if ext_out not in EXT_VOLUME | EXT_SUPERFICIE:
        sys.exit(f"Erro: extensão de saída '{ext_out}' não suportada. "
                 f"Use .vtp (PolyData) ou .vtu (UnstructuredGrid).")
    if args.apex is not None and not os.path.isfile(args.apex):
        sys.exit(f"Erro: arquivo do apex não encontrado: {args.apex}")

    run_pipeline(
        input_mesh           = args.input,
        output_vtp           = args.output,
        apex_csv             = args.apex,
        outdir               = args.outdir,
        z_threshold          = args.z_threshold,
        z_mode               = args.z_mode,
        dist                 = args.dist,
        dot                  = args.dot,
        classe_apex          = args.classe_apex,
        base_value           = args.base_value,
        region_array         = args.region_array,
        keep_cell_class      = args.keep_cell_class,
        keep_original_arrays = args.keep_original_arrays,
        min_ilha             = args.min_ilha,
        triangular           = args.triangular,
        enxuto               = args.enxuto,
        so_superficie        = args.so_superficie,
        apex_mode            = args.apex_mode,
        apex_dist_max        = args.apex_dist_max,
        apex_estrito         = args.apex_estrito,
    )


if __name__ == "__main__":
    main()
