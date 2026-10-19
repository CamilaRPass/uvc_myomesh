"""
uvc_bridge.py
-------------
Ponte entre o `pipeline_UVC_completo.py` (marcação da superfície) e o
pacote `cobiveco/` (cálculo das coordenadas UVC).

Dependências: vtk, numpy, scipy
"""

import os
import shutil

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree


# ════════════════════════════════════════════════════════════════════════════
# Convenção Cobiveco
# ════════════════════════════════════════════════════════════════════════════

ARRAY_NAME = "class"

CLS_BASE, CLS_EPI, CLS_LV, CLS_RV, CLS_APEX, CLS_SEPTO = 1, 2, 3, 4, 5, 6

LABELS_COBIVECO = {
    CLS_BASE:  "Base",
    CLS_EPI:   "Epi",
    CLS_LV:    "Endo tv=0",
    CLS_RV:    "Endo tv=1",
    CLS_APEX:  "Apex",
    CLS_SEPTO: "Septo (face VD)",
}

VTK_TETRA = 10


def secao(titulo: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {titulo}")
    print("═" * 72)


# ════════════════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════════════════

def read_vtp(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    m = r.GetOutput()
    if m is None or m.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Não consegui ler pontos de '{path}'.")
    return m


def read_vtu(path: str) -> vtk.vtkUnstructuredGrid:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    m = r.GetOutput()
    if m is None or m.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Não consegui ler pontos de '{path}'.")
    return m


def write_vtp(mesh: vtk.vtkPolyData, path: str) -> None:
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(mesh)
    w.SetCompressorTypeToZLib()
    w.Write()


def write_vtu(mesh: vtk.vtkUnstructuredGrid, path: str) -> None:
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(path)
    w.SetInputData(mesh)
    w.SetCompressorTypeToZLib()
    w.Write()


# ════════════════════════════════════════════════════════════════════════════
# Acesso a pontos / classes
# ════════════════════════════════════════════════════════════════════════════

def pontos(mesh) -> np.ndarray:
    return vtk_to_numpy(mesh.GetPoints().GetData())


def get_classes(mesh: vtk.vtkPolyData) -> np.ndarray:
    arr = mesh.GetPointData().GetArray(ARRAY_NAME)
    if arr is None:
        nomes = [mesh.GetPointData().GetArrayName(i)
                 for i in range(mesh.GetPointData().GetNumberOfArrays())]
        raise RuntimeError(
            f"Array '{ARRAY_NAME}' ausente em PointData. Arrays: {nomes}. "
            f"Rode o pipeline_UVC_completo.py antes desta etapa.")
    return vtk_to_numpy(arr).astype(np.int32).copy()


def set_classes(mesh: vtk.vtkPolyData, classes: np.ndarray) -> None:
    arr = numpy_to_vtk(np.ascontiguousarray(classes, dtype=np.int32),
                       deep=True, array_type=vtk.VTK_INT)
    arr.SetName(ARRAY_NAME)
    pd = mesh.GetPointData()
    pd.RemoveArray(ARRAY_NAME)
    pd.AddArray(arr)
    pd.SetActiveScalars(ARRAY_NAME)


def distribuicao(classes: np.ndarray, labels: dict, titulo: str) -> None:
    uniq, counts = np.unique(classes, return_counts=True)
    print(titulo)
    for v, c in zip(uniq, counts):
        print(f"    class={int(v):3d}  ({labels.get(int(v), '?'):16s})"
              f"  {c:8d}  ({c / classes.size * 100:5.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
# Arestas da superfície (para vizinhança / comprimento médio)
# ════════════════════════════════════════════════════════════════════════════

def arestas(mesh: vtk.vtkPolyData) -> np.ndarray:
    """Arestas únicas (p0, p1) das células poligonais."""
    polys = mesh.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        raise RuntimeError("A superfície não tem células poligonais.")
    conn = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
    offs = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
    sizes = np.diff(offs)
    cell_of = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    starts = offs[:-1]
    pos = np.arange(len(conn), dtype=np.int64) - starts[cell_of]
    prox = starts[cell_of] + (pos + 1) % sizes[cell_of]
    e = np.sort(np.stack([conn, conn[prox]], axis=1), axis=1)
    return np.unique(e, axis=0)


def comprimento_medio_aresta(mesh: vtk.vtkPolyData) -> float:
    e = arestas(mesh)
    P = pontos(mesh)
    return float(np.mean(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)))


# ════════════════════════════════════════════════════════════════════════════
# Etapa A — Permutação de rótulos para a convenção Cobiveco
# ════════════════════════════════════════════════════════════════════════════

def _papel(nome: str):
    """Deduz o papel anatômico a partir do rótulo textual do pipeline."""
    n = nome.lower()
    if "base" in n:
        return CLS_BASE
    if "sept" in n:
        return CLS_SEPTO
    if "apex" in n or "ápice" in n or "apice" in n:
        return CLS_APEX
    if "epi" in n:
        return CLS_EPI
    if "lv" in n or "ve " in n or n.strip() in ("ve", "vesq"):
        return CLS_LV
    if "rv" in n or "vd " in n or n.strip() in ("vd", "vdir"):
        return CLS_RV
    return None


def permutacao_para_cobiveco(labels_origem: dict) -> dict:
    """Permutação {valor_no_pipeline: valor_cobiveco} deduzida do LABELS_UVC.

    Com o LABELS_UVC atual do pipeline
    ({1:'Base', 2:'Epi', 3:'RV Endo', 4:'LV Endo', 5:'Apex', 6:'Septo'})
    isto devolve {1:1, 2:2, 3:4, 4:3, 5:5, 6:6} — ou seja, troca 3↔4.

    Se você corrigir o REMAP/LABELS_UVC do pipeline para a convenção
    Cobiveco, esta função passa a devolver a identidade sozinha.
    """
    perm = {}
    for valor, nome in labels_origem.items():
        alvo = _papel(str(nome))
        if alvo is None:
            raise ValueError(
                f"Não consegui deduzir o papel anatômico do rótulo "
                f"'{nome}' (class={valor}). Passe a permutação à mão.")
        perm[int(valor)] = int(alvo)

    if sorted(perm.values()) != sorted(set(perm.values())):
        raise ValueError(f"Permutação ambígua (dois rótulos com o mesmo "
                         f"papel): {perm}")
    return perm


def aplicar_permutacao(classes: np.ndarray, perm: dict) -> np.ndarray:
    out = classes.copy()
    for origem, destino in perm.items():
        out[classes == origem] = destino
    return out


# ════════════════════════════════════════════════════════════════════════════
# Diagnóstico: qual endocárdio é o ventrículo esquerdo?
# ════════════════════════════════════════════════════════════════════════════

def diagnosticar_ventriculos(mesh: vtk.vtkPolyData,
                             classes: np.ndarray) -> dict:
    """Relata, pela geometria, qual das classes 3/4 parece ser o VE.

    Critério: espessura de parede. A distância mediana de cada endocárdio ao
    epicárdio é bem maior no VE (parede compacta, ~10 mm) do que no VD
    (parede livre fina, ~3–5 mm). É um indicativo que NÃO depende dos
    rótulos textuais nem da convenção de quem gerou a malha.

    Esta função só informa — não altera nada.
    """
    P = pontos(mesh)
    epi = P[classes == CLS_EPI]
    if len(epi) == 0:
        print("  (sem epicárdio marcado; não dá para estimar espessuras)")
        return {}

    tree = cKDTree(epi)
    esp = {}
    print("\n  Espessura de parede (endocárdio → epicárdio):")
    for v in (CLS_LV, CLS_RV):
        pts = P[classes == v]
        if len(pts) == 0:
            continue
        d = tree.query(pts)[0]
        esp[v] = float(np.median(d))
        print(f"    class {v}: {len(pts):6d} pontos   "
              f"mediana {esp[v]:6.2f} mm   "
              f"p25 {np.percentile(d, 25):6.2f}   "
              f"p75 {np.percentile(d, 75):6.2f}")

    if len(esp) < 2:
        return {"ve_geometrico": None, "espessuras": esp}

    ve = max(esp, key=esp.get)
    razao = esp[ve] / max(min(esp.values()), 1e-9)
    print(f"  → Pela espessura, o VE anatômico parece ser a class {ve} "
          f"(parede {razao:.1f}× mais grossa).")
    if razao < 1.3:
        print("    AVISO: as duas espessuras são parecidas; o indicativo é "
              "fraco nesta malha.")
    print(f"  → O cobiveco vai tratar class {CLS_LV} como o ventrículo de "
          f"tv=0 e class {CLS_RV} como o de tv=1,")
    print(f"    e resolve o rt completo [-π, π] só no lado tv=0 "
          f"(o lado tv=1 recebe apenas ±π/2).")
    if ve != CLS_LV:
        print(f"    OBS: pela geometria o VE é a class {ve}, então o rt "
              f"detalhado vai sair no VD. Se quiser o contrário, use "
              f"--permutar rotulos.")
    return {"ve_geometrico": ve, "espessuras": esp}


# ════════════════════════════════════════════════════════════════════════════
# Etapa B — Lado do septo (face VE → face VD)
# ════════════════════════════════════════════════════════════════════════════

def diagnosticar_septo(mesh: vtk.vtkPolyData, classes: np.ndarray,
                       meanEdg: float) -> dict:
    """De que lado do septo está a class 6?

    O critério é TOPOLÓGICO, não métrico: o septo é sempre um pedaço
    recortado de um dos dois endocárdios, então a borda do recorte encosta,
    aresta a aresta, exatamente na classe de onde ele saiu. Se as arestas
    6–3 dominam, o septo foi recortado do endocárdio do VE (face VE); se as
    6–4 dominam, do endocárdio do VD (face VD, que é o que o Cobiveco quer).

    Um critério por distância não serve aqui: a distância do septo ao outro
    endocárdio é a espessura septal, que na mesma malha é da ordem de duas
    ou três arestas — perto demais de qualquer limiar fixo.
    """
    P = pontos(mesh)
    sep = P[classes == CLS_SEPTO]
    rv = P[classes == CLS_RV]
    lv = P[classes == CLS_LV]

    if len(sep) == 0:
        raise RuntimeError("Nenhum ponto com class=6 (septo) na superfície.")
    if len(rv) == 0 or len(lv) == 0:
        raise RuntimeError("LV Endo ou RV Endo ausente — impossível "
                           "diagnosticar o lado do septo.")

    # --- contagem de arestas de fronteira do septo ---------------------
    e = arestas(mesh)
    ca, cb = classes[e[:, 0]], classes[e[:, 1]]
    n_sep_lv = int(np.sum(((ca == CLS_SEPTO) & (cb == CLS_LV)) |
                          ((cb == CLS_SEPTO) & (ca == CLS_LV))))
    n_sep_rv = int(np.sum(((ca == CLS_SEPTO) & (cb == CLS_RV)) |
                          ((cb == CLS_SEPTO) & (ca == CLS_RV))))

    # --- distâncias, só como diagnóstico ------------------------------
    med_rv = float(np.median(cKDTree(rv).query(sep)[0]))
    med_lv = float(np.median(cKDTree(lv).query(sep)[0]))

    if n_sep_lv == 0 and n_sep_rv == 0:
        face = "rv" if med_rv < med_lv else "lv"
        criterio = "distância (o septo não faz fronteira com nenhum endocárdio)"
    else:
        face = "rv" if n_sep_rv > n_sep_lv else "lv"
        criterio = "arestas de fronteira"

    print(f"  Pontos de septo (class 6): {len(sep)}")
    print(f"  Arestas de fronteira septo–class {CLS_LV} (tv=0): {n_sep_lv:6d}")
    print(f"  Arestas de fronteira septo–class {CLS_RV} (tv=1): {n_sep_rv:6d}")
    print(f"  Distância mediana septo → class {CLS_RV}: {med_rv:7.3f} mm")
    print(f"  Distância mediana septo → class {CLS_LV}: {med_lv:7.3f} mm")
    print(f"  Aresta média da superfície        : {meanEdg:7.3f} mm")
    print(f"  → Critério: {criterio}.")
    print(f"  → O septo foi recortado da class "
          f"{CLS_RV if face == 'rv' else CLS_LV}"
          f"{' — é o lado tv=1, que é o que o cobiveco espera' if face == 'rv' else ' — o lado tv=0'}.")

    return {"face": face, "med_rv": med_rv, "med_lv": med_lv,
            "n_sep_lv": n_sep_lv, "n_sep_rv": n_sep_rv,
            "espessura_estimada": med_rv}


def _suavizar_fronteira(classes: np.ndarray, e: np.ndarray,
                        alvo=(CLS_RV, CLS_SEPTO), passes: int = 2):
    """Voto de maioria entre duas classes vizinhas — tira ilhas da fronteira."""
    a, b = e[:, 0], e[:, 1]
    ea = np.concatenate([a, b])
    eb = np.concatenate([b, a])
    cls = classes.copy()
    c1, c2 = alvo
    for _ in range(passes):
        mutavel = np.isin(cls, alvo)
        v1 = np.zeros(len(cls))
        v2 = np.zeros(len(cls))
        np.add.at(v1, ea, (cls[eb] == c1).astype(float))
        np.add.at(v2, ea, (cls[eb] == c2).astype(float))
        novo = cls.copy()
        tem_voto = (v1 + v2) > 0
        novo[mutavel & tem_voto & (v2 > v1)] = c2
        novo[mutavel & tem_voto & (v1 > v2)] = c1
        if np.array_equal(novo, cls):
            break
        cls = novo
    return cls


def mover_septo_para_face_rv(mesh: vtk.vtkPolyData,
                             classes: np.ndarray,
                             espessura: float = None,
                             fator: float = 1.4,
                             meanEdg: float = None) -> np.ndarray:
    """Reclassifica o septo: face VE (class 6) → face VD (class 6).

    Estratégia: os pontos do endocárdio do VD que estão a menos de uma
    espessura septal do septo-VE atual passam a ser o septo; o septo-VE
    antigo volta a ser endocárdio do VE.

    A espessura é estimada pela mediana da distância septo-VE → VD Endo
    (que é justamente atravessar o septo). A parede livre do VD fica muito
    mais longe — atravessando a cavidade —, daí a separação funcionar.
    """
    P = pontos(mesh)
    if meanEdg is None:
        meanEdg = comprimento_medio_aresta(mesh)

    sep_lv = classes == CLS_SEPTO
    rv_mask = classes == CLS_RV
    if not sep_lv.any():
        raise RuntimeError("Nenhum ponto com class=6 para mover.")
    if not rv_mask.any():
        raise RuntimeError("Nenhum ponto de RV Endo (class=4).")

    idx_rv = np.where(rv_mask)[0]
    d_rv = cKDTree(P[sep_lv]).query(P[idx_rv])[0]

    if espessura is None:
        espessura = float(np.median(cKDTree(P[rv_mask]).query(P[sep_lv])[0]))
    limiar = fator * espessura + 2.0 * meanEdg

    q = np.percentile(d_rv, [5, 25, 50, 75, 95])
    print(f"  Espessura septal estimada : {espessura:.3f} mm")
    print(f"  Limiar de captura          : {limiar:.3f} mm "
          f"({fator} × espessura + 2 × aresta)")
    print(f"  Distância RV Endo → septo-VE, percentis "
          f"[5,25,50,75,95]: {np.round(q, 2).tolist()} mm")

    novos = idx_rv[d_rv <= limiar]
    if len(novos) == 0:
        raise RuntimeError(
            f"Nenhum ponto do VD ficou a menos de {limiar:.2f} mm do "
            f"septo-VE. Ajuste --septo-espessura.")

    out = classes.copy()
    out[sep_lv] = CLS_LV          # septo-VE volta a ser endocárdio do VE
    out[novos] = CLS_SEPTO        # face VD do septo

    out = _suavizar_fronteira(out, arestas(mesh))

    frac = (out == CLS_SEPTO).sum() / max(int(rv_mask.sum()), 1) * 100
    print(f"  Pontos do VD marcados como septo: "
          f"{int((out == CLS_SEPTO).sum())} ({frac:.1f}% do RV Endo)")
    if not (15.0 <= frac <= 75.0):
        print(f"  AVISO: fração septal atípica no VD ({frac:.1f}%). "
              f"Esperado ~30–60%. Confira no ParaView antes de seguir.")
    return out


# ════════════════════════════════════════════════════════════════════════════
# Etapa C — Validação do contrato com o cobiveco
# ════════════════════════════════════════════════════════════════════════════

def extrair_superficie_como_cobiveco(vol: vtk.vtkUnstructuredGrid):
    """Mesma extração que o vu.surface_filter (vtkDataSetSurfaceFilter)."""
    f = vtk.vtkDataSetSurfaceFilter()
    f.SetInputData(vol)
    f.PassThroughPointIdsOn()
    f.Update()
    return f.GetOutput()


def _apex_na_ponta(P: np.ndarray, classes: np.ndarray,
                   minimo: float = 0.75) -> bool:
    """O ponto class=5 está no terço apical, medido ao longo do eixo longo?

    O eixo longo é a normal do plano basal (PCA dos pontos class=1),
    orientada para o corpo do ventrículo. A altura do apex é normalizada
    pela altura máxima da malha nesse eixo: 1.0 = ponto mais apical de
    todos, 0.0 = no plano da base.
    """
    base = P[classes == CLS_BASE]
    apex = P[classes == CLS_APEX]
    if len(base) < 3 or len(apex) == 0:
        return True   # outras checagens já reclamaram

    centroide = base.mean(axis=0)
    _, _, vt = np.linalg.svd(base - centroide, full_matrices=False)
    eixo = vt[-1]
    proj = (P - centroide) @ eixo
    if np.median(proj) < 0:
        eixo, proj = -eixo, -proj

    alt_max = float(proj.max())
    if alt_max <= 1e-9:
        return True
    rel = float(((apex[0] - centroide) @ eixo) / alt_max)

    print(f"\n  Altura do apex ao longo do eixo longo: {rel * 100:.1f}% "
          f"(1.0 = ponto mais apical da malha)")
    if rel >= minimo:
        print(f"  ✓ Apex no terço apical.")
        return True

    print(f"  ✗ O ponto marcado como apex (class=5) está a apenas "
          f"{rel * 100:.1f}% da altura — não é o ápice.")
    print(f"    O abLaplace sai torto e o computeRotational falha em "
          f"'Não foi possível dividir a junção por abLaplace'.")
    print(f"    Rode sem o arquivo de apex (--sem-apex) ou corrija a "
          f"coordenada.")
    return False


def validar_contrato(vtu_path: str, vtp_path: str) -> bool:
    """Reproduz o que o prepareMesh vai fazer e checa se sobrevive."""
    secao("Validação do contrato vtu ↔ vtp ↔ cobiveco")
    ok = True

    vol = read_vtu(vtu_path)
    sur = read_vtp(vtp_path)
    classes = get_classes(sur)

    print(f"  Volume     : {vol.GetNumberOfPoints()} pontos, "
          f"{vol.GetNumberOfCells()} células")
    print(f"  Superfície : {sur.GetNumberOfPoints()} pontos, "
          f"{sur.GetNumberOfCells()} células")

    # 1) o cotmatrix só sabe tetraedros
    tipos = np.unique(vtk_to_numpy(vol.GetCellTypesArray()))
    if not np.array_equal(tipos, np.array([VTK_TETRA])):
        print(f"  ✗ O .vtu tem tipos de célula {tipos.tolist()}; o "
              f"cotmatrix() do cobiveco exige apenas tetraedros "
              f"(VTK_TETRA=10).")
        ok = False
    else:
        print(f"  ✓ Volume puramente tetraédrico.")

    # 2) o mapeamento por vizinho mais próximo do prepareMesh
    bv_sur = extrair_superficie_como_cobiveco(vol)
    P_bv = pontos(bv_sur)
    d, idx = cKDTree(pontos(sur)).query(P_bv)
    cls_map = classes[idx]

    diag = float(np.linalg.norm(P_bv.max(axis=0) - P_bv.min(axis=0)))
    print(f"  Superfície reextraída do .vtu: {bv_sur.GetNumberOfPoints()} "
          f"pontos")
    print(f"  Erro do mapeamento vizinho-mais-próximo: "
          f"máx {d.max():.4f} mm, mediana {np.median(d):.4f} mm "
          f"(diagonal da malha {diag:.1f} mm)")
    if d.max() > 0.01 * diag:
        print(f"  ✗ O .vtp e o .vtu não são da mesma geometria (ou estão em "
              f"unidades diferentes).")
        ok = False
    else:
        print(f"  ✓ Geometrias coincidem.")

    # 3) classes depois do mapeamento — é isto que o cobiveco vai enxergar
    distribuicao(cls_map, LABELS_COBIVECO,
                 "\n  Distribuição de 'class' COMO O COBIVECO VAI LER:")

    for v, nome in LABELS_COBIVECO.items():
        n = int((cls_map == v).sum())
        if n == 0:
            print(f"  ✗ class={v} ({nome}) ausente após o mapeamento.")
            ok = False

    n_apex = int((cls_map == CLS_APEX).sum())
    if n_apex == 0:
        print("  ✗ O apex NÃO sobreviveu ao mapeamento por vizinho mais "
              "próximo → computeApicobasal vai quebrar em "
              "`s2v[cls == 5][0]` (IndexError).")
        ok = False
    else:
        # o cobiveco usa [0], então >1 não quebra, mas vale saber
        print(f"  {'✓' if n_apex == 1 else '~'} Apex: {n_apex} ponto(s) "
              f"após o mapeamento{'' if n_apex == 1 else ' (o cobiveco usa o primeiro)'}.")

    fora = sorted(set(int(v) for v in np.unique(cls_map)) - set(LABELS_COBIVECO))
    if fora:
        print(f"  ✗ Rótulos fora de 1–6: {fora}")
        ok = False

    # 4) o apex está mesmo na ponta?
    #    O abLaplace é um campo que vale 0 no ápice e 1 na base; se o ponto
    #    class=5 não estiver no terço apical, esse campo não acompanha o eixo
    #    longo e o computeRotational não consegue partir a junção VE-VD
    #    ("Não foi possível dividir a junção por abLaplace").
    if not _apex_na_ponta(P_bv, cls_map):
        ok = False

    if ok:
        print("\n  ✓ Contrato OK.")
    else:
        print("\n  ✗ Contrato COM PROBLEMAS — resolva os itens acima antes "
              "de rodar o UVC.")
    return ok


# ════════════════════════════════════════════════════════════════════════════
# Etapa D — Montar o layout de arquivos que o prepareMesh espera
# ════════════════════════════════════════════════════════════════════════════

def preparar_entrada_cobiveco(vtp_marcado: str,
                              vtu_original: str,
                              destino: str,
                              nome: str = "heart",
                              labels_origem: dict = None,
                              permutar: bool = False,
                              permutacao: dict = None,
                              septo_face: str = "auto",
                              septo_espessura: float = None,
                              septo_fator: float = 1.4) -> str:
    """Converte a saída do pipeline na entrada do cobiveco.

    Retorna o `inPrefix` pronto para `UVC(inPrefix=...)`.
    """
    secao("Ponte pipeline → cobiveco")

    os.makedirs(destino, exist_ok=True)
    mesh = read_vtp(vtp_marcado)
    classes = get_classes(mesh)
    meanEdg = comprimento_medio_aresta(mesh)

    # ── A) permutação de rótulos ────────────────────────────────────────
    if permutacao is None and permutar:
        if labels_origem is None:
            raise ValueError("Passe `labels_origem` (o LABELS_UVC do "
                             "pipeline) ou uma `permutacao` explícita.")
        permutacao = permutacao_para_cobiveco(labels_origem)

    if permutacao and permutar:
        identidade = all(k == v for k, v in permutacao.items())
        print(f"\n  Permutação pipeline → cobiveco: {permutacao}"
              f"{'  (identidade)' if identidade else ''}")
        if not identidade:
            for k, v in sorted(permutacao.items()):
                if k != v:
                    print(f"    class {k} ({labels_origem.get(k, '?')}) "
                          f"→ class {v} ({LABELS_COBIVECO[v]})")
        classes = aplicar_permutacao(classes, permutacao)
    else:
        print("\n  Permutação de rótulos: DESATIVADA — as classes vão para o "
              "cobiveco exatamente como o pipeline as produziu.")

    distribuicao(classes, LABELS_COBIVECO,
                 "\n  Classes entregues ao cobiveco:")
    diagnosticar_ventriculos(mesh, classes)

    # ── B) lado do septo ────────────────────────────────────────────────
    print()
    diag = diagnosticar_septo(mesh, classes, meanEdg)
    # 'auto' = deixar na convenção Cobiveco (face VD); o diagnóstico só diz
    # se há trabalho a fazer.
    alvo = "rv" if septo_face == "auto" else septo_face

    if diag["face"] == "rv":
        if alvo == "rv":
            print("\n  Septo já está na face VD — nada a fazer.")
        else:
            print("\n  AVISO: --septo-face lv pedido, mas o septo já está na "
                  "face VD; mover de volta não está implementado. Mantido "
                  "como está.")
    elif alvo == "rv":
        print("\n  Movendo o septo da face VE para a face VD...")
        classes = mover_septo_para_face_rv(
            mesh, classes, espessura=septo_espessura, meanEdg=meanEdg,
            fator=septo_fator)
        distribuicao(classes, LABELS_COBIVECO,
                     "\n  Distribuição após mover o septo:")
    else:
        print("\n  AVISO: o septo ficará na face VE (--septo-face lv). O "
              "computeTransmural do VE trata class 6 como epicárdio (tm=1), "
              "então o tm sai invertido na região septal.")

    # ── C) layout de arquivos ───────────────────────────────────────────
    vtu_destino = os.path.join(destino, f"{nome}.vtu")
    vtp_destino = os.path.join(destino, f"{nome}_for_UVC.vtp")

    set_classes(mesh, classes)
    write_vtp(mesh, vtp_destino)

    if os.path.abspath(vtu_original) != os.path.abspath(vtu_destino):
        shutil.copyfile(vtu_original, vtu_destino)

    print(f"\n  Gravado: {vtu_destino}")
    print(f"  Gravado: {vtp_destino}")

    inPrefix = os.path.join(destino, f"{nome}_for_UVC")
    print(f"  inPrefix para o UVC: '{inPrefix}'")
    return inPrefix


# ════════════════════════════════════════════════════════════════════════════
# Etapa E — Anexar as UVC ao .vtu de ENTRADA (saída final)
# ════════════════════════════════════════════════════════════════════════════

UVC_CAMPOS = ("tv", "tm", "ab", "rt", "rtSin", "rtCos",
              "aha17", "aha17Coronary")


def anexar_uvc_ao_vtu(vtu_original: str,
                      campos: dict,
                      saida: str,
                      pontos_result: np.ndarray = None,
                      prefixo: str = "") -> str:
    """Grava uma cópia do .vtu de ENTRADA com as coordenadas UVC anexadas.

    Nada do arquivo original se perde: geometria, conectividade e todos os
    arrays (fibrose, LAT, f_0/s_0/n_0, region_id, PointData e CellData)
    continuam onde estavam; as UVC entram como PointData novo ao lado deles.

    A correspondência é por ÍNDICE, e isso é legítimo: o `prepareMesh` faz
    `vu.read(vtu)` direto no mesmo arquivo e nunca reordena pontos, então
    `result` tem a ordem do original. Passando `pontos_result`, a função
    confere isso pelo desvio máximo das coordenadas e cai para vizinho mais
    próximo se algo não bater.
    """
    secao("Anexando as UVC ao .vtu de entrada")

    orig = read_vtu(vtu_original)
    n = orig.GetNumberOfPoints()
    P = pontos(orig)
    print(f"  Entrada: {vtu_original}  ({n} pontos, "
          f"{orig.GetNumberOfCells()} células)")

    tamanhos = {k: int(np.asarray(v).size) for k, v in campos.items()}
    idx = None

    if all(t == n for t in tamanhos.values()):
        if pontos_result is not None:
            Pr = np.asarray(pontos_result, dtype=float)
            desvio = float(np.abs(Pr - P).max())
            if desvio <= 1e-6:
                print(f"  Correspondência por índice confirmada "
                      f"(desvio máx das coordenadas: {desvio:.2e} mm).")
            else:
                print(f"  AVISO: as coordenadas não batem ponto a ponto "
                      f"(desvio máx {desvio:.3f} mm); usando vizinho mais "
                      f"próximo.")
                idx = cKDTree(Pr).query(P)[1]
        else:
            print("  Correspondência por índice (mesma contagem de pontos).")
    else:
        if pontos_result is None:
            raise RuntimeError(
                f"Os campos UVC têm tamanhos {tamanhos} mas o .vtu tem {n} "
                f"pontos, e não recebi `pontos_result` para remapear.")
        Pr = np.asarray(pontos_result, dtype=float)
        print(f"  AVISO: o resultado tem {len(Pr)} pontos e o .vtu de entrada "
              f"tem {n}; usando vizinho mais próximo.")
        idx = cKDTree(Pr).query(P)[1]

    pd = orig.GetPointData()
    substituidos, escritos = [], []
    for nome, valores in campos.items():
        v = np.asarray(valores).ravel()
        if idx is not None:
            v = v[idx]
        alvo = f"{prefixo}{nome}"
        if pd.GetArray(alvo) is not None:
            substituidos.append(alvo)
            pd.RemoveArray(alvo)
        arr = numpy_to_vtk(np.ascontiguousarray(v, dtype=np.float32),
                           deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName(alvo)
        pd.AddArray(arr)
        escritos.append(alvo)
        finito = int(np.sum(~np.isfinite(v)))
        print(f"    {alvo:8s} [{np.nanmin(v):+.3f}, {np.nanmax(v):+.3f}]"
              f"{'   AVISO: ' + str(finito) + ' valor(es) não finito(s)' if finito else ''}")

    if substituidos:
        print(f"  AVISO: arrays sobrescritos no arquivo de entrada: "
              f"{substituidos}")

    preservados_pd = [n_ for n_ in _nomes_arrays_vtk(pd) if n_ not in escritos]
    preservados_cd = _nomes_arrays_vtk(orig.GetCellData())
    print(f"  PointData preservado: {preservados_pd or 'nenhum'}")
    print(f"  CellData  preservado: {preservados_cd or 'nenhum'}")

    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    write_vtu(orig, saida)
    print(f"\n  Gravado: {saida}")
    return saida


def _nomes_arrays_vtk(container) -> list:
    return [container.GetArrayName(i)
            for i in range(container.GetNumberOfArrays())
            if container.GetArrayName(i)]


def copiar_arrays_originais(alvo_path: str, vtu_original: str,
                            rotulo: str = "result.vtu") -> None:
    """Devolve a um .vtu produzido pelo cobiveco todas as tags do .vtu de
    entrada.

    O `prepareMesh` faz `delete_data_arrays` no volume, então o `result.vtu`
    sai só com as UVC — sem region_id/elemTag, fenótipo, tecido, fibrose,
    f_0/s_0/n_0. Como `bv.vol` vem de `read(vtu)` direto e `result` é uma
    cópia dele, pontos e células estão na mesma ordem do original, e os
    arrays voltam por índice. Arrays já presentes no alvo (as UVC) não são
    tocados.
    """
    secao(f"Devolvendo as tags do .vtu de entrada ao {rotulo}")

    alvo = read_vtu(alvo_path)
    orig = read_vtu(vtu_original)

    n_ok = alvo.GetNumberOfPoints() == orig.GetNumberOfPoints()
    c_ok = alvo.GetNumberOfCells() == orig.GetNumberOfCells()
    if not n_ok:
        print(f"  ✗ Contagem de pontos difere "
              f"({alvo.GetNumberOfPoints()} vs {orig.GetNumberOfPoints()}); "
              f"nenhum array de ponto foi copiado.")
    if not c_ok:
        print(f"  ✗ Contagem de células difere "
              f"({alvo.GetNumberOfCells()} vs {orig.GetNumberOfCells()}); "
              f"nenhum array de célula foi copiado.")

    if n_ok:
        desvio = float(np.abs(pontos(alvo) - pontos(orig)).max())
        if desvio > 1e-6:
            print(f"  ✗ As coordenadas não batem ponto a ponto (desvio máx "
                  f"{desvio:.3f} mm); nada foi copiado para não desalinhar "
                  f"os campos.")
            n_ok = c_ok = False
        else:
            print(f"  Ordem dos pontos confirmada "
                  f"(desvio máx {desvio:.2e} mm).")

    copiados, ja_existiam = [], []
    for src, dst, tipo, permitido in (
            (orig.GetPointData(), alvo.GetPointData(), "PointData", n_ok),
            (orig.GetCellData(), alvo.GetCellData(), "CellData", c_ok)):
        if not permitido:
            continue
        for nome in _nomes_arrays_vtk(src):
            if dst.GetArray(nome) is not None:
                ja_existiam.append(f"{nome} ({tipo})")
                continue
            dst.AddArray(src.GetArray(nome))
            copiados.append(f"{nome} ({tipo})")

    if not copiados and not ja_existiam:
        print("  O .vtu de entrada não tem arrays para copiar.")
        return

    write_vtu(alvo, alvo_path)
    print(f"  Tags copiadas : {copiados or 'nenhuma'}")
    if ja_existiam:
        print(f"  Já existiam   : {ja_existiam}")
    print(f"  PointData final: {_nomes_arrays_vtk(alvo.GetPointData())}")
    print(f"  CellData  final: {_nomes_arrays_vtk(alvo.GetCellData())}")
    print(f"  Regravado: {alvo_path}")
