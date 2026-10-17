"""
Coordenada rotacional (rt) das UVC — porte de computeRotational.m.

É a etapa mais intrincada: resolve uma sequência de problemas de Laplace
intermediários para localizar as linhas de referência (junção VE-VD,
linha 0, ±pi, ±pi/2, ±pi/2.5) e então monta o ângulo rotacional final no
VE e no VD. Segue a mesma lógica do original (Bayer 2018 / Schuler).

A intuição física: imagine "fatiar" a parede do ventrículo como um relógio.
Cada solução de Laplace intermediária encontra uma agulha de referência;
combinando-as, define-se o ângulo de 0 a ±pi ao redor da cavidade.

─────────────────────────────────────────────────────────────────────────────
A PREMISSA DO VD QUE DESCE ATÉ O ÁPICE
─────────────────────────────────────────────────────────────────────────────
Duas etapas deste arquivo assumem, implicitamente, a geometria da `geo1` do
Cobiveco, em que o ventrículo direito desce quase até a ponta. Em malha de
paciente o VD costuma terminar vários centímetros acima do ápice (`rvApexVec`
de 30-40 mm), o ápice é maciço, e as duas premissas quebram — cada uma com um
erro diferente e nenhum deles legível:

1. `_apex_slice` (fatia apical). O original recorta a ponta da folha de
   cruzamento zero com um limiar fixo, `abLaplace <= 0.4`. Esse número
   pressupõe o `abLaplace` do Cobiveco, cuja banda de Dirichlet apical sobe
   do ápice até o endocárdio. Este port trunca a banda (`cfg.abBandLength`)
   para evitar que `ab = 0` vire uma agulha subindo pelo ventrículo; o preço é
   que, perto do ápice, o `abLaplace` passa a se comportar como um poço
   `1 - a/r`. Com a banda curta o bastante, nenhum tetraedro da folha tem os
   quatro vértices abaixo de 0.4, o `vtkThreshold` com `all_scalars=True`
   devolve malha vazia, e a PCA adiante recebe zero pontos:

       IndexError: index -1 is out of bounds for axis 1 with size 0

2. `_split_sheet` (divisão da folha). A folha é uma "página" que desce do
   anel basal até a ponta, e precisa ser partida no ramo septal e no ramo da
   parede livre. O original faz isso removendo um cordão fino ao longo do
   eixo apical — os nós a menos de `1.5 * meanEdgLen` da reta que passa pelo
   ponto class=5 na direção de `lvApexVec`. Isso funciona porque, na geo1, as
   curvas de junção VE-VD chegam ao ápice e a página converge ali. Quando o
   VD termina bem acima, o Dirichlet de `lap1` também termina lá, a página
   acaba a dezenas de milímetros do ápice marcado, e o cordão não captura
   nó algum: a `connectivity` devolve uma componente só e o código tropeça
   no `sortInd[1]`.

As duas correções mantêm o caminho do Cobiveco como primeira tentativa — em
malha onde ele funciona, nada muda — e só caem para o alternativo quando o
original não produz resultado. O alternativo da divisão é o MESMO critério
que `_split_junction_branch` já usa logo abaixo para partir a curva de
junção: cortar a ponta por `abLaplace` crescente até sobrarem dois ramos
equilibrados. Ele não depende de `normalVec`, de `lvApexVec` nem de
distância alguma ao ponto marcado.
"""

import numpy as np
from scipy.spatial import cKDTree

from . import vtk_utils as vu
from .laplace import solveLaplace


def _mode_int(arr):
    arr = np.asarray(arr).astype(np.int64)
    return int(np.bincount(arr - arr.min()).argmax() + arr.min())


def _apex_slice(zc1, limiar=0.4, min_pts=10, verbose=True):
    """Pedaço apical da folha de cruzamento zero (`zc1Apex` do original).

    Tenta primeiro o limiar do Cobiveco (0.4) e, se a fatia sair vazia ou
    pequena demais para uma PCA confiável, sobe para percentis crescentes do
    próprio `abLaplace` dentro da folha. Veja o item 1 da nota no topo do
    módulo: o limiar fixo só vale para o campo não truncado.

    O candidato escolhido é sempre o MENOR que serve, de modo que a fatia
    continue sendo apical — subir demais faria a "ponta" incluir metade da
    folha e o `normalVec` deixaria de descrever o plano da ponta.
    """
    abL = np.asarray(zc1.point_data['abLaplace'])
    if abL.size == 0:
        raise RuntimeError(
            'A folha de cruzamento zero (zc1) está vazia: a solução lap1 não '
            'tem superfície de nível zero utilizável. Verifique a junção '
            'VE-VD — ela é recortada da class 6 (septo), então um septo mal '
            'marcado derruba esta etapa.')

    candidatos = sorted({float(limiar)} |
                        {float(np.percentile(abL, p))
                         for p in (2, 5, 10, 20, 30, 40)})

    for t in candidatos:
        if t < abL.min():
            continue
        part = vu.threshold(zc1, 'points', 'abLaplace', (-np.inf, t))
        if part.n_points >= min_pts:
            if t > limiar and verbose:
                print(f'\n  AVISO: a fatia apical ficou vazia com '
                      f'abLaplace <= {limiar:.2f}; usando <= {t:.3f} '
                      f'({part.n_points} pontos, {part.n_cells} células).')
                print(f'         O abLaplace mínimo na folha é {abL.min():.3f}'
                      f' — sinal de banda de Dirichlet apical curta.')
                print(f'         Considere aumentar cfg.abBandLength '
                      f'(--ab-banda) nesta malha.')
            return part

    raise RuntimeError(
        f'Não há fatia apical na folha de cruzamento zero: nem com '
        f'abLaplace <= {candidatos[-1]:.3f} aparecem {min_pts} pontos '
        f'(mínimo na folha = {abL.min():.3f}, {zc1.n_points} pontos ao todo). '
        f'A folha não alcança o ápice — revise a marcação do septo (class 6) '
        f'e a posição do ponto class=5.')


def _duas_maiores(malha, min_frac=0.0):
    """As duas maiores componentes conexas de `malha`, ou None.

    `min_frac` exige que a SEGUNDA componente tenha pelo menos essa fração
    dos pontos — é o mesmo teste de equilíbrio que `_split_junction_branch`
    aplica (`> n/4`) ao partir a curva de junção. Com `min_frac=0` qualquer
    segunda componente serve, que é o comportamento do Cobiveco original.

    Retorna (regiao1, regiao2, malha_conectada) ou None.
    """
    if malha.n_points == 0:
        return None
    conectada = vu.connectivity(malha)
    rids = vu.region_id_point(conectada)
    uniq, inv = np.unique(rids, return_inverse=True)
    if len(uniq) < 2:
        return None
    freq = np.bincount(inv)
    ordem = np.argsort(freq)[::-1]
    if freq[ordem[1]] <= min_frac * conectada.n_points:
        return None
    rid1 = int(uniq[ordem[0]])
    rid2 = int(uniq[ordem[1]])
    return (vu.threshold(conectada, 'points', 'RegionId', (rid1, rid1)),
            vu.threshold(conectada, 'points', 'RegionId', (rid2, rid2)),
            conectada)


def _split_sheet(zc1, n_anel, verbose=True):
    """Parte a folha de cruzamento zero em ramo septal e ramo de parede livre.

    Duas estratégias, nesta ordem (veja o item 2 da nota no topo do módulo):

      1. a do Cobiveco — remover o anel apical já marcado em
         `zc1.point_data['apex']`. Se ela partir a folha, é ela que vale, e o
         resultado é idêntico ao do original;
      2. cortar a ponta da folha por `abLaplace` crescente até sobrarem dois
         ramos equilibrados. Não depende de direção nenhuma, e é o critério
         que `_split_junction_branch` já usa na curva de junção.

    Retorna (regiao1, regiao2, folha_cortada).
    """
    if n_anel > 0:
        resto = vu.threshold(zc1, 'points', 'apex', (-np.inf, 0))
        par = _duas_maiores(resto, min_frac=0.0)
        if par is not None:
            return par

    abL = np.asarray(zc1.point_data['abLaplace'])
    cortes = sorted({float(np.percentile(abL, p)) for p in range(1, 61)})

    for t in cortes:
        resto = vu.threshold(zc1, 'points', 'abLaplace', (t, np.inf))
        par = _duas_maiores(resto, min_frac=0.25)
        if par is not None:
            if verbose:
                n1, n2 = par[0].n_points, par[1].n_points
                print(f'\n  AVISO: o anel apical não partiu a folha de '
                      f'cruzamento zero ({n_anel} nó(s) no anel).')
                print(f'         Cortando a ponta por abLaplace >= {t:.3f}: '
                      f'dois ramos de {n1} e {n2} pontos.')
                print(f'         Isso acontece quando o VD termina bem acima '
                      f'do ápice e a folha não')
                print(f'         desce até o ponto class=5 — veja a nota no '
                      f'topo de rotational.py.')
            return par

    raise RuntimeError(
        f'Não foi possível partir a folha de cruzamento zero em dois ramos '
        f'(septal e parede livre): o anel apical capturou {n_anel} nó(s) e '
        f'nenhum corte por abLaplace até o percentil 60 produziu duas '
        f'componentes equilibradas. A folha tem {zc1.n_points} pontos. '
        f'O problema está em lap1 — revise a marcação do septo (class 6) e '
        f'os ramos da junção VE-VD.')


def _split_junction_branch(junc_part, basePoint, meanEdgLen, tol):
    """
    Separa uma curva de junção (epi ou endo) em ramos anterior/posterior.
    Réplica fiel das DUAS estratégias do computeRotational.m:

      - se a connectivity dá 1 região  -> 'default': remove a ponta apical
        (abLaplace baixo) por threshold crescente + connectivity até o laço
        se partir em dois arcos balanceados;
      - se dá >1 região -> 'fallback': corta por um plano que passa pelo
        ápice, com normal = cross(apex->base, eixo de menor variância).

    Retorna (regiao1, regiao2).
    """
    part = vu.connectivity(junc_part)
    nreg = len(np.unique(vu.region_id_point(part)))

    if nreg == 1:
        # ---- default ----
        rid = _mode_int(vu.region_id_point(part))
        part = vu.threshold(part, 'points', 'RegionId', (rid, rid))
        abL = np.asarray(part.point_data['abLaplace'])
        abSorted = np.sort(abL)
        for k in range(len(abSorted)):
            cut = vu.threshold(part, 'points', 'abLaplace',
                               (abSorted[k] + tol, np.inf))
            if cut.n_points == 0:
                break
            cut = vu.connectivity(cut)
            rids = vu.region_id_point(cut)
            uniq, inv = np.unique(rids, return_inverse=True)
            if len(uniq) > 1:
                freq = np.bincount(inv)
                sortInd = np.argsort(freq)[::-1]
                if freq[sortInd[1]] > cut.n_points / 4:
                    rid1 = int(uniq[sortInd[0]])
                    rid2 = int(uniq[sortInd[1]])
                    r1 = vu.threshold(cut, 'points', 'RegionId', (rid1, rid1))
                    r2 = vu.threshold(cut, 'points', 'RegionId', (rid2, rid2))
                    return r1, r2
        raise RuntimeError('Não foi possível dividir a junção por abLaplace.')
    else:
        # ---- fallback: corte por plano através do ápice ----
        pts = part.points
        abL = np.asarray(part.point_data['abLaplace'])
        apexPoint = pts[np.argmin(abL)]
        leftRightVec = vu.pca(pts)[:, -1]
        normalVec = np.cross(apexPoint - basePoint, leftRightVec)
        normalVec = normalVec / np.linalg.norm(normalVec)
        d = (pts - apexPoint) @ normalVec
        part.point_data['d'] = d
        r1 = vu.threshold(part, 'points', 'd', (-np.inf, 0))
        r2 = vu.threshold(part, 'points', 'd', (0, np.inf))
        return r1, r2


def _zero_cross_submesh(o, lap, extra_pdata):
    """Sub-malha de tets que cruzam o nível zero de `lap` (no VE)."""
    lc = o.lv.cells
    signs = (lap[lc] < 0).astype(np.int8)
    crossing = np.any(np.diff(signs, axis=1) != 0, axis=1)
    subset = lc[crossing]
    pdata = {'abLaplace': o.bv.abLaplace[o.lv.ids]}
    pdata.update(extra_pdata)
    mesh, _ = vu.make_tet_submesh(o.lv.vol.points, subset, pdata)
    return mesh


def compute_rotational(o):
    tol = o.cfg.tol
    P = o.bv.vol.points
    Plv = o.lv.vol.points
    verbose = getattr(o.cfg, 'verbose', True)

    # ==================================================================
    # Junção VE-VD e primeira solução de Laplace (lap1)
    # ==================================================================
    lv_sur_cells = vu.threshold(o.lv.sur, 'cells', 'class', (0, 0))
    junc = vu.feature_edges(lv_sur_cells, boundary=True)
    j2bvsur = vu.map_point_ids(o.bv.sur, junc)
    junc.point_data['abLaplace'] = o.bv.abLaplace[o.bv.surToVol[j2bvsur]]
    j2lvsur = vu.map_point_ids(o.lv.sur, junc)
    junc.point_data['class'] = np.asarray(o.lv.sur.point_data['class'])[j2lvsur]

    juncEpi = vu.threshold(junc, 'points', 'class', (2, 2))
    juncEndo = vu.threshold(junc, 'points', 'class', (6, 6))

    # ponto de base do VE (média dos pontos de classe 1 na superfície do VE)
    lv_base_pt = o.lv.sur.points[
        np.asarray(o.lv.sur.point_data['class']) == 1].mean(0)
    juncEpiAnt, juncEpiPost = _split_junction_branch(
        juncEpi, lv_base_pt, o.meanEdgLen, tol)
    juncEndoAnt, juncEndoPost = _split_junction_branch(
        juncEndo, lv_base_pt, o.meanEdgLen, tol)

    clsB = o.bv.sur.point_data['class']
    PsB = o.bv.sur.points
    baseToApex = (PsB[clsB == 1].mean(0) - PsB[clsB == 5][0])
    leftToRight = (PsB[clsB == 3].mean(0) - PsB[clsB == 4].mean(0))
    antToPost = np.cross(baseToApex, leftToRight)
    antToPost = antToPost / np.linalg.norm(antToPost)
    juncCenter = junc.points.mean(0)

    if (juncEpiAnt.points.mean(0) - juncCenter) @ antToPost > 0:
        juncEpiAnt, juncEpiPost = juncEpiPost, juncEpiAnt
    if (juncEndoAnt.points.mean(0) - juncCenter) @ antToPost > 0:
        juncEndoAnt, juncEndoPost = juncEndoPost, juncEndoAnt

    Mdl = cKDTree(Plv)
    idsEpiAnt = np.unique(Mdl.query(juncEpiAnt.points)[1])
    idsEpiPost = np.setdiff1d(Mdl.query(juncEpiPost.points)[1], idsEpiAnt)
    idsEndoAnt = np.setdiff1d(Mdl.query(juncEndoAnt.points)[1],
                              np.concatenate([idsEpiAnt, idsEpiPost]))
    idsEndoPost = np.setdiff1d(Mdl.query(juncEndoPost.points)[1],
                               np.concatenate([idsEpiAnt, idsEpiPost, idsEndoAnt]))

    ids = np.concatenate([idsEpiAnt, idsEndoAnt, idsEpiPost, idsEndoPost])
    val = np.concatenate([np.ones(len(idsEpiAnt) + len(idsEndoAnt)),
                          -np.ones(len(idsEpiPost) + len(idsEndoPost))])
    ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
    o.lv.lap1 = solveLaplace(o.lv.L, ids, val, tol, o.cfg.maxit)

    # ==================================================================
    # Linhas 0 e ±pi e segunda solução de Laplace (lap2)
    # ==================================================================
    zc1 = _zero_cross_submesh(o, o.lv.lap1, {'lap1': o.lv.lap1})
    zc1 = vu.connectivity(zc1)
    rid = _mode_int(vu.region_id_point(zc1))
    zc1 = vu.threshold(zc1, 'points', 'RegionId', (rid, rid))

    # Limiar adaptativo em vez do 0.4 fixo — item 1 da nota no topo.
    zc1Apex = _apex_slice(zc1, verbose=verbose)
    zc1Apex_ids = np.asarray(zc1Apex.point_data['ids'])
    normalVec = vu.pca(zc1Apex.points)[:, -1]
    apexVec = o.lvApexVec - o.lvApexVec @ np.outer(normalVec, normalVec)
    posVec = PsB[clsB == 5][0]
    proj = zc1Apex.points - (zc1Apex.points - posVec) @ np.outer(normalVec, normalVec)
    rel = proj - posVec
    d = np.sqrt(np.sum(np.cross(rel, np.broadcast_to(apexVec, rel.shape))**2, axis=1)
                / np.sum(apexVec**2))
    apex_global = zc1Apex_ids[d < 1.5 * o.meanEdgLen]

    zc1_ids = np.asarray(zc1.point_data['ids'])
    apexFlag = np.zeros(zc1.n_points)
    apexFlag[np.isin(zc1_ids, apex_global)] = 1
    zc1.point_data['apex'] = apexFlag

    # Anel apical primeiro; corte por abLaplace como alternativa — item 2.
    zc1Sept, zc1Free, zc1b = _split_sheet(zc1, len(apex_global),
                                          verbose=verbose)

    if (zc1Sept.points.mean(0) - zc1b.points.mean(0)) @ leftToRight > 0:
        zc1Sept, zc1Free = zc1Free, zc1Sept

    septIds = np.asarray(zc1Sept.point_data['ids'])
    septLap1 = np.asarray(zc1Sept.point_data['lap1'])
    freeIds = np.asarray(zc1Free.point_data['ids'])
    freeLap1 = np.asarray(zc1Free.point_data['lap1'])
    idsZero = septIds[septLap1 < 0]
    idsMinusPi = freeIds[freeLap1 < 0]
    idsPlusPi = freeIds[freeLap1 > 0]

    ids = np.concatenate([idsZero, idsMinusPi, idsPlusPi])
    val = np.concatenate([np.zeros(len(idsZero)),
                          np.full(len(idsMinusPi), -np.pi),
                          np.full(len(idsPlusPi), np.pi)])
    ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
    o.lv.lap2 = solveLaplace(o.lv.L, ids, val, tol, o.cfg.maxit)

    # ==================================================================
    # Terceira/quarta soluções e linhas ±pi/2 e ±pi/2.5
    # ==================================================================
    lvc = o.lv.sur.point_data['class']
    lv_s2v = o.lv.surToVol

    def mean_lap2_at(part):
        idx = vu.map_point_ids(o.lv.vol, part)
        return float(np.mean(o.lv.lap2[idx]))

    meanEndoAnt = mean_lap2_at(juncEndoAnt)
    meanEndoPost = mean_lap2_at(juncEndoPost)
    endo_band = np.intersect1d(
        lv_s2v[lvc == 3],
        np.where((o.lv.lap2 > meanEndoPost) & (o.lv.lap2 < meanEndoAnt))[0])
    idsPlus = np.union1d(endo_band, lv_s2v[lvc == 6])
    idsMinus = np.setdiff1d(lv_s2v[lvc != 1], idsPlus)
    ids = np.concatenate([idsPlus, idsMinus])
    val = np.concatenate([np.ones(len(idsPlus)), -np.ones(len(idsMinus))])
    ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
    o.lv.lap3 = solveLaplace(o.lv.L, ids, val, tol, o.cfg.maxit)

    zc3 = _zero_cross_submesh(o, o.lv.lap3,
                              {'lap2': o.lv.lap2, 'lap3': o.lv.lap3})
    s3 = vu.surface_filter(zc3)
    s3_lap3 = vu.threshold(s3, 'points', 'lap3', (-np.inf, 0))
    minusPi25 = vu.threshold(s3_lap3, 'points', 'lap2', (-np.inf, 0))
    plusPi25 = vu.threshold(s3_lap3, 'points', 'lap2', (0, np.inf))
    idsMinusPi25 = np.union1d(np.asarray(minusPi25.point_data['ids']),
                              vu.map_point_ids(o.lv.vol, juncEndoPost))
    idsPlusPi25 = np.union1d(np.asarray(plusPi25.point_data['ids']),
                             vu.map_point_ids(o.lv.vol, juncEndoAnt))

    meanEpiAnt = mean_lap2_at(juncEpiAnt)
    meanEpiPost = mean_lap2_at(juncEpiPost)
    epi_band = np.intersect1d(
        lv_s2v[lvc == 3],
        np.where((o.lv.lap2 > meanEpiPost) & (o.lv.lap2 < meanEpiAnt))[0])
    scells = vu.get_cells(o.lv.sur)
    sclass = np.asarray(o.lv.sur.cell_data['class'])
    epi0 = lv_s2v[np.unique(scells[sclass == 0])]
    idsPlus = np.union1d(np.union1d(epi_band, lv_s2v[lvc == 6]), epi0)
    idsMinus = np.setdiff1d(lv_s2v[lvc != 1], idsPlus)
    ids = np.concatenate([idsPlus, idsMinus])
    val = np.concatenate([np.ones(len(idsPlus)), -np.ones(len(idsMinus))])
    ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
    o.lv.lap4 = solveLaplace(o.lv.L, ids, val, tol, o.cfg.maxit)

    zc4 = _zero_cross_submesh(o, o.lv.lap4,
                              {'lap2': o.lv.lap2, 'lap4': o.lv.lap4})
    s4 = vu.surface_filter(zc4)
    s4_lap4 = vu.threshold(s4, 'points', 'lap4', (0, np.inf))
    minusPi2 = vu.threshold(s4_lap4, 'points', 'lap2', (-np.inf, 0))
    plusPi2 = vu.threshold(s4_lap4, 'points', 'lap2', (0, np.inf))
    idsPlusPi2 = np.asarray(plusPi2.point_data['ids'])
    idsMinusPi2 = np.asarray(minusPi2.point_data['ids'])

    # ==================================================================
    # Solução final de Laplace no VE (rt) — remove sobreposições por prioridade
    # ==================================================================
    idsZero = np.setdiff1d(idsZero, np.concatenate(
        [idsPlusPi25, idsPlusPi2, idsPlusPi, idsMinusPi, idsMinusPi2, idsMinusPi25]))
    idsPlusPi25 = np.setdiff1d(idsPlusPi25, np.concatenate(
        [idsPlusPi2, idsPlusPi, idsMinusPi, idsMinusPi2, idsMinusPi25]))
    idsPlusPi2 = np.setdiff1d(idsPlusPi2, np.concatenate(
        [idsPlusPi, idsMinusPi, idsMinusPi2, idsMinusPi25]))
    idsPlusPi = np.setdiff1d(idsPlusPi, np.concatenate(
        [idsMinusPi, idsMinusPi2, idsMinusPi25]))
    idsMinusPi = np.setdiff1d(idsMinusPi, np.concatenate(
        [idsMinusPi2, idsMinusPi25]))
    idsMinusPi2 = np.setdiff1d(idsMinusPi2, idsMinusPi25)

    ids = np.concatenate([idsZero, idsPlusPi25, idsPlusPi2, idsPlusPi,
                          idsMinusPi, idsMinusPi2, idsMinusPi25])
    val = np.concatenate([
        np.zeros(len(idsZero)),
        np.full(len(idsPlusPi25), np.pi / 2.5),
        np.full(len(idsPlusPi2), np.pi / 2),
        np.full(len(idsPlusPi), np.pi),
        np.full(len(idsMinusPi), -np.pi),
        np.full(len(idsMinusPi2), -np.pi / 2),
        np.full(len(idsMinusPi25), -np.pi / 2.5)])
    ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
    o.lv.rt = solveLaplace(o.lv.L, ids, val, tol, o.cfg.maxit)

    # ==================================================================
    # Solução final de Laplace no VD (rt)
    # ==================================================================
    rvBound = vu.threshold(o.rv.sur, 'cells', 'class', (0, 0))
    rvBound.point_data['lap2'] = o.lv.lap2[vu.map_point_ids(o.lv.vol, rvBound)]
    rvPlusPi2 = vu.threshold(rvBound, 'points', 'lap2', (0, np.inf))
    rvMinusPi2 = vu.threshold(rvBound, 'points', 'lap2', (-np.inf, 0))

    idsRvPlus = vu.map_point_ids(o.rv.vol, rvPlusPi2)
    idsRvMinus = vu.map_point_ids(o.rv.vol, rvMinusPi2)
    both = np.intersect1d(idsRvPlus, idsRvMinus)
    idsRvPlus = np.setdiff1d(idsRvPlus, both)
    idsRvMinus = np.setdiff1d(idsRvMinus, both)

    ids = np.concatenate([idsRvPlus, idsRvMinus])
    val = np.concatenate([np.full(len(idsRvPlus), np.pi / 2),
                          np.full(len(idsRvMinus), -np.pi / 2)])
    ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
    o.rv.rt = solveLaplace(o.rv.L, ids, val, tol, o.cfg.maxit)

    # ==================================================================
    # Combina VE+VD e preenche buracos por vizinho mais próximo
    # ==================================================================
    rt = np.full(o.bv.vol.n_points, np.nan)
    rt[o.lv.ids] = o.lv.rt
    rt[o.rv.ids] = o.rv.rt
    nan = np.isnan(rt)
    for side_val in (0, 1):
        sideNan = nan & (o.bv.tv == side_val)
        sideOk = np.where(~nan & (o.bv.tv == side_val))[0]
        if sideNan.any() and len(sideOk):
            tree = cKDTree(P[sideOk])
            _, j = tree.query(P[sideNan])
            rt[sideNan] = rt[sideOk[j]]
    o.bv.rt = rt
