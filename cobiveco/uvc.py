"""
Universal Ventricular Coordinates (UVC) — porte Python do Cobiveco.

Baseado no código MATLAB de Steffen Schuler / KIT-IBT (Bayer et al. 2018).
Mantém a mesma sequência de etapas e a mesma lógica do original:

    prepareMesh -> computeTransventricular -> splitMesh
                -> computeTransmural -> computeApicobasal -> computeRotational
                -> computeAHA17

Uso:
    from cobiveco.uvc import UVC
    c = UVC(inPrefix='examples/geo1/heart_for_UVC',
            outPrefix='result_geo1/', exportLevel=2)
    c.computeAll()
    result = c.result   # pointData tv, tm, ab, rt, rtSin, rtCos, aha17

A última etapa, computeAHA17, não resolve nenhum Laplace: apenas "fatia"
(ab, rt) na grade de segmentos. O resultado é BIVENTRICULAR: 1..17 no VE
(AHA, Cerqueira et al. 2002) e 18..34 no VD, 34 regiões ao todo. Veja
cobiveco/aha17.py. Desligue com aha17=False; para ficar só no VE, use
aha17RV=False.

─────────────────────────────────────────────────────────────────────────────
O ÁPICE MARCADO (class = 5) — parâmetro `apexSnap`
─────────────────────────────────────────────────────────────────────────────
O `@uvc/computeApicobasal.m` original NÃO usa o ponto class=5 do arquivo de
entrada: ele o emprega apenas como semente para achar o nó epicárdico do
lado tv=1 (parede livre do VD) mais próximo, e é ESSE nó que vira o ápice:

    idsEpiRv  = intersect(class==2, tv==1);
    idEpiApex = idsEpiRv(argmin |p - p_marcado|);

Na geo1 do Cobiveco isso é inócuo — o VD desce quase até a ponta, e o
deslocamento é de poucos milímetros. Em malha de paciente o VD termina bem
acima do ápice, e a substituição tem duas consequências:

  1. O ápice efetivo migra para a borda apical da parede livre do VD. Como
     `lvApexVec`/`rvApexVec` partem dele, a banda de Dirichlet ab=0 deixa de
     ficar na ponta do VE e o abLaplace sai inclinado.
  2. A informação do usuário é perdida: qualquer coordenada razoável de
     ápice cai no MESMO nó epi-VD, de modo que fornecer ou não o
     `apex_coord.txt` produz exatamente o mesmo resultado.

Daí o parâmetro `apexSnap`:

    'epi'    (PADRÃO) — o ápice é o nó EPICÁRDICO mais próximo do ponto
                        marcado, em qualquer ventrículo. Se o ponto marcado
                        já estiver no epicárdio, ele é usado tal e qual.
                        Respeita o apex do usuário e mantém o contrato do
                        Cobiveco (class 5 é um ponto do epicárdio).
    'marked'          — usa o nó marcado exatamente como veio, sem snap.
                        Só faça isso se você tem certeza de que o ponto está
                        no epicárdio: um ápice sobre o endocárdio deixa
                        `lvApexVec` degenerado.
    'rv-epi'          — comportamento literal do MATLAB original. Mantido
                        para reprodutibilidade e comparação.

O ápice é fixado por `resolveApex()`, que roda depois do tv e ANTES do
`splitMesh` — não mais no meio do `computeApicobasal`, como no original. A
ordem é o que garante que ele valha para TODAS as coordenadas: `lv.sur` e
`rv.sur` recebem a `class` por cópia no momento do corte, de modo que um
ápice decidido depois deixaria os submeshes marcando o nó antigo, e o
`computeTransmural` e o `computeRotational`, que leem `lv.sur`/`rv.sur`,
trabalhariam com um ápice diferente do que o `ab` usou. Depois de
`resolveApex()`, `self.bv.idEpiApex` é o ápice de tm, ab e rt.

O deslocamento entre o ponto marcado e o efetivamente usado é sempre
relatado quando `verbose=True`.

─────────────────────────────────────────────────────────────────────────────
ab = 0 EM UM ÚNICO NÓ — parâmetro `abApexUnique`
─────────────────────────────────────────────────────────────────────────────
Convenção mantida: ab = 0 no ápice, ab = 1 na base (é a que `aha17.py`
assume ao recortar a calota apical com `ab <= ab_apex`).

O Dirichlet ab=0 não é imposto no ápice sozinho — um único nó fixo num
Laplace 3D gera um poço tipo 1/r e o campo degenera —, e sim na BANDA que o
`_apex_band` recorta ao longo da espessura da parede apical. Todos esses
nós saem com abLaplace exatamente 0, e o mínimo de ab acaba sendo um platô
de dezenas de nós em vez de um ponto.

Com `abApexUnique=True` (padrão), o Laplace continua exatamente como está e
só os valores DENTRO do platô são redistribuídos, proporcionalmente à
distância geodésica de cada nó ao ápice e dentro da folga até o menor ab de
fora. O ápice tem distância zero e fica sozinho no mínimo; o resto do campo
não muda e a ordem global é preservada. `False` reproduz o platô original.
"""

import os
import time
import warnings
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

from . import vtk_utils as vu
from .laplace import cotmatrix, solveLaplace


APEX_SNAP_MODES = ('epi', 'marked', 'rv-epi')


class Bunch:
    """Container simples de atributos (faz o papel das structs do MATLAB)."""
    pass


class UVC:
    def __init__(self, inPrefix, outPrefix='', exportLevel=1, verbose=True,
                 mappingTol=0.5, tol=1e-8, maxit=3000,
                 aha17=True, aha17SeptumMode='aha', aha17ApexCap=None,
                 aha17Coronary=False, aha17RV=True,
                 aha17RVModel='mirror17', aha17RVRtMode='stretch',
                 apexSnap='epi', abApexUnique=True, abBandLength=2.5):
        self.cfg = Bunch()
        self.cfg.inPrefix = inPrefix
        self.cfg.outPrefix = outPrefix
        self.cfg.exportLevel = exportLevel
        self.cfg.verbose = verbose
        self.cfg.mappingTol = mappingTol
        self.cfg.tol = tol
        self.cfg.maxit = maxit
        # --- ápice marcado (veja a nota no topo do módulo) ---
        if apexSnap not in APEX_SNAP_MODES:
            raise ValueError(f"apexSnap deve ser um de {list(APEX_SNAP_MODES)}; "
                             f"recebi '{apexSnap}'.")
        self.cfg.apexSnap = apexSnap
        # ab = 0 em exatamente um nó (o ápice) em vez do platô da banda de
        # Dirichlet. False reproduz o comportamento do Cobiveco original.
        self.cfg.abApexUnique = bool(abApexUnique)
        # comprimento maximo da banda de Dirichlet apical, em arestas medias.
        # np.inf reproduz o Cobiveco original (banda ate o endocardio).
        self.cfg.abBandLength = float(abBandLength)
        # --- parcelamento AHA-17 ---
        self.cfg.aha17 = aha17                    # calcular junto com as UVC
        self.cfg.aha17SeptumMode = aha17SeptumMode  # 'aha' ou 'junction'
        self.cfg.aha17ApexCap = aha17ApexCap      # None => estima pelo tm
        self.cfg.aha17Coronary = aha17Coronary    # exporta tambem LAD/RCA/LCX
        self.cfg.aha17RV = aha17RV                # rotula tambem o VD (padrao)
        self.cfg.aha17RVModel = aha17RVModel      # 'mirror17' (18..34) ou
                                                  # 'freewall9' (18..26)
        self.cfg.aha17RVRtMode = aha17RVRtMode    # 'stretch' (estica o arco
                                                  # do rt do VD para a volta
                                                  # inteira) ou 'raw'

        if exportLevel > 0:
            if not outPrefix:
                raise ValueError("Parâmetro 'outPrefix' obrigatório e vazio.")
            outPath = os.path.dirname(outPrefix)
            if outPath and not os.path.isdir(outPath):
                os.makedirs(outPath, exist_ok=True)

        self.bv = Bunch()
        self.lv = Bunch()
        self.rv = Bunch()
        self.result = None

        self.meanEdgLen = None
        self.lvApexVec = None
        self.rvApexVec = None

        # diagnóstico do ápice, preenchido por computeApicobasal
        self.apexInfo = {}

        self.available = Bunch()
        for k in ['mesh', 'transventricular', 'apex', 'splitMesh',
                  'transmural', 'apicobasal', 'rotational', 'aha17']:
            setattr(self.available, k, False)

        self._t0 = time.time()

    # ------------------------------------------------------------------
    def printStatus(self, msg, noPadding=False):
        if self.cfg.verbose:
            if noPadding:
                print(msg, end='', flush=True)
            else:
                print(f'{msg:<41}', end='', flush=True)

    def computeAll(self):
        t = time.time()
        self.prepareMesh()
        self.computeTransventricular()
        self.splitMesh()
        self.computeTransmural()
        self.computeApicobasal()
        self.computeRotational()
        if self.cfg.aha17:
            self.computeAHA17()
        self.printStatus('Total elapsed time:')
        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)

    # ==================================================================
    # 1) prepareMesh
    # ==================================================================
    def prepareMesh(self):
        self.printStatus('Preparing mesh...')
        t = time.time()

        vtu = f'{self.cfg.inPrefix}.vtu'.replace('_for_UVC', '')
        self.bv.vol = vu.delete_data_arrays(vu.read(vtu))
        self.bv.sur = vu.surface_filter(self.bv.vol)
        self.bv.surToVol = vu.original_point_ids(self.bv.sur)
        self.meanEdgLen = vu.mean_edge_length(self.bv.vol)

        sur = vu.read(f'{self.cfg.inPrefix}.vtp')
        ids = vu.map_point_ids(sur, self.bv.sur)
        self.bv.sur.point_data['class'] = np.asarray(sur.point_data['class'])[ids]

        # O apex é UM ponto: se o vizinho-mais-próximo não o alcançar, ele
        # some aqui, silenciosamente, e computeApicobasal quebra lá na frente
        # com um IndexError incompreensível. Melhor reclamar agora.
        n_apex = int(np.sum(np.asarray(self.bv.sur.point_data['class']) == 5))
        if n_apex == 0:
            raise RuntimeError(
                f"O ponto do ápice (class=5) do '{self.cfg.inPrefix}.vtp' não "
                f"sobreviveu ao mapeamento por vizinho mais próximo para a "
                f"superfície extraída de '{os.path.basename(vtu)}'. O .vtp e o "
                f".vtu provavelmente não são da mesma geometria.")

        T = vu.get_cells(self.bv.vol)
        self.bv.L = cotmatrix(self.bv.vol.points, T)
        self.bv.cells = T

        self.result = self.bv.vol.copy()

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)
        self.available.mesh = True

    # ==================================================================
    # 2) computeTransventricular
    # ==================================================================
    def computeTransventricular(self):
        if not self.available.mesh:
            self.prepareMesh()
        self.printStatus('Computing transventricular coordinate...')
        t = time.time()

        cls = self.bv.sur.point_data['class']
        s2v = self.bv.surToVol
        idsEndoLv = s2v[cls == 3]
        idsEndoRv = s2v[cls == 4]
        idsSeptumRv = s2v[cls == 6]

        ids = np.concatenate([idsEndoLv, idsSeptumRv, idsEndoRv])
        val = np.concatenate([np.zeros(len(idsEndoLv)),
                              np.full(len(idsSeptumRv), 0.5),
                              np.ones(len(idsEndoRv))])
        ids, uniq = np.unique(ids, return_index=True)
        val = val[uniq]

        sol = solveLaplace(self.bv.L, ids, val, self.cfg.tol, self.cfg.maxit)
        self.bv.tv = np.round(sol - 0.01)

        self.result.point_data['tv'] = self.bv.tv.astype(np.float32)

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)
        self.available.transventricular = True

    # ==================================================================
    # 3) splitMesh
    # ==================================================================
    def splitMesh(self):
        if not self.available.transventricular:
            self.computeTransventricular()
        # O ápice é fixado AQUI, antes do corte, e não mais dentro do
        # computeApicobasal: assim lv.sur e rv.sur já nascem com o nó certo
        # marcado, e tm, ab e rt passam todos a ser calculados a partir dele.
        if not self.available.apex:
            self.resolveApex()
        self.printStatus('Splitting mesh...')
        t = time.time()

        tmp = self.bv.vol.copy()
        npts = tmp.n_points
        tmp.point_data['ids'] = np.arange(npts, dtype=np.int32)
        cls_full = np.zeros(npts, dtype=np.uint8)
        cls_full[self.bv.surToVol] = self.bv.sur.point_data['class']
        tmp.point_data['class'] = cls_full
        tmp.point_data['tv'] = self.bv.tv

        # ---- LV ----
        self.lv = self._split_one(tmp, side='lv')
        # ---- RV ----
        self.rv = self._split_one(tmp, side='rv')

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)
        self.available.splitMesh = True

    def _split_one(self, tmp, side):
        rng = (-np.inf, 0.5) if side == 'lv' else (0.5, np.inf)
        sub = vu.threshold(tmp, 'points', 'tv', rng)
        sub = vu.connectivity(sub)
        rid = int(np.bincount(vu.region_id_cell(sub)).argmax())
        sub = vu.threshold(sub, 'cells', 'RegionId', (rid, rid))

        b = Bunch()
        b.vol = sub
        b.ids = np.asarray(sub.point_data['ids']).astype(np.int64)
        b.sur = vu.surface_filter(sub)
        b.surToVol = vu.original_point_ids(b.sur)
        # classe nos pontos da superfície do submesh
        b.sur.point_data['class'] = np.asarray(
            sub.point_data['class'])[b.surToVol].astype(np.uint8)
        # classe por célula = mínimo das classes dos pontos da célula
        scells = vu.get_cells(b.sur)
        b.sur.cell_data['class'] = np.min(
            b.sur.point_data['class'][scells], axis=1).astype(np.uint8)
        b.vol = vu.delete_data_arrays(b.vol)
        T = vu.get_cells(b.vol)
        b.cells = T
        b.L = cotmatrix(b.vol.points, T)
        return b

    # ==================================================================
    # 4) computeTransmural
    # ==================================================================
    def computeTransmural(self):
        if not self.available.splitMesh:
            self.splitMesh()
        self.printStatus('Computing transmural coordinate...')
        t = time.time()

        # LV: endo (class 3) -> 0 ; epi (class != 1 e != 3) -> 1
        clsLv = self.lv.sur.point_data['class']
        idsEndoLv = self.lv.surToVol[clsLv == 3]
        idsEpiLv = self.lv.surToVol[(clsLv != 1) & (clsLv != 3)]
        ids = np.concatenate([idsEndoLv, idsEpiLv])
        val = np.concatenate([np.zeros(len(idsEndoLv)), np.ones(len(idsEpiLv))])
        ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
        self.lv.tm = solveLaplace(self.lv.L, ids, val, self.cfg.tol, self.cfg.maxit)

        # RV: endo (class 4) -> 0 ; epi (class 2) -> 1
        # O ápice (class 5) entra como epicárdio quando de fato está no
        # epicárdio; sem isso ele seria o único nó epicárdico do VD sem
        # condição de contorno, bem na ponta onde o tm mais importa.
        clsRv = self.rv.sur.point_data['class']
        idsEndoRv = self.rv.surToVol[clsRv == 4]
        epiRv = (clsRv == 2)
        if getattr(self.bv, 'apexIsEpi', False):
            epiRv = epiRv | (clsRv == 5)
        idsEpiRv = self.rv.surToVol[epiRv]
        ids = np.concatenate([idsEndoRv, idsEpiRv])
        val = np.concatenate([np.zeros(len(idsEndoRv)), np.ones(len(idsEpiRv))])
        ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
        self.rv.tm = solveLaplace(self.rv.L, ids, val, self.cfg.tol, self.cfg.maxit)

        # junta no biventricular e preenche buracos por vizinho mais próximo
        npts = self.bv.vol.n_points
        tm = np.full(npts, np.nan)
        tm[self.lv.ids] = self.lv.tm
        tm[self.rv.ids] = self.rv.tm
        self.bv.tm = self._fill_nan_by_side(tm)

        self.result.point_data['tm'] = self.bv.tm.astype(np.float32)

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)
        self.available.transmural = True

    def _fill_nan_by_side(self, arr):
        P = self.bv.vol.points
        tv = self.bv.tv
        out = arr.copy()
        nan = np.isnan(out)
        for side_val in (0, 1):
            sideNan = nan & (tv == side_val)
            sideOk = np.where(~nan & (tv == side_val))[0]
            if sideNan.any() and len(sideOk):
                tree = cKDTree(P[sideOk])
                _, j = tree.query(P[sideNan])
                out[sideNan] = out[sideOk[j]]
        return out

    # ==================================================================
    # 5) computeApicobasal
    # ==================================================================
    def resolveApex(self):
        """Fixa, de uma vez por todas, o nó que é o ápice (class = 5).

        Roda logo depois do tv e ANTES do splitMesh, e não mais no meio do
        computeApicobasal. A ordem importa: `lv.sur` e `rv.sur` recebem a
        `class` por cópia no momento do corte, então um ápice decidido depois
        deixaria os dois submeshes com o nó antigo marcado — e o
        computeTransmural e o computeRotational, que leem `lv.sur`/`rv.sur`,
        trabalhariam com um ápice diferente do usado pelo ab.

        Depois daqui, `self.bv.idEpiApex` é o ápice de TODAS as coordenadas.
        """
        if not self.available.transventricular:
            self.computeTransventricular()
        self.printStatus('Resolving apex node...')
        t = time.time()

        cls = np.asarray(self.bv.sur.point_data['class']).astype(np.int32)
        s2v = self.bv.surToVol
        P = self.bv.vol.points

        idMarcado, idEpiApex, desloc = self._resolve_apex(cls, s2v, P)

        # a marcação passa a valer para o nó escolhido, e só para ele
        cls[cls == 5] = 2
        cls[s2v == idEpiApex] = 5
        self.bv.sur.point_data['class'] = cls

        self.bv.idEpiApex = idEpiApex
        self.bv.idApexMarcado = idMarcado
        # o apex é epicárdico? decide se o computeTransmural pode usá-lo como
        # condição de contorno do epicárdio do VD (onde só a class 2 entra).
        self.bv.apexIsEpi = bool(
            self.cfg.apexSnap != 'marked'
            or self.apexInfo.get('origem', '').startswith('ponto marcado (já'))

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)

        if self.cfg.verbose:
            info = self.apexInfo
            pm = np.round(info['pontoMarcado'], 2).tolist()
            pu = np.round(info['pontoUsado'], 2).tolist()
            print(f"  apice marcado : id {info['idMarcado']} {pm}")
            print(f"  apice usado   : id {info['idUsado']} {pu}  "
                  f"[apexSnap='{info['modo']}': {info['origem']}]")
            print(f"  deslocamento  : {desloc:.3f} mm "
                  f"({desloc / self.meanEdgLen:.1f} arestas medias)")
            if desloc > 3.0 * self.meanEdgLen:
                print(f"  AVISO: o ponto usado esta a {desloc:.1f} mm do que "
                      f"voce marcou. Com apexSnap='epi' isso quer dizer que o")
                print("         seu ponto nao estava sobre o epicardio; "
                      "confira a coordenada do apex.")

        self.available.apex = True

    def _marcado_e_epicardico(self, cls, s2v, P, idMarcado, epi):
        """O ponto class=5 estava no epicárdio antes de ser marcado?

        A `class` original dele se perdeu ao virar 5, então a pergunta é
        respondida pela geometria: um ponto do epicárdio tem vizinhos
        epicárdicos a uma aresta de distância, enquanto o endocárdio mais
        próximo está uma espessura de parede adiante — vários milímetros. Já
        um ponto marcado SOBRE o endocárdio tem a relação invertida.

        Decidir isso importa: mantido, o ápice é exatamente o que o usuário
        pediu; trocado sem necessidade, ele anda uma aresta sem motivo.
        """
        endo = np.unique(np.concatenate([s2v[cls == c] for c in (3, 4, 6)]))
        endo = endo[endo != idMarcado]
        if len(endo) == 0:
            return True        # sem endocárdio para comparar: confie no ponto

        p = P[idMarcado]
        d_epi = float(np.sqrt(np.min(np.sum((P[epi] - p) ** 2, axis=1))))
        d_endo = float(np.sqrt(np.min(np.sum((P[endo] - p) ** 2, axis=1))))
        return d_epi <= d_endo

    def _resolve_apex(self, cls, s2v, P):
        """Decide qual nó de volume será o ápice epicárdico efetivo.

        Devolve (idMarcado, idEpiApex, deslocamento_mm). Veja a nota sobre
        `apexSnap` no topo do módulo: é aqui que o ponto fornecido pelo
        usuário é respeitado ou substituído.
        """
        marcados = s2v[cls == 5]
        if len(marcados) == 0:
            raise RuntimeError(
                "Nenhum ponto com class=5 (ápice) na superfície. O arquivo "
                "'<prefixo>.vtp' precisa ter exatamente um ponto marcado.")
        if len(marcados) > 1:
            warnings.warn(f'{len(marcados)} pontos com class=5; usando o '
                          f'primeiro, como no Cobiveco original.')
        idMarcado = int(marcados[0])

        modo = self.cfg.apexSnap
        epi = np.unique(s2v[cls == 2])

        def mais_proximo(candidatos):
            d = np.sum((P[candidatos] - P[idMarcado]) ** 2, axis=1)
            return int(candidatos[int(np.argmin(d))])

        if modo == 'marked':
            idEpiApex, origem = idMarcado, 'ponto marcado (sem snap)'

        elif modo == 'rv-epi':
            # comportamento literal do @uvc/computeApicobasal.m
            cand = np.intersect1d(epi, np.where(self.bv.tv == 1)[0])
            if len(cand):
                idEpiApex = mais_proximo(cand)
                origem = 'nó epicárdico do VD (tv=1) mais próximo'
            elif len(epi):
                warnings.warn('Nenhum nó epicárdico com tv=1; caindo para o '
                              'epicárdio inteiro.')
                idEpiApex = mais_proximo(epi)
                origem = 'nó epicárdico mais próximo (não havia tv=1)'
            else:
                idEpiApex, origem = idMarcado, 'ponto marcado (sem epicárdio)'

        else:   # 'epi'
            if len(epi) == 0:
                warnings.warn('Nenhum nó com class=2 (epicárdio); o ponto '
                              'marcado é usado sem snap.')
                idEpiApex, origem = idMarcado, 'ponto marcado (sem epicárdio)'
            elif self._marcado_e_epicardico(cls, s2v, P, idMarcado, epi):
                # O ponto marcado perdeu a class que tinha (virou 5), então não
                # aparece entre os candidatos `class==2` e um snap cego o
                # deslocaria uma aresta à toa. Se ele já ERA epicárdico,
                # mantenha-o exatamente onde está.
                idEpiApex, origem = idMarcado, 'ponto marcado (já epicárdico)'
            else:
                idEpiApex = mais_proximo(epi)
                origem = 'nó epicárdico mais próximo (o ponto marcado não era)'

        desloc = float(np.linalg.norm(P[idEpiApex] - P[idMarcado]))
        self.apexInfo = {
            'modo': modo,
            'origem': origem,
            'idMarcado': idMarcado,
            'idUsado': idEpiApex,
            'pontoMarcado': P[idMarcado].tolist(),
            'pontoUsado': P[idEpiApex].tolist(),
            'deslocamento': desloc,
            'meanEdgLen': self.meanEdgLen,
        }
        return idMarcado, idEpiApex, desloc

    def computeApicobasal(self):
        if not self.available.splitMesh:
            self.splitMesh()
        self.printStatus('Computing apicobasal coordinate...')
        t = time.time()

        cls = self.bv.sur.point_data['class']
        s2v = self.bv.surToVol
        P = self.bv.vol.points

        idsBase = s2v[cls == 1]
        if len(idsBase) == 0:
            raise RuntimeError('Nenhum ponto com class=1 (base) na superfície.')

        # O ápice já foi fixado por resolveApex(), antes do splitMesh, e é o
        # mesmo nó que o tm e o rt enxergam.
        idEpiApex = int(self.bv.idEpiApex)

        idsEndoLv = s2v[cls == 3]
        idEndoLvApex = int(idsEndoLv[np.argmin(
            np.sum((P[idsEndoLv] - P[idEpiApex])**2, axis=1))])
        idsEndoRv = s2v[cls == 4]
        idEndoRvApex = int(idsEndoRv[np.argmin(
            np.sum((P[idsEndoRv] - P[idEpiApex])**2, axis=1))])

        self.lvApexVec = P[idEndoLvApex] - P[idEpiApex]
        self.rvApexVec = P[idEndoRvApex] - P[idEpiApex]

        # Um apexVec degenerado (ápice marcado SOBRE o endocárdio) faria
        # _apex_band dividir por ~0 e a banda de Dirichlet viraria ruído.
        for nome, vec in (('lvApexVec', self.lvApexVec),
                          ('rvApexVec', self.rvApexVec)):
            n = float(np.linalg.norm(vec))
            if n < 0.1 * self.meanEdgLen:
                raise RuntimeError(
                    f'{nome} é degenerado (|{nome}| = {n:.4f} mm, aresta média '
                    f'{self.meanEdgLen:.3f} mm): o ponto usado como ápice está '
                    f'praticamente sobre o endocárdio. Use apexSnap="epi" '
                    f'(padrão) ou marque o ápice no epicárdio.')

        idsApex = self._apex_band(P, idEpiApex, self.lvApexVec)
        idsApex = np.union1d(idsApex,
                             self._apex_band(P, idEpiApex, self.rvApexVec))

        ids = np.concatenate([idsApex, idsBase])
        val = np.concatenate([np.zeros(len(idsApex)), np.ones(len(idsBase))])
        ids, uniq = np.unique(ids, return_index=True); val = val[uniq]
        self.bv.abLaplace = solveLaplace(self.bv.L, ids, val,
                                         self.cfg.tol, self.cfg.maxit)

        # --- geodésica base->ápice na superfície p/ reparametrizar ab ---
        ab = self._reparam_ab_geodesic()

        # --- ab = 0 em um único nó: o ápice -------------------------------
        n_plato = int(np.sum(ab <= ab.min() + 1e-12))
        if self.cfg.abApexUnique:
            ab = self._unique_apex_ab(ab, idEpiApex, idsApex)

        self.bv.ab = ab
        self.result.point_data['ab'] = self.bv.ab.astype(np.float32)

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)

        if self.cfg.verbose:
            n_min = int(np.sum(ab == ab.min()))
            marca = '✓' if n_min == 1 else '✗'
            print(f'  banda de Dirichlet ab=0: {len(idsApex)} nos '
                  f'(plato bruto: {n_plato})')
            print(f'  {marca} ab minimo ({ab.min():.6f}) em {n_min} no(s)'
                  + (f' — id {int(np.argmin(ab))}, o apice'
                     if n_min == 1 else ''))
            print(f'  ab maximo ({ab.max():.6f}) em '
                  f'{int(np.sum(ab == ab.max()))} no(s) — o anel da base')

        self.available.apicobasal = True

    def _unique_apex_ab(self, ab, idEpiApex, idsApex=None):
        """Faz ab = 0 valer para exatamente um nó: o ápice.

        Por que existe
        --------------
        A condição de Dirichlet ab=0 não é imposta no ápice sozinho — um único
        nó fixo num Laplace 3D gera um poço tipo 1/r e o campo inteiro degenera
        —, e sim numa BANDA de dezenas de nós, o cilindro que o `_apex_band`
        recorta ao longo da espessura da parede apical. Todos eles saem do
        solver com abLaplace exatamente 0, e portanto com ab exatamente igual
        depois da reparametrização: o mínimo de ab é um platô, não um ponto.

        A correção não toca no Laplace (que continua bem posto): ela só
        redistribui os valores DENTRO do platô, proporcionalmente à distância
        geodésica de cada nó ao ápice, dentro da folga que vai até o menor ab
        de fora do platô. O ápice tem distância zero e fica sozinho no mínimo;
        os vizinhos sobem de forma monótona, sem criar degrau nem inverter a
        ordem com o restante do campo.
        """
        ab = np.asarray(ab, dtype=float).copy()
        ab_min = float(ab.min())

        # O platô são os nós que o Dirichlet prendeu em zero, e não os que
        # empatam com ab.min() por tolerância absoluta: ab.min() costuma ser
        # um -1e-10 vindo da extrapolação do np.interp em
        # _reparam_ab_geodesic, de modo que `ab <= ab_min + 1e-12` pegava UM
        # nó e deixava intocados todos os que estão em 0.0 exato.
        if idsApex is not None:
            plato = np.asarray(idsApex, dtype=np.int64)
        else:
            escala = max(abs(ab_min), float(ab.max()) - ab_min, 1.0)
            plato = np.where(ab <= ab_min + 1e-9 * escala)[0]
        if len(plato) <= 1:
            return ab

        mascara_fora = np.ones(len(ab), dtype=bool)
        mascara_fora[plato] = False
        fora = ab[mascara_fora]
        if fora.size == 0:
            return ab
        folga = float(fora.min()) - ab_min
        if not np.isfinite(folga) or folga <= 0:
            return ab

        d = self._geodesic_from(idEpiApex, plato)
        d_max = float(d.max())
        if d_max <= 0:
            return ab

        # 0.95 deixa o nó mais alto do platô estritamente abaixo do primeiro
        # valor de fora: a ordem global do campo não muda.
        ab[plato] = ab_min + 0.95 * folga * (d / d_max)
        ab[idEpiApex] = ab_min
        return ab

    def _geodesic_from(self, idOrigem, alvos):
        """Distância, ao longo das arestas dos tets, de `idOrigem` a `alvos`.

        O grafo é montado só com os tetraedros que tocam o conjunto de alvos —
        algumas dezenas de células —, então o Dijkstra é instantâneo mesmo em
        malhas de milhões de nós. Alvo que ficar inalcançável (platô partido
        em duas ilhas) cai para a distância euclidiana, que nessa escala é
        praticamente a mesma coisa.
        """
        import scipy.sparse as sp

        P = self.bv.vol.points
        T = self.bv.cells
        alvos = np.asarray(alvos, dtype=np.int64)

        marca = np.zeros(P.shape[0], dtype=bool)
        marca[alvos] = True
        marca[idOrigem] = True
        tets = T[np.any(marca[T], axis=1)]
        if len(tets) == 0:
            return np.linalg.norm(P[alvos] - P[idOrigem], axis=1)

        pares = [(a, b) for a in range(4) for b in range(a + 1, 4)]
        e0 = np.concatenate([tets[:, a] for a, _ in pares])
        e1 = np.concatenate([tets[:, b] for _, b in pares])
        w = np.linalg.norm(P[e0] - P[e1], axis=1)
        n = P.shape[0]
        G = sp.csr_matrix((np.concatenate([w, w]),
                           (np.concatenate([e0, e1]),
                            np.concatenate([e1, e0]))), shape=(n, n))

        d_todos = dijkstra(G, indices=int(idOrigem), directed=False)
        d = d_todos[alvos]
        ruim = ~np.isfinite(d)
        if ruim.any():
            d[ruim] = np.linalg.norm(P[alvos[ruim]] - P[idOrigem], axis=1)
        return d

    def _apex_band(self, P, idEpiApex, apexVec):
        """Nós dentro da faixa apical ao longo de apexVec (réplica de d1/d2/d3).

        Diferença em relação ao original: o COMPRIMENTO do cilindro é limitado
        a `cfg.abBandLength` arestas médias.

        No Cobiveco original d3 = |apexVec|, e apexVec vai do ápice epicárdico
        até o nó endocárdico mais próximo. Na geo1 isso é a espessura da parede
        apical — poucos milímetros. Em malha de paciente o ápice é maciço (a
        cavidade termina bem acima da ponta), esse vetor passa de 20 mm, e a
        banda prende ab = 0 numa coluna que sobe quase até o meio do
        ventrículo: é a "mancha alongada" que aparece no ParaView em vez de um
        ápice pontual, e ela achata o ab justamente onde ele deveria variar.

        A banda só existe para dar suporte ao Dirichlet (um nó isolado num
        Laplace 3D degenera num poço 1/r). Poucas arestas bastam.
        """
        rel = P - P[idEpiApex]
        d1 = np.sqrt(np.sum(np.cross(rel, np.broadcast_to(apexVec, rel.shape))**2,
                            axis=1) / np.sum(apexVec**2))
        d2 = rel @ apexVec / np.linalg.norm(apexVec)
        d3 = apexVec @ apexVec / np.linalg.norm(apexVec)
        e = self.meanEdgLen
        limite = self.cfg.abBandLength * e
        if d3 > limite:
            if self.cfg.verbose:
                print(f'  banda apical truncada: {d3:.2f} mm -> {limite:.2f} mm '
                      f'({self.cfg.abBandLength:.1f} arestas medias)')
            d3 = limite
        return np.where((d1 < 1.5 * e) & (d2 > -e) & (d2 < d3 + e))[0]

    def _reparam_ab_geodesic(self):
        """
        Reparametriza o campo de Laplace apicobasal para que ab seja
        proporcional à distância geodésica real base->ápice na superfície.

        O original usa fast marching + extração de geodésica (numerical-tours).
        Aqui usamos o caminho geodésico de Dijkstra sobre as arestas da
        superfície (mesma ideia: um caminho base->ápice sobre o tecido).
        """
        sur = self.bv.sur
        cls = sur.point_data['class']
        Psur = sur.points
        cells = vu.get_cells(sur)

        # grafo de arestas da superfície, pesos = comprimento euclidiano
        import scipy.sparse as sp
        e = np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
        w = np.sqrt(np.sum((Psur[e[:, 0]] - Psur[e[:, 1]])**2, axis=1))
        n = sur.n_points
        G = sp.csr_matrix((np.concatenate([w, w]),
                          (np.concatenate([e[:, 0], e[:, 1]]),
                           np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n))

        startIds = np.where(cls == 1)[0]          # base
        endId = int(np.where(cls == 5)[0][0])     # ápice

        dist, pred, sources = dijkstra(G, indices=startIds,
                                       return_predecessors=True,
                                       min_only=True, directed=False)
        # reconstrói caminho do ápice de volta até a base
        path = [endId]
        cur = endId
        while pred[cur] >= 0:
            cur = pred[cur]
            path.append(cur)
        path = np.array(path[::-1])               # base -> ápice

        if len(path) < 2:
            raise RuntimeError(
                'O caminho geodésico base→ápice ficou vazio: o nó do ápice '
                'está desconectado da base na superfície.')

        # abLaplace ao longo do caminho (nos nós da superfície)
        abL_vol = self.bv.abLaplace[self.bv.surToVol]
        abPath = abL_vol[path]
        # distância acumulada normalizada ao longo do caminho
        seg = np.sqrt(np.sum(np.diff(Psur[path], axis=0)**2, axis=1))
        gdist = np.concatenate([[0], np.cumsum(seg)])
        gdist = gdist / gdist[-1]
        # reorienta para casar com a convenção do abLaplace (apice=0, base=1);
        # `path` foi construido base->apice, entao gdist sai com base=0/apice=1
        # sem esta linha, invertendo o sentido final de ab.
        gdist = 1.0 - gdist

        # monotoniza (abLaplace deve crescer base->ápice)
        order = np.argsort(abPath)
        abMono = abPath[order]
        gMono = gdist[order]
        abMono, uidx = np.unique(abMono, return_index=True)
        gMono = gMono[uidx]
        abMono = np.concatenate([[-1e-10], abMono, [1 + 1e-10]])
        gMono = np.concatenate([[-1e-10], gMono, [1 + 1e-10]])

        return np.interp(self.bv.abLaplace, abMono, gMono)

    # ==================================================================
    # 6) computeRotational
    # ==================================================================
    def computeRotational(self):
        if not self.available.apicobasal:
            self.computeApicobasal()
        self.printStatus('Computing rotational coordinate...')
        t = time.time()

        from .rotational import compute_rotational
        compute_rotational(self)

        self.bv.rtSin = np.sin(self.bv.rt)
        self.bv.rtCos = np.cos(self.bv.rt)
        self.result.point_data['rt'] = self.bv.rt.astype(np.float32)
        self.result.point_data['rtSin'] = self.bv.rtSin.astype(np.float32)
        self.result.point_data['rtCos'] = self.bv.rtCos.astype(np.float32)

        # se o AHA-17 vem a seguir, ele grava o result.vtu ja com o campo novo
        if self.cfg.exportLevel > 0 and not self.cfg.aha17:
            vu.write(self.result, f'{self.cfg.outPrefix}result.vtu')

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)
        self.available.rotational = True

    # ==================================================================
    # 7) computeAHA17 — parcelamento biventricular em 34 segmentos
    # ==================================================================
    def computeAHA17(self):
        """
        Rotula cada no: 1..17 no VE (AHA) e 18..34 no VD — 34 regioes.

        Nao resolve nenhum Laplace: as UVC ja contem os dois eixos de que a
        grade precisa. ab da o nivel apicobasal (tercos + calota apical) e rt
        da o setor circunferencial (60 graus no basal/medial, 90 graus no
        apical), com rt = 0 no centro do septo.

        No VD ha um detalhe que precisa de tratamento: como o
        computeRotational so impoe Dirichlet nas duas juncoes VE-VD, o rt da
        parede livre do VD cobre apenas meia volta. Com aha17RVRtMode =
        'stretch' (padrao) esse arco e reparametrizado para a volta inteira
        antes do fatiamento, e ai os 17 setores do VD existem de fato. Veja
        cobiveco/aha17.py.
        """
        if not self.available.rotational:
            self.computeRotational()
        self.printStatus('Computing AHA parcellation (VE+VD)...')
        t = time.time()

        from .aha17 import (aha17_from_uvc, aha17_coronary, level_fractions,
                            segment_fractions, ventricle_fractions,
                            nodal_volumes)

        # peso de cada no = 1/4 do volume dos tets incidentes. Sem isso as
        # estatisticas medem refino de malha, nao massa de miocardio.
        try:
            self.bv.nodeVolume = nodal_volumes(self.bv.vol.points, self.bv.cells)
        except Exception as exc:
            self.bv.nodeVolume = None
            warnings.warn(f'Volumes nodais indisponiveis ({exc}); as '
                          f'estatisticas do AHA-17 vao contar nos.')

        self.bv.aha17, self.bv.aha17Info = aha17_from_uvc(
            self.bv.ab, self.bv.rt, self.bv.tv, tm=self.bv.tm,
            ab_apex=self.cfg.aha17ApexCap,
            septum_mode=self.cfg.aha17SeptumMode,
            rv=self.cfg.aha17RV, rv_model=self.cfg.aha17RVModel,
            rv_rt_mode=self.cfg.aha17RVRtMode, return_info=True)

        self.result.point_data['aha17'] = self.bv.aha17.astype(np.float32)
        if self.cfg.aha17Coronary:
            self.result.point_data['aha17Coronary'] = \
                aha17_coronary(self.bv.aha17,
                               incluir_vd=self.cfg.aha17RV).astype(np.float32)

        if self.cfg.exportLevel > 0:
            vu.write(self.result, f'{self.cfg.outPrefix}result.vtu')

        self.printStatus(f'{time.time()-t:.1f} seconds\n', True)

        if self.cfg.verbose:
            w = self.bv.nodeVolume
            frac = segment_fractions(self.bv.aha17, w)
            vazios = [s for s in range(1, 18) if frac[s] == 0.0]
            partes = ' | '.join(f'{k.split()[0]} {v:.0f}%'
                                for k, v in level_fractions(self.bv.aha17, w).items())
            vf = ventricle_fractions(self.bv.aha17, w)
            print(f'  VE (1-17): {partes}')

            if self.cfg.aha17RV and np.any(self.bv.aha17 >= 18):
                modelo = self.cfg.aha17RVModel
                modo = self.cfg.aha17RVRtMode
                frv = segment_fractions(self.bv.aha17, w, which='rv',
                                        rv_model=modelo, rv_rt_mode=modo)
                partes_vd = ' | '.join(
                    f'{k.split()[0]} {v:.0f}%'
                    for k, v in level_fractions(self.bv.aha17, w, which='rv',
                                                rv_model=modelo,
                                                rv_rt_mode=modo).items())
                ultimo = max(frv)
                print(f'  VD ({min(frv)}-{ultimo}): {partes_vd}')
                arco = self.bv.aha17Info['rv']
                if arco.get('abertura_deg') is not None:
                    print(f"  arco do rt no VD: "
                          f"{arco['abertura_deg']:.0f} graus"
                          + (f", esticado {arco['fator']:.2f}x para a volta "
                             f"inteira" if arco.get('esticar')
                             else ' (sem esticar)'))
                vazios_vd = [s for s in frv if frv[s] == 0.0]
                if vazios_vd and vazios_vd != [ultimo]:
                    print(f'  AVISO: segmentos do VD vazios {vazios_vd}.')
                print(f"  massa: VE {vf['VE']:.0f}% | VD {vf['VD']:.0f}%")

            if vazios:
                print(f'  AVISO: segmentos do VE vazios {vazios} — confira '
                      f'ab/rt no ParaView.')

        self.available.aha17 = True
