"""
Parcelamento biventricular em 34 segmentos a partir das UVC.

VE: AHA-17 (Cerqueira et al., Circulation 2002), rotulos 1..17.
VD: 17 segmentos com a MESMA grade (3 niveis + calota, 6/6/4/1 setores),
    rotulos 18..34 — como no exampleAHAParcellation.m do Cobiveco, que
    numera o VD somando 17 ao rotulo do VE. Total: 34 regioes.

Ideia
-----
O modelo AHA e, no fundo, uma grade cilindrica:

    eixo longo  ->  tercos apicobasais (basal / medial / apical) + calota
    angulo      ->  fatias de 60 graus (basal e medial) ou 90 graus (apical)

As UVC ja entregam esses dois eixos prontos e normalizados, entao a
parcelacao vira um "binning" em (ab, rt). Nenhuma geometria cartesiana e
necessaria: o trabalho pesado foi feito pelas solucoes de Laplace.

Convencao deste pipeline (confirmada em uvc.py / rotational.py)
---------------------------------------------------------------
    ab in [0, 1]        0 = apice, 1 = base
    rt in [-pi, pi]     0 = centro do septo
                        rt > 0 = parede anterior
                        rt < 0 = parede inferior (posterior)
                        rt = +-pi     = centro da parede livre do VE
                        rt = +-pi/2.5 = juncao VE-VD endocardica
                        rt = +-pi/2   = juncao VE-VD epicardica
    tv: 0 = VE, 1 = VD

Uso direto:
    from cobiveco.aha17 import aha17_from_uvc
    seg = aha17_from_uvc(ab, rt, tv, tm=tm)      # 1..17 no VE, 18..34 no VD
"""

import numpy as np

DEG = np.pi / 180.0

# ---------------------------------------------------------------------------
# Nomenclatura oficial do VE (Cerqueira et al. 2002, Figura 4)
# ---------------------------------------------------------------------------
SEGMENT_NAMES = {
    0: 'nao classificado',
    1: 'basal anterior',
    2: 'basal anterosseptal',
    3: 'basal inferosseptal',
    4: 'basal inferior',
    5: 'basal inferolateral',
    6: 'basal anterolateral',
    7: 'medial anterior',
    8: 'medial anterosseptal',
    9: 'medial inferosseptal',
    10: 'medial inferior',
    11: 'medial inferolateral',
    12: 'medial anterolateral',
    13: 'apical anterior',
    14: 'apical septal',
    15: 'apical inferior',
    16: 'apical lateral',
    17: 'apice (apical cap)',
}

# ---------------------------------------------------------------------------
# Ventriculo direito — tres esquemas
# ---------------------------------------------------------------------------
# 'mirror17' + rv_rt_mode='stretch'  (PADRAO)
#     17 segmentos (18..34) sobre a parede livre do VD, com o arco do rt
#     reparametrizado para a volta inteira. Todos os 34 rotulos existem.
#     Nomes por posicao na parede livre (ver nota no topo do modulo).
#
# 'mirror17' + rv_rt_mode='raw'
#     Porte literal do exampleAHAParcellation.m: mesma regra do VE aplicada
#     ao rt cru do VD, com os nomes do VE prefixados por "VD". Como o rt do
#     VD so cobre [-pi/2, +pi/2] nesta implementacao, varios setores ficam
#     vazios — `report` lista quais.
#
# 'freewall9'
#     Esquema proprio, anterior a este modulo: parede livre em 3 niveis x 3
#     setores (18..26). Mantido por compatibilidade. NAO da 34 regioes.

# Espelhamento literal do VE (modo 'raw')
RV_MIRROR17_NAMES = {17 + s: f'VD {SEGMENT_NAMES[s]}' for s in range(1, 18)}

# Nomes honestos do modo 'stretch': posicao dentro da parede livre.
# psi = 0 no centro da parede livre, psi = +-pi nas juncoes VE-VD,
# psi > 0 = lado anterior (mesmo sentido do rt do VE).
_RV_SETOR6 = {1: 'anterior medio',    2: 'anterior central',
              3: 'inferior central',  4: 'inferior medio',
              5: 'inferior juncao',   6: 'anterior juncao'}
_RV_SETOR4 = {13: 'anterior', 14: 'central', 15: 'inferior', 16: 'juncoes'}

RV_STRETCH17_NAMES = {}
for _s, _n in _RV_SETOR6.items():
    RV_STRETCH17_NAMES[17 + _s] = f'VD basal {_n}'
    RV_STRETCH17_NAMES[17 + _s + 6] = f'VD medial {_n}'
for _s, _n in _RV_SETOR4.items():
    RV_STRETCH17_NAMES[17 + _s] = f'VD apical {_n}'
RV_STRETCH17_NAMES[34] = 'VD calota apical'

RV_FREEWALL9_NAMES = {
    18: 'VD basal anterior',
    19: 'VD basal lateral',
    20: 'VD basal inferior',
    21: 'VD medial anterior',
    22: 'VD medial lateral',
    23: 'VD medial inferior',
    24: 'VD apical anterior',
    25: 'VD apical lateral',
    26: 'VD apical inferior',
}

# compatibilidade com versoes anteriores deste modulo
RV_SEGMENT_NAMES = RV_FREEWALL9_NAMES

RV_MODELS = {'mirror17': (18, 34), 'freewall9': (18, 26)}
RV_RT_MODES = ('stretch', 'raw')


def rv_model_info(modelo, rv_rt_mode='stretch'):
    """(nomes, primeiro_rotulo, ultimo_rotulo) de um modelo de VD."""
    if modelo not in RV_MODELS:
        raise ValueError(f"rv_model deve ser {sorted(RV_MODELS)}; recebi "
                         f"'{modelo}'.")
    if rv_rt_mode not in RV_RT_MODES:
        raise ValueError(f"rv_rt_mode deve ser {list(RV_RT_MODES)}; recebi "
                         f"'{rv_rt_mode}'.")
    lo, hi = RV_MODELS[modelo]
    if modelo == 'freewall9':
        nomes = RV_FREEWALL9_NAMES
    else:
        nomes = (RV_STRETCH17_NAMES if rv_rt_mode == 'stretch'
                 else RV_MIRROR17_NAMES)
    return nomes, lo, hi


def all_segment_names(rv_model='mirror17', rv_rt_mode='stretch'):
    """Dicionario completo de nomes (VE + VD) para o esquema escolhido."""
    nomes = dict(SEGMENT_NAMES)
    nomes.update(rv_model_info(rv_model, rv_rt_mode)[0])
    return nomes


ALL_SEGMENT_NAMES = all_segment_names('mirror17', 'stretch')

# Convencao rotacional do Cobiveco (rt em [0,1]), deduzida dos limiares do
# exampleAHAParcellation.m: o centro do septo cai em rt = 5/6 e o rt crescente
# caminha para o lado inferior. Usado so quando rt_in_radians=False.
COBIVECO_SEPTUM_CENTER = 5.0 / 6.0
COBIVECO_DIRECTION = -1

# Territorios coronarianos do VE (Cerqueira et al. 2002, Figura 5)
CORONARY_TERRITORY = {}
CORONARY_TERRITORY.update({s: 'LAD' for s in (1, 2, 7, 8, 13, 14, 17)})
CORONARY_TERRITORY.update({s: 'RCA' for s in (3, 4, 9, 10, 15)})
CORONARY_TERRITORY.update({s: 'LCX' for s in (5, 6, 11, 12, 16)})

# A parede livre do VD e irrigada pelos ramos marginais agudos da coronaria
# direita. Usado so quando `aha17_coronary(..., incluir_vd=True)`.
RV_CORONARY_TERRITORY = {s: 'RCA' for s in range(18, 35)}

# k = ceil(rt / 60deg) - 1 -> setor basal (medial = +6)
_SECTOR6 = {1: 1, 0: 2, -1: 3, -2: 4, -3: 5, 2: 6}
# k = ceil((rt + 45deg) / 90deg) - 1 -> setor apical
_SECTOR4 = {1: 13, 0: 14, -1: 15, 2: 16, -2: 16}

# Valores de rt onde o rotational.py crava as juncoes VE-VD
JUNCTION_ENDO = np.pi / 2.5     # 72 graus  (superficie interna)
JUNCTION_EPI = np.pi / 2.0      # 90 graus  (superficie externa)


# ---------------------------------------------------------------------------
# Utilitarios de coordenada
# ---------------------------------------------------------------------------
def wrap_to_pi(x):
    """Dobra um angulo qualquer para o intervalo (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    return -((-x + np.pi) % (2.0 * np.pi) - np.pi)


def rescale_septum(rt, junction_angle=JUNCTION_ENDO, target_angle=60.0 * DEG):
    """
    Reescala rt (linear por partes, preservando o sinal) para que a juncao
    VE-VD da UVC caia exatamente no limite do septo previsto pelo AHA.

    Por que isso existe
    -------------------
    O AHA assume, por convencao, que o septo ocupa 120 graus (+-60 graus a
    partir do centro), porque nas imagens ele e delimitado pelos pontos de
    insercao do VD. Ja a UVC crava a juncao em +-pi/2.5 = +-72 graus
    (endocardica) e +-pi/2 = +-90 graus (epicardica). Sem corrigir, uma faixa
    de ~12 graus que a UVC considera septo e rotulada como parede anterior ou
    inferior nos segmentos 2, 3, 8 e 9.

    Dentro de |rt| <= junction_angle a escala e uniforme; fora dela tambem,
    com outra inclinacao. O mapa e continuo, monotono e preserva rt = 0 e
    rt = +-pi.
    """
    rt = np.asarray(rt, dtype=float)
    a = np.abs(rt)
    dentro = a * (target_angle / junction_angle)
    fora = target_angle + (a - junction_angle) * (
        (np.pi - target_angle) / (np.pi - junction_angle))
    return np.sign(rt) * np.where(a <= junction_angle, dentro, fora)


def estimate_apex_cap(ab, tm, lv_mask=None, tm_endo_tol=1e-2, percentile=1.0):
    """
    Estima `ab_apex`, o limite superior da calota apical (segmento 17).

    O AHA define o segmento 17 como "o miocardio alem do fim da cavidade".
    Em UVC isso e o ab abaixo do qual nao existe mais superficie endocardica,
    ou seja, o menor ab entre os nos com tm ~ 0. Usa-se um percentil baixo,
    e nao o minimo absoluto, para nao ser capturado por um no espurio.

    Devolve 0.05 se nao houver nos endocardicos identificaveis.
    """
    ab = np.asarray(ab, dtype=float).ravel()
    tm = np.asarray(tm, dtype=float).ravel()
    mask = tm <= tm_endo_tol
    if lv_mask is not None:
        mask = mask & np.asarray(lv_mask, dtype=bool).ravel()
    if not np.any(mask):
        return 0.05
    return float(np.percentile(ab[mask], percentile))


# ---------------------------------------------------------------------------
# Arco rotacional do VD
# ---------------------------------------------------------------------------
def _arc_center_half(phi, mask, percentile=0.5):
    """
    Centro e semi-abertura do arco ocupado por `phi[mask]`, em radianos.

    Faz a media de forma CIRCULAR (via numeros complexos) antes de medir os
    percentis, de modo que um arco que porventura atravesse +-pi nao vire um
    intervalo artificialmente gigante. `percentile` apara as pontas: o
    preenchimento por vizinho mais proximo do rotational.py pode deixar
    alguns nos fora do arco real.

    Retorna (centro, semi_abertura) ou (None, None) se nao houver nos.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    a = phi[np.asarray(mask, dtype=bool).ravel()]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    z = np.exp(1j * a).mean()
    c = float(np.angle(z)) if np.abs(z) > 1e-12 else 0.0
    d = wrap_to_pi(a - c)
    lo = float(np.percentile(d, percentile))
    hi = float(np.percentile(d, 100.0 - percentile))
    half = 0.5 * (hi - lo)
    centro = float(wrap_to_pi(c + 0.5 * (lo + hi)))
    return centro, half


def rv_rotational_info(rt, tv, lv_value=0.0, lv_tol=0.5, percentile=0.5,
                       rt_in_radians=True,
                       rt_septum_center=COBIVECO_SEPTUM_CENTER,
                       rt_direction=COBIVECO_DIRECTION):
    """
    Diagnostico do arco rotacional do VD — serve para explicar (e justificar)
    o 'stretch'.

    Retorna um dict com:
        n           numero de nos do VD
        centro_deg  centro do arco (em graus, no referencial do rt do VE)
        abertura_deg  abertura total do arco
        fator       pi / semi_abertura (quanto o arco e esticado)
        esticar     True se o 'stretch' muda alguma coisa
    """
    phi = _to_phi(rt, rt_in_radians, rt_septum_center, rt_direction)
    vd = _vd_mask(phi, tv, lv_value, lv_tol)
    centro, half = _arc_center_half(phi, vd, percentile)
    n = int(vd.sum())
    if centro is None or half is None or half <= 1e-6:
        return {'n': n, 'centro_deg': None, 'abertura_deg': None,
                'fator': None, 'esticar': False}
    abertura = 2.0 * half
    return {'n': n,
            'centro_deg': centro / DEG,
            'abertura_deg': abertura / DEG,
            'fator': np.pi / half,
            'esticar': bool(abertura < 0.9 * 2.0 * np.pi)}


def _to_phi(rt, rt_in_radians, rt_septum_center, rt_direction):
    rt = np.asarray(rt, dtype=float).ravel()
    if rt_in_radians:
        return wrap_to_pi(rt)
    if rt_direction not in (+1, -1):
        raise ValueError('rt_direction deve ser +1 ou -1.')
    return wrap_to_pi(2.0 * np.pi * rt_direction * (rt - rt_septum_center))


def _vd_mask(phi, tv, lv_value, lv_tol):
    if tv is None:
        return np.zeros(phi.shape, dtype=bool)
    tv = np.asarray(tv, dtype=float).ravel()
    return (np.abs(tv - lv_value) > lv_tol) & np.isfinite(phi)


# ---------------------------------------------------------------------------
# Nucleo
# ---------------------------------------------------------------------------
def _resolve_ab_bounds(ab, tm, lv, ab_apex, ab_bounds):
    """Resolve (ab_apex, b1, b2) — as duas fronteiras dos tercos apicobasais."""
    if ab_apex is None:
        if tm is None:
            raise ValueError('ab_apex=None exige o array `tm`.')
        ab_apex = estimate_apex_cap(ab, tm, lv_mask=lv)
    ab_apex = float(np.clip(ab_apex, 0.0, 0.9))

    if ab_bounds is None:
        span = 1.0 - ab_apex
        return ab_apex, ab_apex + span / 3.0, ab_apex + 2.0 * span / 3.0

    b1, b2 = (float(v) for v in ab_bounds)
    if not (ab_apex <= b1 < b2 <= 1.0):
        raise ValueError('ab_bounds deve satisfazer ab_apex <= b1 < b2 <= 1.')
    return ab_apex, b1, b2


def _sectors(phi):
    """(setor de 60 graus 1..6, setor de 90 graus 13..16) para cada angulo.

    ceil(x)-1 em vez de floor(x) => intervalos (lo, hi], do lado em que o rt
    normalizado do Cobiveco cresce. Isso faz deste um porte exato do
    exampleAHAParcellation.m, inclusive nas bordas.

    O arredondamento antes do ceil nao e cosmetico: um ponto exatamente sobre
    a fronteira de 120 graus vira 120.00000000000001 depois do wrap, e o ceil
    joga o no para o setor errado. Arredondar em 1e-9 de setor absorve isso
    sem afetar no algum que esteja de fato no interior.
    """
    k6 = np.clip(np.ceil(np.round(phi / (60.0 * DEG), 9)).astype(int) - 1,
                 -3, 2)
    sect6 = np.zeros_like(k6)
    for chave, valor in _SECTOR6.items():
        sect6[k6 == chave] = valor

    k4 = np.clip(
        np.ceil(np.round((phi + 45.0 * DEG) / (90.0 * DEG), 9)).astype(int) - 1,
        -2, 2)
    sect4 = np.zeros_like(k4)
    for chave, valor in _SECTOR4.items():
        sect4[k4 == chave] = valor
    return sect6, sect4


def _grid17(phi, is_bas, is_mid, is_api, is_cap):
    """Rotulo 1..17 da grade AHA para um angulo ja no referencial certo."""
    sect6, sect4 = _sectors(phi)
    out = np.zeros(phi.shape, dtype=np.int32)
    out[is_bas] = sect6[is_bas]
    out[is_mid] = sect6[is_mid] + 6
    out[is_api] = sect4[is_api]
    out[is_cap] = 17
    return out


def aha17_from_uvc(ab, rt, tv=None, tm=None,
                   ab_apex=0.05, ab_bounds=None,
                   septum_mode='aha', junction_angle=JUNCTION_ENDO,
                   rt_in_radians=True, lv_value=0.0, lv_tol=0.5,
                   rv=True, rv_model='mirror17', rv_rt_mode='stretch',
                   rv_span_percentile=0.5,
                   rt_septum_center=COBIVECO_SEPTUM_CENTER,
                   rt_direction=COBIVECO_DIRECTION,
                   return_info=False):
    """
    Rotulo de cada no: 1..17 no VE (AHA) e 18..34 no VD — 34 regioes.

    Parametros
    ----------
    ab : array
        Coordenada apicobasal (0 = apice, 1 = base).
    rt : array
        Coordenada rotacional em radianos (-pi, pi]. Se o seu rt estiver
        normalizado em [0, 1], passe `rt_in_radians=False`.
    tv : array, opcional
        Coordenada transventricular. Se None, assume que tudo e VE (e entao
        nao ha VD para rotular).
    tm : array, opcional
        Coordenada transmural; usada so quando `ab_apex=None`, para estimar
        a extensao da calota apical.
    ab_apex : float ou None
        Limite superior do segmento 17. None + `tm` => estimativa automatica.
        0.0 desativa a calota (vira o modelo classico de 16 segmentos).
    ab_bounds : (float, float), opcional
        Fronteiras apical|medial e medial|basal em ab. O padrao divide
        [ab_apex, 1] em tres partes iguais, que e a recomendacao do AHA
        ("equal thirds perpendicular to the long axis") — legitima aqui
        porque o ab do cobiveco e reparametrizado pela distancia geodesica
        apice-base, e nao pelo campo de Laplace cru. As MESMAS fronteiras
        valem para o VD, para que os niveis sejam comparaveis entre os dois
        ventriculos.
    septum_mode : {'aha', 'junction'}
        'aha'      : limites rigidos de 60 graus a partir do centro do septo.
        'junction' : alinha a juncao VE-VD da UVC (+-`junction_angle`) com os
                     +-60 graus do AHA. Veja `rescale_septum`. Afeta so o VE.
    junction_angle : float
        Angulo da juncao a alinhar: `JUNCTION_ENDO` (pi/2.5, padrao) ou
        `JUNCTION_EPI` (pi/2).
    lv_value, lv_tol : float
        Valor de `tv` que marca o VE e tolerancia da comparacao.
    rv : bool
        Se True (padrao), rotula tambem o VD.
    rv_model : {'mirror17', 'freewall9'}
        'mirror17' (padrao): 17 segmentos no VD, rotulos 18..34 — e o que da
        as 34 regioes. 'freewall9': parede livre em 3x3 (18..26).
    rv_rt_mode : {'stretch', 'raw'}
        So para 'mirror17'. 'stretch' (padrao) reparametriza o arco do rt do
        VD para a volta inteira, de modo que os 17 setores existam de fato;
        'raw' usa o rt cru (porte literal do exampleAHAParcellation.m, com
        setores vazios nesta implementacao do computeRotational).
    rv_span_percentile : float
        Percentil usado para aparar as pontas ao medir o arco do VD.
    rt_septum_center, rt_direction : float, int
        So usados quando `rt_in_radians=False` (rt normalizado em [0,1]):
        valor de rt no centro do septo e sentido em que rt cresce (+1 rumo a
        parede anterior, -1 rumo a inferior). Os padroes sao os do Cobiveco.
    return_info : bool
        Se True, devolve `(seg, info)` com o diagnostico do arco do VD e as
        fronteiras apicobasais efetivamente usadas.

    Retorna
    -------
    seg : ndarray de int32, mesmo tamanho de ab.
    """
    ab = np.asarray(ab, dtype=float).ravel()
    rt = np.asarray(rt, dtype=float).ravel()
    if ab.shape != rt.shape:
        raise ValueError('`ab` e `rt` devem ter o mesmo tamanho.')
    if septum_mode not in ('aha', 'junction'):
        raise ValueError("septum_mode deve ser 'aha' ou 'junction'.")

    # --- 1. mascara do VE ---------------------------------------------------
    if tv is None:
        lv = np.ones_like(ab, dtype=bool)
    else:
        tv_arr = np.asarray(tv, dtype=float).ravel()
        if tv_arr.shape != ab.shape:
            raise ValueError('`tv` deve ter o mesmo tamanho de `ab`.')
        lv = np.abs(tv_arr - lv_value) <= lv_tol

    # --- 2. angulo ----------------------------------------------------------
    phi = _to_phi(rt, rt_in_radians, rt_septum_center, rt_direction)

    finito = np.isfinite(ab) & np.isfinite(phi)
    lv = lv & finito
    vd = (~lv) & finito if tv is not None else np.zeros(ab.shape, dtype=bool)

    phi_lv = rescale_septum(phi, junction_angle=junction_angle) \
        if septum_mode == 'junction' else phi
    phi_lv = np.where(finito, phi_lv, 0.0)

    # --- 3. niveis apicobasais (os mesmos nos dois ventriculos) -------------
    ab_apex, b1, b2 = _resolve_ab_bounds(ab, tm, lv, ab_apex, ab_bounds)

    # Intervalos (lo, hi], como no exampleAHAParcellation.m ("ab > 7/10",
    # "ab <= 7/10 && ab > 4/10", ..., "ab <= 1/10").
    is_cap = (ab <= ab_apex) if ab_apex > 0 else np.zeros(ab.shape, bool)
    is_api = (ab > ab_apex) & (ab <= b1)
    is_mid = (ab > b1) & (ab <= b2)
    is_bas = ab > b2

    # --- 4. VE --------------------------------------------------------------
    seg = np.zeros(ab.shape, dtype=np.int32)
    regra_lv = _grid17(phi_lv, is_bas, is_mid, is_api, is_cap)
    seg[lv] = regra_lv[lv]

    info = {'ab_apex': ab_apex, 'ab_bounds': (b1, b2),
            'rv_model': rv_model, 'rv_rt_mode': rv_rt_mode,
            'rv': {'n': int(vd.sum()), 'centro_deg': None,
                   'abertura_deg': None, 'fator': None, 'esticar': False}}

    # --- 5. VD --------------------------------------------------------------
    if rv and vd.any():
        if tv is None:
            raise ValueError('rv=True exige o array `tv` para separar o VD.')
        rv_model_info(rv_model, rv_rt_mode)      # valida os nomes

        if rv_model == 'mirror17':
            psi = np.where(finito, phi, 0.0)

            if rv_rt_mode == 'stretch':
                centro, half = _arc_center_half(phi, vd, rv_span_percentile)
                info['rv']['centro_deg'] = (None if centro is None
                                            else centro / DEG)
                info['rv']['abertura_deg'] = (None if half is None
                                              else 2.0 * half / DEG)
                # Se o arco ja da (quase) a volta inteira, esticar so
                # distorceria: deixa como esta.
                if half is not None and 1e-6 < half < 0.9 * np.pi:
                    info['rv']['fator'] = np.pi / half
                    info['rv']['esticar'] = True
                    psi = np.clip(np.pi * wrap_to_pi(phi - centro) / half,
                                  -np.pi, np.pi)
                    psi = np.where(finito, psi, 0.0)

            regra_rv = _grid17(psi, is_bas, is_mid, is_api, is_cap)
            seg[vd] = regra_rv[vd] + 17
        else:
            # parede livre em 3 niveis x 3 setores de 60 graus (18..26)
            phi_f = np.where(finito, phi, 0.0)
            setor = np.full(ab.shape, 1, dtype=np.int32)   # 0 ant, 1 lat, 2 inf
            setor[phi_f > 30.0 * DEG] = 0
            setor[phi_f < -30.0 * DEG] = 2
            nivel = np.full(ab.shape, 2, dtype=np.int32)   # 0 bas, 1 med, 2 api
            nivel[is_bas] = 0
            nivel[is_mid] = 1
            seg[vd] = 18 + 3 * nivel[vd] + setor[vd]

    return (seg, info) if return_info else seg


# compatibilidade: o nome antigo continua valendo
aha_from_uvc = aha17_from_uvc


def aha17_coronary(seg, incluir_vd=True):
    """Territorio coronariano por no (0=fora, 1=LAD, 2=RCA, 3=LCX).

    `incluir_vd=True` atribui a parede livre do VD a coronaria direita
    (ramos marginais agudos). O AHA define territorios so para o VE.
    """
    seg = np.asarray(seg).ravel()
    codigo = {'LAD': 1, 'RCA': 2, 'LCX': 3}
    tabela = dict(CORONARY_TERRITORY)
    if incluir_vd:
        tabela.update(RV_CORONARY_TERRITORY)
    out = np.zeros_like(seg, dtype=np.int32)
    for s, terr in tabela.items():
        out[seg == s] = codigo[terr]
    return out


# ---------------------------------------------------------------------------
# Verificacao
# ---------------------------------------------------------------------------
def nodal_volumes(points, tets):
    """
    Volume associado a cada no: 1/4 do volume de cada tetraedro incidente.

    Serve de peso nas estatisticas por segmento. Sem isso, contar nos mede
    densidade de malha, nao massa de miocardio: uma regiao mais refinada
    aparece maior do que e.
    """
    P = np.asarray(points, dtype=float)
    T = np.asarray(tets, dtype=np.int64)
    if T.ndim != 2 or T.shape[1] != 4:
        raise ValueError('`tets` deve ser (n, 4). Malha nao tetraedrica?')
    a = P[T[:, 1]] - P[T[:, 0]]
    b = P[T[:, 2]] - P[T[:, 0]]
    c = P[T[:, 3]] - P[T[:, 0]]
    vol = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
    w = np.zeros(len(P))
    np.add.at(w, T.ravel(), np.repeat(vol / 4.0, 4))
    return w


def nodal_volumes_from_mesh(mesh):
    """`nodal_volumes` a partir de uma malha pyvista de tetraedros."""
    try:
        tets = mesh.cells_dict[10]          # 10 = VTK_TETRA
    except Exception:
        celulas = np.asarray(mesh.cells).reshape(-1, 5)
        if not np.all(celulas[:, 0] == 4):
            raise ValueError('A malha nao e puramente tetraedrica.')
        tets = celulas[:, 1:]
    return nodal_volumes(np.asarray(mesh.points), tets)


def detect_rv_model(seg):
    """Deduz o modelo de VD pelos rotulos presentes.

    Só distingue 'mirror17' de 'freewall9': o rt_mode nao deixa rastro nos
    rotulos (so nos nomes), entao passe-o explicitamente quando souber.
    """
    seg = np.asarray(seg).ravel()
    m = int(seg.max(initial=0))
    if m > 26:
        return 'mirror17'
    # 18..26 pode ser freewall9 OU mirror17/'raw' com setores vazios.
    # Na duvida, 'freewall9' e a leitura conservadora.
    return 'freewall9'


def segment_fractions(seg, weights=None, which='lv', rv_model='auto',
                      rv_rt_mode='stretch'):
    """
    Fracao (%) de cada segmento sobre o total do ventriculo correspondente.

    `weights` = volumes nodais (veja `nodal_volumes`).
    `which` = 'lv' (segmentos 1..17) ou 'rv' (18..34 ou 18..26). Cada um
    normaliza pelo seu proprio ventriculo, para que as porcentagens continuem
    somando 100 em cada lado.
    """
    seg = np.asarray(seg).ravel()
    w = np.ones(seg.size) if weights is None else np.asarray(weights, float).ravel()
    if which == 'lv':
        lo, hi = 1, 17
    else:
        if rv_model == 'auto':
            rv_model = detect_rv_model(seg)
        _, lo, hi = rv_model_info(rv_model, rv_rt_mode)
    dentro = (seg >= lo) & (seg <= hi)
    total = w[dentro].sum()
    if total <= 0:
        raise ValueError(f"Nenhum ponto rotulado no ventriculo '{which}'.")
    return {s: 100.0 * w[seg == s].sum() / total for s in range(lo, hi + 1)}


def ventricle_fractions(seg, weights=None):
    """Fracao do volume total em cada ventriculo (e fora de ambos)."""
    seg = np.asarray(seg).ravel()
    w = np.ones(seg.size) if weights is None else np.asarray(weights, float).ravel()
    total = w.sum()
    ve = w[(seg >= 1) & (seg <= 17)].sum()
    vd = w[seg >= 18].sum()
    return {'VE': 100.0 * ve / total, 'VD': 100.0 * vd / total,
            'sem rotulo': 100.0 * (total - ve - vd) / total}


def level_fractions(seg, weights=None, which='lv', rv_model='auto',
                    rv_rt_mode='stretch'):
    """Fracoes agregadas por nivel apicobasal, dentro do ventriculo pedido."""
    f = segment_fractions(seg, weights, which=which, rv_model=rv_model,
                          rv_rt_mode=rv_rt_mode)
    if which == 'lv':
        return {
            'basal (1-6)': sum(f[s] for s in range(1, 7)),
            'medial (7-12)': sum(f[s] for s in range(7, 13)),
            'apical (13-16)': sum(f[s] for s in range(13, 17)),
            'apice (17)': f[17],
        }
    if max(f) > 26:     # mirror17
        return {
            'basal (18-23)': sum(f[s] for s in range(18, 24)),
            'medial (24-29)': sum(f[s] for s in range(24, 30)),
            'apical (30-33)': sum(f[s] for s in range(30, 34)),
            'apice (34)': f[34],
        }
    return {
        'basal (18-20)': sum(f[s] for s in range(18, 21)),
        'medial (21-23)': sum(f[s] for s in range(21, 24)),
        'apical (24-26)': sum(f[s] for s in range(24, 27)),
    }


def report(seg, weights=None, arquivo=None, rv_model='auto',
           rv_rt_mode='stretch', rv_info=None):
    """Imprime (ou grava) um resumo legivel da parcelacao biventricular.

    `weights` = volumes nodais (veja `nodal_volumes`). Sem eles o relatorio
    conta nos, o que mede densidade de malha e nao massa — o cabecalho avisa.
    `rv_info` = o dicionario `info['rv']` devolvido por `aha17_from_uvc(...,
    return_info=True)`; se vier, o arco medido do VD entra no relatorio.
    """
    seg = np.asarray(seg).ravel()
    linhas = []
    f = segment_fractions(seg, weights)
    unidade = '%vol' if weights is not None else '%nos'
    if weights is None:
        linhas.append('  (sem pesos: porcentagens por CONTAGEM DE NOS, '
                      'sensiveis ao refino local da malha)')

    # --- repartição entre os ventrículos -----------------------------------
    vf = ventricle_fractions(seg, weights)
    linhas.append(f"  Volume por ventriculo: VE {vf['VE']:.1f}%  |  "
                  f"VD {vf['VD']:.1f}%  |  sem rotulo "
                  f"{vf['sem rotulo']:.1f}%")
    linhas.append('')

    # --- VE -----------------------------------------------------------------
    linhas.append('  VE — AHA 17 segmentos (Cerqueira et al. 2002). '
                  '% do volume do VE:')
    linhas.append(f"{'seg':>3}  {'nome':<24} {unidade:>6}  {'coronaria':>9}")
    for s in range(1, 18):
        linhas.append(f'{s:>3}  {SEGMENT_NAMES[s]:<24} {f[s]:6.2f}  '
                      f'{CORONARY_TERRITORY[s]:>9}')
    linhas.append('')
    for nivel, frac in level_fractions(seg, weights).items():
        linhas.append(f'  {nivel:<16s} {frac:5.1f} %')
    linhas.append('  (referencia anatomica do AHA para tercos iguais: '
                  '42 / 36 / 21)')
    vazios = [s for s in range(1, 18) if f[s] == 0.0]
    if vazios:
        linhas.append(f'  AVISO: segmentos do VE sem nenhum no: {vazios}')

    # --- simetria anterior x inferior, em torno do eixo do septo -----------
    linhas.append('')
    linhas.append('  Simetria anterior/inferior do VE (razao esperada ~1):')
    for ant, inf, rotulo in ((2, 3, 'septo basal'),
                             (1, 4, 'anterior/inferior basal'),
                             (6, 5, 'lateral basal'), (8, 9, 'septo medial'),
                             (7, 10, 'anterior/inferior medial'),
                             (12, 11, 'lateral medial'),
                             (13, 15, 'anterior/inferior apical')):
        if f[inf] <= 0 or f[ant] <= 0:
            linhas.append(f'    {rotulo:<26} {ant:>2}/{inf:<2} =     - '
                          f'(segmento vazio)')
            continue
        r = f[ant] / f[inf]
        marca = ' <-- assimetrico' if (r > 1.6 or r < 1 / 1.6) else ''
        linhas.append(f'    {rotulo:<26} {ant:>2}/{inf:<2} = {r:5.2f}{marca}')
    for nivel, (sep, livre) in {'basal': ((2, 3), (5, 6)),
                                'medial': ((8, 9), (11, 12))}.items():
        s_ = sum(f[x] for x in sep)
        l_ = sum(f[x] for x in livre)
        razao = f'{s_/l_:5.2f}' if l_ > 0 else '    -'
        linhas.append(f'    septo/parede livre {nivel:<9} '
                      f'{s_:.2f} / {l_:.2f} = {razao}')

    # --- VD ------------------------------------------------------------------
    if np.any(seg >= 18):
        modelo = detect_rv_model(seg) if rv_model == 'auto' else rv_model
        nomes_vd, lo, hi = rv_model_info(modelo, rv_rt_mode)
        frv = segment_fractions(seg, weights, which='rv', rv_model=modelo,
                                rv_rt_mode=rv_rt_mode)
        linhas.append('')
        if modelo == 'mirror17' and rv_rt_mode == 'stretch':
            linhas.append('  VD — 17 segmentos (18-34) sobre a parede livre, '
                          'com o arco do rt esticado para a volta')
            linhas.append('  inteira. Nomes por posicao na parede livre: '
                          '"central" = meio da parede livre,')
            linhas.append('  "juncao" = junto a insercao do VD no VE. '
                          '% do volume do VD:')
        elif modelo == 'mirror17':
            linhas.append('  VD — 17 segmentos espelhados do VE (18-34), rt '
                          'cru (exampleAHAParcellation.m do')
            linhas.append('  Cobiveco; o AHA so define o VE). '
                          '% do volume do VD:')
        else:
            linhas.append('  VD — parede livre, esquema proprio (NAO e AHA). '
                          '% do volume do VD:')
        if rv_info and rv_info.get('abertura_deg') is not None:
            linhas.append(f"  Arco do rt no VD: abertura "
                          f"{rv_info['abertura_deg']:.1f} graus, centro em "
                          f"{rv_info['centro_deg']:+.1f} graus"
                          + (f", esticado {rv_info['fator']:.2f}x"
                             if rv_info.get('esticar') else ', sem esticar'))
        linhas.append(f"{'seg':>5}  {'nome':<28} {unidade:>6}")
        for s in range(lo, hi + 1):
            linhas.append(f'{s:>5}  {nomes_vd[s]:<28} {frv[s]:6.2f}')

        linhas.append('')
        for rotulo, frac in level_fractions(
                seg, weights, which='rv', rv_model=modelo,
                rv_rt_mode=rv_rt_mode).items():
            linhas.append(f'  {rotulo:<16s} {frac:5.1f} %')

        vazios_vd = [s for s in range(lo, hi + 1) if frv[s] == 0.0]
        if vazios_vd:
            linhas.append('')
            linhas.append(f'  AVISO: segmentos do VD sem nenhum no: '
                          f'{vazios_vd}')
            if vazios_vd == [hi] and modelo == 'mirror17':
                linhas.append('  So a calota apical do VD: normal, a parede '
                              'livre do VD costuma terminar antes')
                linhas.append('  do apice do VE, e as fronteiras '
                              'apicobasais sao as do VE.')
            elif modelo == 'mirror17' and rv_rt_mode == 'raw':
                linhas.append('  Causa: no modo rv_rt_mode="raw" o rt do VD '
                              'cobre apenas [-pi/2, +pi/2] (meia')
                linhas.append('  volta), entao os setores fora dessa faixa '
                              'nao existem. Use rv_rt_mode="stretch".')
            else:
                linhas.append('  Confira o arco do rt no VD com: '
                              'rt[tv==1].min(), rt[tv==1].max()')

    texto = '\n'.join(linhas)
    if arquivo:
        with open(arquivo, 'w') as fh:
            fh.write(texto + '\n')
    else:
        print(texto)
    return texto


# ---------------------------------------------------------------------------
# Bull's eye (opcional, requer matplotlib)
# ---------------------------------------------------------------------------
def _cmap(nome):
    import matplotlib.pyplot as plt
    return plt.get_cmap(nome)


def _desenha_bullseye(ax, v, mapa, norm, rotulos):
    """Desenha os 4 aneis (6 + 6 + 4 + calota) num eixo polar.

    theta do matplotlib: 0 = direita. O angulo 0 da grade (centro do septo no
    VE, centro da parede livre no VD) fica a esquerda => theta = pi - angulo.
    """
    def anel(seg_ids, r0, r1, n):
        largura = 2.0 * np.pi / n
        for i, s in enumerate(seg_ids):
            a_c = (i + 0.5) * largura
            if i >= n / 2:
                a_c -= 2 * np.pi
            theta_c = np.pi - a_c
            ax.bar(theta_c, r1 - r0, width=largura, bottom=r0,
                   color=mapa(norm(v[s - 1])), edgecolor='w', linewidth=1.2,
                   align='center')
            ax.text(theta_c, (r0 + r1) / 2, rotulos[s - 1], ha='center',
                    va='center', fontsize=8, color='k')

    anel([2, 1, 6, 5, 4, 3], 0.70, 1.00, 6)        # basal
    anel([8, 7, 12, 11, 10, 9], 0.45, 0.70, 6)     # medial
    anel([14, 13, 16, 15], 0.20, 0.45, 4)          # apical
    ax.bar(0.0, 0.20, width=2 * np.pi, bottom=0.0,
           color=mapa(norm(v[16])), edgecolor='w', linewidth=1.2)
    ax.text(0.0, 0.10, rotulos[16], ha='center', va='center', fontsize=8)

    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)


def _vetor17(valores, lo=1):
    if isinstance(valores, dict):
        return np.array([valores.get(s, np.nan)
                         for s in range(lo, lo + 17)], float)
    v = np.asarray(valores, float).ravel()
    if v.size != 17:
        raise ValueError('`valores` deve ter 17 entradas.')
    return v


def bullseye(valores, titulo='', cmap='viridis', vmin=None, vmax=None,
             arquivo=None, ax=None, which='lv'):
    """
    Desenha o bull's eye padrao com um valor por segmento.

    `valores`: dict {1..17: valor} (VE) ou {18..34: valor} (VD), ou um array
    de 17 posicoes. A orientacao segue a Figura 4 do AHA: anterior em cima,
    angulo 0 a esquerda, inferior embaixo. No VD o angulo 0 e o centro da
    parede livre (veja a nota de nomenclatura no topo do modulo).
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    lo = 1 if which == 'lv' else 18
    v = _vetor17(valores, lo)
    rotulos = [str(lo + i) for i in range(17)]

    criado = ax is None
    if criado:
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                             figsize=(5.2, 5.2))

    vmin = np.nanmin(v) if vmin is None else vmin
    vmax = np.nanmax(v) if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapa = _cmap(cmap)

    _desenha_bullseye(ax, v, mapa, norm, rotulos)
    if titulo:
        ax.set_title(titulo)
    if criado:
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=mapa), ax=ax,
                     shrink=0.7, pad=0.08)
        if arquivo:
            plt.savefig(arquivo, dpi=200, bbox_inches='tight')
        else:
            plt.show()
    return ax


def bullseye_pair(f_lv, f_rv, titulo='', cmap='viridis', vmin=None, vmax=None,
                  arquivo=None):
    """Dois bull's eyes lado a lado: VE (1-17) e VD (18-34), 34 regioes."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    v_lv = _vetor17(f_lv, 1)
    v_rv = _vetor17(f_rv, 18)
    todos = np.concatenate([v_lv, v_rv])
    vmin = np.nanmin(todos) if vmin is None else vmin
    vmax = np.nanmax(todos) if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapa = _cmap(cmap)

    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                             figsize=(10.4, 5.4))
    _desenha_bullseye(axes[0], v_lv, mapa, norm,
                      [str(s) for s in range(1, 18)])
    axes[0].set_title('VE (1-17) — 0 = centro do septo')
    _desenha_bullseye(axes[1], v_rv, mapa, norm,
                      [str(s) for s in range(18, 35)])
    axes[1].set_title('VD (18-34) — 0 = centro da parede livre')
    if titulo:
        fig.suptitle(titulo)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=mapa), ax=axes,
                 shrink=0.7, pad=0.06)
    if arquivo:
        plt.savefig(arquivo, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    return axes
