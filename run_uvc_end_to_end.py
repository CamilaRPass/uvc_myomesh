#!/usr/bin/env python3
"""
run_uvc_end_to_end.py
---------------------
Um comando só: malha biventricular volumétrica (.vtu) → coordenadas UVC.

Encadeia os três blocos que hoje vivem separados:

    pipeline_UVC_completo.py   (.vtu → superfície .vtp com as 6 marcações)
            ↓
    uvc_bridge.py              (permuta rótulos, ajusta o lado do septo,
                                monta <nome>.vtu + <nome>_for_UVC.vtp,
                                valida o contrato)
            ↓
    cobiveco/ (UVC)            (tv, tm, ab, rt, rtSin, rtCos e o
                                parcelamento aha17 — 1-17 no VE e 18-34
                                no VD, 34 regiões → result.vtu)

Uso típico:

    python run_uvc_end_to_end.py Patient_10.vtu --caso patient10

Produz:

    uvc_patient10/preparo/heart.vtu
    uvc_patient10/preparo/heart_for_UVC.vtp     ← entrada do cobiveco
    uvc_patient10/marcacao/...                   ← intermediários do pipeline
    uvc_patient10/result/result.vtu              ← saída com as UVC

Pular etapas:

    --so-marcacao     para depois da marcação (confira no ParaView)
    --so-preparo      para depois da ponte + validação
    --do-vtp X.vtp    parte de um .vtp já marcado (pula o pipeline)

Dependências: vtk, pyvista, numpy, scipy
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

import uvc_bridge as ponte


# ════════════════════════════════════════════════════════════════════════════
# Carregamento dinâmico dos dois pacotes
# ════════════════════════════════════════════════════════════════════════════

def carregar_pipeline(path: str):
    """Importa o pipeline_UVC_completo.py de um caminho arbitrário."""
    if not os.path.isfile(path):
        sys.exit(f"Erro: pipeline não encontrado em '{path}'. "
                 f"Use --pipeline para indicar o caminho.")
    spec = importlib.util.spec_from_file_location("pipeline_UVC_completo",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_UVC_completo"] = mod
    spec.loader.exec_module(mod)
    for atributo in ("run_pipeline", "LABELS_UVC"):
        if not hasattr(mod, atributo):
            sys.exit(f"Erro: '{path}' não expõe '{atributo}'. "
                     f"É mesmo o pipeline_UVC_completo.py?")
    return mod


def carregar_uvc(raiz_cobiveco: str):
    """Importa a classe UVC do pacote cobiveco/."""
    raiz = os.path.abspath(raiz_cobiveco)
    if raiz not in sys.path:
        sys.path.insert(0, raiz)
    try:
        from cobiveco import UVC
    except ImportError as exc:
        sys.exit(f"Erro ao importar o pacote cobiveco a partir de '{raiz}': "
                 f"{exc}\n  Use --cobiveco para indicar o diretório que "
                 f"CONTÉM a pasta cobiveco/.")
    return UVC


# ════════════════════════════════════════════════════════════════════════════
# Resolução da entrada
# ════════════════════════════════════════════════════════════════════════════

def resolver_entrada(caminho: str) -> str:
    """Aceita 'Patient_1' ou 'Patient_1.vtu' — acrescenta a extensão se faltar."""
    if os.path.isfile(caminho):
        if os.path.splitext(caminho)[1].lower() != ".vtu":
            sys.exit(f"Erro: a entrada deve ser um .vtu (volume); "
                     f"recebi '{caminho}'.")
        return caminho
    if os.path.isfile(caminho + ".vtu"):
        return caminho + ".vtu"
    sys.exit(f"Erro: não encontrei '{caminho}' nem '{caminho}.vtu'.")


def procurar_apex(entrada: str):
    """Procura o arquivo do apex ao lado da malha, pelos nomes usuais."""
    base = os.path.splitext(entrada)[0]
    d = os.path.dirname(os.path.abspath(entrada)) or "."
    candidatos = [f"{base}_apex.txt", f"{base}_apex_coord.txt",
                  f"{base}_apex.csv",
                  os.path.join(d, "apex_coord.txt"),
                  os.path.join(d, "apex.txt")]
    for c in candidatos:
        if os.path.isfile(c):
            return c
    return None


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    aqui = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Pipeline de ponta a ponta: .vtu → marcação da "
                    "superfície → UVC (Cobiveco).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── entrada / saída ─────────────────────────────────────────────────
    p.add_argument("input",
                   help="Malha volumétrica de entrada; a extensão .vtu pode "
                        "ser omitida (Patient_1 == Patient_1.vtu)")
    p.add_argument("--caso", default=None,
                   help="Nome do caso (padrão: nome do arquivo de entrada)")
    p.add_argument("--outdir", default=None,
                   help="Diretório raiz das saídas (padrão: uvc_<caso>)")
    p.add_argument("--nome", default="heart",
                   help="Nome base dos arquivos que o cobiveco vai ler "
                        "(<nome>.vtu e <nome>_for_UVC.vtp)")
    p.add_argument("--do-vtp", default=None,
                   help="Parte de um .vtp JÁ marcado (pula o pipeline)")

    # ── localização dos módulos ─────────────────────────────────────────
    p.add_argument("--pipeline",
                   default=os.path.join(aqui, "pipeline_UVC_completo.py"),
                   help="Caminho do pipeline_UVC_completo.py")
    p.add_argument("--cobiveco", default=aqui,
                   help="Diretório que CONTÉM a pasta cobiveco/")

    # ── etapa 1: marcação (repassado ao pipeline) ───────────────────────
    g = p.add_argument_group("marcação da superfície (pipeline)")
    g.add_argument("--region-array", default=None,
                   help="Nome do array de regiões no .vtu")
    g.add_argument("--dist", type=float, default=15.0,
                   help="Distância máxima (mm) LV→RV para o septo")
    g.add_argument("--dot", type=float, default=-0.8,
                   help="Limiar do produto interno normal·dir_LV→RV")
    g.add_argument("--min-ilha", type=int, default=16,
                   help="Tamanho mínimo de ilha conexa (despeckle)")
    g.add_argument("--z-threshold", type=float, default=None,
                   help="Ativa a correção da Base por plano em Z")
    g.add_argument("--z-mode", choices=["below", "above"], default="below")
    g.add_argument("--apex", default=None,
                   help="Arquivo com o apex manual (Point ID ou x,y,z). Se "
                        "omitido, procura apex_coord.txt / <malha>_apex.txt "
                        "ao lado da malha")
    g.add_argument("--sem-apex", action="store_true",
                   help="Ignora qualquer arquivo de apex e deixa o pipeline "
                        "detectar o ápice sozinho")
    g.add_argument("--apex-mode", choices=["auto", "id", "xyz"],
                   default="auto",
                   help="Como LER o arquivo do apex — NÃO recebe um nome de "
                        "arquivo (esse vai em --apex)")
    g.add_argument("--apex-dist-max", type=float, default=0.10,
                   help="Distância máxima (fração da diagonal da malha) entre "
                        "a coordenada do --apex e o vértice mais próximo. "
                        "Acima disso a coordenada é rejeitada e o ápice é "
                        "detectado automaticamente")
    g.add_argument("--apex-estrito", action="store_true",
                   help="Aborta quando a coordenada do --apex está fora da "
                        "malha, em vez de cair para a detecção automática")
    g.add_argument("--triangular", action="store_true",
                   help="Triangula a superfície após a extração")
    g.add_argument("--keep-original-arrays", action="store_true",
                   help="Preserva fenotipo/tecido/f_0/s_0/n_0 na marcação")

    # ── etapa 2: ponte ──────────────────────────────────────────────────
    g = p.add_argument_group("ponte pipeline → cobiveco")
    g.add_argument("--permutar", choices=["nao", "rotulos"], default="nao",
                   help="'nao' (padrão): entrega as classes ao cobiveco "
                        "exatamente como o pipeline as produziu. 'rotulos': "
                        "reordena 1–6 conforme o LABELS_UVC do pipeline, de "
                        "modo que o VE rotulado receba tv=0")
    g.add_argument("--sem-permutar", action="store_true",
                   help="Obsoleto: já é o padrão. Aceito sem efeito")
    g.add_argument("--septo-face", choices=["auto", "lv", "rv"],
                   default="auto",
                   help="Face em que o septo (class 6) deve ficar. 'auto' "
                        "detecta e move para a face VD se necessário")
    g.add_argument("--septo-espessura", type=float, default=None,
                   help="Espessura septal (mm) usada ao mover o septo "
                        "(padrão: estimada da própria malha)")
    g.add_argument("--septo-fator", type=float, default=1.4,
                   help="Limiar de captura = fator × espessura + 2 × aresta. "
                        "Aumente se o septo sair pequeno demais, diminua se "
                        "invadir a parede livre do VD")

    # ── etapa 3: UVC ────────────────────────────────────────────────────
    g = p.add_argument_group("cálculo das UVC (cobiveco)")
    g.add_argument("--export-level", type=int, default=1, choices=[0, 1, 2])
    g.add_argument("--tol", type=float, default=1e-8)
    g.add_argument("--maxit", type=int, default=3000)
    g.add_argument("--mapping-tol", type=float, default=0.5)
    g.add_argument("--ab-banda", default="2.5",
                   help="Comprimento da banda de Dirichlet apical (ab=0), em "
                        "arestas medias. 'inf' reproduz o Cobiveco original "
                        "(banda ate o endocardio). Banda curta demais faz o "
                        "abLaplace subir como um poco 1/r junto ao apice, e o "
                        "computeRotational nao encontra a fatia apical; banda "
                        "longa demais achata o ab na ponta e desloca a "
                        "fronteira da calota apical (segmento 17 do AHA)")
    g.add_argument("--saida", default=None,
                   help="Caminho do .vtu final = entrada + UVC "
                        "(padrão: <outdir>/<caso>_UVC.vtu)")
    g.add_argument("--prefixo", default="",
                   help="Prefixo dos campos UVC no arquivo final "
                        "(ex.: uvc_ → uvc_tv, uvc_tm, ...)")
    g.add_argument("--apex-snap", choices=["epi", "marked", "rv-epi"],
                   default="epi",
                   help="O que o cobiveco faz com o ponto class=5. 'epi' "
                        "(padrão): usa o ápice marcado, fazendo snap para o "
                        "nó epicárdico mais próximo só se ele não estiver no "
                        "epicárdio. 'marked': usa o nó marcado sem snap "
                        "algum. 'rv-epi': comportamento literal do "
                        "computeApicobasal.m do Cobiveco, que troca o seu "
                        "ponto pelo nó epicárdico do VD mais próximo — é o "
                        "que fazia o arquivo do apex não ter efeito")
    g.add_argument("--ab-plato-apical", action="store_true",
                   help="Deixa ab = 0 em toda a banda de Dirichlet apical, "
                        "como no Cobiveco original. Por padrão o mínimo de ab "
                        "é redistribuído dentro da banda para que ab = 0 "
                        "ocorra em exatamente 1 nó: o ápice")
    g.add_argument("--sem-aha17", action="store_true",
                   help="Nao calcula o parcelamento (VE 1-17 + VD 18-34)")
    g.add_argument("--aha17-septo", choices=["aha", "junction"], default="aha",
                   help="'aha': septo rigido de +-60 graus. 'junction': "
                        "realinha a juncao VE-VD da UVC (+-pi/2.5) com os "
                        "+-60 graus do AHA")
    g.add_argument("--aha17-apex", type=float, default=None,
                   help="Limite em ab da tampa apical (segmento 17). Omitido: "
                        "estimado pelo fim do endocardio. 0 desativa o "
                        "segmento 17 (modelo de 16 segmentos)")
    g.add_argument("--aha17-vd", action="store_true",
                   help="Obsoleto: rotular o VD ja e o padrao. Aceito sem "
                        "efeito")
    g.add_argument("--sem-aha17-vd", action="store_true",
                   help="Rotula SO o VE (1-17); o VD fica com 0. Por padrao "
                        "o VD recebe 18-34, totalizando 34 regioes")
    g.add_argument("--aha17-vd-modelo", choices=["mirror17", "freewall9"],
                   default="mirror17",
                   help="'mirror17' (padrao): 17 segmentos no VD, 18-34 — a "
                        "numeracao do exampleAHAParcellation.m do Cobiveco. "
                        "'freewall9': parede livre em 3x3 (18-26)")
    g.add_argument("--aha17-vd-rt", choices=["stretch", "raw"],
                   default="stretch",
                   help="Como tratar o rt do VD, que nesta implementacao do "
                        "computeRotational cobre so meia volta. 'stretch' "
                        "(padrao) estica o arco para a volta inteira, de modo "
                        "que os 17 setores do VD existam; 'raw' usa o rt cru "
                        "e deixa varios segmentos vazios (porte literal do "
                        "exemplo do Cobiveco)")
    g.add_argument("--aha17-coronaria", action="store_true",
                   help="Exporta tambem o territorio coronariano "
                        "(1=LAD, 2=RCA, 3=LCX; parede livre do VD = RCA)")
    g.add_argument("--preservar-arrays", action="store_true",
                   help="Obsoleto: a saída já é o .vtu de entrada com as UVC "
                        "anexadas. Aceito por compatibilidade, sem efeito")

    # ── paradas ─────────────────────────────────────────────────────────
    p.add_argument("--so-marcacao", action="store_true",
                   help="Para depois da marcação da superfície")
    p.add_argument("--so-preparo", action="store_true",
                   help="Para depois da ponte + validação")
    p.add_argument("--ignorar-validacao", action="store_true",
                   help="Segue para o UVC mesmo com o contrato reprovado")

    args = p.parse_args()

    try:
        ab_banda = float(args.ab_banda)
    except ValueError:
        sys.exit(f"Erro: --ab-banda deve ser um numero ou 'inf'; "
                 f"recebi '{args.ab_banda}'.")
    if ab_banda <= 0:
        sys.exit("Erro: --ab-banda deve ser positivo.")

    entrada = resolver_entrada(args.input)

    if args.sem_apex:
        apex = None
    elif args.apex:
        if not os.path.isfile(args.apex):
            sys.exit(f"Erro: arquivo do apex não encontrado: {args.apex}")
        apex = args.apex
    else:
        apex = procurar_apex(entrada)

    caso = args.caso or os.path.splitext(os.path.basename(entrada))[0]
    raiz = args.outdir or f"uvc_{caso}"
    dir_marcacao = os.path.join(raiz, "marcacao")
    dir_preparo = os.path.join(raiz, "preparo")
    dir_result = os.path.join(raiz, "result")
    os.makedirs(raiz, exist_ok=True)

    t_total = time.time()

    ponte.secao(f"UVC DE PONTA A PONTA — caso '{caso}'")
    print(f"Entrada     : {entrada}")
    print(f"Apex        : {apex or 'detecção automática pelo pipeline'}"
          f"{'  (encontrado ao lado da malha)' if apex and not args.apex else ''}")
    print(f"Apex no UVC : --apex-snap {args.apex_snap}"
          f"{'  (o ponto marcado é respeitado)' if args.apex_snap != 'rv-epi' else '  (comportamento do Cobiveco original: o ponto marcado é descartado)'}")
    print(f"Banda ab=0  : {ab_banda} arestas medias"
          f"{'  (banda do Cobiveco original)' if not np.isfinite(ab_banda) else ''}")
    print(f"Saídas em   : {raiz}/")
    print(f"Pipeline    : {args.pipeline}")
    print(f"Cobiveco em : {os.path.abspath(args.cobiveco)}")

    # ════════════════════════════════════════════════════════════════════
    # 1) Marcação da superfície
    # ════════════════════════════════════════════════════════════════════
    pipeline = carregar_pipeline(args.pipeline)

    if args.do_vtp:
        if not os.path.isfile(args.do_vtp):
            sys.exit(f"Erro: .vtp não encontrado: {args.do_vtp}")
        vtp_marcado = args.do_vtp
        ponte.secao("Marcação da superfície (IGNORADA)")
        print(f"  Partindo do .vtp já marcado: {vtp_marcado}")
    else:
        vtp_marcado = os.path.join(dir_marcacao, f"{caso}_marcado.vtp")
        os.makedirs(dir_marcacao, exist_ok=True)
        pipeline.run_pipeline(
            input_mesh           = entrada,
            output_vtp           = vtp_marcado,
            apex_csv             = apex,
            outdir               = dir_marcacao,
            z_threshold          = args.z_threshold,
            z_mode               = args.z_mode,
            dist                 = args.dist,
            dot                  = args.dot,
            classe_apex          = pipeline.UVC_APEX,
            base_value           = pipeline.UVC_BASE,
            region_array         = args.region_array,
            keep_cell_class      = False,
            keep_original_arrays = args.keep_original_arrays,
            min_ilha             = args.min_ilha,
            triangular           = args.triangular,
            enxuto               = False,
            so_superficie        = False,
            apex_mode            = args.apex_mode,
            apex_dist_max        = args.apex_dist_max,
            apex_estrito         = args.apex_estrito,
        )

    if args.so_marcacao:
        ponte.secao("PARADA SOLICITADA (--so-marcacao)")
        print(f"  • {vtp_marcado}")
        return

    # ════════════════════════════════════════════════════════════════════
    # 2) Ponte + validação
    # ════════════════════════════════════════════════════════════════════
    inPrefix = ponte.preparar_entrada_cobiveco(
        vtp_marcado      = vtp_marcado,
        vtu_original     = entrada,
        destino          = dir_preparo,
        nome             = args.nome,
        labels_origem    = pipeline.LABELS_UVC,
        permutar         = (args.permutar == "rotulos"),
        septo_face       = args.septo_face,
        septo_espessura  = args.septo_espessura,
        septo_fator      = args.septo_fator,
    )

    ok = ponte.validar_contrato(f"{os.path.join(dir_preparo, args.nome)}.vtu",
                                f"{inPrefix}.vtp")

    if args.so_preparo:
        ponte.secao("PARADA SOLICITADA (--so-preparo)")
        print(f"  inPrefix = '{inPrefix}'")
        return

    if not ok and not args.ignorar_validacao:
        sys.exit("\nAbortado: o contrato não passou. Corrija os itens acima "
                 "ou rode de novo com --ignorar-validacao.")

    # ════════════════════════════════════════════════════════════════════
    # 3) UVC
    # ════════════════════════════════════════════════════════════════════
    ponte.secao("Cálculo das coordenadas UVC (cobiveco)")
    UVC = carregar_uvc(args.cobiveco)

    outPrefix = os.path.join(dir_result, "")
    c = UVC(inPrefix=inPrefix,
            outPrefix=outPrefix,
            exportLevel=args.export_level,
            verbose=True,
            mappingTol=args.mapping_tol,
            tol=args.tol,
            maxit=args.maxit,
            aha17=not args.sem_aha17,
            aha17SeptumMode=args.aha17_septo,
            aha17ApexCap=args.aha17_apex,
            aha17Coronary=args.aha17_coronaria,
            aha17RV=not args.sem_aha17_vd,
            aha17RVModel=args.aha17_vd_modelo,
            aha17RVRtMode=args.aha17_vd_rt,
            apexSnap=args.apex_snap,
            abApexUnique=not args.ab_plato_apical,
            abBandLength=ab_banda)
    c.computeAll()

    result_path = os.path.join(dir_result, "result.vtu")

    # ── o result cru do cobiveco também recebe as tags de volta ─────────
    if args.export_level > 0 and os.path.isfile(result_path):
        ponte.copiar_arrays_originais(result_path, entrada)

    # ── saída final: o .vtu de ENTRADA + os campos UVC ──────────────────
    campos = {k: np.asarray(c.result.point_data[k])
              for k in ponte.UVC_CAMPOS if k in c.result.point_data}
    saida_final = args.saida or os.path.join(raiz, f"{caso}_UVC.vtu")
    ponte.anexar_uvc_ao_vtu(entrada, campos, saida_final,
                            pontos_result=np.asarray(c.result.points),
                            prefixo=args.prefixo)

    # ════════════════════════════════════════════════════════════════════
    # 4) Resumo
    # ════════════════════════════════════════════════════════════════════
    ponte.secao("Resumo das coordenadas UVC")
    r = c.result
    esperado = {"tv": (0.0, 1.0), "tm": (0.0, 1.0),
                "ab": (0.0, 1.0), "rt": (-np.pi, np.pi)}
    for nome, (lo, hi) in esperado.items():
        a = np.asarray(r.point_data[nome])
        nan = int(np.isnan(a).sum())
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        dentro = (mn >= lo - 1e-3) and (mx <= hi + 1e-3)
        marca = "✓" if (dentro and nan == 0) else "✗"
        print(f"  {marca} {nome:>3}: [{mn:+.3f}, {mx:+.3f}]  "
              f"média={float(np.nanmean(a)):+.3f}  NaNs={nan}  "
              f"(esperado [{lo:+.3f}, {hi:+.3f}])")

    ab = np.asarray(r.point_data["ab"])
    n_apice = int(np.sum(ab == ab.min()))
    print(f"\n  {'✓' if n_apice == 1 else '✗'} ab mínimo (o ápice) em "
          f"{n_apice} nó(s){'' if n_apice == 1 else ' — esperado 1'}; "
          f"ab máximo (a base) em {int(np.sum(ab == ab.max()))} nós")
    if n_apice == 1:
        pts_res = np.asarray(c.result.points)
        print(f"      ápice em {pts_res[int(np.argmin(ab))].round(3).tolist()}")

    if "aha17" in r.point_data:
        from cobiveco.aha17 import report as aha_report
        seg = np.asarray(r.point_data["aha17"]).astype(int)
        n_regioes = len(set(seg[seg > 0].tolist()))
        ponte.secao(f"Parcelamento VE+VD — {n_regioes} regioes rotuladas "
                    f"(max esperado: 34)")
        info = getattr(c.bv, "aha17Info", {})
        aha_report(seg,
                   weights=getattr(c.bv, "nodeVolume", None),
                   rv_model=args.aha17_vd_modelo,
                   rv_rt_mode=args.aha17_vd_rt,
                   rv_info=info.get("rv"))

    ponte.secao(f"CONCLUÍDO em {time.time() - t_total:.1f} s")
    campos_txt = ", ".join(f"{args.prefixo}{k}" for k in campos)
    print(f"  SAÍDA PRINCIPAL")
    print(f"  • {saida_final}")
    print(f"      = {os.path.basename(entrada)} intacto + {campos_txt}")
    print(f"\n  Intermediários")
    print(f"  • {vtp_marcado}  (superfície marcada, confira no ParaView)")
    print(f"  • {inPrefix}.vtp  (entrada do cobiveco)")
    if args.export_level > 0:
        print(f"  • {result_path}  (result do cobiveco, também com as tags "
              f"do .vtu de entrada)")


if __name__ == "__main__":
    main()
