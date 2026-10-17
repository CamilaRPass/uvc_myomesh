# Cobiveco / UVC em Python

Porte do pipeline **Universal Ventricular Coordinates** (UVC) do Cobiveco
(MATLAB, KIT-IBT — Bayer et al. 2018 / Schuler et al. 2021) para Python.

A estratégia foi **fidelidade, não reimplementação**: o `vtkToolbox` (MATLAB)
e o `pyvista` (Python) são apenas casca para a *mesma* biblioteca VTK em C++.
Por isso os filtros `vtkThreshold`, `vtkConnectivityFilter`,
`vtkDataSetSurfaceFilter`, `vtkFeatureEdges` e `vtkMapPointIds` traduzem
quase 1:1. O que foi reconstruído do zero é só a matemática pura
(`cotmatrix`, `solveLaplace`).

## Estrutura

```
cobiveco/
  laplace.py      cotmatrix (Laplaciano cotangente P1-FEM) + solveLaplace (bicgstab+ILU+RCM)
  vtk_utils.py    casca pyvista p/ os filtros vtkToolbox + baryInterpMat + pca
  uvc.py          classe UVC: prepareMesh, transventricular, splitMesh,
                  transmural, apicobasal (etapas 1–5)
  rotational.py   computeRotational (etapa 6) — a mais intrincada
run_geo1.py       exemplo de ponta a ponta
geo1/             malha de teste (heart.vtu + heart_for_UVC.vtp)
```

## Uso

```python
from cobiveco import UVC
c = UVC(inPrefix='geo1/heart_for_UVC', outPrefix='result_geo1/', exportLevel=2)
c.computeAll()
# c.result tem pointData: tv, tm, ab, rt, rtSin, rtCos
```

Dependências: `pyvista`, `scipy`, `numpy`, `vtk`.

## Status de validação (na malha geo1: 61885 nós, 294309 tets)

**Validado numericamente (alta confiança):**

- `cotmatrix` — simetria 1e-16, soma de linhas 1e-15, reproduz funções
  lineares no interior até 1e-13 (propriedade que *define* o Laplaciano
  cotangente correto).
- `solveLaplace` — valores de borda exatos, solução em [0,1], sem NaN.
- `tv` (transventricular) ∈ {0,1}, `tm` (transmural) ∈ [0,1],
  `ab` (apicobasal) ∈ [0, 0.99] — todas com 0 NaN e os intervalos corretos.

**Validado estruturalmente (etapa rotacional):**

- `rt` ∈ [−π, π], 0 NaN.
- septo em rt ≈ 0.08 rad (~5°): o septo é a linha de referência zero, como
  esperado.
- cobertura angular completa [−π, π] na parede do VE.
- campo suave: salto angular por aresta com mediana ~1.4° (só o corte de
  ramo natural em ±π gera saltos grandes).

**O que ainda precisa da sua verificação:**

A coordenada **rotacional** (`computeRotational`) é a parte mais complexa, com
vários ramos condicionais (default vs. fallback na divisão da junção, splits
por PCA, soluções de Laplace intermediárias). As propriedades acima indicam
forte correção, mas **não houve comparação numérica direta com a saída do
MATLAB original** (o repositório não traz um resultado de referência). Para
fechar a validação, rode o `example_geo1_uvc.m` no MATLAB e compare
`result.vtu` nó a nó (especialmente `rt`). As etapas `tv`/`tm`/`ab` já estão
ancoradas pelos intervalos corretos e pela validação do núcleo numérico.

## Diferença técnica relevante

A reparametrização geodésica do `ab` usa o caminho geodésico de Dijkstra
sobre as arestas da superfície, no lugar do *fast marching* do numerical-tours
usado no MATLAB. A ideia é a mesma (distância geodésica base→ápice), mas a
discretização do caminho difere ligeiramente — vale conferir se você precisar
de correspondência exata nessa etapa.
