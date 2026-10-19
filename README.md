# uvc_myomesh
 
Dada uma malha biventricular e o ponto do ápice, resolve as coordenadas ventriculares universais (UVC) e divide os ventrículos esquerdo e direito nos segmentos da AHA.
 
## Uso
 
```bash
python run_uvc_end_to_end.py malha.vtu --apex apex.txt
```
 
- `malha.vtu` — malha biventricular tetraédrica (VTK Unstructured Grid)
- `apex.txt` — coordenadas do ápice, uma linha: `x y z`
A saída é um `.vtu` com os campos `ab` (apicobasal), `tm` (transmural), `rt` (rotacional), `tv` (transventricular) e `aha`, visualizáveis no ParaView.
 
## Referências
 
- Bayer, J., Prassl, A. J., Pashaei, A., Gomez, J. F., Frontera, A., Neic, A., Plank, G., Vigmond, E. J. *Universal ventricular coordinates: A generic framework for describing position within the heart and transferring data.* Medical Image Analysis, 45:83–93, 2018. [doi:10.1016/j.media.2018.01.005](https://doi.org/10.1016/j.media.2018.01.005)
- Schuler, S., Pilia, N., Potyagaylo, D., Loewe, A. *Cobiveco: Consistent biventricular coordinates for precise and intuitive description of position in the heart.* Medical Image Analysis, 74:102247, 2021. [doi:10.1016/j.media.2021.102247](https://doi.org/10.1016/j.media.2021.102247)
- Cerqueira, M. D., Weissman, N. J., Dilsizian, V., Jacobs, A. K., Kaul, S., Laskey, W. K., Pennell, D. J., Rumberger, J. A., Ryan, T., Verani, M. S. *Standardized myocardial segmentation and nomenclature for tomographic imaging of the heart: A statement for healthcare professionals from the Cardiac Imaging Committee of the Council on Clinical Cardiology of the American Heart Association.* Circulation, 105(4):539–542, 2002. [doi:10.1161/hc0402.102975](https://doi.org/10.1161/hc0402.102975)
