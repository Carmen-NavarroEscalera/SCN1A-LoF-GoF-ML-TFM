# Datos — SCN1A LoF/GoF Machine Learning

Este archivo documenta la estructura y procedencia del conjunto de datos utilizado en el proyecto, ya que el archivo original no se incluye en el repositorio por razones de tamaño y para preservar la integridad de los datos.

## Descripción del conjunto de datos

El archivo de datos principal (`SCN1A_ML_complete_dbnsfp.xlsx`) no está incluido en este repositorio por razones de tamaño y para garantizar la integridad de los datos originales.

El conjunto de datos contiene **56 variantes missense** del gen SCN1A con caracterización funcional experimental validada mediante patch-clamp, recopiladas de la literatura científica publicada entre 2002 y 2023.

---

## Estructura del archivo de datos

El archivo Excel contiene una única hoja (`Sheet1`) con las siguientes columnas:

### Información de la variante
| Columna | Descripción | Ejemplo |
|---|---|---|
| rsID | Identificador de la variante en dbSNP | rs121918770 |
| Posicion | Posición genómica en GRCh38 | 166054710 |
| Cambio_cDNA | Cambio en el ADN complementario (notación HGVS) | c.530G>C |
| Cambio_Proteico | Cambio aminoacídico | G177A |
| Dominio | Segmento proteico del canal Nav1.1 | DI S2 segment |
| Clase_Funcional | Clasificación funcional experimental | LoF / GoF |
| PMID | Identificador PubMed del estudio de origen | 30735520 |
| Año | Año de publicación del estudio | 2019 |
| Alelo_Referencia | Alelo de referencia | C |
| Alelo_Alternativo | Alelo alternativo | G |

### Predictores in silico (rankscores de dbNSFP)
| Columna | Herramienta | Descripción |
|---|---|---|
| VARITY_ER_rankscore | VARITY-ER | Predictor de patogenicidad |
| ESM1b_converted_rankscore | ESM1b | Modelo de lenguaje proteico |
| AlphaMissense_rankscore | AlphaMissense | Predictor basado en AlphaFold |
| MutPred2_rankscore | MutPred2 | Mecanismos moleculares |
| MPC_rankscore | MPC | Missense badness + constraint |
| PrimateAI_rankscore | PrimateAI | Red neuronal profunda |
| REVEL_rankscore | REVEL | Metapredictor |
| MetaRNN_rankscore | MetaRNN | Red neuronal recurrente |
| SIFT4G_converted_rankscore | SIFT4G | Tolerancia evolutiva |
| Polyphen2_HDIV_rankscore | PolyPhen-2 HDIV | Impacto estructural |

> Todos los rankscores están normalizados en el rango [0, 1], donde valores más altos indican mayor patogenicidad predicha.

---

## Distribución del conjunto de datos

| Clase | N | % |
|---|---|---|
| LoF | 38 | 67.9% |
| GoF | 18 | 32.1% |
| **Total** | **56** | **100%** |

---

## Fuentes de datos

### Variantes y clasificación funcional
- PubMed  
- ClinVar  
- Estudios con caracterización electrofisiológica mediante patch-clamp  

### Predictores in silico
- dbNSFP v4.x  
- Referencia: Liu X et al., Genome Medicine (2020)

---

## Cómo reproducir el conjunto de datos

1. Buscar variantes missense de SCN1A con caracterización funcional experimental.  
2. Extraer rsID, posición, cambio cDNA, cambio proteico, dominio, clasificación funcional, PMID, año y alelos.  
3. Consultar dbNSFP para obtener los rankscores.  
4. Construir un archivo Excel con la estructura descrita arriba.

---

## Criterios de inclusión y exclusión

**Inclusión:**  
- Variantes missense  
- Patch-clamp disponible  
- Clasificación LoF o GoF inequívoca  

**Exclusión:**  
- Nonsense, frameshift, splicing  
- Sin caracterización experimental  
- Clasificación ambigua  

---

## Archivo esperado

| Archivo | Descripción |
|---|---|
| `SCN1A_ML_complete_dbnsfp.xlsx` | Dataset completo con 56 variantes y predictores |
