# 7. Random Forest / árboles — Sección 12, Ejercicio 6: verificación del árbol

**Datos.** La Tabla 5.1 del capítulo de Random Forest y XGBoost (10 observaciones) es la base del ejercicio. La variable objetivo es *Gonna Be an Astronaut* (5 Sí y 5 No) y el índice de impureza es Gini, $\mathrm{Gini}(\text{nodo})=1-\sum_i p_i^2$, con $\mathrm{Gini}_{\text{split}}=\frac{N_1}{N}\mathrm{Gini}(D_1)+\frac{N_2}{N}\mathrm{Gini}(D_2)$.

| Age | Like Dogs | Like Gravity | Astronaut |
|:--:|:--:|:--:|:--:|
| 24 | 0 | 0 | 0 |
| 30 | 1 | 1 | 1 |
| 36 | 0 | 1 | 1 |
| 36 | 0 | 0 | 0 |
| 42 | 0 | 0 | 0 |
| 44 | 1 | 1 | 1 |
| 46 | 1 | 0 | 0 |
| 47 | 1 | 1 | 1 |
| 47 | 0 | 1 | 1 |
| 51 | 1 | 1 | 0 |

## 7.1 Umbrales de *Age*

Con las edades ordenadas **incluyendo repetidas** (24, 30, 36, 36, 42, 44, 46, 47, 47, 51), los puntos medios son 27, 33, 36, 39, 43, 45, 46.5, 47 y 49, que es la lista de la solución. Los umbrales 36 y 47 son puntos medios de dos valores **iguales** y producen la misma partición que otro umbral (36 equivale a 33, y 47 equivale a 46.5). Es una redundancia, no un error de cálculo. Lo habitual es usar los puntos medios de valores **distintos** (27, 33, 39, 43, 45, 46.5 y 49).

## 7.2 Recálculo de todos los umbrales

Aquí $D_1$ es age $<t$ y $D_2$ es age $\ge t$. Los pares indican (Sí, No).

| Umbral | $\lvert D_1\rvert$ | (Sí, No) en $D_1$ | $\lvert D_2\rvert$ | (Sí, No) en $D_2$ | Gini($D_1$) | Gini($D_2$) | Gini(split) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 27 | 1 | (0, 1) | 9 | (5, 4) | 0.000 | 0.494 | **0.444** |
| 33 | 2 | (1, 1) | 8 | (4, 4) | 0.500 | 0.500 | 0.500 |
| 36 | 2 | (1, 1) | 8 | (4, 4) | 0.500 | 0.500 | 0.500 (igual que 33) |
| 39 | 4 | (2, 2) | 6 | (3, 3) | 0.500 | 0.500 | 0.500 |
| 43 | 5 | (2, 3) | 5 | (3, 2) | 0.480 | 0.480 | 0.480 |
| 45 | 6 | (3, 3) | 4 | (2, 2) | 0.500 | 0.500 | 0.500 |
| 46.5 | 7 | (3, 4) | 3 | (2, 1) | 0.490 | 0.444 | 0.476 |
| 47 | 7 | (3, 4) | 3 | (2, 1) | 0.490 | 0.444 | 0.476 (igual que 46.5) |
| 49 | 9 | (5, 4) | 1 | (0, 1) | 0.494 | 0.000 | **0.444** |

Los cuatro diagramas que aparecen en el documento (umbrales 27, 33, 36 y 39) **coinciden con estos valores**: conteos, Gini de cada hijo y Gini ponderado. Valores exactos: umbral 27 y 49, $\frac{9}{10}\cdot\frac{40}{81}=\frac49\approx0.444$; umbral 46.5 y 47, $\frac{7}{10}\cdot\frac{24}{49}+\frac{3}{10}\cdot\frac49=\frac{10}{21}\approx0.476$.

## 7.3 Las otras variables y el nodo raíz

| Variable | Partición | $\lvert D_1\rvert$ (Sí) | $\lvert D_2\rvert$ (Sí) | Gini($D_1$) | Gini($D_2$) | Gini(split) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Like Dogs | $=0$ / $=1$ | 5 (2) | 5 (3) | 0.480 | 0.480 | 0.480 |
| Like Gravity | $=0$ / $=1$ | 4 (0) | 6 (5) | 0.000 | 0.278 | **0.167** |

El Gini del nodo raíz es $1-0.5^2-0.5^2=0.5$. La mejor división es **Like Gravity** (Gini 0.167, ganancia $0.5-0.167=0.333$), mejor que la mejor de *Age* (0.444) y que *Like Dogs* (0.480). Dentro de *Age*, los umbrales 27 y 49 **empatan** en 0.444.

## 7.4 Conclusión sobre "dónde está el error"

- Los diagramas visibles en el documento (27, 33, 36, 39) están bien. Si hay un error, está en un árbol que **no aparece en el documento**: los umbrales 43, 45, 46.5, 47 o 49, o las variables *Like Dogs* y *Like Gravity*.
- La tabla 7.2 y la 7.3 sirven de referencia para compararlas con esas figuras. Puntos concretos que conviene revisar en la solución original: (i) los conteos de 43 deben ser (2, 3) y (3, 2) con Gini 0.48; (ii) el umbral 46.5 y el 47 deben dar el mismo valor, 0.476; (iii) el umbral 49 debe dar 0.444 y empatar con el 27; (iv) *Like Gravity* debe dar 0.167 y ser la división elegida en la raíz.

## 7.5 Código de verificación

```python
import numpy as np, pandas as pd

df = pd.DataFrame({
    "Age":     [24, 30, 36, 36, 42, 44, 46, 47, 47, 51],
    "Dogs":    [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    "Gravity": [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    "Astro":   [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]})

def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1 - p**2 - (1 - p)**2

def gini_split(mask):
    y = df.Astro.values
    a, b = y[mask], y[~mask]
    return len(a)/len(y)*gini(a) + len(b)/len(y)*gini(b)

a = np.sort(df.Age.values)
for t in [(a[i] + a[i+1]) / 2 for i in range(len(a) - 1)]:
    print(t, round(gini_split(df.Age.values < t), 4))
for v in ["Dogs", "Gravity"]:
    print(v, round(gini_split(df[v].values == 0), 4))
```
