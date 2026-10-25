# Demostraciones de Machine Learning

Este libro reúne las demostraciones matemáticas de la tarea de Machine Learning. Cada capítulo corresponde a un ítem del documento *Demostraciones Tarea*.

| Capítulo | Tema | Qué se demuestra |
|:--|:--|:--|
| 1 | k-NN | Error verdadero y empírico (Definición 2.3) y optimalidad del predictor de Bayes |
| 2 | k-NN | Desigualdad $2\eta(1-\eta)\le 2\min\{\eta,1-\eta\}$, caso $1-\eta(\boldsymbol{x})$ |
| 3 | Ridge | Propiedades del estimador (3.12): esperanza, covarianza, sesgo y error cuadrático |
| 4 | Ridge | Ecuación (3.8) y $Z^T\mathbf{1}=0$ |
| 5 | Ridge | Igualdades (3.11) y $\beta_0^\star=0$ |
| 6 | Lasso | Estimación de parámetros: KKT, umbral suave y LARS |
| 7 | Random Forest | Ejercicio 6 (Gini sobre la Tabla 5.1): verificación del árbol |
| 8 | AdaBoost | Derivación del Algoritmo 5.1 |

```{note}
Las referencias teóricas son {cite}`shalev2014,cover1967` (k-NN), {cite}`hoerl1970,tibshirani1996,efron2004` (Ridge y Lasso), {cite}`breiman1984,theodoridis2020` (árboles) y {cite}`freund1997,friedman2000` (AdaBoost).
```

## Verificación numérica

Las fórmulas de ridge, lasso (umbral suave, solución por conjunto activo y algoritmo LARS) y los valores de Gini del capítulo 7 se contrastaron con código. Los *scripts* están en la carpeta `verificacion/` del repositorio:

- `verificacion/ridge_lasso_lars.py`: compara las fórmulas con `scikit-learn` (`Lasso`, `lars_path`).
- `verificacion/gini_ejercicio6.py`: recalcula todos los umbrales y variables del ejercicio 6.

## Cómo compilar

```bash
pip install -r requirements.txt
jupyter-book build book/
```

El resultado queda en `book/_build/html/index.html`.
