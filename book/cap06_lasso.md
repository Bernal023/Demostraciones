# 6. Regresión Lasso — Estimación de parámetros (ejercicio para el lector)

**Problema.** Con los datos estandarizados como en (3.11),

$$S_\lambda(\boldsymbol{\beta})=(\boldsymbol{y}-X\boldsymbol{\beta})^T(\boldsymbol{y}-X\boldsymbol{\beta})+\lambda\sum_{j=1}^p|\beta_j|,\qquad\lambda\ge0.$$

**Intercepto.** Igual que en ridge (sección 4), la penalización no incluye el intercepto y $\mathbf{1}^TX=\mathbf{0}^T$; por tanto $\partial S_\lambda/\partial\beta_0^\star=-2n\bar y+2n\beta_0^\star$ y $\beta_0^\star=\bar y=0$. Basta estimar $\boldsymbol{\beta}$.

## 6.1 Condiciones de optimalidad (subgradiente)

$S_\lambda$ es convexa (suma de una cuadrática convexa y una norma $L^1$), pero no es diferenciable donde algún $\beta_j=0$. Un punto $\hat{\boldsymbol{\beta}}$ es mínimo global si y solo si $\mathbf{0}\in\partial S_\lambda(\hat{\boldsymbol{\beta}})$. Como el subdiferencial de $|t|$ es $\{\mathrm{sgn}(t)\}$ si $t\ne0$ y $[-1,1]$ si $t=0$, la condición es, con $c_j=X_j^T(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})$ (correlación de la columna $j$ con el residuo):

$$c_j=\frac{\lambda}{2}\,\mathrm{sgn}(\hat\beta_j)\quad\text{si }\hat\beta_j\neq0,\qquad |c_j|\le\frac{\lambda}{2}\quad\text{si }\hat\beta_j=0.\tag{KKT}$$

(Se obtiene de $-2X_j^T(\boldsymbol{y}-X\boldsymbol{\beta})+\lambda\,s_j=0$ con $s_j\in\partial|\beta_j|$.) Esto ya explica por qué lasso produce ceros exactos: basta que la correlación de una variable con el residuo no supere $\lambda/2$ en valor absoluto.

## 6.2 Solución cerrada dado el conjunto activo y los signos

Sea $\mathcal{A}=\{j:\hat\beta_j\ne0\}$ y $\boldsymbol{s}_\mathcal{A}=\mathrm{sgn}(\hat{\boldsymbol{\beta}}_\mathcal{A})$. De (KKT) en el bloque activo, $X_\mathcal{A}^T(\boldsymbol{y}-X_\mathcal{A}\hat{\boldsymbol{\beta}}_\mathcal{A})=\frac\lambda2\boldsymbol{s}_\mathcal{A}$, es decir,

$$\hat{\boldsymbol{\beta}}_\mathcal{A}=(X_\mathcal{A}^TX_\mathcal{A})^{-1}\Big(X_\mathcal{A}^T\boldsymbol{y}-\frac\lambda2\boldsymbol{s}_\mathcal{A}\Big),\qquad\hat{\boldsymbol{\beta}}_{\mathcal{A}^c}=\mathbf{0},$$

siempre que se cumplan las consistencias $\mathrm{sgn}(\hat{\boldsymbol{\beta}}_\mathcal{A})=\boldsymbol{s}_\mathcal{A}$ y $|X_j^T(\boldsymbol{y}-X_\mathcal{A}\hat{\boldsymbol{\beta}}_\mathcal{A})|\le\lambda/2$ para $j\notin\mathcal{A}$. Compárese con ridge, $\hat{\boldsymbol{\beta}}_R=(X^TX+\lambda I)^{-1}X^T\boldsymbol{y}$: lasso resta una cantidad fija $\frac\lambda2\boldsymbol{s}_\mathcal{A}$ en lugar de multiplicar por un factor de contracción. El problema es que $\mathcal{A}$ y $\boldsymbol{s}_\mathcal{A}$ no se conocen de antemano, y por eso en general no hay fórmula cerrada en términos de $\lambda$ solamente.

## 6.3 Caso de columnas ortogonales: umbral suave (soft-thresholding)

Si $X^TX=nI_p$ (columnas ortogonales con $\sum_ix_{ij}^2=n$), sea $b_j=X_j^T\boldsymbol{y}/n$ el coeficiente de mínimos cuadrados. Entonces

$$\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2=\|\boldsymbol{y}\|^2-2n\sum_j\beta_jb_j+n\sum_j\beta_j^2=\text{cte}+n\sum_j(\beta_j-b_j)^2,$$

y el problema se separa por coordenadas: minimizar $n(\beta_j-b_j)^2+\lambda|\beta_j|$.

- Si $\beta_j>0$: la derivada $2n(\beta_j-b_j)+\lambda=0$ da $\beta_j=b_j-\frac{\lambda}{2n}$, válido si $b_j>\frac{\lambda}{2n}$.
- Si $\beta_j<0$: $2n(\beta_j-b_j)-\lambda=0$ da $\beta_j=b_j+\frac{\lambda}{2n}$, válido si $b_j<-\frac{\lambda}{2n}$.
- En otro caso, $|b_j|\le\frac{\lambda}{2n}$ y (KKT) se cumple con $\hat\beta_j=0$ (aquí $|c_j|=n|b_j|\le\lambda/2$).

En resumen,

$$\hat\beta_j^{\,\text{lasso}}=\mathrm{sgn}(b_j)\Big(|b_j|-\frac{\lambda}{2n}\Big)_+,\qquad \hat\beta_j^{\,\text{ridge}}=\frac{n\,b_j}{n+\lambda}=\frac{b_j}{1+\lambda/n}.$$

Ridge contrae todos los coeficientes proporcionalmente y nunca los anula, mientras que lasso resta una constante y anula los coeficientes pequeños. Además, todos los coeficientes son cero si y solo si $\lambda\ge\lambda_{\max}=2\max_j|X_j^T\boldsymbol{y}|$.

## 6.4 Caso general: algoritmo LARS (Efron et al., 2004) con la modificación lasso

Para $X$ general la trayectoria $\lambda\mapsto\hat{\boldsymbol{\beta}}(\lambda)$ es **lineal a trozos**, porque en cada tramo el conjunto activo y los signos son constantes y 6.2 es una función afín de $\lambda$. LARS la calcula exactamente. Sea $\boldsymbol{r}=\boldsymbol{y}-X\boldsymbol{\beta}$ el residuo y $\boldsymbol{c}=X^T\boldsymbol{r}$.

1. **Inicio.** $\boldsymbol{\beta}=\mathbf{0}$, $\boldsymbol{r}=\boldsymbol{y}$, $C=\max_j|c_j|$, $\mathcal{A}=\{j:|c_j|=C\}$ y $\boldsymbol{s}_\mathcal{A}=\mathrm{sgn}(\boldsymbol{c}_\mathcal{A})$. Aquí $\lambda=2C$.
2. **Dirección equiangular.** $\boldsymbol{d}_\mathcal{A}=(X_\mathcal{A}^TX_\mathcal{A})^{-1}\boldsymbol{s}_\mathcal{A}$ (cero fuera de $\mathcal{A}$) y $\boldsymbol{u}=X\boldsymbol{d}$. Se mueve $\boldsymbol{\beta}\leftarrow\boldsymbol{\beta}+\gamma\boldsymbol{d}$. Para $j\in\mathcal{A}$, $X_j^T\boldsymbol{u}=s_j$, luego la correlación evoluciona como $c_j(\gamma)=s_j(C-\gamma)$: **todas las variables activas mantienen la misma correlación absoluta** $C-\gamma$ con el residuo, de ahí "equiangular". Para $j\notin\mathcal{A}$, $c_j(\gamma)=c_j-\gamma a_j$ con $a_j=X_j^T\boldsymbol{u}$.
3. **Paso.** Se toma $\gamma=\min(\gamma_{\text{entra}},\gamma_{\text{cero}})$, donde
$$\gamma_{\text{entra}}=\min_{j\notin\mathcal{A}}{}^{+}\left\{\frac{C-c_j}{1-a_j},\ \frac{C+c_j}{1+a_j}\right\}\quad(\text{primer instante en que }|c_j(\gamma)|=C-\gamma),$$
$$\gamma_{\text{cero}}=\min_{j\in\mathcal{A}}{}^{+}\left\{-\frac{\beta_j}{d_j}\right\}\quad(\text{primer instante en que un coeficiente activo llega a }0).$$
Aquí $\min^{+}$ significa mínimo sobre valores positivos.
4. **Actualización.** $\boldsymbol{r}\leftarrow\boldsymbol{r}-\gamma\boldsymbol{u}$, $C\leftarrow C-\gamma$. Si ganó $\gamma_{\text{entra}}$, la variable $j$ entra a $\mathcal{A}$. **Si ganó $\gamma_{\text{cero}}$, la variable sale de $\mathcal{A}$**: esta regla es la modificación que convierte LARS en lasso, ya que de lo contrario se violaría $\mathrm{sgn}(\hat\beta_j)=\mathrm{sgn}(c_j)$ de (KKT).
5. **Parada.** Se repite hasta $C=0$ (se alcanza el ajuste de mínimos cuadrados si $n>p$).

**Relación con $\lambda$.** En (KKT) las variables activas cumplen $|c_j|=\lambda/2$; en LARS valen $C$. Por tanto $\lambda=2C$, y cada nudo de la trayectoria ocurre en un valor de $\lambda$ que decrece desde $\lambda_{\max}$ hasta 0. Entre dos nudos, $\hat{\boldsymbol{\beta}}(\lambda)$ varía linealmente.

:::{tip}
**Verificación numérica.** Se implementaron las fórmulas anteriores (6.2, 6.3 y los pasos 1 a 5 de LARS) y se compararon con `sklearn.linear_model.Lasso` y `lars_path(method="lasso")`: coinciden los coeficientes en 6.2 y 6.3 y todos los nudos $\lambda=2C$ de la trayectoria. En scikit-learn el parámetro es $\alpha=\lambda/(2n)$, porque minimiza $\frac1{2n}\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2+\alpha\|\boldsymbol{\beta}\|_1$.
:::
