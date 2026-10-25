# 5. Regresión Ridge — Verificación de las igualdades (3.11) y de $\beta_0^\star=0$

**Estandarización.** Sean $\bar y=\frac1n\sum_iy_i$, $\bar x_j=\frac1n\sum_ix_{ij}$ y $s_j^2=\frac1n\sum_i(x_{ij}-\bar x_j)^2>0$. Se definen

$$\tilde y_i=y_i-\bar y,\qquad \tilde x_{ij}=\frac{x_{ij}-\bar x_j}{s_j}.$$

**(a)** $\displaystyle\sum_{i=1}^n\tilde y_i=\sum_{i=1}^ny_i-n\bar y=n\bar y-n\bar y=0.$

**(b)** $\displaystyle\sum_{i=1}^n\tilde x_{ij}=\frac1{s_j}\Big(\sum_{i=1}^nx_{ij}-n\bar x_j\Big)=0,\quad j=1,\dots,p.$

**(c)** $\displaystyle\sum_{i=1}^n\tilde x_{ij}^{\,2}=\frac1{s_j^2}\sum_{i=1}^n(x_{ij}-\bar x_j)^2=\frac{n\,s_j^2}{s_j^2}=n.$

$\blacksquare$

:::{note}
**Nota.** La igualdad $\sum_i\tilde x_{ij}^2=n$ exige definir $s_j$ con divisor $n$, que es la definición del libro, $s_j^2=n^{-1}\sum_i(x_{ij}-\bar x_j)^2$. Si se usa el divisor $n-1$ (desviación estándar muestral), la suma vale $n-1$.
:::

**Consecuencia: $\beta_0^\star=\bar y=\frac1n\sum_iy_i=0$.** Por la sección 4, el intercepto óptimo es la media de la respuesta con la que se trabaja. Con la respuesta estandarizada, esa media es $\frac1n\sum_i\tilde y_i=0$ por (a), luego $\beta_0^\star=0$ y el modelo estandarizado no necesita intercepto.
