# 4. Regresión Ridge — Ecuación (3.8) y verificación de $Z^T\mathbf{1}=\mathbf{1}^TZ=0$

**Planteamiento (ecuaciones 3.4 a 3.7 del libro).** Con $\bar x_j=\frac1n\sum_ix_{ij}$, $s_j^2=\frac1n\sum_i(x_{ij}-\bar x_j)^2$ y $z_{ij}=(x_{ij}-\bar x_j)/s_j$, el modelo se reescribe como $\boldsymbol{y}=\beta_0^\star\mathbf{1}+Z\boldsymbol{\beta}_s+\boldsymbol{\varepsilon}$, con $\beta_0^\star=\beta_0+\sum_j\beta_j\bar x_j$ y $\boldsymbol{\beta}_s=(s_1\beta_1,\dots,s_p\beta_p)^T$. El estimador ridge minimiza, sin penalizar el intercepto,

$$S_\lambda(\beta_0^\star,\boldsymbol{\beta}_s)=\sum_{i=1}^n\big(y_i-\beta_0^\star-\boldsymbol{z}_i^T\boldsymbol{\beta}_s\big)^2+\lambda\,\boldsymbol{\beta}_s^T\boldsymbol{\beta}_s=\|\boldsymbol{y}-\beta_0^\star\mathbf{1}-Z\boldsymbol{\beta}_s\|^2+\lambda\|\boldsymbol{\beta}_s\|^2.$$

## 4.1 Verificación: $Z^T\mathbf{1}=\mathbf{1}^TZ=0$

*Por entradas.* La entrada $j$ de $Z^T\mathbf{1}$ es

$$\sum_{i=1}^n z_{ij}=\sum_{i=1}^n(x_{ij}-\bar x_j)=n\bar x_j-n\bar x_j=0.$$

Si además se divide por $s_j$, la suma queda dividida por $s_j$ y sigue valiendo 0.

*En forma matricial.* $Z=CX$ con la matriz de centrado $C=I_n-\frac1n\mathbf{1}\mathbf{1}^T$, y

$$\mathbf{1}^TC=\mathbf{1}^T-\tfrac1n(\mathbf{1}^T\mathbf{1})\mathbf{1}^T=\mathbf{1}^T-\mathbf{1}^T=\mathbf{0}^T\ \Longrightarrow\ \mathbf{1}^TZ=\mathbf{1}^TCX=\mathbf{0}^T,$$

y transponiendo, $Z^T\mathbf{1}=\mathbf{0}$. $\blacksquare$

## 4.2 Derivada respecto a $\beta_0^\star$ (ecuación 3.8)

Derivando $S_\lambda$ (el término de penalización no depende de $\beta_0^\star$):

$$\frac{\partial S_\lambda}{\partial\beta_0^\star}=-2\sum_{i=1}^n\big(y_i-\beta_0^\star-\boldsymbol{z}_i^T\boldsymbol{\beta}_s\big)=-2n\bar y+2n\beta_0^\star+2\,\mathbf{1}^TZ\boldsymbol{\beta}_s.$$

Por 4.1, $\mathbf{1}^TZ=\mathbf{0}^T$, así que

$$\frac{\partial S_\lambda(\beta_0^\star,\boldsymbol{\beta}_s)}{\partial\beta_0^\star}=-2n\bar y+2n\beta_0^\star=0\ \Longrightarrow\ \beta_0^\star=\bar y.\qquad\blacksquare$$

## 4.3 Complemento: derivada de una forma cuadrática y derivada respecto a $\boldsymbol{\beta}_s$

Para $A\in\mathbb{R}^{p\times p}$, $\boldsymbol{x}^TA\boldsymbol{x}=\sum_{i,j}a_{ij}x_ix_j$ y

$$\frac{\partial}{\partial x_k}\boldsymbol{x}^TA\boldsymbol{x}=\sum_j a_{kj}x_j+\sum_i a_{ik}x_i=(A\boldsymbol{x})_k+(A^T\boldsymbol{x})_k\ \Longrightarrow\ \frac{\partial}{\partial\boldsymbol{x}}\boldsymbol{x}^TA\boldsymbol{x}=(A+A^T)\boldsymbol{x},$$

que es $2A\boldsymbol{x}$ si $A$ es simétrica. Aplicando esto a $S_\lambda$ con $\beta_0^\star=\bar y$ y usando $Z^T\mathbf{1}=\mathbf{0}$:

$$\frac{\partial S_\lambda}{\partial\boldsymbol{\beta}_s}=-2Z^T(\boldsymbol{y}-\beta_0^\star\mathbf{1}-Z\boldsymbol{\beta}_s)+2\lambda\boldsymbol{\beta}_s=-2Z^T\boldsymbol{y}+2(Z^TZ+\lambda I_p)\boldsymbol{\beta}_s=0,$$

de donde $\hat{\boldsymbol{\beta}}_s=(Z^TZ+\lambda I_p)^{-1}Z^T\boldsymbol{y}$. La matriz $Z^TZ+\lambda I_p$ es definida positiva, y por tanto este punto crítico es el mínimo único.
