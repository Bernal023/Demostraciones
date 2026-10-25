# 3. Regresión Ridge — Propiedades del estimador (3.12)

**Modelo.** Es el del libro tras estandarizar (sin intercepto): $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$, con $X\in\mathbb{R}^{n\times p}$ fija, $\mathbb{E}[\boldsymbol{\varepsilon}]=\mathbf{0}$ y $\mathrm{Cov}(\boldsymbol{\varepsilon})=\sigma^2 I_n$. Para $\lambda>0$ sea $A=X^TX+\lambda I_p$, simétrica y definida positiva (por tanto invertible) y

$$\hat{\boldsymbol{\beta}}_R=A^{-1}X^T\boldsymbol{y}.$$

**(i) Esperanza.** Como $X$ es fija, $\mathbb{E}[\boldsymbol{y}]=X\boldsymbol{\beta}$ y

$$\mathbb{E}(\hat{\boldsymbol{\beta}}_R)=A^{-1}X^T\mathbb{E}[\boldsymbol{y}]=(X^TX+\lambda I_p)^{-1}X^TX\boldsymbol{\beta}.$$

**(ii) Covarianza.** Para una matriz fija $M$, $\mathrm{Cov}(M\boldsymbol{y})=M\,\mathrm{Cov}(\boldsymbol{y})\,M^T$. Con $M=A^{-1}X^T$ y $\mathrm{Cov}(\boldsymbol{y})=\sigma^2I_n$, y usando $A^{-1}$ simétrica:

$$\mathrm{Cov}(\hat{\boldsymbol{\beta}}_R)=\sigma^2A^{-1}X^T\,X A^{-1}=\sigma^2(X^TX+\lambda I_p)^{-1}X^TX(X^TX+\lambda I_p)^{-1}.$$

**(iii) Sesgo.** Como $X^TX=A-\lambda I_p$,

$$\mathbb{E}(\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta})=A^{-1}X^TX\boldsymbol{\beta}-\boldsymbol{\beta}=A^{-1}(X^TX-A)\boldsymbol{\beta}=-\lambda(X^TX+\lambda I_p)^{-1}\boldsymbol{\beta}.$$

**(iv) Error cuadrático esperado.** Sea $\boldsymbol{b}=\mathbb{E}(\hat{\boldsymbol{\beta}}_R)-\boldsymbol{\beta}$ el sesgo. Escribiendo $\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta}=(\hat{\boldsymbol{\beta}}_R-\mathbb{E}\hat{\boldsymbol{\beta}}_R)+\boldsymbol{b}$, el término cruzado tiene esperanza $2\boldsymbol{b}^T\mathbb{E}(\hat{\boldsymbol{\beta}}_R-\mathbb{E}\hat{\boldsymbol{\beta}}_R)=0$, luego

$$\mathbb{E}\big[(\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta})\big]=\mathrm{tr}\,\mathrm{Cov}(\hat{\boldsymbol{\beta}}_R)+\boldsymbol{b}^T\boldsymbol{b}.$$

*Término de varianza.* Sea $X^TX=PLP^T$ la descomposición espectral, con $P$ ortogonal y $L=\mathrm{diag}(l_1,\dots,l_p)$, $l_j\ge0$. Entonces $A=P(L+\lambda I)P^T$ y $A^{-1}=P(L+\lambda I)^{-1}P^T$, de modo que

$$A^{-1}X^TXA^{-1}=P\,(L+\lambda I)^{-1}L\,(L+\lambda I)^{-1}P^T=P\,\mathrm{diag}\!\left(\frac{l_j}{(l_j+\lambda)^2}\right)P^T.$$

Como la traza es invariante bajo conjugación ortogonal,

$$\mathrm{tr}\,\mathrm{Cov}(\hat{\boldsymbol{\beta}}_R)=\sigma^2\sum_{j=1}^p\frac{l_j}{(l_j+\lambda)^2}.$$

*Término de sesgo.* Por (iii), con $A^{-1}$ simétrica,

$$\boldsymbol{b}^T\boldsymbol{b}=\lambda^2\boldsymbol{\beta}^TA^{-1}A^{-1}\boldsymbol{\beta}=\lambda^2\boldsymbol{\beta}^T(X^TX+\lambda I_p)^{-2}\boldsymbol{\beta}.$$

Sumando ambos términos:

$$\mathbb{E}\big[(\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta})\big]=\sigma^2\sum_{j=1}^p\frac{l_j}{(l_j+\lambda)^2}+\lambda^2\boldsymbol{\beta}^T(X^TX+\lambda I_p)^{-2}\boldsymbol{\beta}.\qquad\blacksquare$$

:::{note}
**Comentario.** Con $\boldsymbol{\alpha}=P^T\boldsymbol{\beta}$ el sesgo al cuadrado es $\lambda^2\sum_j\alpha_j^2/(l_j+\lambda)^2$. Cuando $\lambda$ crece, la varianza decrece y el sesgo crece, y para $\lambda=0$ se recupera el resultado de mínimos cuadrados (sesgo nulo, varianza $\sigma^2\sum_j 1/l_j$). En el documento la covarianza se escribe $\mathrm{Cov}(\boldsymbol{\beta}_R)$; se entiende como $\mathrm{Cov}(\hat{\boldsymbol{\beta}}_R)$.
:::
