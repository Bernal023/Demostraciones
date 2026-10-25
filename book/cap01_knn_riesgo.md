# 1. k-NN — Error verdadero y error empírico (Definición 2.3)

**Qué se demuestra.** Que las tres expresiones que aparecen en la ecuación (2.2) del error verdadero son la misma cantidad, que el error empírico es un estimador insesgado del error verdadero cuando $h$ no depende de la muestra y, como complemento, la afirmación "se puede verificar" que sigue a la ecuación (2.3): el predictor de Bayes es óptimo.

**Notación.** $Z=(\boldsymbol{x},y)\sim\mathcal{D}$ sobre $\mathcal{X}\times\mathcal{Y}$, $\mathcal{Y}=\{0,1\}$, y $h:\mathcal{X}\to\mathcal{Y}$ medible. Sea

$$A_h=\{(\boldsymbol{x},y)\in\mathcal{X}\times\mathcal{Y}:\ h(\boldsymbol{x})\neq y\}.$$

## 1.1 Igualdad de las tres expresiones del riesgo

**Afirmación.**

$$L_{\mathcal{D}}(h)=\mathbb{E}_{(\boldsymbol{x},y)\sim\mathcal{D}}\big[\mathbf{1}[h(\boldsymbol{x})\neq y]\big]=\mathbb{P}_{(\boldsymbol{x},y)\sim\mathcal{D}}\big[h(\boldsymbol{x})\neq y\big]=\mathcal{D}(A_h).$$

**Demostración.** La función $\mathbf{1}[h(\boldsymbol{x})\neq y]$ es la función indicadora $\mathbf{1}_{A_h}(Z)$, que vale 1 sobre $A_h$ y 0 sobre su complemento $A_h^{c}$. Por definición de esperanza de una variable que toma solo los valores 0 y 1,

$$\mathbb{E}\big[\mathbf{1}_{A_h}(Z)\big]=1\cdot\mathcal{D}(A_h)+0\cdot\mathcal{D}(A_h^{c})=\mathcal{D}(A_h).$$

Por otra parte, como $Z$ tiene ley $\mathcal{D}$, la probabilidad del evento $\{h(\boldsymbol{x})\neq y\}=\{Z\in A_h\}$ es, por definición, $\mathbb{P}[Z\in A_h]=\mathcal{D}(A_h)$. Las tres cantidades son iguales a $\mathcal{D}(A_h)$. En particular, $L_{\mathcal{D}}(h)=\mathbb{E}[\ell(h,(\boldsymbol{x},y))]$ con la pérdida 0-1 $\ell(h,(\boldsymbol{x},y))=\mathbf{1}[h(\boldsymbol{x})\neq y]$, y $0\le L_{\mathcal{D}}(h)\le 1$. $\blacksquare$

## 1.2 El error empírico es la misma definición con la medida empírica

Sea $S=((\boldsymbol{x}_1,y_1),\dots,(\boldsymbol{x}_m,y_m))$ y $\mathcal{D}_S=\frac1m\sum_{i=1}^m\delta_{(\boldsymbol{x}_i,y_i)}$ la medida empírica. Entonces

$$L_S(h)=\frac{|\{i\in[m]:h(\boldsymbol{x}_i)\neq y_i\}|}{m}=\frac1m\sum_{i=1}^m\mathbf{1}[h(\boldsymbol{x}_i)\neq y_i]=\mathcal{D}_S(A_h),$$

es decir, $L_S$ es el riesgo $L_{\mathcal{D}}$ calculado con $\mathcal{D}_S$ en lugar de $\mathcal{D}$.

## 1.3 Insesgadez para una hipótesis fija

**Afirmación.** Si $h$ se fija sin mirar $S$ y $S\sim\mathcal{D}^m$ (muestra i.i.d.), entonces $\mathbb{E}_{S\sim\mathcal{D}^m}[L_S(h)]=L_{\mathcal{D}}(h)$.

**Demostración.** Por linealidad de la esperanza y porque cada $(\boldsymbol{x}_i,y_i)\sim\mathcal{D}$,

$$\mathbb{E}_S[L_S(h)]=\frac1m\sum_{i=1}^m\mathbb{E}\big[\mathbf{1}[h(\boldsymbol{x}_i)\neq y_i]\big]=\frac1m\sum_{i=1}^m L_{\mathcal{D}}(h)=L_{\mathcal{D}}(h).\qquad\blacksquare$$

:::{note}
**Observación.** La hipótesis se usa de forma esencial: si $h=h_S$ se elige a partir de $S$ (por ejemplo ERM), $\mathbf{1}[h_S(\boldsymbol{x}_i)\neq y_i]$ ya no tiene esperanza $L_{\mathcal{D}}(h_S)$ y $L_S(h_S)$ resulta, en general, optimista. Por eso se usan conjuntos de validación y de prueba.
:::

## 1.4 Optimalidad del predictor de Bayes (ecuación 2.3)

El PDF define $f_{\mathcal{D}}(\boldsymbol{x})=\mathbf{1}[\eta(\boldsymbol{x})\ge 1/2]$ con $\eta(\boldsymbol{x})=\mathbb{P}[y=1\mid\boldsymbol{x}]$ y afirma que, para todo clasificador $g:\mathcal{X}\to\{0,1\}$, $L_{\mathcal{D}}(f_{\mathcal{D}})\le L_{\mathcal{D}}(g)$.

**Demostración.** Fijado $\boldsymbol{x}$, el error condicional de un clasificador $g$ es

$$\mathbb{P}[g(\boldsymbol{x})\neq y\mid\boldsymbol{x}]=\begin{cases}1-\eta(\boldsymbol{x}), & g(\boldsymbol{x})=1,\\ \eta(\boldsymbol{x}), & g(\boldsymbol{x})=0.\end{cases}$$

Para $f_{\mathcal{D}}$ este error vale $\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}$, porque elige $1$ cuando $1-\eta\le\eta$ y $0$ en caso contrario. Si $g(\boldsymbol{x})=f_{\mathcal{D}}(\boldsymbol{x})$ los errores condicionales coinciden. Si $g(\boldsymbol{x})\neq f_{\mathcal{D}}(\boldsymbol{x})$, la diferencia es $|2\eta(\boldsymbol{x})-1|\ge0$: por ejemplo, si $f_{\mathcal{D}}(\boldsymbol{x})=1$ (es decir, $\eta\ge\frac12$) y $g(\boldsymbol{x})=0$, entonces $\eta-(1-\eta)=2\eta-1\ge0$, y el caso $f_{\mathcal{D}}(\boldsymbol{x})=0$ es simétrico. En resumen,

$$\mathbb{P}[g(\boldsymbol{x})\neq y\mid\boldsymbol{x}]-\mathbb{P}[f_{\mathcal{D}}(\boldsymbol{x})\neq y\mid\boldsymbol{x}]=|2\eta(\boldsymbol{x})-1|\,\mathbf{1}[g(\boldsymbol{x})\neq f_{\mathcal{D}}(\boldsymbol{x})]\ \ge 0.$$

Tomando esperanza respecto de $\boldsymbol{x}$ (ley de la esperanza total),

$$L_{\mathcal{D}}(g)-L_{\mathcal{D}}(f_{\mathcal{D}})=\mathbb{E}_{\boldsymbol{x}}\big[|2\eta(\boldsymbol{x})-1|\,\mathbf{1}[g(\boldsymbol{x})\neq f_{\mathcal{D}}(\boldsymbol{x})]\big]\ge0.\qquad\blacksquare$$

Además $L_{\mathcal{D}}(f_{\mathcal{D}})=\mathbb{E}_{\boldsymbol{x}}[\min\{\eta,1-\eta\}]$, que es el resultado que se usa en el capítulo siguiente.
