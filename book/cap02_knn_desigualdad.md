# 2. k-NN — La desigualdad $2\eta(1-\eta)\le 2\min\{\eta,1-\eta\}$: caso $1-\eta(\boldsymbol{x})$

**Contexto.** En la demostración del Lema 2.1 (regla 1-NN) se necesita

$$\mathbb{E}_{\boldsymbol{x}}\big[2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\big]\le 2\,\mathbb{E}_{\boldsymbol{x}}\big[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}\big]=2L_{\mathcal{D}}(h^\star).$$

Basta probar la desigualdad puntual para todo $t=\eta(\boldsymbol{x})\in[0,1]$:

$$2t(1-t)\le 2\min\{t,1-t\}.\tag{*}$$

**Caso A: $\min\{t,1-t\}=t$** (es decir, $t\le 1-t$). Ya demostrado en el documento. Se puede cerrar directamente:

$$2t-2t(1-t)=2t^{2}\ge 0\ \Longrightarrow\ 2t(1-t)\le 2t=2\min\{t,1-t\}.$$

**Caso B: $\min\{t,1-t\}=1-t$** (es decir, $1-t\le t$). Se calcula la diferencia:

$$2(1-t)-2t(1-t)=2(1-t)\,(1-t)=2(1-t)^{2}\ge 0\ \Longrightarrow\ 2t(1-t)\le 2(1-t)=2\min\{t,1-t\}.$$

Los casos A y B cubren todo $t\in[0,1]$, luego $(\ast)$ vale. $\blacksquare$

**Forma unificada.** Para $t\in[0,1]$, $2t(1-t)=2\min\{t,1-t\}\cdot\max\{t,1-t\}\le 2\min\{t,1-t\}$, porque $\max\{t,1-t\}\le 1$.

:::{warning}
**Corrección a la cadena del documento (importante).** En la nota se escribe, para el caso $\eta\le 1-\eta$,
$$2\eta(1-\eta)\le 2(1-\eta)^2\le 2\min\{\eta,1-\eta\}.$$
La primera desigualdad es correcta (se multiplica $\eta\le 1-\eta$ por $2(1-\eta)\ge 0$), pero la segunda es **falsa** en general: con $\eta=0.1$ se obtiene $2(0.9)^2=1.62>0.2=2\min\{\eta,1-\eta\}$. La conclusión $(\ast)$ sí es cierta, pero debe justificarse con la cadena $2\eta(1-\eta)\le 2\eta=2\min\{\eta,1-\eta\}$ del caso A. La cadena análoga para el caso B ($2\eta(1-\eta)\le 2\eta^2\le 2(1-\eta)$) también fallaría en su segundo paso, por eso se usa la del caso B de arriba.
:::

## 2.1 Por qué $\mathbb{E}_{\boldsymbol{x}}[\min\{\eta,1-\eta\}]=L_{\mathcal{D}}(h^\star)$

Condicionando en $\boldsymbol{x}$: si $h(\boldsymbol{x})=0$ el error condicional es $\mathbb{P}(y=1\mid\boldsymbol{x})=\eta(\boldsymbol{x})$; si $h(\boldsymbol{x})=1$ es $1-\eta(\boldsymbol{x})$. La regla de Bayes $h^\star(\boldsymbol{x})=\mathbf{1}[\eta(\boldsymbol{x})\ge 1/2]$ escoge en cada punto la opción de menor error condicional, que vale $\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}$. Por la ley de la esperanza total, $L_{\mathcal{D}}(h^\star)=\mathbb{E}_{\boldsymbol{x}}[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}]$.

## 2.2 Verificación de la identidad algebraica usada antes en el Lema

Con $\eta=\eta(\boldsymbol{x})$ y $\eta'=\eta(\boldsymbol{x}')$:

$$\eta'(1-\eta)+(1-\eta')\eta=\eta+\eta'-2\eta\eta'=2\eta(1-\eta)+(\eta-\eta')(2\eta-1),$$

ya que $2\eta-2\eta^2+(2\eta^2-\eta-2\eta\eta'+\eta')=\eta+\eta'-2\eta\eta'$. Con $|2\eta-1|\le1$ y la $c$-Lipschitzness, $|\eta-\eta'|\,|2\eta-1|\le c\|\boldsymbol{x}-\boldsymbol{x}'\|$, y combinando con $(\ast)$:

$$\mathop{\mathbb{E}}_{S\sim\mathcal{D}^m}[L_{\mathcal{D}}(h_S)]\le 2L_{\mathcal{D}}(h^\star)+c\mathop{\mathbb{E}}_{\boldsymbol{x},S_x}\big[\|\boldsymbol{x}-\boldsymbol{x}_{\pi_1(\boldsymbol{x})}\|\big].$$
