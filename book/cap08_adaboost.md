# 8. AdaBoost — Derivación del Algoritmo 5.1

**Objetivo.** Demostrar que los pasos del algoritmo ($P_i$, $a_i=\frac12\ln\frac{1-P_i}{P_i}$, actualización y normalización de pesos) resultan de minimizar la pérdida exponencial de forma aditiva por etapas.

**Planteamiento.** Datos $(y_n,\boldsymbol{x}_n)$, $y_n\in\{-1,1\}$, clasificadores base $\phi(\cdot;\boldsymbol{\theta})\in\{-1,1\}$ y $F_i=F_{i-1}+a_i\phi(\cdot;\boldsymbol{\theta}_i)$, con $F_0\equiv0$. La pérdida es

$$J(F)=\sum_{n=1}^N\exp\big(-y_nF(\boldsymbol{x}_n)\big).$$

## 8.1 Reducción a una suma ponderada

En la etapa $i$, $F_{i-1}$ es conocida y se escoge $(a,\boldsymbol{\theta})$ para minimizar

$$J=\sum_{n=1}^N\exp\big(-y_nF_{i-1}(\boldsymbol{x}_n)\big)\exp\big(-y_na\,\phi(\boldsymbol{x}_n;\boldsymbol{\theta})\big)=\sum_{n=1}^N W_n^{(i)}\exp\big(-y_na\,\phi(\boldsymbol{x}_n;\boldsymbol{\theta})\big),$$

con $W_n^{(i)}=\exp(-y_nF_{i-1}(\boldsymbol{x}_n))$ ($W_n^{(1)}=1$ porque $F_0\equiv0$). Dividiendo por la constante positiva $\sum_mW_m^{(i)}$ (no cambia los minimizadores) se trabaja con los pesos normalizados $\omega_n^{(i)}=W_n^{(i)}/\sum_mW_m^{(i)}$; en particular $\omega_n^{(1)}=1/N$ (inicialización del algoritmo).

## 8.2 Elección de $\boldsymbol{\theta}_i$: minimizar $P_i$

Como $y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta})=+1$ si el clasificador acierta y $-1$ si falla, definiendo el error ponderado

$$P_i(\boldsymbol{\theta})=\sum_{n:\,y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta})=-1}\omega_n^{(i)},$$

se separan aciertos y errores:

$$J_i(a,\boldsymbol{\theta})\propto e^{-a}\big(1-P_i(\boldsymbol{\theta})\big)+e^{a}P_i(\boldsymbol{\theta})=e^{-a}+\big(e^{a}-e^{-a}\big)P_i(\boldsymbol{\theta}).$$

Para $a>0$, $e^{a}-e^{-a}>0$, de modo que $J_i$ es creciente en $P_i$: para cualquier $a>0$ el mejor $\boldsymbol{\theta}$ es el que **minimiza $P_i$** (pasos 1 y 2 del algoritmo).

## 8.3 Elección de $a_i$

Con $P=P_i(\boldsymbol{\theta}_i)\in(0,1)$ fijo, sea $f(a)=e^{-a}(1-P)+e^{a}P$. Entonces

$$f'(a)=-e^{-a}(1-P)+e^{a}P=0\ \Longleftrightarrow\ e^{2a}=\frac{1-P}{P}\ \Longleftrightarrow\ a_i=\frac12\ln\frac{1-P_i}{P_i},$$

y $f''(a)=e^{-a}(1-P)+e^{a}P>0$, así que es el mínimo global. Se tiene $a_i>0$ si y solo si $P_i<\tfrac12$, es decir, si el clasificador base es mejor que el azar (si $P_i>\tfrac12$ basta invertir el signo de $\phi$), lo cual justifica el supuesto $a>0$ de 8.2. El valor mínimo es

$$f(a_i)=\sqrt{P_i(1-P_i)}+\sqrt{P_i(1-P_i)}=2\sqrt{P_i(1-P_i)}.$$

## 8.4 Actualización de pesos y normalización

Como $F_i=F_{i-1}+a_i\phi_i$,

$$W_n^{(i+1)}=\exp(-y_nF_i(\boldsymbol{x}_n))=W_n^{(i)}\exp\big(-y_na_i\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i)\big),$$

que es el paso 5.1 del algoritmo (aplicado sobre pesos normalizados). La suma $Z_i=\sum_n\omega_n^{(i)}\exp(-y_na_i\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i))$ (paso 5.2) es exactamente $f(a_i)$, y dividir cada peso entre $Z_i$ (pasos 7 y 8) restablece $\sum_n\omega_n^{(i+1)}=1$. Es decir,

$$Z_i=e^{-a_i}(1-P_i)+e^{a_i}P_i=2\sqrt{P_i(1-P_i)}.$$

Los pesos de las observaciones mal clasificadas se multiplican por $e^{a_i}>1$ y los de las bien clasificadas por $e^{-a_i}<1$, de modo que el siguiente clasificador se concentra en los errores. El clasificador final es $f(\boldsymbol{x})=\mathrm{sgn}\{\sum_{k=1}^Ka_k\phi(\boldsymbol{x};\boldsymbol{\theta}_k)\}$. $\blacksquare$

## 8.5 Complemento: cota del error de entrenamiento

Como $\mathbf{1}[\mathrm{sgn}F(\boldsymbol{x}_n)\neq y_n]\le\exp(-y_nF(\boldsymbol{x}_n))$ (si hay error, $y_nF\le0$ y $e^{-y_nF}\ge1$; si no, el lado derecho es positivo),

$$\frac1N\sum_{n=1}^N\mathbf{1}[\mathrm{sgn}F_K(\boldsymbol{x}_n)\neq y_n]\le\frac1N\sum_{n=1}^N\exp(-y_nF_K(\boldsymbol{x}_n))=\prod_{i=1}^KZ_i.$$

La última igualdad sale de $\sum_nW_n^{(i+1)}=\big(\sum_nW_n^{(i)}\big)Z_i$ y $\sum_nW_n^{(1)}=N$. Si $P_i=\tfrac12-\gamma_i$, entonces $Z_i=\sqrt{1-4\gamma_i^2}\le e^{-2\gamma_i^2}$ (por $1-x\le e^{-x}$), y se concluye

$$\frac1N\sum_{n=1}^N\mathbf{1}[\mathrm{sgn}F_K(\boldsymbol{x}_n)\neq y_n]\le\exp\Big(-2\sum_{i=1}^K\gamma_i^2\Big),$$

de modo que el error de entrenamiento decrece exponencialmente mientras cada clasificador base sea mejor que el azar por un margen $\gamma_i>0$.
