# 第1章：预备定理

*编辑：赵志民, 李一飞*

------

本章将对书中出现或用到的重要定理进行回顾，并简要解释其证明和应用场景。对于可能不熟悉相关基础知识的读者，建议参考附录中的基础知识部分。通过这些定理的阐述，希望帮助读者更好地理解数学推导的核心原理，并为后续章节的学习打下坚实基础。**大数定律**（Law of Large Numbers）和**集中不等式**（Concentration Inequality）密切相关，二者共同揭示了随机变量偏离其期望值的行为。大数定律说明，当样本量足够大时，样本均值会以概率收敛于总体的期望值，反映了长期平均结果的稳定性。而集中不等式（定理 1.8 至 1.18）则更进一步，为随机变量在有限样本量下偏离其期望值的可能性提供了精确的上界。这些不等式描述了随机变量偏离期望值的程度有多大，通过对概率的约束，确保这种偏离发生的概率较小，从而为各种随机现象提供了更细致的控制。集中不等式在大数定律的基础上提供了有力的工具，用于分析有限样本中的波动。

## 1.1 Jensen 不等式

对于任意凸函数 $f$，则有：
$$
\begin{equation}
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
\end{equation}
$$
成立。

### 证明

设 $p(x)$ 为 $X$ 的概率密度函数。由 Taylor 展开式及 $f$ 的凸性，可知 $\exists \xi$ 使得：
$$
\begin{equation}
\begin{align*}
f(x) &= f(\mathbb{E}[X]) + f^{\prime}(\mathbb{E}[X])(x-\mathbb{E}[X]) + \frac{f^{\prime \prime}(\xi)}{2}(x-\mathbb{E}[X])^{2} \\
& \geq f(\mathbb{E}[X]) + f^{\prime}(\mathbb{E}[X])(x-\mathbb{E}[X])
\end{align*}
\end{equation}
$$
对上式取期望，得到：
$$
\begin{equation}
\begin{align*}
\mathbb{E}[f(X)] &= \int p(x) f(x) \,dx \\
&\geq f(\mathbb{E}[X]) \int p(x) \,dx + f^{\prime}(\mathbb{E}[X]) \int p(x)(x-\mathbb{E}[X]) \,dx \\
&= f(\mathbb{E}[X])
\end{align*}
\end{equation}
$$
因此，原不等式得证。

如果 $f$ 是凹函数，则 Jensen 不等式变为：
$$
\begin{equation}
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
\end{equation}
$$
这一结论可以通过将上述证明中的 $f$ 替换为 $-f$ 得到。$\square$



## 1.2 Hölder 不等式

对于任意 $p, q \in \mathbb{R}^{+}$，且满足 $\frac{1}{p} + \frac{1}{q} = 1$，则有：
$$
\begin{equation}
\mathbb{E}[|XY|] \leq (\mathbb{E}[|X|^p])^{\frac{1}{p}} (\mathbb{E}[|Y|^q])^{\frac{1}{q}}
\end{equation}
$$
成立。

### 证明

设 $f(x)$ 和 $g(y)$ 分别为 $X$ 和 $Y$ 的概率密度函数，定义：
$$
\begin{equation}
M = \frac{|x|}{(\int_X |x|^p f(x) \,dx)^{\frac{1}{p}}}, \quad N = \frac{|y|}{(\int_Y |y|^q g(y) \,dy)^{\frac{1}{q}}}
\end{equation}
$$
代入 Young 不等式：
$$
\begin{equation}
MN \leq \frac{1}{p}M^p + \frac{1}{q}N^q
\end{equation}
$$
对该不等式两边同时取期望：
$$
\begin{equation}
\begin{align*}
\frac{\mathbb{E}[|XY|]}{(\mathbb{E}[|X|^p])^{\frac{1}{p}} (\mathbb{E}[|Y|^q])^{\frac{1}{q}}} &= \frac{\int_{XY} |xy| f(x)g(y) \,dx\,dy}{(\int_X |x|^p f(x) \,dx)^{\frac{1}{p}} (\int_Y |y|^q g(y) \,dy)^{\frac{1}{q}}} \\
&\leq \frac{\int_X |x|^p f(x) \,dx}{p \int_X |x|^p f(x) \,dx} + \frac{\int_Y |y|^q g(y) \,dy}{q \int_Y |y|^q g(y) \,dy} \\
&= \frac{1}{p} + \frac{1}{q} \\
&= 1
\end{align*}
\end{equation}
$$
因此，Hölder 不等式得证。$\square$
  
  
  
## 1.3 Cauchy-Schwarz 不等式

当 $p = q = 2$ 时，Hölder 不等式退化为 Cauchy-Schwarz 不等式：
$$
\begin{equation}
\mathbb{E}[|XY|] \leq \sqrt{\mathbb{E}[X^{2}] \cdot \mathbb{E}[Y^{2}]}
\end{equation}
$$



## 1.4 Lyapunov 不等式

对于任意 $0 \lt r \leq s$，有：
$$
\begin{equation}
\sqrt[r]{\mathbb{E}[|X|^{r}]} \leq \sqrt[s]{\mathbb{E}[|X|^{s}]}
\end{equation}
$$

### 证明

由 Hölder 不等式：
对任意 $p \geq 1$，有：
$$
\begin{equation}
\begin{align*}
\mathbb{E}[|X|^{r}] &= \mathbb{E}[|X \cdot 1|^{r}] \\
&\leq (\mathbb{E}[|X|^{rp}])^{\frac{1}{p}} \cdot (\mathbb{E}[1^q])^{\frac{1}{q}} \\
&= (\mathbb{E}[|X|^{rp}])^{\frac{1}{p}}
\end{align*}
\end{equation}
$$
记 $s = rp \geq r$，则：
$$
\begin{equation}
\mathbb{E}[|X|^{r}] \leq (\mathbb{E}[|X|^{s}])^{\frac{r}{s}}
\end{equation}
$$
因此，原不等式得证。$\square$



## 1.5 Minkowski 不等式

对于任意 $p \geq 1$，有：
$$
\begin{equation}
\sqrt[p]{\mathbb{E}[|X+Y|^p]} \leq \sqrt[p]{\mathbb{E}[|X|^p]} + \sqrt[p]{\mathbb{E}[|Y|^p]}
\end{equation}
$$

### 证明

由三角不等式和 Hölder 不等式，可得：
$$
\begin{equation}
\begin{align*}
\mathbb{E}[|X+Y|^p] &\leq \mathbb{E}[(|X|+|Y|)|X+Y|^{p-1}] \\
&= \mathbb{E}[|X\|X+Y|^{p-1}] + \mathbb{E}[|Y\|X+Y|^{p-1}] \\
&\leq (\mathbb{E}[|X|^p])^{\frac{1}{p}} (\mathbb{E}[|X+Y|^{(p-1)q}])^{\frac{1}{q}} + (\mathbb{E}[|Y|^p])^{\frac{1}{p}} (\mathbb{E}[|X+Y|^{(p-1)q}])^{\frac{1}{q}} \\
&= [(\mathbb{E}[|X|^p])^{\frac{1}{p}} + (\mathbb{E}[|Y|^p])^{\frac{1}{p}}] \cdot \frac{\mathbb{E}[|X+Y|^p]}{(\mathbb{E}[|X+Y|^p])^{\frac{1}{p}}}
\end{align*}
\end{equation}
$$
化简后即得证。$\square$



## 1.6 Bhatia-Davis 不等式

对 $X \in [a,b]$，有：
$$
\begin{equation}
\mathbb{V}[X] \leq (b - \mathbb{E}[X])(\mathbb{E}[X] - a) \leq \frac{(b-a)^2}{4}
\end{equation}
$$

### 证明

因为 $a \leq X \leq b$，所以有：
$$
\begin{equation}
\begin{align*}
0 &\leq \mathbb{E}[(b-X)(X-a)] \\
&= -\mathbb{E}[X^2] - ab + (a+b)\mathbb{E}[X]
\end{align*}
\end{equation}
$$
因此，
$$
\begin{equation}
\begin{align*}
\mathbb{V}[X] &= \mathbb{E}[X^2] - \mathbb{E}[X]^2 \\
&\leq -ab + (a+b)\mathbb{E}[X] - \mathbb{E}[X^2] \\
&= (b - \mathbb{E}[X])(\mathbb{E}[X] - a)
\end{align*}
\end{equation}
$$

考虑 AM-GM 不等式：
$$
\begin{equation}
xy \leq (\frac{x+y}{2})^2
\end{equation}
$$
将 $x = b - \mathbb{E}[X]$ 和 $y = \mathbb{E}[X] - a$ 带入并化简即得证。$\square$



## 1.7 Union Bound（Boole's）不等式

对于任意事件 $X$ 和 $Y$，有：
$$
\begin{equation}
P(X \cup Y) \leq P(X) + P(Y)
\end{equation}
$$

### 证明

根据概率的加法公式：
$$
\begin{equation}
P(X \cup Y) = P(X) + P(Y) - P(X \cap Y) \leq P(X) + P(Y)
\end{equation}
$$
由于 $P(X \cap Y) \geq 0$，因此不等式得证。$\square$
  


## 1.8 Markov 不等式

若 $X \geq 0$，则对于任意 $\varepsilon \gt 0$，有：
$$
\begin{equation}
P(X \geq \varepsilon) \leq \frac{\mathbb{E}[X]}{\varepsilon}
\end{equation}
$$

### 证明

由定义可得：
$$
\begin{equation}
\mathbb{E}[X] = \int_{0}^{\infty} x p(x) \,dx \geq \int_{\varepsilon}^{\infty} x p(x) \,dx \geq \varepsilon \int_{\varepsilon}^{\infty} p(x) \,dx = \varepsilon P(X \geq \varepsilon)
\end{equation}
$$
因此，原不等式得证。$\square$



## 1.9 Chebyshev 不等式

对于任意 $\varepsilon \gt 0$，有：
$$
\begin{equation}
P(|X-\mathbb{E}[X]| \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\varepsilon^{2}}
\end{equation}
$$

### 证明

利用 Markov 不等式，得到：
$$
\begin{equation}
P(|X-\mathbb{E}[X]| \geq \varepsilon) = P((X-\mathbb{E}[X])^2 \geq \varepsilon^{2}) \leq \frac{\mathbb{E}[(X-\mathbb{E}[X])^2]}{\varepsilon^{2}} = \frac{\mathbb{V}[X]}{\varepsilon^{2}}
\end{equation}
$$
因此，Chebyshev 不等式得证。$\square$



## 1.10 Cantelli 不等式

对于任意 $\varepsilon \gt 0$，有：
$$
\begin{equation}
P(X-\mathbb{E}[X] \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\mathbb{V}[X]+\varepsilon^{2}}
\end{equation}
$$

### 证明

设 $Y = X - \mathbb{E}[X]$，则对于任意 $\lambda \geq 0$，有：
$$
\begin{equation}
\begin{align*}
P(X-\mathbb{E}[X] \geq \varepsilon) &= P(Y \geq \varepsilon) \\
&= P(Y+\lambda \geq \varepsilon+\lambda) \\
&= P((Y+\lambda)^{2} \geq (\varepsilon+\lambda)^{2}) \\
&\leq \frac{\mathbb{E}[(Y+\lambda)^{2}]}{(\varepsilon+\lambda)^{2}} = \frac{\mathbb{V}[X]+\lambda^{2}}{(\varepsilon+\lambda)^{2}}
\end{align*}
\end{equation}
$$
通过对 $\lambda$ 求导，得右端在 $\lambda = \frac{\mathbb{V}[X]}{\varepsilon}$ 时取得最小值 $\frac{\mathbb{V}[X]}{\mathbb{V}[X]+\varepsilon^{2}}$，因此：
$$
\begin{equation}
P(X-\mathbb{E}[X] \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\mathbb{V}[X]+\varepsilon^{2}}
\end{equation}
$$
原不等式得证。$\square$

值得注意的是，Cantelli 不等式是 Chebyshev 不等式的加强版，也称为单边 Chebyshev 不等式。通过类似的构造方法，可以推导出比 Cantelli 不等式更严格的上界。



## 1.11 Chernoff 界（Chernoff-Cramér 界）

对于任意 $\lambda \gt 0, \varepsilon \gt 0$，有：
$$
\begin{equation}
P(X \geq \varepsilon) \leq \min_{\lambda \gt 0} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \varepsilon}}
\end{equation}
$$
对于任意 $\lambda \lt 0, \varepsilon \gt 0$，有：
$$
\begin{equation}
P(X \leq \varepsilon) \leq \min_{\lambda \lt 0} \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \varepsilon}}
\end{equation}
$$

### 证明

应用 Markov 不等式，有：
$$
\begin{equation}
P(X \geq \varepsilon) = P(e^{\lambda X} \geq e^{\lambda \varepsilon}) \leq \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \varepsilon}}, \quad \lambda \gt 0, \varepsilon \gt 0
\end{equation}
$$
同理，
$$
\begin{equation}
P(X \leq \varepsilon) = P(e^{\lambda X} \leq e^{\lambda \varepsilon}) \leq \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda \varepsilon}}, \quad \lambda \lt 0, \varepsilon \gt 0
\end{equation}
$$
因此，Chernoff 界得证。$\square$

基于上述 Chernoff 界的技术，我们可以进一步定义次高斯性：

**定义 1** (随机变量的次高斯性)：若一个期望为零的随机变量 $X$ 的矩母函数满足 $\forall \lambda \in \mathbb{R}^+$：
$$
\begin{equation}
\mathbb{E}[e^{\lambda X}] \leq \exp(\frac{\sigma^2\lambda^2}{2})
\end{equation}
$$
则称 $X$ 服从参数为 $\sigma$ 的次高斯分布。

实际上，Hoeffding 引理中的随机变量 $X$ 服从 $\frac{(b-a)}{2}$ 的次高斯分布。Hoeffding 引理也是次高斯分布的直接体现。次高斯性还有一系列等价定义，这里不作详细讨论。

次高斯分布有一个直接的性质：假设两个独立的随机变量 $X_1, X_2$ 都是次高斯分布的，分别服从参数 $\sigma_1, \sigma_2$，那么 $X_1 + X_2$ 就是服从参数为 $\sqrt{\sigma_1^2 + \sigma_2^2}$ 的次高斯分布。这个结果的证明可以直接利用定义来完成。

显然，并非所有常见的随机变量都是次高斯的，例如指数分布。为此可以扩大定义：

**定义 2** (随机变量的次指数性)：若非负的随机变量 $X$ 的矩母函数满足 $\forall \lambda \in (0,a)$：
$$
\begin{equation}
\mathbb{E}[e^{\lambda X}] \leq \frac{a}{a - \lambda}
\end{equation}
$$
则称 $X$ 服从参数为 $(\mathbb{V}[X], 1/a)$ 的次指数分布。

同样地，次指数性也有一系列等价定义。一种不直观但更常用的定义如下：存在 $(\sigma^2, b)$，使得 $\forall |s| \lt 1/b$：
$$
\begin{equation}
\mathbb{E}[e^{s(X−\mathbb{E}[X])}] \leq \exp ( \frac{s^2\sigma^2}{2} )
\end{equation}
$$

常见的次指数分布包括：指数分布，Gamma 分布，以及**任何有界随机变量**。

类似地，次指数分布对于加法也是封闭的：如果 $X_1, X_2$ 分别是服从 $(\sigma_1^2, b_1)$ 和 $(\sigma_2^2, b_2)$ 的次指数分布，那么 $X_1 + X_2$ 是服从 $(\sigma_1^2 + \sigma_2^2, \max(b_1, b_2))$ 的次指数分布。在高维统计问题中，次高斯分布和次指数分布的尾端控制能得到一些重要的结论。


  
## 1.12 Chernoff 不等式（乘积形式）

对于 $m$ 个独立同分布的随机变量 $x_i \in [0, 1], i \in [m]$，设 $X = \sum_{i=1}^m X_i$，$\mu \gt 0$ 且 $r \leq 1$。若对所有 $i \leq m$ 都有 $\mathbb{E}[x_i] \leq \mu$，则：
$$
\begin{equation}
\begin{align*}
P(X \geq (1+r)\mu m) \leq e^{-\frac{r^2 \mu m}{3}}, \quad r \geq 0 \\
P(X \leq (1-r)\mu m) \leq e^{-\frac{r^2 \mu m}{2}}, \quad r \geq 0
\end{align*}
\end{equation}
$$

### 证明

应用 Markov 不等式，有：
$$
\begin{equation}
P(X \geq (1+r)\mu m) = P((1+r)^X \geq (1+r)^{(1+r)\mu m}) \leq \frac{\mathbb{E}[(1+r)^X]}{(1+r)^{(1+r)\mu m}}
\end{equation}
$$
由于 $x_i$ 之间是独立的，可得：
$$
\begin{equation}
\mathbb{E}[(1+r)^X] = \prod_{i=1}^m \mathbb{E}[(1+r)^{x_i}] \leq \prod_{i=1}^m \mathbb{E}[1+rx_i] \leq \prod_{i=1}^m (1+r\mu) \leq e^{r\mu m}
\end{equation}
$$
其中，第二步使用了 $\forall x \in [0,1]$ 都有 $(1+r)^x \leq 1+rx$，第三步使用了 $\mathbb{E}[x_i] \leq \mu$，第四步使用了 $\forall x \in [0,1]$ 都有 $1+x \leq e^x$。

又由于 $\forall r \in [0,1]$，有 $\frac{e^r}{(1+r)^{1+r}} \leq e^{-\frac{r^2}{3}}$，综上所述：
$$
\begin{equation}
P(X \geq (1+r)\mu m) \leq (\frac{e^r}{(1+r)^{(1+r)}})^{\mu m} \leq e^{-\frac{r^2 \mu m}{3}}
\end{equation}
$$

当我们将 $r$ 替换为 $-r$ 时，根据之前的推导，并利用 $\forall r \in [0,1]$ 有 $\frac{e^r}{(1-r)^{1-r}} \leq e^{-\frac{r^2}{2}}$，可得第二个不等式的证明。$\square$



## 1.13 最优 Chernoff 界

如果 $X$ 是一个随机变量，并且 $\mathbb{E}[e^{\lambda(X-\mathbb{E}X)}] \leq e^{\phi(\lambda)}$ 对于所有 $\lambda \geq 0$ 成立，则有以下结论：
$$
\begin{equation}
P(X - \mathbb{E}X \geq \varepsilon) \leq e^{-\phi^*(\varepsilon)}, \quad \varepsilon \geq 0
\end{equation}
$$
或
$$
\begin{equation}
P(X - \mathbb{E}X \leq (\phi^*)^{-1}(\ln(1/\delta))) \geq 1 - \delta, \quad \delta \in [0,1]
\end{equation}
$$
其中，$\phi^*$ 是 $\phi$ 的凸共轭函数，即 $\phi^*(x) = \sup_{\lambda \geq 0}(\lambda x - \phi(\lambda))$。

### 证明

根据 Chernoff 不等式，有：
$$
\begin{equation}
\begin{align*}
P(X - \mathbb{E}X \geq \varepsilon) &\leq \inf_{\lambda \geq 0} e^{-\lambda \varepsilon} \mathbb{E}[e^{\lambda(X-\mathbb{E}X)}] \\
&\leq \inf_{\lambda \geq 0} e^{\phi(\lambda) - \lambda \varepsilon} \\
&= e^{-\sup_{\lambda \geq 0}(\lambda \varepsilon - \phi(\lambda))} \\
&= e^{-\phi^*(\varepsilon)}
\end{align*}
\end{equation}
$$
因此，最优 Chernoff 界得证。$\square$


  
## 1.14 Hoeffding 不等式

设有 $m$ 个独立随机变量 $X_{i} \in [a_{i}, b_{i}]$，令 $\bar{X}$ 为 $X_{i}$ 的均值。Hoeffding 不等式表示：

$$
\begin{equation}
P(\bar{X} - \mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp (-\frac{2 m^{2} \varepsilon^{2}}{\sum_{i=1}^{m}(b_{i} - a_{i})^{2}})
\end{equation}
$$

### 证明

首先，我们引入一个引理 (Hoeffding 定理)：

对于 $\mathbb{E}[X] = 0$ 且 $X \in [a, b]$ 的随机变量，对于任意 $\lambda \in \mathbb{R}$，有：

$$
\begin{equation}
\mathbb{E}[e^{\lambda X}] \leq \exp( \frac{\lambda^2(b-a)^2}{8} )
\end{equation}
$$

由于 $e^x$ 是凸函数，对于任意 $x \in [a, b]$，可以写为：

$$
\begin{equation}
e^{\lambda x} \leq \frac{b-x}{b-a}e^{\lambda a} + \frac{x-a}{b-a}e^{\lambda b}
\end{equation}
$$

对上式取期望，得到：

$$
\begin{equation}
\mathbb{E}[e^{\lambda X}] \leq \frac{b-\mathbb{E}[X]}{b-a}e^{\lambda a} + \frac{\mathbb{E}[X]-a}{b-a}e^{\lambda b} = \frac{be^{\lambda a} - ae^{\lambda b}}{b - a}
\end{equation}
$$

记 $\theta = -\frac{a}{b-a}$，$h = \lambda(b-a)$，则：

$$
\begin{equation}
\frac{be^{\lambda a} - ae^{\lambda b}}{b - a} = [1-\theta + \theta e^{h}]e^{-\theta h} = e^{\ln(1-\theta + \theta e^{h})}e^{-\theta h} = e^{\ln(1-\theta + \theta e^{h}) -\theta h}
\end{equation}
$$

定义函数 $\varphi(\theta, h) = \ln(1-\theta + \theta e^{h}) -\theta h$。注意到 $\theta$ 实际上与 $h$ 无关。对 $h$ 求偏导数：

$$
\begin{equation}
\frac{\partial \varphi}{\partial h} = \frac{\theta e^h}{1 - \theta + \theta e^h} - \theta
\end{equation}
$$

显然有 $\frac{\partial \varphi}{\partial h}\big|_{h=0^+} = 0$。同理，利用链式法则可得：

$$
\begin{equation}
\frac{\partial^2 \varphi}{\partial h^2} = \frac{\theta e^h(1 - \theta + \theta e^h) - \theta^2e^{2h}}{(1 - \theta + \theta e^h)^2} = \frac{\theta e^h}{1 - \theta + \theta e^h}(1- \frac{\theta e^h}{1 - \theta + \theta e^h}) \leq \frac{1}{4}
\end{equation}
$$

根据泰勒展开式，可以得到：

$$
\begin{equation}
\varphi(\theta, h) \leq \frac{h^2}{8} = \frac{\lambda^2(b-a)^2}{8}
\end{equation}
$$

由 Markov 不等式可知，对于任意 $\lambda \gt 0$：

$$
\begin{equation}
P(\bar{X} - \mathbb{E}[\bar{X}] \geq \varepsilon) = P(e^{\lambda(\bar{X} - \mathbb{E}[\bar{X}])} \geq e^{\lambda \varepsilon}) \leq \frac{\mathbb{E}[e^{\lambda(\bar{X} - \mathbb{E}[\bar{X}])}]}{e^{\lambda \varepsilon}}
\end{equation}
$$

利用随机变量的独立性及 Hoeffding 引理，有：

$$
\begin{equation}
\frac{\mathbb{E}[e^{\lambda(\bar{X} - \mathbb{E}[\bar{X}]})]}{e^{\lambda \varepsilon}} = e^{-\lambda \varepsilon} \prod_{i=1}^{m} \mathbb{E}[e^{\lambda(X_{i} - \mathbb{E}[X_{i}]) / m}] \leq e^{-\lambda \varepsilon} \prod_{i=1}^{m} \exp (\frac{\lambda^{2}(b_{i} - a_{i})^{2}}{8 m^{2}})
\end{equation}
$$

考虑二次函数 $g(\lambda) = -\lambda \varepsilon + \frac{\lambda^{2}}{8 m^{2}} \sum_{i=1}^{m}(b_{i} - a_{i})^{2}$，其最小值为 $-\frac{2 m^{2} \varepsilon^{2}}{\sum_{i=1}^{m}(b_{i} - a_{i})^{2}}$。

因此可以得到：

$$
\begin{equation}
P(\bar{X} - \mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp (-\frac{2 m^{2} \varepsilon^{2}}{\sum_{i=1}^{m}(b_{i} - a_{i})^{2}})
\end{equation}
$$
$\square$

注意，这里并未要求随机变量同分布，因此Hoeffding 不等式常用来解释集成学习的基本原理。



## 1.15 McDiarmid 不等式

对于 $m$ 个独立随机变量 $X_{i} \in \mathcal{X}$，若函数 $f$ 是差有界的，则对于任意 $\varepsilon \gt 0$，有：
$$
\begin{equation}
P(f(X_{1}, \cdots, X_{m})-\mathbb{E}[f(X_{1}, \cdots, X_{m})] \geq \varepsilon) \leq \exp (-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}})
\end{equation}
$$

### 证明

构造一个鞅差序列：
$$
\begin{equation}
D_j = \mathbb{E}[f(X) \mid X_1, \cdots, X_j] - \mathbb{E}[f(X) \mid X_1, \cdots, X_{j-1}]
\end{equation}
$$
容易验证：
$$
\begin{equation}
f(X) - \mathbb{E}[f(X)] = \sum_{i=1}^m D_i
\end{equation}
$$
由于 $f$ 是差有界的，因此满足 Azuma-Hoeffding 引理。代入后可得：
$$
\begin{equation}
P(f(X_1, \cdots, X_m) - \mathbb{E}[f(X_1, \cdots, X_m)] \geq \varepsilon) \leq \exp( -\frac{\varepsilon^2}{2\sum_{i=1}^m c_i^2} )
\end{equation}
$$
原不等式得证。$\square$



## 1.16 Bennett 不等式

对于 $m$ 个独立随机变量 $X_{i}$，令 $\bar{X}$ 为 $X_{i}$ 的均值，若存在 $b \gt 0$，使得 $|X_i-\mathbb{E}[X_i]| \lt b$，则有：
$$
\begin{equation}
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp (-\frac{m \varepsilon^{2}}{2(\sum_{i=1}^{m} \mathbb{V}[X_{i}] / m + b \varepsilon / 3)})
\end{equation}
$$

### 证明

首先，Bennett 不等式是 Hoeffding 不等式的一个加强版，对于独立随机变量的条件可以放宽为弱独立条件，结论仍然成立。

这些 Bernstein 类的集中不等式更多地反映了在非渐近观点下的大数定律表现，即它们刻画了样本均值如何集中在总体均值附近。

如果将样本均值看作是样本（数据点的函数），即令 $f(X_{1}, \cdots, X_{m}) = \sum_{i=1}^{m} X_{i} / m$，那么 Bernstein 类不等式刻画了如下的概率：
$$
\begin{equation}
P(f(X_{1}, \cdots, X_{m}) - \mathbb{E}[f(X_{1}, \cdots, X_{m})] \geq \varepsilon)
\end{equation}
$$
为了在某些泛函上也具有类似 Bernstein 类的集中不等式形式，显然 $f$ 需要满足某些特定性质。差有界性是一种常见的约束条件。

### 定义 3: 差有界性

函数 $f: \mathcal{X}^{m} \rightarrow \mathbb{R}$ 满足对于每个 $i$，存在常数 $c_{i} \lt \infty$，使得：
$$
\begin{equation}
|f(x_{1}, \cdots, x_{i}, \cdots, x_{m})-f(x_{1}, \cdots, x_{i}^{\prime}, \cdots, x_{m})| \leq c_{i}
\end{equation}
$$
则称 $f$ 是差有界的。

为了证明这些结果，需要引入一些新的数学工具。

### 定义 4: 离散鞅

若离散随机变量序列（随机过程）$Z_m$ 满足：

1. $\mathbb{E}[|Z_{i}|] \lt \infty$
2. $\mathbb{E}[Z_{m+1} \mid Z_{1}, \cdots, Z_{m}] = \mathbb{E}[Z_{m+1} \mid \mathcal{F}_{m}] = Z_{m}$

则称序列 $Z_i$ 为离散鞅。

### 引理 2: Azuma-Hoeffding 定理

对于鞅 $Z_{i}$，若 $\mathbb{E}[Z_{i}] = \mu, Z_{1} = \mu_{\circ}$，则构造鞅差序列 $X_{i} = Z_{i} - Z_{i-1}$，且 $|X_{i}| \leq c_{i}$，则对于任意 $\varepsilon \gt 0$，有：
$$
\begin{equation}
P(Z_{m}-\mu \geq \varepsilon) = P(\sum_{i=1}^{m} X_{i} \geq \varepsilon) \leq \exp (-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}})
\end{equation}
$$

### 证明

首先，若 $\mathbb{E}[X \mid Y] = 0$，则有 $\forall \lambda \gt 0$：
$$
\begin{equation}
\mathbb{E}[e^{\lambda X} \mid Y] \leq \mathbb{E}[e^{\lambda X}]
\end{equation}
$$
因此，由恒等式 $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$ 及 Chernoff 一般性技巧，对于任意 $\lambda \gt 0$：
$$
\begin{equation}
\begin{align*}
P(Z_{m}-\mu \geq \varepsilon) &\geq e^{-\lambda \varepsilon} \mathbb{E}[e^{\lambda(Z_{m}-\mu)}] \\
& = e^{-\lambda \varepsilon} \mathbb{E}[\mathbb{E}[e^{\lambda(Z_{m}-\mu)} \mid \mathcal{F}_{m-1}]] \\
& = e^{-\lambda \varepsilon} \mathbb{E}[e^{\lambda(Z_{m-1}-\mu)}\mathbb{E}[e^{\lambda (Z_{m}-Z_{m-1})} \mid \mathcal{F}_{m-1}]]
\end{align*}
\end{equation}
$$

由于 $\{X_{i}\}$ 是鞅差序列，因此 $\mathbb{E}[X_{m} \mid \mathcal{F}_{m-1}] = 0, \mathbb{E}[X_{i}] = 0$。再结合不等式 $\mathbb{E}[e^{\lambda X} \mid Y] \leq \mathbb{E}[e^{\lambda X}]$ 及 Hoeffding 引理，有：
$$
\begin{equation}
\begin{align*}
P(Z_{m}-\mu \geq \varepsilon) & \leq e^{-\lambda \varepsilon} \mathbb{E}[e^{\lambda(Z_{m-1}-\mu)}] \mathbb{E}[e^{\lambda X_{n}}] \\
& \leq e^{-\lambda \varepsilon} \mathbb{E}[e^{\lambda(Z_{m-1}-\mu)}] \exp (\frac{\lambda^{2} c_{m}^{2}}{2})
\end{align*}
\end{equation}
$$
迭代上不等式可得：
$$
\begin{equation}
P(Z_{m}-\mu \geq \varepsilon) \leq e^{-\lambda \varepsilon} \prod_{i=1}^{m} \exp (\frac{\lambda^{2} c_{i}^{2}}{2})
\end{equation}
$$
当 $\lambda = \frac{\varepsilon}{\sum_{i=1}^{m} c_{i}^{2}}$ 时，上式右端取得极小值：
$$
\begin{equation}
P(Z_{m}-\mu \geq \varepsilon) \leq \exp (-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}})
\end{equation}
$$
原不等式得证。$\square$

  
  
## 1.17 Bernstein 不等式

考虑 $m$ 个独立同分布的随机变量 $X_i, i \in [m]$。令 $\bar{X} = \frac{\sum_{i=1}^{m} X_i}{m}$。若存在常数 $b > 0$，使得对所有 $k \geq 2$，第 $k$ 阶矩满足 $\mathbb{E}[|X_i|^k] \leq \frac{k! b^{k-2}}{2} \mathbb{V}[X_1]$，则该不等式成立：

$$
\begin{equation}
\mathbb{P}(\bar{X} \geq \mathbb{E}[\bar{X}] + \epsilon) \leq \exp\left(\frac{-m\epsilon^2}{2 \mathbb{V}[X_1] + 2b\epsilon}\right)
\end{equation}
$$

### 证明

首先，我们需要将**矩条件**（Moment Condition）转换为**亚指数条件**（Sub-exponential Condition），以便进一步推导，即：

- **矩条件：**
    对于随机变量 $X$，其 $k$-阶中心矩 满足如下条件：
    $$
    \begin{equation}
    \mathbb{E}\left[|X - \mathbb{E}[X]|^k\right] \leq \frac{k! \, b^{k-2}}{2} \, \mathbb{V}[X], \quad \forall k \geq 2
    \end{equation}
    $$
    其中：
    1. **中心矩**：随机变量 $X$ 的 $k$ 阶中心矩为 $\mathbb{E}\left[|X - \mathbb{E}[X]|^k\right]$，表示 $X$ 偏离其期望值的 $k$ 次幂的期望值。中心矩用于衡量随机变量的分布形状，尤其是描述其尾部行为。当 $k = 2$ 时，中心矩即为随机变量的方差。
    2. $\frac{k!}{2}$ 是阶乘项，随着 $k$ 增大迅速增长。
    3. $b^{k-2}$ 是一个修正因子，其中 $b$ 为常数，用以控制高阶矩的增长速率。
    4. $\mathbb{V}[X]$ 表示随机变量 $X$ 的方差，它作为标准的离散度量来标定中心矩的大小。

- **亚指数条件**：
    给定随机变量 $X$，其均值为 $\mathbb{E}[X]$，方差为 $\mathbb{V}[X]$，则其偏离均值的随机变量 $X - \mathbb{E}[X]$ 的矩母函数（MGF）满足如下不等式：
    $$
    \begin{equation}
    \mathbb{E}\left[e^{\lambda (X - \mathbb{E}[X])}\right] \leq \exp\left(\frac{\mathbb{V}[X] \lambda^2}{2(1 - b\lambda)}\right), \quad \forall \lambda \in \left[0, \frac{1}{b}\right)
    \end{equation}
    $$
    其中：
    1. **矩母函数**：这是一个重要的工具，用于控制随机变量的尾部概率。矩母函数的形式是 $\mathbb{E}[e^{\lambda X}]$，它通过调整 $\lambda$ 来捕捉不同程度的偏差行为。
    2. **方差主导项**：不等式右边的表达式包含一个方差主导的项 $\frac{\mathbb{V}[X] \lambda^2}{2}$，类似于高斯分布的尾部特性，表明当 $\lambda$ 较小时，$X$ 的偏差行为主要由其方差控制，尾部概率呈现指数衰减。
    3. **修正项 $(1 - b\lambda)$**：该项显示，当 $\lambda$ 接近 $\frac{1}{b}$ 时，尾部偏差的控制变得更加复杂。这种形式通常出现在亚指数条件中，意味着随机变量的尾部行为介于高斯分布和重尾分布之间，尾部衰减较慢但仍比重尾分布快。

---

- **步骤 1：中心化随机变量**

设：
$$
\begin{equation}
Y = X - \mathbb{E}[X]
\end{equation}
$$

我们的目标是对 $Y$ 的矩母函数（MGF）进行上界：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right]
\end{equation}
$$

---

- **步骤 2：展开指数矩**

将 MGF 展开为幂级数（Taylor展开）：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] = \mathbb{E}\left[\sum_{k=0}^\infty \frac{(\lambda Y)^k}{k!}\right] = \sum_{k=0}^\infty \frac{\lambda^k}{k!} \mathbb{E}[Y^k]
\end{equation}
$$

由于 $\mathbb{E}[Y] = 0$，故 $k = 1$ 项消失：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] = 1 + \sum_{k=2}^\infty \frac{\lambda^k}{k!} \mathbb{E}[Y^k]
\end{equation}
$$

---

- **步骤 3：使用矩条件对中心矩进行上界**

根据矩条件：
$$
\begin{equation}
\mathbb{E}\left[|Y|^k\right] \leq \frac{k! \, b^{k-2}}{2} \, \mathbb{V}[X]
\end{equation}
$$

因此：
$$
\begin{equation}
|\mathbb{E}[Y^k]| \leq \mathbb{E}\left[|Y|^k\right] \leq \frac{k! \, b^{k-2}}{2} \, \mathbb{V}[X]
\end{equation}
$$

---

- **步骤 4：代入 MGF 展开式**

将上界代入 MGF 展开式：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] \leq 1 + \sum_{k=2}^\infty \frac{\lambda^k}{k!} \cdot \frac{k! \, b^{k-2}}{2} \, \mathbb{V}[X] = 1 + \frac{\mathbb{V}[X]}{2} \sum_{k=2}^\infty (b\lambda)^{k-2} \lambda^2
\end{equation}
$$

通过令 $j = k - 2$ 进行简化：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] \leq 1 + \frac{\mathbb{V}[X] \lambda^2}{2} \sum_{j=0}^\infty (b\lambda)^j
\end{equation}
$$

---

- **步骤 5：求解几何级数的和**

当 $b\lambda < 1$ 时，几何级数收敛：
$$
\begin{equation}
\sum_{j=0}^\infty (b\lambda)^j = \frac{1}{1 - b\lambda}
\end{equation}
$$

因此：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] \leq 1 + \frac{\mathbb{V}[X] \lambda^2}{2(1 - b\lambda)}
\end{equation}
$$

---

- **步骤 6：应用指数不等式**

使用不等式 $1 + x \leq e^{x}$ 对所有实数 $x$ 成立：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] \leq \exp\left(\frac{\mathbb{V}[X] \lambda^2}{2(1 - b\lambda)}\right)
\end{equation}
$$

这与**亚指数条件**相符：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] \leq \exp\left(\frac{\mathbb{V}[X] \lambda^2}{2(1 - b\lambda)}\right), \quad \forall \lambda \in \left[0, \frac{1}{b}\right)
\end{equation}
$$

---

接下来我们完成在给定矩条件下的**Bernstein 不等式**的证明，即：

**陈述：**

给定 $m$ 个独立同分布的随机变量 $X_i, i \in [m]$，令 $\bar{X} = \frac{1}{m}\sum_{i=1}^{m} X_i$。若存在常数 $b > 0$，使得对所有 $k \geq 2$，
$$
\begin{equation}
\mathbb{E}\left[|X_i - \mathbb{E}[X_i]|^k\right] \leq \frac{k! \, b^{k-2}}{2} \, \mathbb{V}[X_1],
\end{equation}
$$

则对于任意 $\epsilon > 0$，
$$
\begin{equation}
\mathbb{P}\left(\bar{X} \geq \mathbb{E}[\bar{X}] + \epsilon\right) \leq \exp\left(\frac{-m\epsilon^2}{2 \mathbb{V}[X_1] + 2b\epsilon}\right)
\end{equation}
$$

---

- **步骤 1：定义单侧 Bernstein 条件**

首先，回顾对于参数 $b > 0$ 的**单侧 Bernstein 条件**：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda(Y)}\right] \leq \exp\left(\frac{\mathbb{V}[Y] \lambda^2 / 2}{1 - b\lambda}\right), \quad \forall \lambda \in \left[0, \frac{1}{b}\right)
\end{equation}
$$
其中 $Y = X - \mathbb{E}[X]$。

根据**矩条件**，我们已经证明 $Y$ 满足**亚指数条件**：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda Y}\right] \leq \exp\left(\frac{\mathbb{V}[Y] \lambda^2}{2(1 - b\lambda)}\right), \quad \forall \lambda \in \left[0, \frac{1}{b}\right)
\end{equation}
$$

因此，$Y$ 满足**单侧 Bernstein 条件**，且 $\mathbb{V}[Y] = \mathbb{V}[X]$。

- **步骤 2：应用 Chernoff 界**

考虑 $m$ 个独立同分布随机变量 $Y_i = X_i - \mathbb{E}[X_i]$ 的和：
$$
\begin{equation}
S_m = \sum_{i=1}^{m} Y_i = m(\bar{X} - \mathbb{E}[\bar{X}])
\end{equation}
$$

我们的目标是对概率 $\mathbb{P}(S_m \geq m\epsilon)$ 进行上界，这等价于 $\mathbb{P}(\bar{X} \geq \mathbb{E}[\bar{X}] + \epsilon)$。

使用**Chernoff 界**：
$$
\begin{equation}
\mathbb{P}(S_m \geq m\epsilon) \leq \inf_{\lambda > 0} \exp(-\lambda m \epsilon) \mathbb{E}\left[e^{\lambda S_m}\right]
\end{equation}
$$

- **步骤 3：对和的矩母函数进行上界**

由于 $Y_i$ 是独立的：
$$
\begin{equation}
\mathbb{E}\left[e^{\lambda S_m}\right] = \prod_{i=1}^{m} \mathbb{E}\left[e^{\lambda Y_i}\right] \leq \left[\exp\left(\frac{\mathbb{V}[Y_i] \lambda^2}{2(1 - b\lambda)}\right)\right]^m = \exp\left(\frac{m \mathbb{V}[Y] \lambda^2}{2(1 - b\lambda)}\right)
\end{equation}
$$

因此：
$$
\begin{equation}
\mathbb{P}(S_m \geq m\epsilon) \leq \inf_{\lambda > 0} \exp\left(-\lambda m \epsilon + \frac{m \mathbb{V}[Y] \lambda^2}{2(1 - b\lambda)}\right)
\end{equation}
$$

- **步骤 4：对 $\lambda$ 进行优化**

为了找到最紧的界，我们需要对 $\lambda$ 进行优化。最优的 $\lambda$ 是使指数最小的值：
$$
\begin{equation}
-\lambda m \epsilon + \frac{m \mathbb{V}[Y] \lambda^2}{2(1 - b\lambda)}
\end{equation}
$$

对 $\lambda$ 求导并令其为零：
$$
\begin{equation}
-\epsilon + \frac{\mathbb{V}[Y] \lambda}{1 - b\lambda} + \frac{\mathbb{V}[Y] \lambda^2 b}{2(1 - b\lambda)^2} = 0
\end{equation}
$$

然而，直接求解该方程较为复杂。我们可以选择：
$$
\begin{equation}
\lambda = \frac{\epsilon}{\mathbb{V}[Y] + b\epsilon}
\end{equation}
$$

此时 $\lambda$ 满足 $\left[0, \frac{1}{b}\right)$ 的范围，因为：
$$
\begin{equation}
\lambda b = \frac{b\epsilon}{\mathbb{V}[Y] + b\epsilon} < 1
\end{equation}
$$

- **步骤 5：将最优的 $\lambda$ 代入界中**

将 $\lambda = \frac{\epsilon}{\mathbb{V}[Y] + b\epsilon}$ 代入指数中：
$$
\begin{equation}
-\lambda m \epsilon + \frac{m \mathbb{V}[Y] \lambda^2}{2(1 - b\lambda)} = -\frac{m \epsilon^2}{\mathbb{V}[Y] + b\epsilon} + \frac{m \mathbb{V}[Y] \left(\frac{\epsilon}{\mathbb{V}[Y] + b\epsilon}\right)^2}{2\left(1 - \frac{b\epsilon}{\mathbb{V}[Y] + b\epsilon}\right)}
\end{equation}
$$

在第二项中简化分母：
$$
\begin{equation}
1 - b\lambda = 1 - \frac{b\epsilon}{\mathbb{V}[Y] + b\epsilon} = \frac{\mathbb{V}[Y]}{\mathbb{V}[Y] + b\epsilon}
\end{equation}
$$

现在，代入回去：
$$
\begin{equation}
-\frac{m \epsilon^2}{\mathbb{V}[Y] + b\epsilon} + \frac{m \epsilon^2}{2(\mathbb{V}[Y] + b\epsilon)} = -\frac{m \epsilon^2}{2(\mathbb{V}[Y] + b\epsilon)}
\end{equation}
$$

因此：
$$
\begin{equation}
\mathbb{P}(S_m \geq m\epsilon) \leq \exp\left(-\frac{m \epsilon^2}{2(\mathbb{V}[Y] + b\epsilon)}\right)
\end{equation}
$$

- **步骤 6：回到样本均值**

回忆：
$$
\begin{equation}
S_m = m(\bar{X} - \mathbb{E}[\bar{X}])
\end{equation}
$$

因此：
$$
\begin{equation}
\mathbb{P}\left(\bar{X} - \mathbb{E}[\bar{X}] \geq \epsilon\right) = \mathbb{P}(S_m \geq m\epsilon) \leq \exp\left(-\frac{m \epsilon^2}{2(\mathbb{V}[Y] + b\epsilon)}\right)
\end{equation}
$$

由于 $\mathbb{V}[Y] = \mathbb{V}[X]$，我们得到：
$$
\begin{equation}
\mathbb{P}\left(\bar{X} \geq \mathbb{E}[\bar{X}] + \epsilon\right) \leq \exp\left(-\frac{m \epsilon^2}{2(\mathbb{V}[X] + b\epsilon)}\right)
\end{equation}
$$
$\square$



## 1.18 Azuma–Hoeffding（Azuma）不等式

对于均值为 $Z_0 = \mu$ 的鞅差序列 $\{Z_m, m \geq 1\}$，若 $|Z_i - Z_{i-1}| \leq c_i$，其中$c_i \gt 0$为已知常数，则对于任意 $\varepsilon \gt 0$，有：
$$
\begin{equation}
\begin{align*}
P(Z_{m} - \mu \geq \varepsilon) &\leq \exp(-\frac{\varepsilon^{2}}{2\sum_{i=1}^{m} c_{i}^{2}}) \\
P(Z_{m} - \mu \leq -\varepsilon) &\leq \exp(-\frac{\varepsilon^{2}}{2\sum_{i=1}^{m} c_{i}^{2}})
\end{align*}
\end{equation}
$$

### 证明

1. **构造指数鞅**

   考虑参数 $s \gt 0$，构造如下的指数鞅：

   $$
   \begin{equation}
   M_m = \exp(s(Z_m - \mu) - \frac{s^2}{2}\sum_{i=1}^{m} c_i^2)
   \end{equation}
   $$

   我们需要证明 $\{M_m\}_{m \geq 0}$ 是一个超鞅。

2. **验证鞅性质**

   对于任意 $m \geq 1$，有

   $$
   \begin{equation}
   \mathbb{E}[M_m \mid \mathcal{F}_{m-1}] = \mathbb{E}[\exp(s(Z_m - Z_{m-1})) \mid \mathcal{F}_{m-1}] \cdot \exp(s(Z_{m-1} - \mu) - \frac{s^2}{2}\sum_{i=1}^{m} c_i^2)
   \end{equation}
   $$

   由于 $|Z_m - Z_{m-1}| \leq c_m$，并且 $\mathbb{E}[Z_m - Z_{m-1} \mid \mathcal{F}_{m-1}] = 0$（鞅性质），可以应用 Hoeffding 引理得到：

   $$
   \begin{equation}
   \mathbb{E}[\exp(s(Z_m - Z_{m-1})) \mid \mathcal{F}_{m-1}] \leq \exp(s\mathbb{E}[Z_m - Z_{m-1} \mid \mathcal{F}_{m-1}] + \frac{s^2(c_m-(-c_m))^2}{8}) = \exp(\frac{s^2 c_m^2}{2})
   \end{equation}
   $$

   因此，

   $$
   \begin{equation}
   \mathbb{E}[M_m \mid \mathcal{F}_{m-1}] \leq \exp(\frac{s^2 c_m^2}{2}) \cdot \exp(s(Z_{m-1} - \mu) - \frac{s^2}{2}\sum_{i=1}^{m} c_i^2) = M_{m-1}
   \end{equation}
   $$

   这表明 $\{M_m\}$ 是一个超鞅。

3. **应用鞅不等式**

   由于 $\{M_m\}$ 是一个超鞅，且 $M_0 = \exp(0) = 1$，根据超鞅的性质，有

   $$
   \begin{equation}
   \mathbb{E}[M_m] \le M_0 = 1
   \end{equation}
   $$

   对于事件 $\{Z_m - \mu \geq \varepsilon\}$，有

   $$
   \begin{equation}
   M_m = \exp(s(Z_m - \mu) - \frac{s^2}{2}\sum_{i=1}^{m} c_i^2) \geq \exp(s \varepsilon - \frac{s^2}{2}\sum_{i=1}^{m} c_i^2)
   \end{equation}
   $$

   我们令 $a = \exp\left(s \varepsilon - \frac{s^2}{2}\sum_{i=1}^{m} c_i^2\right)$，由于 $\{Z_m - \mu \geq \varepsilon\}$ 蕴含了 $\{M_m \geq a\}$，所以：
   
   $$
   \begin{equation}
   P\left(Z_m - \mu \geq \varepsilon\right) \leq P\left(M_m \geq a\right)
   \end{equation}
   $$

   结合已知的 $\mathbb{E}[M_m] \leq 1$，应用 Markov 不等式可得：

   $$
   \begin{equation}
   P\left(M_m \geq a\right) \leq \frac{1}{a} = \exp\left(-s \varepsilon + \frac{s^2}{2}\sum_{i=1}^{m} c_i^2\right)
   \end{equation}
   $$
   
   因此，我们得到：

   $$
   \begin{equation}
   P\left(Z_m - \mu \geq \varepsilon\right) \leq \exp\left(-s \varepsilon + \frac{s^2}{2}\sum_{i=1}^{m} c_i^2\right)
   \end{equation}
   $$

4. **优化参数 $s$**

   为了得到最优的上界，选择 $s$ 使得表达式 $-s \varepsilon + \frac{s^2}{2}\sum c_i^2$ 最小化。对 $s$ 求导并取零：

   $$
   \begin{equation}
   -\varepsilon + s \sum_{i=1}^{m} c_i^2 = 0 \quad \Rightarrow \quad s = \frac{\varepsilon}{\sum_{i=1}^{m} c_i^2}
   \end{equation}
   $$

   代入得：

   $$
   \begin{equation}
   P(Z_m - \mu \geq \varepsilon) \leq \exp(-\frac{\varepsilon^2}{2\sum_{i=1}^{m} c_i^2})
   \end{equation}
   $$

   这即是 Azuma 不等式的上侧不等式。

5. **下侧不等式的证明**

   对于下侧不等式，可以类似地考虑 $-Z_m$ 作为鞅，应用相同的方法得到：

   $$
   \begin{equation}
   P(Z_m - \mu \leq -\varepsilon) \leq \exp(-\frac{\varepsilon^2}{2\sum_{i=1}^{m} c_i^2})
   \end{equation}
   $$

   因此，Azuma 不等式得证。$\square$



## 1.19 Slud 不等式

若 $X \sim B(m,p)$，则有：
$$
\begin{equation}
P(\frac{X}{m} \geq \frac{1}{2}) \geq \frac{1}{2}[1 - \sqrt{1-\exp(-\frac{m\varepsilon^{2}}{1-\varepsilon^{2}})}]
\end{equation}
$$
其中 $p = \frac{1-\varepsilon}{2}$。

### 证明

二项随机变量 $X$ 表示在 $m$ 次独立伯努利试验中成功的次数，成功概率为 $p$。对于大的 $m$，二项分布 $B(m,p)$ 可以近似为均值 $\mu=mp$ 和方差 $\sigma^2=mp(1-p)$ 的正态分布：
$$
\begin{equation}
\begin{align*}
\mu &= \frac{m(1-\varepsilon)}{2} \\
\sigma^2 &= \frac{m(1-\varepsilon^2)}{4}
\end{align*}
\end{equation}
$$
令 $Z=\frac{X-\mu}{\sigma}$，代入 $\mu$ 和 $\sigma$，有：
$$
\begin{equation}
P[\frac{X}{m} \geq \frac{1}{2}] = P[Z \geq \frac{\frac{m}{2}-\mu}{\sigma}] = P[Z \geq \frac{\varepsilon\sqrt{m}}{\sqrt{1-\varepsilon^2}}]
\end{equation}
$$
根据正态分布不等式（定理 21），有：
$$
\begin{equation}
P[Z \geq x] \geq \frac{1}{2}[1 - \sqrt{1-\exp(-\frac{2x^2}{\pi})}] \geq \frac{1}{2}[1 - \sqrt{1-\exp(-x^2)}]
\end{equation}
$$
代入可得：
$$
\begin{equation}
P[Z \geq \frac{\varepsilon\sqrt{m}}{\sqrt{1-\varepsilon^2}}] \geq \frac{1}{2}[1 - \sqrt{1-\exp(-\frac{m\varepsilon^2}{1-\varepsilon^2})}]
\end{equation}
$$
$\square$



## 1.20 上界不等式之加性公式

若 $\sup(f)$ 和 $\sup(g)$ 分别为函数 $f$ 和 $g$ 的上界，则有：
$$
\begin{equation}
\sup(f+g) \leq \sup(f) + \sup(g)
\end{equation}
$$

### 证明

假设 $f,g$ 分别有相同的定义域 $D_f,D_g$。根据上确界的定义，对于每一个 $x \in D_f \cap D_g$，我们有
$$
\begin{equation}
g(x) \leq \sup_{y \in D_g} g(y),
\end{equation}
$$
从而
$$
\begin{equation}
f(x) + g(x) \leq f(x) + \sup_{y \in D_g} g(y).
\end{equation}
$$
因为这对于每一个 $x \in D_f \cap D_g$ 都成立，我们可以在不等式的两边取上确界，得到：
$$
\begin{equation}
\sup_{x \in D_f \cap D_g}(f(x) + g(x)) \leq \sup_{x \in D_f \cap D_g} f(x) + \sup_{y \in D_g} g(y) \leq \sup_{z \in D_f} f(z) + \sup_{y \in D_g} g(y).
\end{equation}
$$
这里我们使用了 $\sup_{x \in D_f \cap D_g} f(x) \leq \sup_{z \in D_f} f(z)$，因为 $D_f \cap D_g \subset D_f$。$\square$

值得注意的是，该不等式在（4.33）中利用过两次，且原推导并没有用到 Jensen 不等式的任何性质。

另外，加性公式有几个常见的变形，例如：
$$
\begin{equation}
\sup(f-g) - \sup(f-k) \leq \sup(k-g)
\end{equation}
$$
该不等式在（4.29）中出现过。



## 1.21 正态分布不等式

若 $X$ 是一个服从标准正态分布的随机变量，那么对于任意 $u \geq 0$，有：
$$
\begin{equation}
\mathbb{P}[X \leq u] \leq \frac{1}{2}\sqrt{1-e^{-\frac{2}{\pi}u^2}}
\end{equation}
$$

### 证明

令 $G(u)=\mathbb{P}[X \leq u]$，则有：
$$
\begin{equation}
2G(u) = \int_{-u}^u(2\pi)^{-1/2}e^{-x^2/2}\,dx = \int_{-u}^u(2\pi)^{-1/2}e^{-y^2/2}\,dy
\end{equation}
$$
因此：
$$
\begin{equation}
2\pi[2G(u)]^2 = \int_{-u}^u \int_{-u}^u e^{-(x^2+y^2)/2}\,dx\,dy
\end{equation}
$$
让我们考虑更一般的积分形式：
$$
\begin{equation}
2\pi[2G(u)]^2 = \iint_R e^{-(x^2+y^2)/2}\,dx\,dy
\end{equation}
$$
此时 $R$ 为任意面积为 $4u^2$ 的区域。通过反证法可以证明，只有当 $R$ 为以原点为中心的圆形区域 $R_0$ 时，积分值最大：
$$
\begin{equation}
R_0 = \{(x,y):\pi(x^2+y^2)\leq 4u^2\}
\end{equation}
$$
此时，有：
$$
\begin{equation}
\begin{align*}
2\pi[2G(u)]^2 &\leq \iint_{R_0} e^{-(x^2+y^2)/2}\,dx\,dy \\
&=\int_0^{2\pi}\int_0^{2u\pi^{-1/2}} e^{-r^2/2}r\,dr\,d\varphi \\
&= 2\pi(1-e^{-2u^2/\pi})
\end{align*}
\end{equation}
$$
因此，有：
$$
\begin{equation}
G(u) = \mathbb{P}[X \leq u] \leq \frac{1}{2}\sqrt{1-e^{-\frac{2}{\pi}u^2}}
\end{equation}
$$
进一步，我们可以得到：
$$
\begin{equation}
\mathbb{P}[X \geq u] \geq \frac{1}{2}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})
\end{equation}
$$
$\square$



## 1.22 AM-GM 不等式

算术平均数和几何平均数的不等式，简称 AM-GM 不等式。该不等式指出非负实数序列的算术平均数大于等于该序列的几何平均数，当且仅当序列中的每个数相同时，等号成立。形式上，对于非负实数序列 $\{x_n\}$，其算术平均值定义为：
$$
\begin{equation}
A_n=\frac{1}{n}\sum_{i=1}^n x_i
\end{equation}
$$
其几何平均值定义为：
$$
\begin{equation}
G_n=\sqrt[n]{\prod_{i=1}^n x_i}
\end{equation}
$$
则 AM-GM 不等式成立：
$$
\begin{equation}
A_n \geq G_n
\end{equation}
$$

### 证明

我们可以通过 Jensen 不等式来证明 AM-GM 不等式。首先，我们考虑函数 $f(x)=-\ln x$，该函数是凸函数，因此有：
$$
\begin{equation}
\frac{1}{n}\sum_{i=1}^n -\ln x_i \geq -\ln(\frac{1}{n}\sum_{i=1}^n x_i)
\end{equation}
$$
即：
$$
\begin{equation}
\begin{align*}
\ln(\frac{1}{n}\sum_{i=1}^n x_i) &\geq \frac{1}{n}\sum_{i=1}^n \ln x_i = \ln(\sqrt[n]{\prod_{i=1}^n x_i}) \\
\Rightarrow \frac{1}{n}\sum_{i=1}^n x_i &\geq \sqrt[n]{\prod_{i=1}^n x_i}
\end{align*}
\end{equation}
$$
当取 $x_1 = x_2 = \cdots = x_n$ 时，等号成立。特别地，当 $n=2$ 时，我们有：
$$
\begin{equation}
\frac{x_1 + x_2}{2} \geq \sqrt{x_1 x_2}
\end{equation}
$$
$\square$



## 1.23 Young 不等式

对于任意 $a, b \geq 0$ 且 $p, q \gt 1$，若 $\frac{1}{p} + \frac{1}{q} = 1$，则有：
$$
\begin{equation}
ab \leq \frac{a^p}{p} + \frac{b^q}{q}
\end{equation}
$$
当且仅当 $a^p = b^q$ 时，等号成立。

### 证明

我们可以通过 Jensen 不等式来证明 Young 不等式。首先，当 $ab = 0$ 时，该不等式显然成立。当 $a, b \gt 0$ 时，我们令 $t = 1/p, 1-t = 1/q$，根据 $\ln(x)$ 的凹性，我们有：
$$
\begin{equation}
\begin{align*}
\ln(t a^p + (1-t) b^q) &\geq t\ln(a^p) + (1-t)\ln(b^q) \\
&= \ln(a) + \ln(b) \\
&= \ln(ab)
\end{align*}
\end{equation}
$$
当且仅当 $a^p = b^q$ 时，等号成立。$\square$



## 1.24 Bayes 定理

贝叶斯定理是概率论中的一个重要定理，它描述了在已知某些条件下更新事件概率的数学方法。贝叶斯定理的公式为：
$$
\begin{equation}
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\end{equation}
$$
其中：
- $P(A|B)$ 是在事件 B 发生的情况下事件 A 发生的后验概率。
- $P(B|A)$ 是在事件 A 发生的情况下事件 B 发生的似然函数。
- $P(A)$ 是事件 A 的先验概率。
- $P(B)$ 是事件 B 的边缘概率。

### 证明

根据条件概率的定义，事件 A 在事件 B 发生下的条件概率 $P(A|B)$ 表示为：
$$
\begin{equation}
P(A|B) = \frac{P(A \cap B)}{P(B)}
\end{equation}
$$

同样地，事件 B 在事件 A 发生下的条件概率 $P(B|A)$ 表示为：
$$
\begin{equation}
P(B|A) = \frac{P(A \cap B)}{P(A)}
\end{equation}
$$

通过这两个公式可以得到联合概率 $P(A \cap B)$ 的两种表示方式：
$$
\begin{equation}
P(A \cap B) = P(A|B) \cdot P(B)
\end{equation}
$$

以及：
$$
\begin{equation}
P(A \cap B) = P(B|A) \cdot P(A)
\end{equation}
$$

由于联合概率的性质，我们可以将上述两个等式等同：
$$
\begin{equation}
P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
\end{equation}
$$

将上述等式两边同时除以 $P(B)$，得到贝叶斯定理：
$$
\begin{equation}
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\end{equation}
$$
$\square$

通过先验和后验的更新过程，贝叶斯统计提供了一种动态的、不断修正认知的不确定性量化方法。



## 1.25 广义二项式定理

广义二项式定理（Generalized Binomial Theorem）是二项式定理的扩展：
$$
\begin{equation}
(x + y)^r = \sum_{k=0}^{\infty} \binom{r}{k} x^{r-k} y^k, \quad |x| \lt |y|, \quad k \in \mathbb{N}, \quad r \in \mathbb{R}
\end{equation}
$$
其中我们令 $\binom{r}{k} := \frac{(r)_k}{k!}$，$(r)_k = r(r-1) \cdots (r-k+1)$ 为递降阶乘（falling factorial）。

### 证明

首先代入定义，易证：
$$
\begin{equation}
(r-k) \binom{r}{k} + (r-(k-1)) \binom{r}{k-1} = r \binom{r}{k}
\end{equation}
$$

我们从特殊情况 $y = 1$ 开始。首先我们证明只要 $|x| \lt 1$，后者级数就会收敛。

通过使用幂级数收敛半径的商式来证明这一点，由于绝对值的连续性使我们可以先在绝对值内部计算极限，可得：
$$
\begin{equation}
\lim_{k \to \infty} \frac{|a_k|}{|a_{k+1}|} = \lim_{k \to \infty} | \frac{k+1}{r-k} | = |-1| = 1
\end{equation}
$$
因此我们有一个为 1 的收敛半径。这种收敛使我们能够在 $|x| \lt 1$ 的收敛区域内应用逐项求导，得到：
$$
\begin{equation}
\frac{d}{dx} \sum_{k=0}^\infty \binom{r}{k} x^k = \sum_{k=1}^\infty (r-(k-1)) \binom{r}{k-1} x^{k-1}
\end{equation}
$$
如果我们将我们正在考虑的级数定义的函数记为 $g(x)$，我们得到：
$$
\begin{equation}
\begin{align*}
(1 + x) \frac{d}{dx} g(x) &= \sum_{k=1}^\infty (r-(k-1)) \binom{r}{k-1} x^{k-1} + \sum_{k=1}^\infty (r-(k-1)) \binom{r}{k-1} x^k \\
&= r + \sum_{k=1}^\infty ( (r-k) \binom{r}{k} + (r-(k-1)) \binom{r}{k-1} ) x^k \\
&= r + r \sum_{k=1}^\infty \binom{r}{k} x^k \\
&= r g(x),
\end{align*}
\end{equation}
$$
上式的推导使用了前述引理。

现在定义 $f(x) = (1 + x)^r$，我们通过通常的求导规则得到：
$$
\begin{equation}
\frac{d}{dx} ( \frac{g(x)}{f(x)} ) = \frac{g'(x) f(x) - f'(x) g(x)}{f(x)^2} = \frac{r\frac{g(x)}{x+1}(1+x)^r - rg(x)(1 + x)^{r-1}}{f(x)^2} = 0
\end{equation}
$$
$|x| \lt 1$ 意味着 $f(x) \neq 0$，因此 $g/f$ 为常数。又 $f(0) = g(0) = 1$ 可得 $f(x) = g(x)$。

对于一般的 $x, y \in \mathbb{R}$ 且 $|x| \lt |y|$，我们有：
$$
\begin{equation}
\frac{(x + y)^r}{y^r} = (\frac{x}{y} + 1)^r = \sum_{k=0}^\infty \binom{r}{k} (\frac{x}{y})^k;
\end{equation}
$$
收敛性由假设 $|x/y| \lt 1$ 保证。为了得到原定理的形式，我们只需乘以 $y^r$ 即可。$\square$



## 1.26 Stirling 公式

Stirling 公式是用于近似计算阶乘的一种公式，即使在 $n$ 很小时也有很高的精度。Stirling 公式的一种形式为：
$$
\begin{equation}
n! = \sqrt{2\pi} n^{n+1/2} e^{-n} e^{r_n}
\end{equation}
$$
其中，$\frac{1}{12n + 1} \lt r_n \lt \frac{1}{12n}$。

### 证明

我们令：
$$
\begin{equation}
S_n = \ln(n!) = \sum_{p=1}^{n-1} \ln(p+1)
\end{equation}
$$
且
$$
\begin{equation}
\ln(p+1) = A_p + b_p - \varepsilon_p
\end{equation}
$$
其中：
$$
\begin{equation}
\begin{align*}
A_p &= \int_{p}^{p+1} \ln x \, dx \\
b_p &= \frac{1}{2} [\ln(p+1) - \ln(p)] \\
\varepsilon_p &= \int_{p}^{p+1} \ln x \, dx - \frac{1}{2} [\ln(p+1) + \ln(p)]
\end{align*}
\end{equation}
$$
此时：
$$
\begin{equation}
S_n = \sum_{p=1}^{n-1} (A_p + b_p - \varepsilon_p)
= \int_{1}^{n} \ln x \, dx + \frac{1}{2} \ln n - \sum_{p=1}^{n-1} \varepsilon_p
\end{equation}
$$
易证 $\int \ln x \, dx = x \ln x - x + C, \, C \in \mathbb{R}$，故：
$$
\begin{equation}
S_n = (n+1/2)\ln n - n + 1 - \sum_{p=1}^{n-1} \varepsilon_p
\end{equation}
$$
此时：
$$
\begin{equation}
\varepsilon_p = \frac{2p+1}{2} \ln(\frac{p+1}{p}) - 1
\end{equation}
$$

接下来我们对 $\ln(\frac{p+1}{p})$ 进行级数展开，根据广义二项式定理，即：

令 $a = -1, \, t = \frac{1}{p}, \, t \in (-1, 1)$，则有：
$$
\begin{equation}
\frac{1}{1 + t} = 1 - t + t^2 - t^3 + t^4 - \cdots
\end{equation}
$$
对上式两边同时进行积分，我们有：
$$
\begin{equation}
\ln(1 + t) = t - \frac{1}{2} t^2 + \frac{1}{3} t^3 - \frac{1}{4} t^4 + \cdots
\end{equation}
$$
如果我们令 $-t$ 来代替 $t$，则有：
$$
\begin{equation}
\ln \frac{1}{1 - t} = t + \frac{1}{2} t^2 + \frac{1}{3} t^3 + \frac{1}{4} t^4 + \cdots 
\end{equation}
$$
将两式相加，我们有：
$$
\begin{equation}
\frac{1}{2} \ln \frac{1 + t}{1 - t} = t + \frac{1}{3} t^3 + \frac{1}{5} t^5 + \cdots
\end{equation}
$$

回到我们的问题，我们令 $t = (2p + 1)^{-1} \in (0, 1)$，如此才满足 $\frac{1+t}{1-t} = \frac{p+1}{p}$，带入前式：
$$
\begin{equation}
\varepsilon_p = \frac{1}{3(2p+1)^2} + \frac{1}{5(2p+1)^4} + \frac{1}{7(2p+1)^6} + \cdots
\end{equation}
$$
因此：
$$
\begin{equation}
\varepsilon_p \lt \frac{1}{3(2p+1)^2} \sum_{i=0}^{\infty} \frac{1}{(2p+1)^{2i}} 
= \frac{1}{3(2p+1)^2} \frac{1}{1 - \frac{1}{(2p+1)^2}} 
= \frac{1}{3[(2p+1)^2 - 1]} 
= \frac{1}{12} (\frac{1}{p} - \frac{1}{p+1})
\end{equation}
$$
且
$$
\begin{equation}
\varepsilon_p \gt \frac{1}{3(2p+1)^2} \sum_{i=0}^{\infty} \frac{1}{[3(2p+1)^2]^{i}} 
= \frac{1}{3(2p+1)^2} \frac{1}{1 - \frac{1}{3(2p+1)^2}} 
= \frac{1}{3(2p+1)^2 - 1}
\end{equation}
$$
易证
$$
\begin{equation}
(p+\frac{1}{12})(p+1+\frac{1}{12})
= p^2 + \frac{7}{6}p + \frac{13}{144}
\gt p^2 + p + \frac{1}{6}
= \frac{1}{12} [3(2p+1)^2 - 1], \quad p \in \mathbb{N}^+
\end{equation}
$$
因此：
$$
\begin{equation}
\varepsilon_p \gt \frac{1}{12} (\frac{1}{p+\frac{1}{12}} - \frac{1}{p+1+\frac{1}{12}})
\end{equation}
$$
我们令：
$$
\begin{equation}
B = \sum_{p=1}^{\infty} \varepsilon_p, \quad r_n = \sum_{p=n}^{\infty} \varepsilon_p
\end{equation}
$$
那么易得：
$$
\begin{equation}
\frac{1}{13} \lt B \lt \frac{1}{12}, \quad \frac{1}{12(n+1)} \lt r_n \lt \frac{1}{12n}
\end{equation}
$$
带入 $S_n$ 的表达式：
$$
\begin{equation}
S_n = (n+\frac{1}{2})\ln n - n + 1 - B + r_n
\end{equation}
$$
可得：
$$
\begin{equation}
n! = e^{1-B} n^{n+1/2} e^{-n} e^{r_n}
\end{equation}
$$
令 $C = e^{1-B}$，我们可知常数 $C$ 的取值范围为 $(e^{11/12}, e^{12/13})$，此处我们取 $C = \sqrt{2\pi}$，该公式得证。$\square$



## 1.27 散度定理

散度定理（Divergence Theorem），也称为高斯定理（Gauss's Theorem），是向量分析中的重要定理，它将体积积分和曲面积分联系起来。

具体而言，如果考虑一个 $n$-维球体（$n$-ball）$B^n$ 的体积为 $V$，其表面为 $S^{n-1}$，对于一个位于 $n$-维空间中的光滑向量场 $\mathbf{F}$，则有：

$$
\int_{B^n} (\nabla \cdot \mathbf{F}) \, dV = \oint_{S^{n-1}} \mathbf{F} \cdot \mathbf{n} \, dS
$$

其中：
- $\nabla \cdot \mathbf{F}$ 是向量场 $\mathbf{F}$ 的散度。
- $dV$ 是体积元素。
- $dS$ 是边界表面的面积元素。
- $\mathbf{n}$ 是边界的单位外法向量。

体积积分计算的是在 $n$-球内的散度，而表面积分计算的是在 $n-1$ 维球面上的通量。
这种形式的散度定理在物理学和工程学中广泛应用，比如电磁学中的高斯定理、流体力学中的质量守恒等。



## 1.28 分离超平面定理

如果有两个不相交的非空凸集，则存在一个超平面能够将它们完全分隔开，这个超平面叫做分离超平面（Separating Hyperplane）。形式上，设 $A$ 和 $B$ 是 $\mathbb{R}^n$ 中的两个不相交的非空凸集，那么存在一个非零向量 $v$ 和一个实数 $c$，使得：
$$
\begin{equation}\langle x, v \rangle \geq c \, \text{且} \, \langle y, v \rangle \leq c\end{equation}
$$
对所有 $x \in A$ 和 $y \in B$ 都成立。即超平面 $\langle \cdot, v \rangle = c$ 以 $v$ 作为分离轴（Separating Axis），将 $A$ 和 $B$ 分开。

进一步，如果这两个集合都是闭集，并且至少其中一个是紧致的，那么这种分离可以是严格的，即存在 $c_1 \gt c_2$ 使得：
$$
\begin{equation}\langle x, v \rangle \gt c_1 \, \text{且} \, \langle y, v \rangle \lt c_2\end{equation}
$$

在不同情况下，我们可以通过调整 $v$ 和 $c$ 来使得分离超平面的边界更加清晰。

| A             | B            | $\langle x, v \rangle$    | $\langle y, v \rangle$    |
|---------------|--------------|---------------------------|---------------------------|
| 闭紧集        | 闭集         | $\gt c_1$                   | $\lt c_2$ 且 $c_2 \lt c_1$    |
| 闭集          | 闭紧集       | $\gt c_1$                   | $\lt c_2$ 且 $c_2 \lt c_1$    |
| 开集          | 闭集         | $\gt c$                     | $\leq c$                  |
| 开集          | 开集         | $\gt c$                     | $\lt c$                     |

在支持向量机的背景下，最佳分离超平面（或最大边缘超平面）是分离两个点凸包并且与两者等距的超平面。

### 证明

证明基于以下引理：

设 $A$ 和 $B$ 是 $\mathbb{R}^n$ 中两个不相交的闭集，且假设 $A$ 是紧致的。则存在点 $a_0 \in A$ 和 $b_0 \in B$ 使得 $\|a - b\|$ 在 $a \in A$ 和 $b \in B$ 之间取最小值。

我们给出引理的证明：

令 $a \in A$ 和 $b \in B$ 是任意一对点，并令 $r_1 = \|b - a\|$。由于 $A$ 是紧致的，它被包含在以 $a$ 为中心的一些球中，设该球的半径为 $r_2$。令 $S = B \cap \overline{B_{r_1 + r_2}(a)}$ 为 $B$ 与以 $a$ 为中心、半径为 $r_1 + r_2$ 的闭球的交集。那么 $S$ 是紧致且非空的，因为它包含 $b$。由于距离函数是连续的，存在点 $a_0$ 和 $b_0$ 使得 $\|a_0 - b_0\|$ 在所有 $A \times S$ 的点对中取最小值。现在要证明 $a_0$ 和 $b_0$ 实际上在所有 $A \times B$ 的点对中具有最小距离。假设存在点 $a'$ 和 $b'$ 使得 $\|a' - b'\| \lt \|a_0 - b_0\|$。则特别地，$\|a' - b'\| \lt r_1$，并且根据三角不等式，$\|a - b'\| \leq \|a - a'\| + \|a' - b'\| \lt r_1 + r_2$。因此 $b'$ 包含在 $S$ 中，这与 $a_0$ 和 $b_0$ 在 $A \times S$ 中的最小距离相矛盾。

<div style="text-align: center;">
  <img src="images/separating_hyperplane_theorem.png" alt="separating_hyperplane_theorem" width="400" height="300"/>
</div>

不失一般性地，假设 $A$ 是紧致的。根据引理，存在点 $a_0 \in A$ 和 $b_0 \in B$ 使得它们之间的距离最小。由于 $A$ 和 $B$ 是不相交的，我们有 $a_0 \neq b_0$。现在，构造两条与线段 $[a_0, b_0]$ 垂直的超平面 $L_A, L_B$，其中 $L_A$ 穿过 $a_0$，$L_B$ 穿过 $b_0$。我们声称 $A$ 和 $B$ 都没有进入 $L_A, L_B$ 之间的空间，因此与 $(a_0, b_0)$ 垂直的超平面满足定理的要求。

代数上，超平面 $L_A, L_B$ 由向量 $v:= b_0 - a_0$ 定义，并由两个常数 $c_A := \langle v, a_0\rangle \lt c_B := \langle v, b_0\rangle$ 确定，使得 $L_A = \{x: \langle v, x\rangle = c_A\}, L_B = \{x: \langle v, x\rangle = c_B\}$。我们的主张是 $\forall a\in A, \langle v, a\rangle \leq c_A$ 并且 $\forall b\in B, \langle v, b\rangle \geq c_B$。

假设存在某个 $a\in A$ 使得 $\langle v, a\rangle \gt c_A$，则令 $a'$ 为从 $b_0$ 到线段 $[a_0, a]$ 的垂足。由于 $A$ 是凸集，$a'$ 在 $A$ 内部，并且根据平面几何，$a'$ 比 $a_0$ 更接近 $b_0$，这与 $a_0$ 和 $b_0$ 的最小距离相矛盾。类似的论证适用于 $B$。$\square$



## 1.29 支撑超平面定理

对于一个凸集，支撑超平面（Supporting Hyperplane）是与凸集边界切线的超平面，即它“支撑”了凸集，使得所有的凸集内的点都位于支撑超平面的一侧。形式上，若 $S$ 是非空凸集，且 $x_0$ 是 $S$ 的边界上的一点，那么存在一个包含 $x_0$ 的支撑超平面。
如果 $x^* \in X^* \backslash \{0\}$（$X^*$ 是 $X$ 的对偶空间，$x^*$ 是一个非零的线性泛函），并且对于所有 $x \in S$ 都有 $x^*(x_0) \geq x^*(x)$，那么 $H = \{x \in X: x^*(x) = x^*(x_0)\}$ 定义了一个支撑超平面。

### 证明

定义 $T$ 为所有支撑闭合半空间的交集，显然 $S \subset T$。现在令 $y \not \in S$，证明 $y \not \in T$。 

设 $x \in \mathrm{int}(S)$，并考虑线段 $[x, y]$。令 $t$ 为最大的数，使得 $[x, t(y-x) + x]$ 被包含在 $S$ 中。则 $t \in (0, 1)$。令 $b = t(y-x) + x$，那么 $b \in \partial S$。在 $b$ 处画一条支撑超平面，令其表示为一个非零线性泛函 $f: \mathbb{R}^n \to \mathbb{R}$，使得 $\forall a \in T, f(a) \geq f(b)$。由于 $x \in \mathrm{int}(S)$，我们有 $f(x) \gt f(b)$。因此，由 $\frac{f(y) - f(b)}{1-t} = \frac{f(b) - f(x)}{t - 0} \lt 0$，我们得到 $f(y) \lt f(b)$，所以 $y \not \in T$。$\square$
