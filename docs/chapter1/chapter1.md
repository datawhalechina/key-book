# 第1章：预备知识

*Edit: J. Hu*

*Update: 08/06/2020*

---

强大数定律表明：在样本数量足够多时，样本均值以概率1收敛于总体的期望值。集中不等式主要量化地研究随机变量与其期望的偏离程度，在机器学习理论中常用于考察经验误差与泛化误差的偏离程度，由此刻画学习模型对新数据的处理能力。集中不等式是学习理论的基本分析工具。本节将列出学习理论研究中常用的集中不等式及其简要证明。

记：X 和 Y 表示随机变量，$\mathbb{E}[X]$ 表示 X 的数学期望，$\mathbb{V}[X]$ 表示 X 的方差。



## Theorem 1:  Jensen 不等式

对手任意凸函数 $f,$ 则有:
$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
$$
成立。



$Proof. $

记 $p(x)$ 为 $X$ 的概率密度函数。由 Taylor 公式及 $f$ 的凸性，$\exists \xi$ s.t.
$$
\begin{aligned}
f(x) &=f(\mathbb{E}[X])+f^{\prime}(\mathbb{E}[X])(x-\mathbb{E}[X])+\frac{f^{\prime \prime}(\xi)}{2}(x-\mathbb{E}[X])^{2} \\
& \geq f(\mathbb{E}[X])+f^{\prime}(\mathbb{E}[X])(x-\mathbb{E}[X])
\end{aligned}
$$
对上式取期望：
$$
\begin{aligned}
\mathbb{E}[f(X)]=\int p(x) f(x) d x & \geq f(\mathbb{E}[X]) \int p(x) d x+f^{\prime}(\mathbb{E}[X]) \int p(x)(x-\mathbb{E}[X]) d x \\
&=f(\mathbb{E}[X])
\end{aligned}
$$
原不等式得证。



## Theorem 2:  Holder 不等式

$\forall p, q \in \mathbb{R}^{+}, \frac{1}{p}+\frac{1}{q}=1$，则有：
$$
\mathbb{E}[|X Y|] \leq\left(\mathbb{E}\left[|X|^{p}\right]\right)^{\frac{1}{p}}\left(\mathbb{E}\left[|Y|^{q}\right]\right)^{\frac{1}{q}}
$$
成立。



$Proof. $

记 $f(x), g(x)$ 分别为 $X,Y$ 的概率密度函数，
$$
M=\left(\int|x|^{p} f(x) d x\right)^{\frac{1}{p}}, N=\left(\int|x|^{q} g(x) d x\right)^{\frac{1}{q}}
$$
由杨氏不等式:
$$
\frac{|X|}{M} \frac{|Y|}{N} \leq \frac{1}{p}\left(\frac{|X|}{M}\right)^{p}+\frac{1}{q}\left(\frac{|Y|}{N}\right)^{q}
$$
对上式取期望:
$$
\mathbb{E}\left[\frac{|X|}{M} \frac{|Y|}{N}\right] \leq \frac{1}{p M^{p}} \mathbb{E}\left[|X|^{p}\right]+\frac{1}{q N^{q}} \mathbb{E}\left[|Y|^{q}\right]=1
$$
原不等式得证。



## Theorem 3: Cauchy-Schwarz 不等式

特别的，$p = q = 2 $ 时，Holder不等式退化为 Cauchy-Schwarz 不等式：
$$
\mathbb{E}[|X Y|] \leq \sqrt{\mathbb{E}\left[X^{2}\right] \mathbb{E}\left[Y^{2}\right]}
$$



## Theorem 4: Lyapunov 不等式

$\forall 0<r \leq s$，有：
$$
\sqrt[r]{\mathbb{E}\left[|X|^{r}\right]} \leq \sqrt[r]{\mathbb{E}\left[|X|^{s}\right]}
$$

$Proof.$ 
由Holder不等式：
$\forall p \geq 1:$
$$
\begin{aligned}
\mathbb{E}\left[|X|^{r}\right] &=\mathbb{E}\left[|X \times 1|^{r}\right] \\
& {\leq}\left(\mathbb{E}\left[\left(|X|^{r}\right)^{p}\right]\right)^{1 / p} \times 1 \\
&=\left(\mathbb{E}\left[|X|^{r p}\right]\right)^{1 / p}
\end{aligned}
$$
记 $s=r p \geq r,$ 则 :
$$
\mathbb{E}\left[|X|^{r}\right] \leq\left(\mathbb{E}\left[|X|^{s}\right]\right)^{r / s}
$$
原不等式得证。



## Theorem 5: Minkowski 不等式

$\forall p \geq 1,$ 有：
$$
\sqrt[p]{\mathbb{E}\left[|X+Y|^{p}\right]} \leq \sqrt[p]{\mathbb{E}\left[|X|^{p}\right]}+\sqrt[p]{\mathbb{E}\left[|Y|^{p}\right]}
$$


$Proof.$
由三角不等式及Holder不等式：
$$
\begin{aligned}
\mathbb{E}\left[|X+Y|^{p}\right] & {\leq}\mathbb{E}\left[(|X|+|Y|)|X+Y|^{p-1}\right] \\
&=\mathbb{E}\left[|X||X+Y|^{p-1}\right]+\mathbb{E}\left[|Y||X+Y|^{p-1}\right] \\
& {\leq}\left(\mathbb{E}\left[|X|^{p}\right]\right)^{1 / p}\left(\mathbb{E}\left[|X+Y|^{(p-1) q}\right]\right)^{1 / q}+\left(\mathbb{E}\left[|Y|^{p}\right]\right)^{1 / p}\left(\mathbb{E}\left[|X+Y|^{(p-1) q}\right]\right)^{1 / q} \\
& = \left[\left(\mathbb{E}\left[|X|^{p}\right]\right)^{1 / p}+\left(\mathbb{E}\left[|Y|^{p}\right]\right)^{1 / p}\right] \frac{\mathbb{E}\left[|X+Y|^{p}\right]}{\left(\mathbb{E}\left[|X+Y|^{p}\right]\right)^{1 / p}}
\end{aligned}
$$
化简上式即得证。



## Theorem 6: Markov 不等式

若 $X \geq 0, \forall \varepsilon>0,$ 有：
$$
P(X \geq \varepsilon) \leq \frac{\mathbb{E}[X]}{\varepsilon}
$$



$Proof.$
$$
\mathbb{E}[X]=\int_{0}^{\infty} x p(x) d x \geq \int_{\varepsilon}^{\infty} x p(x) d x \geq \int_{\varepsilon}^{\infty} \varepsilon p(x) d x=\varepsilon P(X \geq \varepsilon)
$$



## Theorem 7: Chebyshev 不等式

$\forall \varepsilon>0,$ 有：
$$
P(|X-\mathbb{E}[X]| \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\varepsilon^{2}}
$$


$Proof.$

取 $Y=(X-\mathbb{E}[X])^{2}$，则可以使用Markov 不等式证明。



## Theorem 8: Cantelli 不等式

$\forall \varepsilon>0,$ 有 :
$$
P(X-\mathbb{E}[X] \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\mathbb{V}[X]+\varepsilon^{2}}
$$


$Proof.$

记 $Y=X-\mathbb{E}[X],$ 则对 $\forall \lambda \geq 0$ 有 :
$$
\begin{aligned}
P(X-\mathbb{E}[X] \geq \varepsilon) &=P(Y+\lambda \geq \varepsilon+\lambda) \\
&=P\left((Y+\lambda)^{2} \geq(\varepsilon+\lambda)^{2}\right) \\
& \quad \leq \frac{\mathbb{E}\left[(Y+\lambda)^{2}\right]}{(\varepsilon+\lambda)^{2}}=\frac{\mathbb{V}[X]+\lambda^{2}}{(\varepsilon+\lambda)^{2}}
\end{aligned}
$$
通过求导可知，上式右端在 $\lambda=\frac{\mathrm{V}[X]}{\varepsilon}$ 时取得最小值 $\frac{\mathrm{V}[X]}{\mathrm{V}[X]+\varepsilon^{2}},$ 于是：
$$
P(X-\mathbb{E}[X] \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\mathbb{V}[X]+\varepsilon^{2}}
$$
原不等式得证。

Note: Cantelli 不等式是 Chebyshev 不等式的加强，也称单边 Chebyshev 不等式。通过类似的构造，可以求得诸多比 Cantelli 不等式更严格的界。



## Theorem 9: Chernoff 不等式

$\forall \lambda>0, \varepsilon>0,$ 有 :
$$
P(X \geq \varepsilon)=P\left(e^{\lambda X} \geq e^{\lambda \varepsilon}\right) \leq \frac{\mathbb{E}\left[e^{\lambda X}\right]}{e^{\lambda \varepsilon}}
$$
$\forall \lambda<0, \varepsilon>0,$ 有 :
$$
P(X \leq \varepsilon)=P\left(e^{\lambda X} \geq e^{\lambda \varepsilon}\right) \leq \frac{\mathbb{E}\left[e^{\lambda X}\right]}{e^{\lambda \varepsilon}}
$$


$Proof. $

取 $Y=e^{\lambda X},$ 应用 Markov 不等式即得证。



## Theorem 10: Hoeffding 不等式

### 引理1 (Hoeffding 引理)
若$\mathbb{E}[X] = 0, X\in[a,b]$，则$\forall \lambda \in \mathbb{R}$有：
$$
\mathbb{E}[e^{\lambda X}] \leq \exp\left( \frac{\lambda^2(b-a)^2}{8} \right)
$$



$Proof. $
由于$e^x$为凸函数，则显然$\forall x\in[a,b]$：
$$
	e^{\lambda x} \leq \frac{b-x}{b-a}e^{\lambda a} + \frac{x-a}{b-a}e^{\lambda b}
$$
对上式取期望有：
$$
\mathbb{E}[e^{\lambda X}] \leq \frac{b-\mathbb{E}[X]}{b-a}e^{\lambda a} + \frac{\mathbb{E}[X]-a}{b-a}e^{\lambda b} = \frac{be^{\lambda a} - ae^{\lambda b}}{b - a}
$$

记$\theta = -\frac{a}{b-a} > 0, h = \lambda(b-a)$，则：
$$
\frac{be^{\lambda a} - ae^{\lambda b}}{b - a} = [1-\theta + \theta e^{h}]e^{-\theta h} = 
e^{\ln(1-\theta + \theta e^{h})}e^{-\theta h} = e^{\ln(1-\theta + \theta e^{h}) -\theta h}
$$

记函数$\varphi(\theta, h) = \ln(1-\theta + \theta e^{h}) -\theta h$，注意到实际上$a$也是变量，因而$\theta$ 与$h$无关。考察关于$h$的偏导数：
$$
\frac{\partial \varphi}{\partial h}  = 
\frac{\theta e^h}{1 - \theta + \theta e^h} - \theta
$$
显然有：$\frac{\partial \varphi}{\partial h}|_{h=0^+} = 0$。同理使用链式法则可计算：
$$
\frac{\partial^2 \varphi}{\partial h^2} = 
\frac{\theta e^h(1 - \theta + \theta e^h) - \theta^2e^{2h}}{(1 - \theta + \theta e^h)^2} = 
\frac{\theta e^h}{1 - \theta + \theta e^h}(1- \frac{\theta e^h}{1 - \theta + \theta e^h}) \leq \frac{1}{4}
$$
由泰勒公式可得：
$$
\varphi(\theta, h) \leq 0 + 0 + \frac{h^2}{8} = \frac{\lambda^2(b-a)^2}{8}
$$
原不等式得证。



### Hoeffding 不等式

对 $m$ 个独立随机变量 $X_{i} \in\left[a_{i}, b_{i}\right],$ 令 $\bar{X}$ 为 $X_{i}$ 均值，则有：
$$
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp \left(-\frac{2 m^{2} \varepsilon^{2}}{\sum_{i=1}^{m}\left(b_{i}-a_{i}\right)^{2}}\right)
$$


$Proof. $

由 Markov 不等式知， $\forall \lambda>0$ :
$$
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon)=P\left(e^{\lambda(\bar{X}-\mathbb{E}[\bar{X}])} \geq e^{\lambda \varepsilon}\right) \leq \frac{\mathbb{E}\left[e^{\lambda(\bar{X}-\mathbb{E}[\bar{X}])}\right]}{e^{\lambda \varepsilon}}
$$
由独立性及 Hoeffding 引理：
$$
\frac{\mathbb{E}\left[e^{\lambda(\bar{X}-\mathbb{E}[\bar{X}])}\right]}{e^{\lambda \varepsilon}}=e^{-\lambda \varepsilon} \prod_{i=1}^{m} \mathbb{E}\left[e^{\lambda\left(X_{i}-\mathbb{E}\left[X_{i}\right]\right) / m}\right] \leq e^{-\lambda \varepsilon} \prod_{i=1}^{m} \exp \left(\frac{\lambda^{2}\left(b_{i}-a_{i}\right)^{2}}{8 m^{2}}\right)
$$
考察二次函数 $g(\lambda)=-\lambda \varepsilon+\frac{\lambda^{2}}{8 m^{2}} \sum_{i=1}^{m}\left(b_{i}-a_{i}\right)^{2},$ 容易可求得最小值 $-\frac{2 m^{2} \varepsilon^{2}}{\sum_{i=1}^{m}\left(b_{i}-a_{i}\right)^{2}}$
于是：
$$
P((\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon)) \leq \exp (g(\lambda)) \leq \exp \left(-\frac{2 m^{2} \varepsilon^{2}}{\sum_{i=1}^{m}\left(b_{i}-a_{i}\right)^{2}}\right)
$$
定理得证。

Note：注意这里没有限定随机变量同分布，下同。可以使用 Hoeffding 不等式解释集成学习的原理。



### 随机变量的次高斯性

上述使用 Markov 不等式的技术称为 Chernoff 界的一般技巧，得到的界称之为 Chernoff Bound。其核心即是对其矩母函数进行控制。于是有定义：

**Definition 1** (随机变量的次高斯性). 若一个期望为零的随机变量$X$其矩母函数满足，$\forall \lambda \in \mathbb{R}^+$:
$$
		\mathbb{E}[e^{\lambda X}] \leq \frac{\sigma^2\lambda^2}{2}
$$
则称$X$服从参数为$\sigma$的次高斯分布。

实际上 Hoeffding 引理中的随机变量$X$服从$\frac{(b-a)}{2}$的次高斯分布， Hoeffding 引理也是次高斯分布的直接体现。次高斯性还有一系列等价定义方式，这里不是本笔记讨论的重点。

次高斯分布有一个直接的性质：假设两个独立的随机变量$X_1,X_2$都是次高斯分布的，分别服从参数$\sigma_1,\sigma_2$，那么$X_1+X_2$就是服从参数为$\sqrt{\sigma_1 + \sigma_2}$的次高斯分布。这个结果的证明直接利用定义即可。



### 随机变量的次指数性

显然，不是所有常见的随机变量都是次高斯的，例如指数分布。为此可以扩大定义：
**Definition 2** (随机变量的次指数性). 若非负的随机变量$X$其矩母函数满足，$\forall \lambda \in (0,a)$:
$$
		\mathbb{E}[e^{\lambda X}] \leq \frac{a}{a - \lambda}
$$
则称$X$服从参数为$(\mathbb{V}[X], 1/a)$的次指数分布。

同样的，次高斯性还有一系列等价定义方式。一种不直观但是更常用的定义方式如下：$\exists (\sigma^2, b)$，s.t.$\forall |s| < 1/b$有:
$$
		\mathbb{E}[e^{s(X−\mathbb{E}[X]})]\leq \exp \left( \frac{s^2\sigma^2}{2} \right)
$$

常见的次指数分布包括:指数分布，Gamma 分布，以及**任何的有界随机变量**。
类似地，次指数分布也对加法是保持的：如果$X_1,X_2$分别是服从$(\sigma_1^2,b_1)$, $(\sigma_2^2,b_2)$的次指数分布，那么$X_1+X_2$是服从$(\sigma_1^2+\sigma_2^2, \max(b_1,b_2))$的次指数分布。

在高维统计的问题中需要利用次高斯分布，次指数分布的尾端控制得到一些重要的结论。



## Theorem 11: Bernstein 不等式

对 $m$ 个独立随机变量 $X_{i},$ 令 $\bar{X}$ 为 $X_{i}$ 均值，若 $\exists b>0,$ s.t. $\forall k \geq 2$ 有 Bound 矩约束 $(\text {Bernstein Condition}):$
$$
\mathbb{E}\left[\left|X_{i}-\mathbb{E}\left[X_{i}\right]\right|^{k}\right] \leq k ! b^{k-2} \frac{\mathbb{V}\left[X_{i}\right]}{2}
$$
则有：
$$
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp \left(-\frac{m \varepsilon^{2}}{2\left(\sum_{i=1}^{m} \mathbb{V}\left[X_{i}\right] / m+b \varepsilon\right)}\right)
$$
成立。



$Proof. $

考察有界随机变量的生成函数，如果不利用指数函数的凸性，而是直接 Taylor 展开，那么对 $\forall \lambda \in[0,1 / b)$ 有：
$$
\begin{aligned}
\mathbb{E}\left[e^{\lambda(X-\mathbb{E}[X])}\right] &=1+\lambda \mathbb{E}[X-\mathbb{E}[X]]+\frac{\lambda^{2}}{2} \mathbb{E}[X-\mathbb{E}[X]]^{2}+\sum_{k=3}^{\infty} \frac{\lambda^{k} \mathbb{E}[X-\mathbb{E}[X]]^{k}}{k !} \\
&=1+\frac{\lambda^{2}}{2} \mathbb{V}[X]+\sum_{k=3}^{\infty} \frac{\lambda^{k} \mathbb{E}[X-\mathbb{E}[X]]^{k}}{k !}
\end{aligned}
$$
由 Bound 矩约束：
$$
\begin{aligned}
\mathbb{E}\left[e^{\lambda(X-\mathbb{E}[X])}\right] & \leq 1+\frac{\lambda^{2}}{2} \mathbb{V}[X]+\sum_{k=3}^{\infty} \frac{\lambda^{k} k ! b^{k-2} \frac{\mathrm{V}[X]}{2}}{k !} \\
&=1+\frac{\lambda^{2}}{2} \mathbb{V}[X]+\frac{\lambda^{2}}{2} \mathbb{V}[X] \sum_{k=3}^{\infty} \lambda^{k-2} b^{k-2} \\
& \leq 1+\frac{\lambda^{2}}{2} \mathbb{V}[X] \frac{1}{1-\lambda b} \\
& \leq \exp \left(\frac{\lambda^{2} \mathbb{V}[X]}{2(1-\lambda b)}\right)
\end{aligned}
$$
则利用这个结果和 Chernoff 界的一般技巧有:
$$
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon) \leq e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda(\bar{X}-\mathbb{E}[\bar{X}])}\right] \leq \exp \left(-\lambda \varepsilon+\sum_{i=1}^{m} \frac{(\lambda / m)^{2} \mathbb{V}\left[X_{i}\right]}{2(1-\lambda b / m)}\right)
$$
取 $\lambda=\frac{\varepsilon}{b \varepsilon+\sum_{i=1}^{m} \mathbb{V}\left[X_{i}\right] / m} \in[0,1 / b)$ ，原不等式得证。

Note：首先，上式右端项小于 $e^{\lambda^{2} \mathrm{V}[X]},$ 这表明这一类随机变量是服从参数为 $(\sqrt{2 \mathbb{V}[X]}, 2 b)$ 的次指数分布。
其次，$|X-\mathbb{E}[X]|<b$ 显然满足 $\mathbb{E}\left[\left|X_{i}-\mathbb{E}\left[X_{i}\right]\right|^{k}\right] \leq k ! b^{k-2} \frac{\mathbb{V}\left[X_{i}\right]}{2}$  那么：
$$
\begin{aligned}
\mathbb{E}\left[e^{\lambda(X-\mathbb{E}[X])}\right] &=1+\frac{\lambda^{2}}{2} \mathbb{V}[X]+\sum_{k=3}^{\infty} \frac{\lambda^{k} \mathbb{E}[X-\mathbb{E}[X]]^{k}}{k !} \\
& \leq 1+\frac{\lambda^{2}}{2} \mathbb{V}[X]+\frac{\lambda^{2}}{2} \mathbb{V}[X] \sum_{k=3}^{\infty} \frac{2 \lambda^{k-2} b^{k-2}}{k !} \\
& \leq 1+\frac{\lambda^{2}}{2} \mathbb{V}[X]+\frac{\lambda^{2}}{2} \mathbb{V}[X] \sum_{k=3}^{\infty} \frac{\lambda^{k-2} b^{k-2}}{3} \\
& \leq 1+\frac{\lambda^{2}}{2} \mathbb{V}[X]\left(1+\frac{\lambda b}{3(1-\lambda b)}\right)
\end{aligned}
$$
则可得 Bernstein 不等式的另一弱化表述，即Bennett 不等式。
最后，Bernstein 不等式可以应用在PAC理论中样本复杂度的下界估计。



## Theorem 12: Bennett 不等式

对 $m$ 个独立随机变量 $X_{i},$ 令 $\bar{X}$ 为 $X_{i}$ 均值, 若 $\exists b>0,$ s.t.$|X-\mathbb{E}[X]|<b$

则有，
$$
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp \left(-\frac{m \varepsilon^{2}}{2\left(\sum_{i=1}^{m} \mathbb{V}\left[X_{i}\right] / m+b \varepsilon / 3\right)}\right)
$$
成立。

Remark: Bernstein 不等式实际是 Hoeffding 不等式的加强。对于个各随机变量独立的条件可以放宽为弱独立结论仍成立。

上述几个 Bernstein 类集中不等式，更多的是在非渐近观点下看到的大数定律的表现。也即是，这些不等式更多刻画了样本均值如何集中在总体均值的附近。

如果把样本均值看成是样本（数据点的函数），即令 $f\left(X_{1}, \cdots, X_{m}\right)=$ $\sum_{i=1}^{m} X_{i} / m,$ 那么 Bernstein 类不等式刻画了如下的概率：
$$
P\left(f\left(X_{1}, \cdots, X_{m}\right)-\mathbb{E}\left[f\left(X_{1}, \cdots, X_{m}\right)\right] \geq \varepsilon\right)
$$
为考察在某个泛函上也具有类似 Bernstein 类集中不等式的形式，很显然 f 需要满足一些很好的性质。这类性质有很多，但是我们尝试在一个最常见的约束下进行尝试:

**Definition 3** (差有界). 函数 $f: \mathcal{X}^{m} \rightarrow \mathbb{R}, \forall i, \exists c_{i}<\infty,$ s.t.
$$
\left|f\left(x_{1}, \cdots, x_{i}, \cdots, x_{m}\right)-f\left(x_{1}, \cdots, x_{i}^{\prime}, \cdots, x_{m}\right)\right| \leq c_{i}
$$
则称 f 是差有界的。

为此，需要引入一些新的数学工具。

**Definition 4** (离散鞅). 若离散随机变量序列(随机过程)$Z_m$满足:

1. $\mathbb{E}\left[\left|Z_{i}\right|\right]<\infty$
2. $\mathbb{E}\left[Z_{m+1} \mid Z_{1}, \cdots, Z_{m}\right]=\mathbb{E}\left[Z_{m+1} \mid \mathcal{F}_{m}\right]=Z_{m}$

则称序列 $Z_i$为离散鞅。

**Lemma 2** (Azuma-Hoeffding 引理). 对于鞅 $Z_{i}, \mathbb{E}\left[Z_{i}\right]=\mu, Z_{1}=\mu_{\circ}$ 作鞅差序列 $X_{i}=Z_{i}-Z_{i-1}, \quad$ 且 $\left|X_{i}\right| \leq c_{i}$ 。 则 $\forall \varepsilon>0$ 有：
$$
P\left(Z_{m}-\mu \geq \varepsilon\right)=P\left(\sum_{i=1}^{m} X_{i} \geq \varepsilon\right) \leq \exp \left(-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}}\right)
$$


$Proof. $

首先，若 $\mathbb{E}[X \mid Y]=0,$ 则有 $\forall \lambda>0:$
$$
\mathbb{E}\left[e^{\lambda X} \mid Y\right] \leq \mathbb{E}\left[e^{\lambda X}\right]
$$
于是，由恒等式$\mathbb{E}[\mathbb{E}[X \mid Y]]=\mathbb{E}[X] $及 Chernoff 一般性技巧 $\forall \lambda>0:$
$$
\begin{aligned}
P\left(Z_{m}-\mu>\varepsilon\right) &<e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda\left(Z_{m}-\mu\right)}\right] \\
& = e^{-\lambda \varepsilon} \mathbb{E}\left[\mathbb{E}\left[e^{\lambda\left(Z_{m}-\mu\right)} \mid \mathcal{F}_{m-1}\right]\right] \\
& \leq e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda\left(Z_{m-1}-\mu\right)}\right] \mathbb{E}\left[e^{\lambda X_{m}} \mid \mathcal{F}_{m-1}\right]
\end{aligned}
$$


又因为 $ X_{i}$ 为鞅差序列，则 $ \mathbb{E}\left[X_{m} \mid \mathcal{F}_{m-1}\right]=0, \mathbb{E}\left[X_{i}\right]=0$ ，再结合不等式$\mathbb{E}\left[e^{\lambda X} \mid Y\right] \leq \mathbb{E}\left[e^{\lambda X}\right]$及 Hoeffding 引理，有：
$$
\begin{aligned}
P\left(Z_{m}-\mu \geq \varepsilon\right) & \leq e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda\left(Z_{m-1}-\mu\right)}\right] \mathbb{E}\left[e^{\lambda X_{n}}\right] \\
& {\leq} e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda\left(Z_{m-1}-\mu\right)}\right] \exp \left(\frac{\lambda^{2} c_{m}^{2}}{2}\right)
\end{aligned}
$$
迭代上不等式可得:
$$
P\left(Z_{m}-\mu \geq \varepsilon\right) \leq e^{-\lambda \varepsilon} \prod_{i=1}^{m} \exp \left(\frac{\lambda^{2} c_{i}^{2}}{2}\right)
$$
则显然当 $\lambda=\frac{\varepsilon}{\sum_{i=1}^{m} c_{i}^{2}}$ 时，上式右端取得极小值：
$$
P\left(Z_{m}-\mu \geq \varepsilon\right) \leq \exp \left(-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}}\right)
$$
原不等式得证。



## Theorem 13: McDiarmid 不等式

对 $m$ 个独立随机变量 $X_{i} \in \mathcal{X},$ 函数 $f$ 为 差有界的，则 $\forall \varepsilon>0$ 有：
$$
P\left(f\left(X_{1}, \cdots, X_{m}\right)-\mathbb{E}\left[f\left(X_{1}, \cdots, X_{m}\right)\right] \geq \varepsilon\right) \leq \exp \left(-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}}\right)
$$


$Proof. $

构造一个鞅差序列：
$$
D_j = \mathbb{E}[f(X)|X_1,\cdots,X_j] - \mathbb{E}[f(X)|X_1,\cdots,X_{j-1}]
$$
容易验证：
$$
f(X) - \mathbb{E}[f(X)]=\sum_{i=1}^mD_i
$$
且 $f$ 为差有界的，则满足 Azuma-Hoeffding 引理，代入则得到：
$$
	P(f(X_1, \cdots, X_m) - \mathbb{E}[f(X_1, \cdots, X_m)] \geq \varepsilon) \leq 
	\exp\left( -\frac{\varepsilon^2}{2\sum_{i=1}^mc_i^2} \right) 
$$
则原不等式得证。



## Theorem 14: Slud 不等式

若$X\sim B(m,p)$，则有：
$$
		P(\frac{X}{m} \geq \frac{1}{2}) \geq \frac{1}{2}\left[1 - \sqrt{1-\exp\left(-\frac{m\varepsilon^{2}}{1-\varepsilon^{2}}\right)}\right]
$$
其中$p = (1- \varepsilon)/2$。

该定理的证明使用了正态分布的标准尾边界，其所需前序知识超出了本笔记的讨论范围。详细证明可以阅读Slud在1977年发表的[论文](https://projecteuclid.org/download/pdf_1/euclid.aop/1176995801)。



## Theorem 15: Johnson-Lindenstrauss 引理

首先借用上述工具考察一个示例：
### $\chi_m^2$随机变量的集中度
若随机变量$Z\sim \chi_m^2$，则$\forall \varepsilon \in (0, 3)$有：
$$
P\left((1-\varepsilon) \leq \frac{Z}{m} \leq (1 + \varepsilon)\right) \leq \exp(-\frac{m\varepsilon^2}{6})
$$


$Proof. $
若$X\sim N(0,1)$，则显然$\forall \lambda > 0$：
$$
	\mathbb{E}[e^{-\lambda X^2}] \leq 1 - \lambda\mathbb{E}[X^2] + \frac{\lambda^2}{2}\mathbb{E}[X^4] = 1 - \lambda + \frac{3}{2}\lambda^2  \leq e^{-\lambda + \frac{3}{2}\lambda^2}
$$
类似地使用 Chernoff 一般性技巧，在$\lambda = \varepsilon/3$时可以证得左端不等式。
对于右端不等式，考察矩母函数$\forall \lambda < 1/2$：
$$
	\mathbb{E}[e^{\lambda X^2}] = (1-2\lambda)^{-m/2}
$$
再次使用 Chernoff 一般性技巧，取$\lambda = \varepsilon/6$即可得证。
Note: 实际上可以通过卡方分布的次指数性得到一个更强且更普适的界$\forall \varepsilon \in (0, 4)$：
$$
	P\left((1-\varepsilon) \leq \frac{Z}{m} \leq (1 + \varepsilon)\right) \leq \exp(-\frac{m\varepsilon^2}{8})
$$
但和上面的结论没有本质区别。
这一结果实际上是高维情况下一个反直觉但常见的现象：这告诉我们标准的n维正态分布，随着n不断变大，这些点主要都分布在一个半径是$\sqrt{n}$的高维球面附近。 这一现象直接导致了一个更加深刻的结果。



### Johnson-Lindenstrauss 引理

$\forall \varepsilon \in (0,1), n \in \mathbb{N}^+$，若正整数$k$满足：
$$
		k\geq \frac{4\ln n}{\varepsilon^2/2 - \varepsilon^3/3}
$$
那么对于任意$\mathbb{R}^d$空间中的$n$个点构成的集合$V$，始终存在一个映射$f:\mathbb{R}^d\to \mathbb{R}^k$，s.t. $\forall u,v \in V$，有：
$$
	(1−\varepsilon)\|u−v\|_2^2\leq \|f(u)−f(v)\|_2^2≤(1+\varepsilon)\|u−v\|_2^2
$$
且该映射可以在多项式时间内被找到。

该定理的证明其所需前序知识超出了本笔记的讨论范围，详细证明及映射的构造可以阅读南京大学的一个[讲义](http://tcs.nju.edu.cn/wiki/index.php/%E9%9A%8F%E6%9C%BA%E7%AE%97%E6%B3%95_(Fall_2011)/Johnson-Lindenstrauss_Theorem)。

该定理的适用性极其广泛，例如在稀疏感知领域直接导致了约束等距性条件 (RIP条件)，即非凸的 $L_0$范数最小化问题与 $L_1$范数最小化问题等价性条件；在流形学习和优化理论中也有重要的应用。而其在学习理论中最重要的应用是对降维任务的估计。 



