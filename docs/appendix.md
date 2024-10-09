# 附录：重要概念

*编辑：赵志民, 李一飞*

------

## 范数

范数（norm）是数学中用于为向量空间中的每个非零向量分配严格正长度或大小的函数。几何上，范数可理解为向量的长度或大小。例如，绝对值是实数集上的一种范数。与之相对的是半范数（seminorm），它可以将非零向量赋予零长度。

向量空间上的半范数需满足以下条件：

1. **半正定性（非负性）**：任何向量的范数总是非负的，对于任意向量 $v$，$\|v\| \geq 0$。
2. **可伸缩性（齐次性）**：对于任意标量 $a$ 和任何向量 $v$，标量乘法 $av$ 的范数等于标量的绝对值乘以向量的范数，即 $\|av\| = |a|\|v\|$。
3. **次可加性（三角不等式）**：对于任何向量 $v$ 和 $w$，向量和 $u=v+w$ 的范数小于或等于向量 $v$ 和 $w$ 的范数之和，即 $\|v+w\| \leq \|v\| + \|w\|$。

范数在具备上述半范数特性的基础上，还要求：对于任意向量 $v$，若 $\|v\|=0$，则 $v$ 必须为零向量。换句话说，所有范数都是半范数，但它们可以将非零向量与零向量区分开来。

常用的向量范数包括：

1. **$\ell_0$ 范数**：向量 $x$ 中非零元素的个数，表示为 $\|x\|_0=\sum_{i=1}^n \mathbb{I}(x_i\neq 0)$。
2. **$\ell_1$ 范数**：向量 $x$ 中各元素绝对值之和，表示为 $\|x\|_1=\sum_{i=1}^n |x_i|$。
3. **$\ell_2$ 范数（欧几里得范数）**：向量 $x$ 各元素绝对值的平方和再开平方，表示为 $\|x\|_2=\sqrt{\sum_{i=1}^n x_i^2}$。
4. **$\ell_p$ 范数**：向量 $x$ 各元素绝对值的 $p$ 次方和再开 $p$ 次方，表示为 $\|x\|_p=(\sum_{i=1}^n |x_i|^p)^{\frac{1}{p}}$。
5. **$\ell_\infty$ 范数（极大范数）**：向量 $x$ 中各元素绝对值的最大值，表示为 $\|x\|_\infty=\max_{i=1,\cdots,n} |x_i|$。
6. **加权范数**：设 $A$ 为 $n$ 阶 Hermite 正定矩阵，则向量 $x$ 的加权范数定义为 $\|x\|_A=\sqrt{x^T A x}$。此类范数在本书第 8.3.2 和 8.4.2 节中经常使用。



## 凸集合

凸集合（convex set）是向量空间（如欧几里得空间）中的一个子集，对于集合中的任意两点，连接它们的线段完全位于该集合内。换句话说，若一个集合包含了连接集合内任意两点的线段上的所有点，则该集合是凸集合。

形式化地说，考虑向量空间 $\mathcal{V}$。若对于该空间中的任意两点 $x$ 和 $y$，以及满足 $\alpha\in[0,1]$ 的任意标量 $\alpha$，点 $\alpha x+(1-\alpha)y$ 也属于 $\mathcal{D}$，那么集合 $\mathcal{D}\subseteq\mathcal{V}$ 是凸集合。

凸集合具有非扩张性（non-expansiveness），即对于集合内的任意两点，连接这两点的线段完全包含在集合内。这种性质使得凸集合在许多数学环境中易于处理，特别是在优化问题中：在凸集合中找到的最小值或最大值必为全局值，没有局部最小值或最大值，从而简化了搜索过程。

不仅凸集合具有非扩张性，映射到凸集合的投影操作也是非扩张的，即两点在凸集合上的投影之间的距离不大于两点本身之间的距离。形式上，对于闭合凸集合 $K\subseteq\mathbb{R}^D$，投影算子 $\Pi:\mathbb{R}^D\rightarrow K$ 定义为：
$$
\begin{equation}
\Pi(x)=\arg \min_{y\in K} \| x-y\|_2
\end{equation}
$$
即将一个向量映射到最接近它的凸集合中的点。投影算子 $\Pi$ 在 $\ell_2$ 范数下是非扩张的，即对于任意 $x,x'\in\mathbb{R}^D$，有：
$$
\begin{equation}
\| \Pi(x) - \Pi(x')\|_2 \leq \| x - x'\|_2
\end{equation}
$$

该性质证明如下：  
令 $y=\Pi(x)$，易知 $x$ 和 $K$ 分处于通过 $y$ 的超平面 $H=\{z\in\mathbb{R}^D:\langle z-y,x-y\rangle=0\}$ 的两侧。因此，对于 $K$ 中的任意 $u$，有以下不等式成立：
$$
\begin{equation}
\langle x-y,u-y\rangle \leq 0
\end{equation}
$$
同理，令 $y'=\Pi(x')$，对于 $K$ 中的任意 $u'$，有以下不等式成立：
$$
\begin{equation}
\langle x'-y',u'-y'\rangle \leq 0
\end{equation}
$$
此时，令 $u=y'$ 且 $u'=y$，则有：
$$
\begin{equation}
\langle x-y,y'-y\rangle \leq 0 \\
\langle x'-y',y-y'\rangle \leq 0
\end{equation}
$$
将两个不等式相加可得：
$$
\begin{equation}
\langle (x-x')+(y'-y),y'-y\rangle \leq 0
\end{equation}
$$
根据 Cauchy-Schwarz 不等式，有：
$$
\begin{equation}
\begin{align*}
&\|y-y'\|_2^2 \leq \langle x-x',y-y'\rangle \leq \|x-x'\|_2\,\|y-y'\|_2\\
\Rightarrow &\|y-y'\|_2 \leq \|x-x'\|_2 \\
\Rightarrow &\|\Pi(x) - \Pi(x')\|_2 \leq \|x-x'\|_2
\end{align*}
\end{equation}
$$

这种投影映射经常用于凸优化中，因为它能将问题简化为凸优化问题，从而提高算法效率，并在许多情况下保证全局最优解。



## Hessian 矩阵

Hessian 矩阵 $H_f$ 是由函数 $f(x)$ 的二阶偏导数组成的方阵，即：
$$
\begin{equation}
\mathbf H_f= \begin{bmatrix}
  \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]
  \vdots & \vdots & \ddots & \vdots \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.
\end{equation}
$$
其中，$x=[x_1,x_2,\cdots,x_n]$。


## 凸函数

凸函数（convex function）是定义在凸集上的实值函数，满足以下性质：对于定义域内的任意两个点 $x$ 和 $y$ 以及满足 $\alpha\in[0,1]$ 的任意标量 $\alpha$，函数图像上这两点之间的线段位于或位于函数图像上方，即：
$$
\begin{equation}
f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha) f(y)
\end{equation}
$$
该不等式被称为凸性条件。

除了上述定义，凸函数还有以下几种等价的定义方式：

1. **一阶条件**：若一个定义在凸集上的函数 $f(x)$ 满足下述条件：
$$
\begin{equation}
f(y) \geq f(x) + \nabla f(x)^T(y - x)
\end{equation}
$$
其中，$\nabla f(x)$ 表示函数 $f(x)$ 在点 $x$ 处的梯度。几何上，这意味着函数的图像位于任意一点处的切线之上。

2. **二阶条件**：若函数 $f(x)$ 是二次可微的，则它是凸函数当且仅当其 Hessian 矩阵 $H_f$ 在其定义域内的所有点 $x$ 上都是半正定的（即矩阵的所有特征值均为非负）。

3. **Jensen 不等式**：若 $f(x)$ 是凸函数，则对于定义域内的任意一组点 ${x_1, x_2, \cdots, x_n}$ 和归一化的非负权重 ${w_1, w_2, \cdots, w_n}$，即 $\sum_{i=1}^n w_i=1$，有：
$$
\begin{equation}
f(\sum_{i=1}^n w_i x_i) \leq \sum_{i=1}^n w_i f(x_i)
\end{equation}
$$

4. **上图集定义**：凸函数与凸集合的概念密切相关。函数 $f$ 是凸函数，当且仅当其上图集（epigraph）是一个凸集。上图集是位于函数图像上方的点的集合，定义为：
$$
\begin{equation}
epi(f) = \{(x, y) | x \in dom(f)，y \geq f(x)\}
\end{equation}
$$
其中，$dom(f)$ 是函数 $f$ 的定义域。

凸函数的一些特性包括：

1. **正比例性质**：若函数 $f(x)$ 是凸函数，则对于任意常数 $\alpha \gt 0$，函数 $\alpha f(x)$ 也是凸函数。
2. **正移位性质**：若函数 $f(x)$ 是凸函数，则对于任意常数 $c \gt 0$，函数 $f(x) - c$ 也是凸函数。
3. **加法性质**：若 $f(x)$ 和 $g(x)$ 均为凸函数，则它们的和 $f(x) + g(x)$ 也是凸函数。



## 凹函数

凹函数（concave function）的定义与凸函数相反。对于其定义域内的任意两个点 $x$ 和 $y$ 以及满足 $\alpha\in[0,1]$ 的任意标量 $\alpha$，满足以下不等式：
$$
\begin{equation}
f(\alpha x + (1-\alpha)y) \geq \alpha f(x) + (1-\alpha) f(y)
\end{equation}
$$
此不等式被称为凹性条件。

其他定义与凸函数类似，这里不再赘述。值得注意的是，若函数 $f(x)$ 为凹函数，则 $-f(x)$ 为凸函数。因此，可以将凹函数问题转化为凸函数问题，从而利用凸函数的性质来求解凹函数问题。



## 强凸函数

若$f(x)$为定义在凸集上的强凸函数，则对于任意$x,y \in dom(f)$，$\alpha \in [0,1]$，存在$\lambda \gt 0$，使得：
$$
\begin{equation}
f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha)f(y) - \frac{\lambda}{2}\alpha(1-\alpha)\|x-y\|_2^2
\end{equation}
$$
此时，称 $f(x)$ 为 $\lambda$-强凸（strongly convex）函数，其中 $\lambda$ 为强凸系数。

强凸函数的其他等价定义包括：

1. **Hessian 矩阵条件**：若一个两次可微的函数 $f(x)$ 的 Hessian 矩阵 $H_f$ 在凸集中的所有 $x$ 处均为正定的（即矩阵的所有特征值为正），则该函数是强凸的。

2. **梯度条件**：若一个可微函数 $f(x)$ 是强凸的，则存在一个常数 $m$，使得对于凸集中的任意 $x,y$，有 $\|\nabla f(x) - \nabla f(y)\|_2 \geq m \|x - y\|_2$。其中，$\nabla f(x)$ 表示 $f(x)$ 在点 $x$ 处的梯度。

直观上，对于强凸函数 $f(x)$，可以在任意一点处构造一个二次函数作为其下界。这一性质使得优化算法更加高效，并具有类似于 **90页** 中定理 7.2 的良好性质。

以下给出定理 7.2 的证明：

根据强凸函数的定义，取 $x = w$，$y = w^*$，然后两边除以 $\alpha$，可得：
$$
\begin{equation}
\begin{align*}
&\frac{f(\alpha w + (1-\alpha)w^*)}{\alpha} \leq f(w) + \frac{1-\alpha}{\alpha}f(w^*) - \frac{\lambda}{2}(1-\alpha)\|w-w^*\|_2^2 \\
\Rightarrow &\frac{\lambda}{2}(1-\alpha)\|w-w^*\|_2^2 \leq f(w) - f(w^*) - \frac{f(w^* + (w-w^*)\alpha) - f(w^*)}{\alpha}
\end{align*}
\end{equation}
$$
令 $\alpha \rightarrow 0^+$，则有：
$$
\begin{equation}
\begin{align*}
&\lim_{\alpha\rightarrow 0^+}\frac{\lambda}{2}(1-\alpha)\|w-w^*\|_2^2 \leq f(w) - f(w^*) + \lim_{\alpha\rightarrow 0^+}\frac{f(w^* + (w-w^*)\alpha) - f(w^*)}{\alpha} \\
\Rightarrow &\frac{\lambda}{2}\|w-w^*\|_2^2 \leq f(w) - f(w^*) + \nabla f(w^*)^T(w-w^*)
\end{align*}
\end{equation}
$$
其中 $\Delta = (w-w^*)\alpha$。

由于 $w^*$ 为最优解，因此 $\nabla f(w^*) = 0$，则有：
$$
\begin{equation}
f(w) - f(w^*) \geq \frac{\lambda}{2}\|w-w^*\|_2^2
\end{equation}
$$



## 指数凹函数

若函数 $f(x)$ 的指数 $\exp(f(x))$ 为凹函数，则称 $f(x)$ 为指数凹（exponentially concave）函数。注意，当 $\exp(f(x))$ 是凹函数时，$f(x)$ 本身不一定是凹函数。
若 $f(x)$ 为指数凹函数，则 $\exp(-f(x))$ 必为凸函数。因此，指数凹是一种弱于强凸但强于凸的性质。

指数凹函数的一些特性包括：

1. **正比例性质**：若函数 $f(x)$ 为指数凹函数，则对于任意常数 $\alpha$，函数 $\alpha f(x)$ 也是指数凹函数。
2. **负移位性质**：若函数 $f(x)$ 为指数凹函数，且 $c$ 为常数，则函数 $f(x) - c$ 也是指数凹函数。

指数凹函数提供了一种灵活且富有表现力的方式来建模各种现象。它能捕捉广泛的形状和行为。例如，在凸优化中使用指数凹函数可以加快迭代优化算法（如梯度下降或牛顿法）的收敛速度。因此，指数凹函数在处理概率模型或存在不确定性的场景中具有重要意义，特别是在限制或量化不确定性方面。



## 凸优化

凸优化（convex optimization）是优化理论的一个分支，研究的是在凸函数的凸集上进行优化的问题。凸优化的目标是在满足一组凸约束条件的情况下，找到凸目标函数的最小值。

一般形式的凸优化问题可以表示为：
$$
\begin{equation}
\begin{align*}
&\min\ &f_0(x) \\
&s.t.\ &f_i(x) \leq 0, &i\in[m] \\
&\ &g_j(x) = 0, &j\in[n]
\end{align*}
\end{equation}
$$
其中，$f_0(x)$ 是凸目标函数，$f_i(x)$ 是凸不等式约束条件，$g_j(x)$ 是仿射等式约束条件。

凸优化具有以下有利特性，使其成为一个被广泛研究和应用的领域：

1. **全局最优性**：凸优化问题的一个关键性质是，任何局部最小值也是全局最小值。此性质确保凸优化算法找到的解是给定凸集中的最优解。

2. **高效算法**：凸优化拥有多项式时间内找到最优解的高效算法。这些算法基于凸目标函数和约束条件的凸性，能够有效解决复杂的优化问题。

3. **广泛应用**：凸优化在工程学、金融学、机器学习、运筹学和信号处理等领域有着广泛的应用。它被用于解决如投资组合优化、信号重构、资源分配和机器学习模型训练等问题。凸优化技术，如线性规划、二次规划和半定规划，构成了许多优化算法的基础，为高效解决复杂优化问题提供了强大工具。

以下证明凸函数任何局部最优解均为全局最优解的性质。

假设 $f(x)$ 是凸函数，$x^*$ 是 $f$ 在凸集合 $\mathcal{D}$ 中的局部最优解。由于凸集的性质，对于任意 $y$，$y-x^*$ 是一个可行方向。因此，总可以选择足够小的 $\alpha \gt 0$，使得：
$$
\begin{equation}
f(x^*) \leq f(x^* + \alpha(y-x^*))
\end{equation}
$$
由 $f$ 的凸性可得:
$$
\begin{equation}
f(x^* + \alpha(y-x^*)) = f((1-\alpha)x^* + \alpha y) \leq (1-\alpha)f(x^*) + \alpha f(y)
\end{equation}
$$
结合以上两式，可得：
$$
\begin{equation}
\begin{align*}
&f(x^*) \leq (1-\alpha)f(x^*) + \alpha f(y) \\
rightarrow &f(x^*) \leq f(y)
\end{align*}
\end{equation}
$$
由于 $y$ 是凸集合 $\mathcal{D}$ 中的任意点，故 $x^*$ 为全局最优解。对于 $f(x)$ 的全局最大解，可以通过考虑函数 $-f(x)$ 的局部最优解得到类似的结论。



## 仿射

仿射变换（Affine transformation），又称仿射映射，是指在几何中，对一个向量空间进行一次线性变换并加上一个平移，变换为另一个向量空间。若该线性映射被表示为矩阵 $A$，平移被表示为向量 $\vec{b}$，则仿射映射 $f$ 可表示为：
$$
\begin{equation}
\vec{y} = f(\vec{x}) = A\vec{x} + \vec{b}
\end{equation}
$$
其中，$A$ 被称为仿射变换矩阵或投射变换矩阵。

仿射变换具有以下性质：

1. **点之间的共线性**：在同一条直线上的三个或更多的点（即共线点）在变换后依然位于同一条直线上（共线）。
2. **直线的平行性**：两条或以上的平行直线在变换后仍保持平行。
3. **集合的凸性**：凸集合在变换后依然是凸集合，且最初的极值点被映射到变换后的极值点集。
4. **平行线段的长度比例恒定**：两条由点 $p_1, p_2, p_3, p_4$ 定义的平行线段，其长度比例在变换后保持不变，即 $\frac{\overrightarrow{p_1p_2}}{\overrightarrow{p_3p_4}} = \frac{\overrightarrow{f(p_1)f(p_2)}}{\overrightarrow{f(p_3)f(p_4)}}$。
5. **质心位置恒定**：不同质量的点组成集合的质心位置在仿射变换后保持不变。

仿射集（affine set）是指欧氏空间 $R^n$ 中具有以下性质的点集 $S$：对于任意 $x,y\in S$，以及 $\forall \lambda\in[0,1]$，有 $(1-\lambda)x+\lambda y\in S$。容易证明，包含原点的仿射集 $S$ 是 $R^n$ 的子空间。

仿射包（affine hull/span）是包含集合 $S$ 的所有仿射集的交集，也是集合 $S$ 中元素通过不断连接直线所形成的所有元素的集合。仿射包是包含集合 $S$ 的最小仿射集，记为 $aff(S)$，即：
$$
\begin{equation}
aff(S) = \{\sum_{i=1}^k \alpha_i x_i \mid k\gt0, x_i\in S, \alpha_i\in R, \sum_{i=1}^k \alpha_i = 1\}
\end{equation}
$$
仿射包具有以下性质：

1. $aff(aff(S)) = aff(S)$
2. $aff(S + T) = aff(S) + aff(T)$
3. 若 $S$ 为有限维度，则 $aff(S)$ 为闭集合。



## Slater条件/定理

关于强对偶性的讨论，**11页** 已给出了详细说明，此处不再赘述。这里着重讨论 **11页** 左下角附注提到的 Slater 条件，即：

存在一点 $x\in \text{relint}(D)$，该点称为 Slater 向量，有：
$$
\begin{equation}
\begin{align*}
f_i(x) \lt 0, &\quad i\in[m]
\end{align*}
\end{equation}
$$
其中，$D = \bigcap_0^m \text{dom}(f_i)$，$relint(D)$ 为 $D$ 的相对内部，即其仿射包的内部所有点，即 $relint(D) = \text{int}(aff(D))$。

当满足 Slater 条件且原始问题为凸优化问题时：

1. 强对偶性成立。
2. 对偶最优解集合非空且有界。

这就是 Slater 定理。

### 证明

首先证明对偶间隙（Duality Gap）为零，即原始问题与对偶问题的目标函数值之差 $p^* - d^* = 0$。考虑集合 $\mathcal{V}\subset \mathbb{R}^m \times \mathbb{R}$，满足：
$$
\begin{equation}
\mathcal{V}:=\{(u,w)\in\mathbb{R}^m \times \mathbb{R} \mid f_0(x) \le w, f_i(x) \le u_i, \forall i\in[m], \forall x\}
\end{equation}
$$
集合 $\mathcal{V}$ 具有以下性质：

1. 它是凸集合，可由 $f_i,\ i\in\{0\}\cup[m]$ 的凸性质得出。
2. 若 $(u,w)\in\mathcal{V}$，且 $(u',w')\succeq(u,w)$，则 $(u',w')\in\mathcal{V}$。

易证向量 $(0,p^*)\notin int(\mathcal{V})$，否则一定存在 $\varepsilon \gt 0$，使得 $(0,p^*-\varepsilon)\in int(\mathcal{V})$，这明显与 $p^*$ 为最优解矛盾。因此，必有 $(0,p^*)\in \partial\mathcal{V}$ 或 $(0,p^*)\notin\mathcal{V}$。应用支撑超平面定理（定理 23），可知存在一个非零点 $(\lambda,\lambda_0)\in \mathbb{R}^m \times \mathbb{R}$，满足以下条件：
$$
\begin{equation}
(\lambda,\lambda_0)^T(u,w) = \lambda^Tu + \lambda_0w \ge \lambda_0p^*, \forall(u,w)\in\mathcal{V}
\end{equation}
$$
在此情况下，必然有 $\lambda \succeq 0$ 和 $\lambda_0 \geq 0$。这是因为，若 $\lambda$ 和 $\lambda_0$ 中的分量出现任何负数，根据集合 $\mathcal{V}$ 的性质二，$(u, w)$ 的分量可以在集合 $\mathcal{V}$ 内取得任意大的值，从而导致上式不一定成立。

因此，只需考虑两种情况：

1. **$\lambda_0 = 0$**：此时根据上式，可得
$$
\begin{equation}
\inf_{(u,w)\in\mathcal{V}}\lambda^Tu = 0
\end{equation}
$$
另一方面，根据 $\mathcal{V}$ 的定义，$\lambda\succeq 0$ 且 $\lambda \neq 0$，可得：
$$
\begin{equation}
\inf_{(u,w)\in\mathcal{V}}\lambda^Tu = \inf_{x}\sum_{i=1}^m \lambda_i f_i(x) \leq \sum_{i=1}^m \lambda_i f_i(\bar{x}) \lt 0
\end{equation}
$$
其中，$\bar{x}$ 是 Slater 向量，而最后一个不等式依据 Slater 条件得出。此时，两个结论互相矛盾，因此 $\lambda_0 \neq 0$。

2. **$\lambda_0 \gt 0$**：对上式左右两边除以 $\lambda_0$，得：
$$
\begin{equation}
\inf_{(u,w)\in\mathcal{V}}\{\tilde\lambda^Tu + w\} \ge p^*
\end{equation}
$$
其中，$\tilde\lambda := \frac{\lambda}{\lambda_0}\succeq 0$。

考虑拉格朗日函数 $L:\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}$：
$$
\begin{equation}
L(x,\tilde\lambda) := f_0(x) + \sum_{i=1}^m \tilde\lambda_i f_i(x)
\end{equation}
$$
其对偶函数为：
$$
\begin{equation}
g(\tilde\lambda) := \inf_{x} L(x,\tilde\lambda) \ge p^*
\end{equation}
$$
其对偶问题为：
$$
\begin{equation}
\max_{\lambda} g(\lambda), \lambda\succeq 0
\end{equation}
$$
因此，可得 $d^* \geq p^*$。根据弱对偶性，$d^* \leq p^*$，从而推断出 $d^* = p^*$。

接着证明对偶问题最优解集合非空且有界。对于任意对偶最优解 $\tilde\lambda\succeq 0$，有：
$$
\begin{equation}
\begin{align*}
d^* = g(\tilde\lambda) &= \inf_{x} \{f_0(x) + \sum_{i=1}^m \tilde\lambda_i f_i(x)\} \\
&\leq f_0(\bar{x}) + \sum_{i=1}^m \tilde\lambda_i f_i(\bar{x}) \\
&\leq f_0(\bar{x}) + \max_{i\in[m]}\{f_i(\bar{x})\}[\sum_{i=1}^m \tilde\lambda_i]
\end{align*}
\end{equation}
$$
因此，有：
$$
\begin{equation}
\min_{i\in[m]}\{-f_i(\bar{x})\}[\sum_{i=1}^m \tilde\lambda_i] \leq f_0(\bar{x}) - d^*
\end{equation}
$$
进而得出：
$$
\begin{equation}
\|\tilde\lambda\| \leq \sum_{i=1}^m \tilde\lambda_i \leq \frac{f_0(\bar{x}) - d^*}{\min_{i\in[m]}\{-f_i(\bar{x})\}} \lt \infty
\end{equation}
$$
其中，最后一个不等式依据 Slater 条件得出。$\square$



## KKT条件

KKT条件（Karush-Kuhn-Tucker条件）在凸优化领域具有至关重要的地位。虽然在**12-13页** 中对其进行了基本解释，此处将进行更为深入的分析。KKT条件中的符号 $\lambda_i,\ i\in[m]$ 和 $\mu_i,\ i\in[n]$ 被视为 KKT 乘子。特别地，当 $m=0$ 时，即不存在不等式约束条件时，KKT条件退化为拉格朗日条件，此时 KKT 乘子也被称为拉格朗日乘子。

### 证明

首先，对于 $x^*,(\mu^*,\lambda^*)$ 满足 KKT 条件等价于它们构成一个纳什均衡。

固定 $(\mu^*,\lambda^*)$，并变化 $x$，均衡等价于拉格朗日函数在 $x^*$ 处的梯度为零，即主问题的稳定性（stationarity）。

固定 $x$，并变化 $(\mu^*,\lambda^*)$，均衡等价于主问题的约束（feasibility）和互补松弛条件。

**充分性**：若解对 $x^*,(\mu^*,\lambda^*)$ 满足 KKT 条件，则它们构成一个纳什均衡，从而消除对偶间隙。

**必要性**：任意解对 $x^*,(\mu^*,\lambda^*)$ 必然消除对偶间隙，因此它们必须构成一个纳什均衡，从而满足 KKT 条件。$\square$

在此对 KKT 和 Slater 条件进行区分：

1. **KKT条件** 是一组用于确定约束优化问题中解的最优性的条件。它们通过将约束纳入条件，扩展了无约束优化中设定目标函数梯度为零的思路到约束优化问题中。  
   **Slater条件** 是凸优化中确保强对偶性的特定约束条件，即主问题和对偶问题最优解的等价性。

2. KKT条件包括对偶问题的约束、互补松弛条件、主问题约束和稳定性。它们整合了目标和约束函数的梯度以及 KKT 乘子，以形成最优性条件。  
   Slater 条件要求存在一个严格可行点，即严格满足所有不等式约束的点。

3. 当点满足 KKT 条件时，表明问题的局部最优解已找到。这些条件弥合了主问题和对偶问题之间的差距，对于分析和解决约束优化问题至关重要。  
   满足 Slater 条件时，确保凸优化问题的强对偶性，对于简化和解决这些问题至关重要。Slater 条件并不直接提供最优性条件，但为强对偶性铺平了道路，之后可以利用强对偶性寻找最优解。

4. **KKT条件** 较为通用，适用于更广泛的优化问题类别，包括非凸问题。  
   **Slater条件** 则特定于凸优化问题，用于确保这些问题中的强对偶性。

5. 对于凸且可微的问题，满足 KKT 条件意味着最优性和强对偶性。相反，最优性和强对偶性意味着所有问题的 KKT 条件得到满足。  
   当 Slater 条件成立时，KKT 条件是最优解的充要条件，此时强对偶性成立。

KKT条件和 Slater 条件通常被归类为“正则条件”（regularity condition）或“约束资格”（constraint qualification）。这些条件为优化问题提供了一个结构化的框架，以便在约束情况下分析和确定解的最优性。更多的正则条件详见参考文献：[On regularity conditions in mathematical programming](https://link.springer.com/chapter/10.1007/BFb0120988)。



## 偏序集

序理论（order theory）是研究捕捉数学排序直觉的各种二元关系的数学分支。在序理论中，一个偏序集（partial order set，简称 poset）包含一个非空集合 $P$ 和一个满足特定条件的二元关系 $\leq$。这个二元关系称为偏序关系，它必须满足以下三个条件：

1. **自反性（Reflexivity）**：对于 $P$ 中的任意元素 $a$，都有 $a \leq a$。
2. **反对称性（Antisymmetry）**：对于 $P$ 中的任意元素 $a$ 和 $b$，如果 $a \leq b$ 且 $b \leq a$，那么 $a = b$。
3. **传递性（Transitivity）**：对于 $P$ 中的任意元素 $a$、$b$ 和 $c$，如果 $a \leq b$ 且 $b \leq c$，那么 $a \leq c$。

这些条件定义了偏序关系，使其与全序（total order）关系不同。在偏序集中，可能存在某些元素是不可比较的，即对于 $P$ 中的某些 $a$ 和 $b$，既不满足 $a \leq b$，也不满足 $b \leq a$。



## 上下界

上界（upper bound 或 majorant）是与偏序集有关的特殊元素，指偏序集中大于或等于其子集中一切元素的元素。若数集 $S$ 为实数集 $R$ 的子集且有上界，则显然有无穷多个上界，其中最小的上界常常具有重要作用，称为数集 $S$ 的上确界（tight upper bound 或 supremum）。同理，可以定义下界（lower bound 或 minorant）和下确界（tight lower bound 或 infimum）。



## 尾界

**尾界（tail bound）**是指给定一个随机变量，其概率分布尾部部分的界限。上尾界（upper tail bound）描述随机变量在其分布上尾处的概率上限，而下尾界（lower tail bound）描述随机变量在其分布下尾处的概率上限。Chebyshev 不等式、Hoeffding 不等式和 Bernstein 不等式都是尾界的例子，它们提供了随机变量偏离其期望值的概率界限。



## 置信界

**置信界（confidence bound）**是在估计一个未知参数时，给出一个包含该参数的区间，并且这个区间具有特定的置信水平。例如，一个95%的置信区间意味着我们有95%的信心该区间包含真实的参数值。置信界可以是上置信界（upper confidence bound），下置信界（lower confidence bound），或同时包含上下界的置信区间（confidence interval）。上置信界提供对参数估计的可能最大值的上限，下置信界提供对参数估计的可能最小值的下限。



## 连续性

连续性（continuity）表示函数在某处的变化不会突然中断或跳跃。形式上，如果函数 $f(x)$ 在 $x = a$ 处满足以下条件，则称其在该点连续：

1. 函数 $f(x)$ 在 $x = a$ 处有定义。
2. 当 $x$ 趋近于 $a$ 时，$f(x)$ 的极限存在且等于 $f(a)$。

连续性意味着输入的微小变化导致输出的微小变化。如果一个函数在其定义域的每个点上都是连续的，则称其为连续函数。

Lipschitz 连续性是连续性的更强形式，它要求函数在变化速度方面有界。具体而言，如果存在一个常数 $L$，使得函数在任意两点的函数值之间的绝对差小于等于 $L$ 乘以两点之间的距离，则称该函数为 $L-Lipschitz$ 连续，即：
$$
\begin{equation}
\forall x,y\in \text{dom}(f),\ \exists L \gt 0\ \text{使得}\ \|f(x)-f(y)\|_2 \leq L\|x-y\|_2
\end{equation}
$$
其中，$L$ 称为 Lipschitz 常数，表示函数的最大变化率。若 $L$ 较大，函数可以快速变化；若 $L$ 较小，函数变化更渐进。

事实上，如果一个函数的导数有界，那么它一定是 Lipschitz 连续的；反之，如果一个可微函数是 Lipschitz 连续的，那么它的导数一定有界。

证明如下：

1. 若函数 $f(x)$ 的导数有界，即存在常数 $L \ge 0$，使得对于任意 $x$，有 $|f'(x)| \leq L$。根据微分中值定理，对于任意 $x \le y$，存在 $c \in [x,y]$，使得：
$$
\begin{equation}
\begin{align*}
&\|f(x)-f(y)\|_2 = \|f'(c)\|_2\|x-y\|_2 \\
\Rightarrow &\|f(x)-f(y)\|_2 \le L \|x-y\|_2
\end{align*}
\end{equation}
$$
此时，函数是 $L-Lipschitz$ 连续的。

2. 若函数 $f(x)$ 是 $L-Lipschitz$ 连续的，即对于任意 $x,y$，有
$$
\begin{equation}
\|f(x)-f(y)\|_2 \le L\|x-y\|_2
\end{equation}
$$
根据微分中值定理，对于任意 $x \le y$，存在 $c \in [x,y]$，使得：
$$
\begin{equation}
\|f(x)-f(y)\|_2 = \|f'(c)\|_2\|x-y\|_2
\end{equation}
$$
不妨令 $x \rightarrow y$，则 $c \rightarrow y$。因为 $f(y)$ 可微，可得：
$$
\begin{equation}
\|f'(y)\|_2 = \|\lim_{x \rightarrow y}\frac{f(x)-f(y)}{x-y}\|_2 = \lim_{x \rightarrow y}\frac{\|f(x)-f(y)\|_2}{\|x-y\|_2} \le \lim_{x \rightarrow y} L = L
\end{equation}
$$
因为 $y$ 的任意性，所以函数的导数有界。

连续性关注函数图像中跳跃或中断的缺失，而 Lipschitz 连续性关注函数的变化速度。因此，Lipschitz 连续性是比连续性更严格的条件。一个连续函数不一定是 Lipschitz 连续的，因为连续性不要求函数变化速度有界。然而，一个 Lipschitz 连续的函数必然是连续的，因为 Lipschitz 连续性蕴含连续性。

Lipschitz 连续性的性质在数学的各个领域中广泛应用，如分析、优化和微分方程研究。它在保证某些数学问题的解的存在性、唯一性和稳定性方面起着关键作用。



## 光滑性

在数学分析中，函数的光滑性（smoothness）通过函数在某个域（称为可微性类）上的连续导数的数量来衡量。最基本的情况下，如果一个函数在每个点上都可导（因此连续），则可以认为它是光滑的。
一方面，光滑性确保了梯度下降等优化算法能够更快收敛，并减少可能遇到的梯度震荡或发散的情况。
另一方面，光滑性提供了函数曲率的信息，从而帮助设计更有效的优化算法，如加速梯度下降法或牛顿法。

在优化理论中，$L$-光滑函数是指它的梯度具有 $L$-Lipschitz 连续性，这意味着函数的梯度在其定义域中的变化速率被 $L$ 所限制。
形式上，对于任意 $x,y \in \mathbb{R}^n$，存在 $L \gt 0$，使得：
$$
\begin{equation}
\|\nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2
\end{equation}
$$
或者等价地，
$$
\begin{equation}
\|\nabla^2 f(x)\|_2 \leq L
\end{equation}
$$
或者等价地，
$$
\begin{equation}
f(y) \leq f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|y - x\|_2^2
\end{equation}
$$
以上三种定义方式是等价的，且 $L$ 被称为光滑系数。
由定义3，我们可以看出，在光滑函数的任意一点处都可以构造一个二次函数作为其上界。

接下来我们证明这些定义的等价性。首先，我们证明定义1可以推导出定义2。

考虑函数 $f$ 的梯度 $\nabla f(x)$ 的二阶泰勒展开：
$$
\begin{equation}
\nabla f(y) = \nabla f(x) + \nabla^2 f(\xi)(y - x)
\end{equation}
$$
其中 $\xi$ 是 $x$ 和 $y$ 之间的一点，$\nabla^2 f(\xi)$ 表示在点 $\xi$ 处的 Hessian 矩阵。

根据 $L$-光滑性的定义1，我们有：
$$
\begin{equation}
\|\nabla f(y) - \nabla f(x)\|_2 \leq L \|y - x\|_2
\end{equation}
$$

将二阶泰勒展开的结果代入其中：
$$
\begin{equation}
\|\nabla^2 f(\xi)(y - x)\|_2 \leq L \|y - x\|_2
\end{equation}
$$

对于任意的非零向量 $v = y - x$，定义：
$$
\begin{equation}
v' = \frac{v}{\|v\|_2}
\end{equation}
$$
我们得到：
$$
\begin{equation}
\|\nabla^2 f(\xi) v'\|_2 \leq L
\end{equation}
$$

由于 $v'$ 是一个单位向量，这意味着 Hessian 矩阵 $\nabla^2 f(\xi)$ 作用在任意单位向量上时的范数不超过 $L$，因此 Hessian 矩阵的谱范数（即最大特征值的绝对值）满足：
$$
\begin{equation}
\|\nabla^2 f(\xi)\|_2 \leq L
\end{equation}
$$
其中，由于 $\xi$ 是 $x$ 和 $y$ 之间的一点，因此我们可以将上述结论推广到整个定义域。

接下来我们证明定义2可以推导出定义3。由定义2，给定 $f$ 是 $L$-光滑的，对任意的 $x, y \in \mathbb{R}^n$，我们有：
$$
\begin{equation}
f(y) \leq f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2} \|y - x\|_2^2
\end{equation}
$$

将定义中的 $x$ 和 $y$ 互换，得到：
$$
\begin{equation}
f(x) \leq f(y) + \langle \nabla f(y), x - y \rangle + \frac{L}{2} \|x - y\|_2^2
\end{equation}
$$

将两个不等式相加可得：
$$
\begin{equation}
\langle \nabla f(x) - \nabla f(y), x - y \rangle \leq L \|x - y\|_2^2
\end{equation}
$$

注意到不等式左侧的内积无论如何取值，该不等式均成立。
根据 Cauchy-Schwarz 不等式，当 $y - x$ 与 $\nabla f(x) - \nabla f(y)$ 平行时左侧内积取到最大值，即 $\|\nabla f(x) - \nabla f(y)\|_2 \|x - y\|_2$，代入可得：
$$
\begin{equation}
\|\nabla f(x) - \nabla f(y)\|_2 \|x - y\|_2 \leq L \|x - y\|_2^2
\end{equation}
$$
化简后即得证。

这里对光滑性和 $Lipschitz$ 连续性进行一些比较：
- $Lipschitz$ 连续性关注的是函数值变化的速度，即函数值的“陡峭程度”，而光滑性关注的是梯度变化的速度，即函数的“曲率”或二阶变化。
- $Lipschitz$ 连续性表示函数变化不会太快，确保函数的整体平滑性，而光滑性表示梯度变化不会太快，确保函数曲面没有急剧的弯曲。



## 次梯度

次梯度（subgradient）是凸函数导数的推广形式。某些凸函数在特定区域内可能不存在导数，但我们依旧可以用次梯度来表示该区域内函数变化率的下界。形式上，对于凸函数 $f(x)$，在任意点 $x_0$ 处的次梯度 $c$ 必须满足以下不等式：
$$
\begin{equation}
f(x) - f(x_0) \geq c(x - x_0)
\end{equation}
$$
根据微分中值定理的逆命题，$c$ 通常在 $[a,b]$ 之间取值，其中 $a,b$ 是函数 $f(x)$ 在 $x_0$ 处的左右导数，即：
$$
\begin{equation}
a = \lim_{x \rightarrow x_0^-}\frac{f(x) - f(x_0)}{x - x_0},\ b = \lim_{x \rightarrow x_0^+}\frac{f(x) - f(x_0)}{x - x_0}
\end{equation}
$$
此时，次梯度 $c$ 的集合 $[a,b]$ 被称为次微分，即 $\partial f(x_0)$。当 $a = b$ 时，次梯度 $c$ 退化为导数。

次梯度在机器学习领域广泛应用，特别是在训练支持向量机（SVM）和其他具有非可微损失函数的模型中。它们还构成了随机次梯度方法的基础，这些方法在处理大规模机器学习问题时非常有效。



## 对偶空间

线性泛函（linear functional）是指从向量空间 $V$ 到对应标量域 $k$ 的线性映射，满足加法和数乘的性质，即对于任意向量 $x,y \in V$ 和标量 $\alpha \in k$，有：
$$
\begin{equation}
\begin{align*}
&f(x+y) = f(x) + f(y) \\
&f(\alpha x) = \alpha f(x)
\end{align*}
\end{equation}
$$
所有从 $V$ 到 $k$ 的线性泛函构成的集合称为 $V$ 的对偶空间（dual space），记为 $V^* = \text{Hom}_k(V,k)$，对偶空间中的元素称为对偶向量。



## Legendre变换

将函数转换为另一种函数，常用于改变其定义域和属性，使问题更简单或更易分析。Legendre 变换（Legendre transform）常用于将一组独立变量转换为另一组独立变量，特别是在经典力学和热力学中。以下是 Legendre 变换的基本概念和步骤：

1. **定义函数**：假设有一个凸函数 $f(x)$，其自变量为 $x$。
2. **定义共轭变量**：定义新的变量 $p$，它是原函数 $f(x)$ 的导数，即 $p = \frac{d f(x)}{dx}$。
3. **定义共轭函数**：定义新的函数 $g(p)$，其形式为：$g(p) = x \cdot p - f(x)$。这里，$x$ 是 $f(x)$ 的自变量，同时也是 $g(p)$ 的隐含变量。
4. **变换关系**：通过 Legendre 变换，从原来的函数 $f(x)$ 得到新的函数 $g(p)$，这个新的函数 $g(p)$ 依赖于共轭变量 $p$。



## 共轭函数

凸共轭（convex conjugate）是 Legendre 变换的一种推广，因此也被称为 Legendre-Fenchel 变换（Legendre-Fenchel transform）。通过凸共轭变换，原函数可以转换为凸函数，从而利用凸函数的性质来解决原问题。

形式上，对于函数 $f(x)$，其共轭函数 $f^*(y)$ 定义为：
$$
\begin{equation}
f^*(y) = \sup_{x \in \text{dom}(f)}(y^T x - f(x))
\end{equation}
$$
其中，$\text{dom}(f)$ 是函数 $f(x)$ 的定义域。

共轭函数具有以下一些有用的性质：

1. **凸性**：函数 $f(x)$ 的共轭函数 $f^*(y)$ 一定是凸函数。证明如下：
$$
\begin{equation}
\begin{align*}
f^*(\lambda y_1+(1-\lambda)y_2) &= \sup_{x\in \text{dom}(f)}\{x^T(\lambda y_1+(1-\lambda)y_2)-f(x)\}\\
&\leq \lambda \sup_{x\in \text{dom}(f)}\{x^T y_1 - f(x)\} + (1-\lambda)\sup_{x\in \text{dom}(f)}\{x^T y_2 - f(x)\}\\
&= \lambda f^*(y_1) + (1-\lambda)f^*(y_2)
\end{align*}
\end{equation}
$$
其中的不等式利用了凸性的性质。

2. **逆序性**：对于定义域中所有元素 $x$，若 $f(x) \leq g(x)$，则 $f^*(y) \geq g^*(y)$。证明如下：

由于 $f(x) \leq g(x)$，因此 $x^T y - f(x) \geq x^T y - g(x)$。两边同时取上确界，根据定义有：
$$
\begin{equation}
f^*(y) = \sup_{x\in \text{dom}(f)}\{x^T y - f(x)\} \geq \sup_{x\in \text{dom}(f)}\{x^T y - g(x)\} = g^*(y)
\end{equation}
$$

3. **极值变换**：若 $f$ 可微，则对于 $\forall y$，有：
$$
\begin{equation}
f^*(y) \leq f^*(\nabla f(x)) = \nabla f^*(x)^T x - f(x) = -[f(x) + \nabla f(x)^T(0 - x)]
\end{equation}
$$
此性质即书中的（1.10），完整证明如下：

为了在 $f^*$ 的定义中找到上确界，对右侧的 $x$ 求导，并将其设置为零以找到极大值点：
$$
\begin{equation}
\frac{d}{dx}(x^T y − f(x)) = y − \nabla f(x) = 0
\end{equation}
$$
此时有 $y = \nabla f(x)$，得证。



## σ-代数

σ-代数（或 σ-域）是测度论和概率论中的一个重要概念。σ-代数是一个满足特定封闭性质的集合族，使我们能够对这些集合定义一致的测度（如概率）。具体来说，σ-代数是一个集合族，满足以下三个性质：

1. **包含全集**：如果 $\mathcal{F}$ 是定义在集合 $X$ 上的一个 σ-代数，那么 $X$ 本身属于 $\mathcal{F}$，即 $X \in \mathcal{F}$。
2. **对补集封闭**：如果 $A$ 是 $\mathcal{F}$ 中的一个集合，那么它的补集 $X \setminus A$ 也属于 $\mathcal{F}$，即 $A \in \mathcal{F} \implies X \setminus A \in \mathcal{F}$。
3. **对可数并封闭**：如果 $A_1, A_2, A_3, \ldots$ 是 $\mathcal{F}$ 中的集合，那么它们的可数并集 $\bigcup_{i=1}^{\infty} A_i$ 也属于 $\mathcal{F}$，即 $A_i \in \mathcal{F}$ 对所有 $i \in \mathbb{N}$，则 $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$。

σ-代数在测度论中尤为重要，因为它为定义测度提供了必要的框架。测度是定义在 σ-代数上的集合函数，用于度量集合的“大小”。在概率论中，σ-代数用于定义事件空间，从而定义概率测度。

### 过滤

σ-代数 $\mathcal{F}$ 是一个固定的集合族，满足特定的封闭性质，表示我们在某一时刻可以知道的所有信息。过滤（filtration）是关于随着时间推移而观察信息的概念，通常与随机过程（stochastic processes）相关。具体来说，过滤是一个按时间参数索引的 σ-代数序列 $\{\mathcal{F}_t\}_{t \in T}$，表示随时间变化的可观测事件的集合，满足以下性质：

1. **每个 $\mathcal{F}_t$ 是一个 σ-代数**：对于每个时刻 $t$，$\mathcal{F}_t$ 是定义在某个固定集合 $X$ 上的一个 σ-代数。
2. **单调性**：对于任意的 $t_1 \leq t_2$，有 $\mathcal{F}_{t_1} \subseteq \mathcal{F}_{t_2}$。这意味着随着时间的推移，所包含的信息只会增加，不会减少。



## 鞅

鞅（Martingale）是概率论中的一个重要概念，用于描述某些类型的随机过程。鞅过程的特点是，其未来期望值在已知当前信息的条件下等于当前值。

### 形式化定义

设 $\{X_t\}$ 是一个随机过程，$\{\mathcal{F}_t\}$ 是一个随时间 $t$ 变化的过滤（即包含随时间增加的所有信息的 σ-代数的序列）。当这个随机过程 $\{X_t\}$ 是鞅时，必须满足以下条件：

1. **适应性（Adaptedness）**：对于每一个 $t$，$X_t$ 是 $\mathcal{F}_t$-可测的（即 $X_t$ 的值在时间 $t$ 时刻是已知信息的函数）。
2. **积分性（Integrability）**：对于所有 $t$，$\mathbb{E}[|X_t|] \lt \infty$。
3. **鞅性质（Martingale Property）**：对于所有 $t$ 和 $s \geq t$，有 $\mathbb{E}[X_s \mid \mathcal{F}_t] = X_t$。这意味着在已知当前时刻 $t$ 的信息 $\mathcal{F}_t$ 条件下，未来某个时刻 $s$ 的期望值等于当前时刻 $t$ 的值。

### 直观解释

鞅的定义保证了在已知当前信息的条件下，未来值的期望等于当前值，这反映了一种“无偏性”。因此，鞅过程可以被看作是一种“公平游戏”。设想一个赌徒在赌场中进行赌博，如果这个赌徒的资金变化形成一个鞅过程，那么在任何时刻，给定当前的资金情况，未来资金的期望值都是当前的资金，表示没有系统性的赢或输的趋势。

### 举例说明

考虑一个简单的随机游走过程，其中 $X_{t+1} = X_t + Z_{t+1}$，其中 $Z_{t+1}$ 是一个独立同分布的随机变量，取值为 $+1$ 或 $-1$，且概率各为 $50\%$。在这种情况下，如果设 $X_0 = 0$，那么 $\{X_t\}$ 是一个鞅，因为每一步的期望值都是零。

### 鞅的类型

除了标准的鞅，还有两个相关的概念：

1. **超鞅（Submartingale）**：若对于所有 $t$ 和 $s \geq t$，有 $\mathbb{E}[X_s \mid \mathcal{F}_t] \geq X_t$，则称 $\{X_t\}$ 为超鞅（或上鞅）。
2. **亚鞅（Supermartingale）**：若对于所有 $t$ 和 $s \geq t$，有 $\mathbb{E}[X_s \mid \mathcal{F}_t] \leq X_t$，则称 $\{X_t\}$ 为亚鞅（或下鞅）。

一个区分超鞅和亚鞅的记忆方法是：“生活是一个超鞅：随着时间的推进，期望降低。”

### 鞅差序列

鞅差 $D_t$ 定义为 $D_t = X_t - X_{t-1}$，鞅差序列（Martingale Difference Sequence）$\{D_t\}$ 则满足以下条件：

1. **适应性（Adaptedness）**：对于每一个 $t$，$D_t$ 是 $\mathcal{F}_t$-可测的。
2. **零条件期望（Zero Conditional Expectation）**：对于所有 $t$，有 $\mathbb{E}[D_t \mid \mathcal{F}_{t-1}] = 0$，即在已知过去信息 $\mathcal{F}_{t-1}$ 的条件下，$D_t$ 的条件期望为零。这意味着当前的观察值不提供对未来观察值的系统性偏差，即每一步的变化是纯随机的。

虽然鞅差序列中的每个元素的条件期望为零，但这并不意味着这些元素是独立的。相反，它们可以有复杂的依赖关系。鞅差序列的关键性质是每个元素在条件期望下为零，这使得它在分析鞅和集中不等式（如 Bernstein 不等式）中非常有用。



## KL 散度

KL 散度（Kullback-Leibler 散度），也称为相对熵，是一种用于衡量两个概率分布之间差异的非对称度量，在信息论和统计学中广泛应用。KL 散度衡量的是在使用近似分布时，相比于使用真实分布，所增加的“信息损失”或“不确定性”。

### 定义

假设有两个概率分布 $P$ 和 $Q$，它们定义在同一个概率空间上。$P$ 通常被认为是“真实”分布，而 $Q$ 是近似分布。KL 散度 $D_{KL}(P \| Q)$ 表示为：
$$
\begin{equation}
D_{KL}(P \| Q) = \sum_{x} P(x) \ln \frac{P(x)}{Q(x)}
\end{equation}
$$
对于连续分布：
$$
\begin{equation}
D_{KL}(P \| Q) = \int_{-\infty}^{+\infty} p(x) \ln \frac{p(x)}{q(x)} \, dx
\end{equation}
$$
其中，$P(x)$ 和 $Q(x)$ 分别是分布 $P$ 和 $Q$ 在 $x$ 处的概率密度函数（或概率质量函数）。

### 性质

1. **非负性**：KL 散度总是非负的，即 $D_{KL}(P \| Q) \geq 0$，只有当 $P$ 和 $Q$ 完全相同时，KL 散度才为零。

### 非负性的证明

KL 散度的非负性可以通过 Jensen 不等式来证明。首先，考虑离散情况下的 KL 散度定义：
$$
\begin{equation}
D_{KL}(P \| Q) = \sum_{x} P(x) \ln \frac{P(x)}{Q(x)}
\end{equation}
$$
由于对数函数是一个凹函数，可以应用 Jensen 不等式。对于凹函数 $f$ 和随机变量 $X$，有：
$$
\begin{equation}
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
\end{equation}
$$

将 $f(x) = \ln(x)$，并令 $X = \frac{Q(x)}{P(x)}$。则有：
$$
\begin{equation}
\ln(\mathbb{E}[\frac{Q(x)}{P(x)}]) \geq \mathbb{E}[\ln(\frac{Q(x)}{P(x)})]
\end{equation}
$$

因为 $\sum_{x} P(x) = 1$ 且 $Q(x) \geq 0$，所以：
$$
\begin{equation}
\mathbb{E}[\frac{Q(x)}{P(x)}] = \sum_{x} P(x) \frac{Q(x)}{P(x)} = \sum_{x} Q(x) = 1
\end{equation}
$$

于是，有：
$$
\begin{equation}
0 = \ln(1) \geq \sum_{x} P(x) \ln(\frac{Q(x)}{P(x)})
\end{equation}
$$
即：
$$
\begin{equation}
D_{KL}(P \| Q) = \sum_{x} P(x) \ln(\frac{P(x)}{Q(x)}) \geq 0
\end{equation}
$$

2. **非对称性**：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$，即 KL 散度不是对称的，交换 $P$ 和 $Q$ 一般会导致不同的结果。

### 应用

- **机器学习**：在训练过程中，KL 散度常用于优化目标函数，例如变分自编码器（VAE）和生成对抗网络（GAN）。通过最小化 KL 散度，可以使近似分布 $Q$ 尽可能接近真实分布 $P$，从而提高模型的准确性和效率。
- **信息论**：用于测量编码方案的效率，评估数据压缩方案等。
- **统计学**：用于假设检验和模型选择。



## 先验和后验

先验（Prior）和后验（Posterior）是贝叶斯统计中的两个核心概念，用于描述不确定性和信息更新的过程。

### 先验概率（Prior Probability）

**定义**：先验概率是指在获得新数据之前，根据已有的知识或经验对某一事件或参数的初始估计。先验概率反映了在观察到新数据之前，我们对某一事件或参数的不确定性。

**表示方法**：用 $P(\theta)$ 表示，其中 $\theta$ 代表参数或事件。

**作用**：先验概率提供了一个起点，在进行贝叶斯推断时，它与新的数据结合，更新我们的认知。

### 后验概率（Posterior Probability）

**定义**：后验概率是指在获得新数据之后，根据贝叶斯定理更新的某一事件或参数的概率分布。后验概率反映了在观察到新数据之后，我们对某一事件或参数的不确定性。

**表示方法**：用 $P(\theta \mid D)$ 表示，其中 $\theta$ 代表参数或事件， $D$ 代表新观察到的数据。

**计算方法**：根据贝叶斯定理，后验概率可以通过先验概率、似然函数和边际似然计算得到：
$$
\begin{equation}
P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)}
\end{equation}
$$
其中：
- $P(\theta \mid D)$ 是后验概率。
- $P(D \mid \theta)$ 是似然函数，表示在给定参数 $\theta$ 时观察到数据 $D$ 的概率。
- $P(\theta)$ 是先验概率。
- $P(D)$ 是边际似然，表示观察到数据 $D$ 的总体概率。



## 拓扑向量空间

拓扑向量空间（Topological Vector Space，简称 TVS）是一个定义在拓扑域 $\mathbb{K}$（通常是带有标准拓扑的实数或复数）上的向量空间，该空间被赋予了一个拓扑结构，使得向量加法 $\cdot\, + \,\cdot\; : X \times X \to X$ 和标量乘法 $\cdot : \mathbb{K} \times X \to X$ 是连续函数（这些函数的定义域赋予了乘积拓扑）。这样的拓扑被称为 $X$ 上的**向量拓扑**或**TVS 拓扑**。

拓扑向量空间是数学分析和函数空间理论中的重要概念，它们将向量空间的代数结构与拓扑空间的结构相结合，从而使我们能够更好地理解向量空间中的连续性和收敛性。



## 超平面

超平面（Hyperplane）是指一个比所在拓扑向量空间少一维的平滑仿射子空间。  
半空间（Half Space）是指拓扑向量空间被超平面划分出的两个区域之一。

假设有一个超平面，其由以下方程定义：
$$
\begin{equation}
\mathbf{n} \cdot \mathbf{x} = c
\end{equation}
$$
其中，$\mathbf{n}$ 是垂直于超平面的法向量，$\mathbf{x}$ 是空间中的一个点，$c$ 是一个常数。

两个半空间分别由以下不等式定义：
$$
\begin{equation}
\mathbf{n} \cdot \mathbf{x} \geq c
\end{equation}
$$
和
$$
\begin{equation}
\mathbf{n} \cdot \mathbf{x} \leq c
\end{equation}
$$
这些不等式中的每一个代表了超平面两侧的一个半空间，满足其中一个不等式的点位于相应的半空间中。



## 紧空间

紧空间（Compact Space）在数学中是一种具有特殊性质的空间，即它在某种意义上表现得像“有限的”，即使它可能看起来非常大，甚至是无限的。

一个空间被称为紧致的，如果可以用有限数量的小而重叠的片段完全覆盖整个空间。换句话说，即使这个空间本身可能非常大或无限大，但紧致性意味着总能用有限数量的部分来描述它的全貌。

紧空间可以理解为一种“有限”或“被包含”的空间。这种空间不会让你“无限延伸”，而是会将你限制在某个范围内。想象你在一个小岛上，无论你走到哪里，总会遇到岛的边缘——你不能无限制地前进，总有一个尽头。这类似于紧空间。

相反地，如果你在一片无边无际的沙漠中，可以一直走下去而永远不会到达尽头，这类似于非紧空间。在紧空间中，总有一种“有限”的感觉，而在非紧空间中，感觉像是没有尽头的延伸。



## Taylor展开

**Taylor展开**（Taylor Expansion）是用多项式来近似一个函数的工具。它表示一个函数在某一点附近的值为该函数在该点的导数信息的线性组合，从而通过简单的多项式来逼近复杂的函数。

### 定义：
给定一个在某点 $a$ 处可导多次的函数 $f(x)$，它的 **Taylor 展开** 在点 $a$ 处的表达式为：

$$
\begin{equation}
f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f^{(3)}(a)}{3!}(x - a)^3 + \dots + \frac{f^{(n)}(a)}{n!}(x - a)^n + R_n(x)
\end{equation}
$$

其中：
- $f^{(n)}(a)$ 表示函数 $f(x)$ 在点 $a$ 处的第 $n$ 阶导数，
- $R_n(x)$ 是剩余项（余项），它表示截断后，未被包含的误差部分。

当 $x$ 足够接近 $a$ 时，截取足够多项的 Taylor 展开可以非常准确地逼近函数值。

### 特殊情况：麦克劳林（Maclaurin）展开
当 $a = 0$ 时，Taylor 展开被称为 **麦克劳林展开**，形式为：

$$
\begin{equation}
f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f^{(3)}(0)}{3!}x^3 + \dots
\end{equation}
$$

### 例子：
1. **指数函数的 Taylor 展开**（以 $a = 0$ 为例，即 麦克劳林展开）：
   $$
   \begin{equation}
   e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots
   \end{equation}
   $$

2. **正弦函数的 Taylor 展开**（在 $a = 0$ 处）：
   $$
   \begin{equation}
   \sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots
   \end{equation}
   $$

通过 Taylor 展开，我们可以在某个点附近用有限项多项式来近似复杂的函数。这在数值计算和分析中非常有用。