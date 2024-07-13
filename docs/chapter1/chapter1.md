# 第1章：预备知识  
  
*Edit: J. Hu, 李一飞, 赵志民*  
  
---  
  
强大数定律表明：在样本数量足够多时，样本均值以概率1收敛于总体的期望值。集中不等式主要量化地研究随机变量与其期望的偏离程度，在机器学习理论中常用于考察经验误差与泛化误差的偏离程度，由此刻画学习模型对新数据的处理能力。集中不等式是学习理论的基本分析工具。本节将列出学习理论研究中常用的集中不等式及其简要证明。  



# 常用概念


## 1. 范数

在数学中，范数（norm）是一个函数，它为向量空间中的每个非零向量分配一个严格的正长度或大小。
从几何上来说，范数是向量的长度或大小。例如，绝对值是实数集上的一个范数。
另一方面，半范数（seminorm）可以为非零的向量赋予零长度。

向量空间上的半范数必须满足以下条件：
1. 半正定性（即非负性）：任何向量的范数总是非负的，对于任意向量$v$，$\|v\| >= 0$。
2. 可伸缩性（即齐次性）：对于任意标量$a$和任何向量$v$，标量乘法$av$的范数等于标量的绝对值乘以向量的范数，即$\|av\| = |a|\|v\|$。
3. 次可加性（即三角不等式）：对于任何向量$v$和$w$，向量和$u=v+w$的范数小于或等于向量$v$和$w$的范数之和，即$\|v+w\| \leq \|v\| + \|w\|$。

范数是一个半范数加上额外性质：对于任何向量$v$，如果$\|v\|=0$，则$v$必须是零向量。
换句话说，所有范数都是半范数，它可以将非零向量与零向量区分开来。

常用向量范数包括：
1. $\ell_0$范数：向量$x$中非零元素的个数，即$\|x\|_0=\sum_{i=1}^n \mathbb{I}(x_i\neq 0)$。
2. $\ell_1$范数：向量$x$中各元素绝对值之和，即$\|x\|_1=\sum_{i=1}^n |x_i|$。
3. $\ell_2$范数（即欧几里得范数）：向量$x$各元素绝对值的平方和再开方，即$\|x\|_2=\sqrt{\sum_{i=1}^n x_i^2}$。
4. $\ell_p$范数：向量$x$各元素绝对值的$p$次方和再开$p$次方，即$\|x\|_p=(\sum_{i=1}^n |x_i|^p)^{\frac{1}{p}}$。
5. $\ell_\infty$范数（即极大范数）：向量$x$中各元素绝对值的最大值，即$\|x\|_\infty=\max_{i=1,\cdots,n} |x_i|$。
6. 加权范数：设$A$是$n$阶Hermite正定矩阵，则向量$x$的加权范数定义为$\|x\|_A=\sqrt{x^TAx}$。该类范数在本书8.3.2和8.4.2中经常被使用。



## 2. 凸集合

凸集（convex set）是向量空间（如欧几里得空间）的一个子集，在该集合中，对于集合内的任意两点，连接它们的线段完全位于该集合内。换句话说，如果一个集合包含连接集合内任意两点的线段上的所有点，则该集合是凸集。

更正式地说，考虑一个向量空间$\mathcal{V}$。如果对于该空间中的任意两点$x$和$y$，以及满足$\alpha\in[0,1]$的任意标量$\alpha$，点$\alpha x+(1-\alpha)y$也属于$\mathcal{D}$，那么集合$\mathcal{D}\subseteq\mathcal{V}$是凸集。

凸集合的这种性质叫做非扩张性（non-expansiveness），对于凸集内的任何两点，连接这两点的线段完全包含在集合内。这种性质使得凸集在许多数学环境中变得非常可预测，因此更容易处理。
例如，如果你正在尝试在一个凸集内找到一个最小值或最大值（如在优化问题中），你可以确定没有局部最小值或最大值，只有一个全局值。这大大简化了搜索过程。

不仅凸集合具有非扩张性，映射到凸集的投影操作也是非扩张的，这意味着两点在凸集上的投影之间的距离小于或等于两点本身之间的距离。
形式上，对于闭合凸集合$K\subseteq\mathbb{R}^D$，投影算子$\Pi:\mathbb{R}^D\rightarrow K$定义为：
$$
\Pi(x)=\arg \min_{y\in K} \| x-y\|_2
$$
即将一个向量映射到最接近它的凸集合中的点。投影算子$\Pi$在$\ell_2$范式下是非扩张的，即对于任意$x,x'\in\mathbb{R}^D$，都有：
$$
\| \Pi(x) - \Pi(x')\|_2 \leq \| x - x'\|_2, \forall x,x'\in \mathbb{R}^D.
$$
该性质的证明如下：
让$y=\Pi(x)$，易知$x$和$K$在通过$y$的超平面$H=\{z\in\mathbb{R}^D:\langle z-y,x-y\rangle=0\}$的两侧。因此，对于$K$中的任意$u$，则有以下不等式成立：
$$
\langle x-y,u-y\rangle \le 0
$$
同理，让$y'=\Pi(x')$，对于$K$中的任意$u'$，有以下不等式成立：
$$
\langle x'-y',u'-y'\rangle \le 0
$$
此时，不妨令$u=y'$且$u'=y$，则有：
$$
\langle x-y,y'-y\rangle \le 0 \\
\langle x'-y',y-y'\rangle \le 0
$$ 
将两个不等式相加可得：
$$
\langle (x-x')+(y'-y),y'-y\rangle \le 0
$$
根据 Cauchy-Schwarz 不等式（1.14），我们有：
$$
\begin{aligned}
&\|y-y'\|_2^2 \le \langle x-x',y-y'\rangle \le \|x-x'\|_2\,\|y-y'\|_2\\
\Rightarrow &\|y-y'\|_2 \le \|x-x'\|_2 \\
\Rightarrow &\|\Pi(x) - \Pi(x')\|_2 \le \|x-x'\|_2
\end{aligned}
$$

该性质在凸优化中经常被使用，因为这种投影映射可以将一个优化问题转化为一个凸优化问题。
凸集合提供了一种良好的结构，可以实现高效的优化算法，并在许多情况下保证全局最优解。



## 2. 凸函数

凸函数（convex function）是定义在凸集上的实值函数，满足函数图像上任意两点间的线段位于或位于函数图像上方，即对于其定义域内的任意两个点 $x$ 和 $y$，以及满足$\alpha\in[0,1]$的任意标量$\alpha$，有以下不等式成立：
$$
f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha) f(y)
$$
这个不等式被称为凸性条件。

除了上述的线段定义法，凸函数还有几种等价的定义：
1. 一阶条件：如果一个定义在凸集上的函数 $f(x)$，对于其定义域内的任意两个点 $x$ 和 $y$，以下不等式成立：
$$
f(y) ≥ f(x) + \nabla f(x)^T(y - x)
$$
其中，$\nabla f(x)$ 表示函数 $f(x)$ 在点 x 处的梯度。几何上，这个条件表示函数的图像位于任意一点处的切线之上。

2. 二阶条件：如果函数 $f(x)$ 具有二次可微性，那么它是凸函数当且仅当其 Hessian 矩阵 $H_f$ 在其定义域内的所有点 $x$ 上都是半正定的（即矩阵的所有元素非负），半正定性保证了 Hessian 矩阵的所有特征值都是非负的。
这里，Hessian 矩阵$H_f$是函数 $f(x)$ 的二阶偏导数构成的方阵：
$$
\mathbf H_f= \begin{bmatrix}
  \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]
  \vdots & \vdots & \ddots & \vdots \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.
$$
其中，$x=[x_1,x_2,\cdots,x_n]$

3. Jensen不等式：如果函数$f(x)$是凸函数，则对于其定义域内的任意一组点${x_1, x_2, \cdots, x_n}$和归一化的非负权重${w_1, w_2, \cdots, w_n}$，即$\sum_{i=1}^n w_i=1$，则有：
$$
f(\sum_{i=1}^n w_i x_i) ≤ \sum_{i=1}^n w_i f(x_i)
$$

4. 上凸集定义：凸函数与凸集合的概念紧密相连，函数$f$是凸函数当且仅当其上图集（epigraph）是一个凸集。
上图集是位于函数图像上方的点的集合，定义为：
$$
epi(f) = \{(x, y) | x ∈ dom(f)，y ≥ f(x)\}
$$
其中，$dom(f)$ 是函数$f$的定义域。

[comment]: <> (是否在此给出不同定义的等价性证明？)

凸函数的一些特性包括：
1. 正比例性质：如果函数$f(x)$是凸函数，那么对于任何常数$\alpha\gt0$，函数$\alpha f(x)$也是凸函数。
2. 正移位性质：如果函数$f(x)$是凸函数，那么对于任何常数$c\gt0$，那么函数$f(x)-c$也是凸函数。
3. 加法性质：如果 $f(x)$ 和 $g(x)$ 都是凸函数，那么他们的和 $f(x)+g(x)$ 也是凸函数。



## 3. 凹函数

凹函数（concave function）的定义恰好与凸函数相反，即对于其定义域内的任意两个点 $x$ 和 $y$，以及满足$\alpha\in[0,1]$的任意标量$\alpha$，有以下不等式成立：
$$
f(\alpha x + (1-\alpha)y) \geq \alpha f(x) + (1-\alpha) f(y)
$$
这个不等式被称为凹性条件。

其他的定义与性质与凸函数类似，这里不再赘述。值得注意的是，若函数$f(x)$为凹函数，那么$-f(x)$为凹函数。
我们因此可以将凹函数问题转化为凸函数问题，从而可以使用凸函数的性质来解决凹函数问题。



## 4. 强凸函数

对于定义在凸集上的函数$f(x)$，如果它满足以下性质，则称为强凸函数：
$$
\forall x,y\in dom(f),\alpha\in[0,1],\exists \lambda\gt0\\
f(\alpha x+(1-\alpha)y)\leq \alpha f(x)+(1-\alpha)f(y)-\frac{\lambda}{2}\alpha(1-\alpha)||x-y||_2^2
$$
则称 $f(x)$ 为$\lambda$-强凸（strongly convex）函数，其中$\lambda$ 为强凸系数。

强凸函数的其他等价定义包括：

1. Hessian 矩阵条件：如果一个两次可微的函数 $f(x)$ 的 Hessian 矩阵 $H_f$ 在凸集中的所有 $x$ 上都是正定的（即矩阵的所有元素为正），则它是强凸的。

2. 梯度条件：如果一个可微函数 $f(x)$ 是强凸的，那么存在一个正常数 $m$，使得对于凸集中的任意 $x,y$，有 $||\nabla f(x) - \nabla f(y)||_2 ≥ m ||x - y||_2$ 成立。这里，$\nabla f(x)$ 表示 $f(x)$ 在点 $x$ 处的梯度。

[comment]: <> (是否在此给出不同定义的等价性证明？)

直观上，对强凸函数$f(x)$，可以在任意一点处构造一个二次函数作为其下界。这个性质使得优化算法更加高效，且具有类似**P90**中定理7.2的优良性质。

这里给出定理7.2的证明：根据强凸函数的定义，我们取$x=w,y=w^*$，然后两边除以$\alpha$可得：
$$
\begin{aligned}
&\frac{f(\alpha w+(1-\alpha)w^*)}{\alpha}\leq f(w)+\frac{1-\alpha}{\alpha}f(w^*)-\frac{\lambda}{2}(1-\alpha)||w-w^*||_2^2\\
\Rightarrow&\frac{\lambda}{2}(1-\alpha)||w-w^*||_2^2\le f(w)-f(w^*)-\frac{f(w^* +(w-w^*)\alpha)-f(w^*)}{\alpha}
\end{aligned}
$$
令$\alpha\rightarrow 0^+$，则有：
$$
\begin{aligned}
&lim_{\alpha\rightarrow 0^+}\frac{\lambda}{2}(1-\alpha)||w-w^*||_2^2\le f(w)-f(w^*)+lim_{\alpha\rightarrow 0^+}\frac{f(w^* +(w-w^*)\alpha)-f(w^*)}{\alpha}\\
\Rightarrow&\frac{\lambda}{2}||w-w^*||_2^2\le f(w)-f(w^*)+lim_{\Delta\rightarrow 0^+}\frac{f(w^* +\Delta)-f(w^*)}{\Delta}(w-w^*)\\
\Rightarrow&\frac{\lambda}{2}||w-w^*||_2^2\le f(w)-f(w^*)+\nabla f(w^*)^T(w-w^*)
\end{aligned}
$$
其中$\Delta=(w-w^*)\alpha$

因为$w^*$为最优解，所以$\nabla f(w^*)=0$，因此有：
$$
f(w)-f(w^*)\ge\frac{\lambda}{2}||w-w^*||_2^2
$$



## 5. 指数凹函数

对于函数$f(x)$，若 $\exp(f(x))$ 是凹函数，则称其为指数凹（exponentially concave）函数。注意到，当$\exp(f(x))$ 是凹函数时，$f(x)$ 不一定是凹函数。
如果函数$f(x)$是指数凹函数，则 $\exp(-f(x))$ 一定是凸函数。因此，指数凹是一种弱于强凸性质，但强于凸性质的约束。

指数凹函数的一些特性包括：
1. 正比例性质：如果函数$f(x)$是指数凹函数，那么对于任何正常数$\alpha$，函数$\alpha f(x)$也是指数凹函数。
2. 负移位性质：如果函数$f(x)$是指数凹函数且$c$是正常数，那么函数$f(x)-c$也是指数凹函数。

指数凹函数可以提供一种非常灵活和富有表现力的方式来建模各种现象，因为它可以捕捉到广泛的形状和行为。
比如，在凸优化中使用指数凹函数可以使迭代优化算法（如梯度下降或牛顿法）更快地收敛。
因此，在概率模型或存在不确定性的场景中，指数凹函数对于限制或量化不确定性非常重要。



## 6. 凸优化

凸优化（convex optimization）是优化领域的一个分支，它处理的问题是在凸函数的凸集上进行优化。凸优化涉及在满足一组凸约束条件的情况下，寻找凸目标函数的最小值。

凸优化问题的一般形式可以表示为：
$$
\begin{aligned}
&min &f_0(x)\\
&s.t. &f_i(x) \le 0,i\in[m]\\
&&g_j(x) = 0,j\in[n]
\end{aligned}
$$
其中，$f_0(x)$是凸目标函数，$f_i(x)$是凸不等式约束条件，$g_j(x)$是仿射等式约束条件。

凸优化具有一些有益的特性，使其成为一个被广泛研究和应用的领域：

全局最优性：凸优化问题具有这样的性质，即任何局部最小值也是全局最小值。这个特性确保凸优化算法找到的解是给定凸集中的最佳解。

高效算法：凸优化具有高效的算法，可以在多项式时间内找到最优解或提供接近最优的解。这些算法基于凸目标函数和约束条件的凸性质。

广泛应用：凸优化在工程学、金融学、机器学习、运筹学和信号处理等各个领域有广泛的应用。它用于解决诸如投资组合优化、信号重构、资源分配和机器学习模型训练等问题。凸优化技术，如线性规划、二次规划和半定规划，构成了许多优化算法的基础，为以计算高效方式解决复杂优化问题提供了强大的工具。

我们在这里证明第一个性质，即凸函数任何局部最优解都是全局最优解。

假设$f(x)$是凸函数，$x^*$是$f$在凸集合$\mathcal{D}$中的局部最小解。因为凸集合的性质，对于任意$y$，$y-x^*$都是一个可行的方向。因此，我们总是可以选择一个足够小的$\alpha>0$，满足：
$$
f(x^*)\leq f(x^*+\alpha(y-x^*))
$$
由$f$的凸函数性质可知:
$$
f(x^*+\alpha(y-x^*))=f((1-\alpha)x^*+\alpha y)\leq (1-\alpha)f(x^*)+\alpha f(y)
$$
结合以上两式，我们有：
$$
\begin{aligned}
&f(x^*)\leq (1-\alpha)f(x^*)+\alpha f(y)\\
\Leftrightarrow &f(x^*)\leq f(y)
\end{aligned}
$$
因为$y$是凸集合$\mathcal{D}$中的任意点，所以$x^*$是全局最小解。
对于$f(x)$的全局最大解，我们可以通过考虑函数$-f(x)$的局部最小解来得到类似结论。



## 7. 仿射

仿射变换（Affine transformation），又称仿射映射，是指在几何中，对一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。
假如该线性映射被表示为一矩阵$A$，平移被表示为向量$\vec{b}$，则仿射映射$f$可被表示为：
$$
\vec{y}=f(\vec{x})=A\vec{x}+\vec{b}
$$
此处$A$被称为仿射变换矩阵或投射变换矩阵。

仿射变换具有以下性质：
1. 点之间的共线性：在同一条直线上的三个或更多的点（称为共线点）在变换后依然在同一条直线上（共线）。
2. 直线的平行性：两条或以上的平行直线，在变换后依然平行。
3. 集合的凸性：凸集合变换后依然是凸集合，且最初的极值点被映射到变换后的极值点集。
4. 平行线段的长度比例恒定：两条由点$p_1,p_2,p_3,p_4$定义的平行线段,其长度比例在变换后保持不变，即$\frac{\overrightarrow{p_1p_2}}{\overrightarrow{p_3p_4}}=\frac{\overrightarrow{f(p_1)f(p_2)}}{\overrightarrow{f(p_3)f(p_4)}}$
5. 质心位置恒定：不同质量的点组成集合的质心位置不变。

仿射集（affine set）是指欧氏空间$R^n$中具有以下性质的点集$S$，对任意$x,y\in S$，以及$\forall\lambda\in[0,1]$，有$(1-\lambda)x+\lambda y\in S$。
易证，包含原点的仿射集$S$是$R^n$的子空间。

仿射集（affine hull/span）是所有包含集合$S$的仿射集的全体的交集，也是集合$S$中的元素的不断用直线连结后的元素全体。它是包含集合$S$的最小仿射集合，记为$aff(S)$，即：
$$
aff(S) = \{\sum_{i=1}^k \alpha_i x_i|k>0,x_i\in S,\alpha_i\in R,\sum_{i=1}^k \alpha_i=1\}
$$
仿射包具有以下性质：
1. aff(aff(S)) = aff(S)
2. aff(S+T) = aff(S) + aff(T)
3. 若$S$为有限维度，则aff(S)为闭集合。



## 8. Slater条件/定理

关于强对偶性的讨论，原书已有详细说明，故不再赘述。
这里着重讨论下11页左下角附注提到的slater条件，即：

存在一点$x\in relint(D)$，该点又叫做Slater向量，有：
$$
\begin{aligned}
&f_i(x)\lt0,i\in[m]\\
\end{aligned}
$$
此处$D=\cap_0^m dom(f_i)$。

其中，$relint(D)$为$D$的相对内部，即其仿射包的内部所有点，即$relint(D)=int(aff(D))$。

相应地，当满足Slater条件且原始问题为凸优化问题时，
1. 强对偶性成立。
2. 对偶最优解集合非空且有界。

这就是Slater定理。

$Proof.$

首先证明对偶间隙（Duality Gap）为零，即原始问题与对偶问题的目标函数值之差$p^*-d^*=0$。
考虑集合$\mathcal{V}\subset\mathbb{R}^m\times\mathbb{R}$满足：
$$
\mathcal{V}:=\{(u,w)\in\mathbb{R}^m\times\mathbb{R}:f_0(x)\le w,f_i(x)\le u_i,\forall i\in[m],\forall x\}
$$
集合$\mathcal{V}$有以下几个性质：
1. 它是凸集合，由$f_i,i\in\{0\}\cup[m]$的凸性质可知。
2. 若$(u,w)\in\mathcal{V}$，且$(u',w')\succeq(u,w)$，则$(u',w')\in\mathcal{V}$。

易证向量$(0,p^*)\notin int(\mathcal{V})$，否则一定存在$\varepsilon>0$，使得$(0,p^*-\varepsilon)\in int(\mathcal{V})$，这明显与$p^*$为最优解矛盾。
因此，必有$(0,p^*)\in \partial\mathcal{V}$或$(0,p^*)\notin\mathcal{V}$。
应用支撑超平面定理（定理23），我们可以得知，存在一个非零点$(\lambda,\lambda_0)\in\mathbb{R}^m\times\mathbb{R}$，满足以下条件：
$$
\begin{equation}
(\lambda,\lambda_0)^T(u,w)=\lambda^Tu+\lambda_0w\ge\lambda_0p^*,\forall(u,w)\in\mathcal{V}
\end{equation}
$$
在此情况下，必然有 $\lambda \succeq 0$ 和 $\lambda_0 \geq 0$。
这是因为，如果在 $\lambda$ 和 $\lambda_0$ 的分量中出现任何负数，根据集合 $\mathcal{V}$ 的性质二，$(u, w)$ 的分量可以在集合 $\mathcal{V}$ 内取得任意大的值，从而导致式（1）不一定成立。
因此，我们只需要考虑两种情况：
1. $\lambda_0=0$：此时根据（1），我们可知
$$
\begin{equation}
\inf_{(u,w)\in\mathcal{V}}\lambda^Tu=0
\end{equation}
$$。
另一方面，根据$\mathcal{V}$的定义，$\lambda\succeq0$且$\lambda\neq0$，可得：
$$
\inf_{(u,w)\in\mathcal{V}}\lambda^Tu=\inf_{x}\sum_{i=1}^m\lambda_i f_i(x)\le\sum_{i=1}^m\lambda_i f_i(\bar{x})\lt0
$$
其中，$\bar{x}$是Slater向量，而最后一个不等式是依据Slater条件得出的。
此结论刚好与（2）矛盾，因此$\lambda_0\neq0$。

2. $\lambda_0\gt0$：我们对（1）左右两边除以$\lambda_0$，有：
$$
\inf_{(u,w)\in\mathcal{V}}\{\tilde\lambda^Tu+w\}\ge p^*
$$
此处，$\tilde\lambda:=\frac{\lambda}{\lambda_0}\succeq0$。

考虑拉格朗日函数$L:\mathbb{R}^n\times\mathbb{R}^n\rightarrow\mathbb{R}$：
$$
L(x,\tilde\lambda):=f_0(x)+\sum_{i=1}^m\tilde\lambda_if_i(x)
$$
其对偶函数为：
$$
g(\tilde\lambda):=\inf_{x}L(x,\tilde\lambda)\ge p^*
$$
其对偶问题为：
$$
\max_{\lambda}g(\lambda),\lambda\succeq0
$$
因此，我们可以得到：$d^* \geq p^*$。根据弱对偶性，我们知道 $d^* \leq p^*$，从而可以推断出 $d^* = p^*$。

其次证明对偶问题最优解集合非空且有界。对于任意对偶最优解$\tilde\lambda\succeq0$，有：
$$
\begin{aligned}
d^*=g(\tilde\lambda)&=\inf_{x}\{f_0(x)+\sum_{i=1}^m\tilde\lambda_if_i(x)\}\\
&\le f_0(\bar{x})+\sum_{i=1}^m\tilde\lambda_if_i(\bar{x}) \\
&\le f_0(\bar{x})+\max_{i\in[m]}\{f_i(\bar{x})\}[\sum_{i=1}^m\tilde\lambda_i]
\end{aligned}
$$
因此，我们有：
$$
\min_{i\in[m]}\{-f_i(\bar{x})\}[\sum_{i=1}^m\tilde\lambda_i]\le f_0(\bar{x})-d^*
$$
进而，有：
$$
\|\tilde\lambda\|\le\sum_{i=1}^m\tilde\lambda_i\le\frac{f_0(\bar{x})-d^*}{\min_{i\in[m]}\{-f_i(\bar{x})\}}\lt\infty
$$
其中，最后一个不等式是依据Slater条件得出的。



## 9. KKT条件

KKT条件（Karush-Kuhn-Tucker条件）在凸优化领域具有至关重要的地位。虽然在原书的第12至13页中对其进行了基本解释，但在这里我们将进行更为深刻的分析。
在KKT条件中，符号$\lambda_i,i\in[m]$和$\mu_i,i\in[n]$被视为KKT乘数子。
特别地，在$m=0$的情况下，也就是不等式约束条件不存在的情况下，KKT条件退化为拉格朗日条件。此时，KKT乘数子也被称为拉格朗日乘数子。
下面给出KKT条件的证明：

$Proof.$

首先，对于 $x^*,(\mu^*,\lambda^*)$ 满足KKT条件等价于它们构成一个纳什均衡。
固定$(\mu^*,\lambda^*)$，并变化$x$，均衡等价于拉格朗日函数在$x^*$处的梯度为0，即主问题稳定性（Stationarity）。
固定$x$，并变化$(\mu^*,\lambda^*)$，均衡等价于主问题约束（feasibility）和互补松弛条件。
充分性: 解对 $x^*,(\mu^*,\lambda^*)$ 满足KKT条件，因此是一个纳什均衡，从而消除了对偶间隙。
必要性: 任何解对 $x^*,(\mu^*,\lambda^*)$ 必然消除对偶间隙，因此它们必须构成一个纳什均衡，因此它们满足KKT条件。

这里对KKT和Slater条件进行区分：
1. KKT条件是一组用于确定约束优化问题中解的最优性的条件。它们通过将约束纳入条件，将无约束优化中将目标函数的梯度设为零的想法扩展到约束优化问题中。
Slater条件是凸优化中用于确保强对偶性的特定约束条件，即主问题和对偶问题最优解的等价性。
2. KKT条件包括对偶问题约束，互补松弛条件，主问题约束和稳定性。它们整合了目标和约束函数的梯度，以及KKT乘数子，以形成最优性条件。
Slater条件要求存在一个严格可行点，即严格满足所有不等式约束的点。
3. 当点满足KKT条件时，它表明了该问题的局部最优解。它们弥合了主问题和对偶问题之间的差距，对于分析和解决约束优化问题至关重要。
当满足Slater条件时，它确保了凸优化问题中的强对偶性，这对于简化和解决这些问题至关重要。它不直接提供最优性条件，但为强对偶性铺平了道路，然后利用强对偶性来寻找最优解。
4. KKT条件较为通用，适用于更广泛的优化问题类别，包括非凸问题。
Slater条件特定于凸优化问题，用于确保这些问题中的强对偶性。
5. 对于凸且可微问题，满足KKT条件意味着最优性和强对偶性。相反地，最优性和强对偶性意味着所有问题的KKT条件得到满足。
当Slater条件成立时，KKT条件是最优解的充要条件，此时强对偶性成立。

KKT条件和Slater条件通常被归类为“正则条件”（regularity condition）或“约束资格”（constraint qualification）。
这些条件为优化问题提供了一个结构化的框架，以便在约束的情况下分析和确定解的最优性。更多的正则条件详见[论文](https://link.springer.com/chapter/10.1007/BFb0120988)。



## 10. 连续性

连续性（continuity）表示该函数的在某处的变化不会突然中断或跳跃。
形式上，如果函数$f(x)$在$x = a$处满足以下任意条件，则称其在该点连续：
1. 函数$f(x)$在$x = a$处有定义。
2. 当$x$趋近于$a$时，$f(x)$的极限存在且等于$f(a)$。

连续性意味着输入的微小变化会导致输出的微小变化，如果一个函数在其定义域的每个点上都是连续的，那么它被称为连续函数。

Lipschitz连续性是连续性的一个更强的形式，它要求函数在变化速度方面有界。具体而言，如果存在一个正常数L，使得函数在任意两点处的函数值之间的绝对差小于等于L乘以这两点之间的距离，那么该函数被称为$L$-Lipschitz连续，即：
$$
\forall x,y\in dom(f),\exists L>0\\
||f(x)-f(y)||_2 \leq L||x-y||_2
$$
这里，$L$ 被称为Lipschitz常数，表示函数的最大变化率。如果$L$较大，函数可以快速变化，而较小的$L$表示更渐进的变化。

事实上，如果一个函数的导数有界，那么它一定是Lipschitz连续的；反之，如果一个可微函数是Lipschitz连续的，那么它的导数一定有界。这里给出证明：
1. 如果函数$f(x)$的导数有界，即存在常数$L\ge0$，使得对于任意$x$，有$|f'(x)|\leq L$。
根据微分中值定理，对于任意$x\le y$，存在$c\in[x,y]$，使得：
$$
\begin{aligned}
&\|f(x)-f(y)\|_2=\|f'(c)\|_2\|x-y\|_2\\
\Rightarrow&\|f(x)-f(y)\|_2\le L \|x-y\|_2
\end{aligned}
$$
此时，函数是$L$-Lipschitz连续的。

2. 如果函数$f(x)$是$L$-Lipschitz连续的，即对于任意$x,y$，有
$$
\|f(x)-f(y)\|_2\le L\|x-y\|_2
$$
根据微分中值定理，对于任意$x\le y$，存在$c\in[x,y]$，使得：
$$
\|f(x)-f(y)\|_2=\|f'(c)\|_2\|x-y\|_2
$$
不妨令$x\rightarrow y$，则$c\rightarrow y$，因为$f(y)$具有可微的性质，可得：
$$
\|f'(y)\|_2=\|\lim_{x\rightarrow y}\frac{f(x)-f(y)}{x-y}\|_2=\lim_{x\rightarrow y}\frac{\|f(x)-f(y)\|_2}{\|x-y\|_2}\le\lim_{x\rightarrow y}L=L
$$
因为$y$的任意性，所以函数的导数是有界的。

连续性关注函数图像中的跳跃或中断的缺失，而Lipschitz连续性关注函数的变化速度。因此，Lipschitz连续性是比连续性更严格的条件。
一个连续函数不一定是Lipschitz连续的，因为连续性不要求函数的变化速度有界。然而，一个Lipschitz连续的函数必然是连续的，因为Lipschitz连续性蕴含着连续性。

Lipschitz连续性的性质在数学的各个领域中经常被应用，例如分析、优化和微分方程的研究。它在保证某些数学问题的解的存在性、唯一性和稳定性方面起着关键作用。



## 11. 光滑性

在数学分析中，函数的光滑性（smoothness）是通过函数在某个域（称为可微性类）上的连续导数的数量来衡量的属性。
最基本的情况下，如果一个函数在每个点上都可导（因此连续），则可以认为它是光滑的。

在优化理论中，L-光滑函数是指具有$L$-Lipschitz连续性的函数，这意味着函数的梯度的幅度在其定义域中的任何地方都被L所限制。
形式上，函数$f(x)$被称为$L$-光滑，则必须满足以下不等式：
$$
\forall x,y\in dom(f),\exists L>0\\
f(y) \le f(x) + \nabla f(x)(y-x) + \frac{L}{2}||y-x||_2^2
$$
这里，$L$被称为光滑系数。上式表明，对光滑函数$F(x)$，可以在任意一点处构造一个二次函数作为其上界。

如果一个函数的梯度是$L$-Lipschitz连续的，那么它就是L-光滑的。因此，L-光滑性是比连续性更强的条件。换句话说，所有L-光滑的函数都是连续的，但并非所有连续函数都是L-光滑的。
光滑性关注导数的存在和规则性，而Lipschitz连续性关注限制函数的变化速度。Lipschitz连续性保证变化速度有界，而光滑性确保函数具有定义良好的导数。

L-光滑函数在优化中非常有用，因为它们可以加快梯度下降算法的收敛速度。此外，L-光滑性是许多优化算法的重要特性，包括随机梯度下降算法。



## 12. 次梯度

次梯度（subgradient）是凸函数导数的一种推广形式，某些凸函数在特定区域内导数可能并不存在，但我们依旧可以用次梯度来表示此区域内函数变化率的下界。
形式上，对于凸函数 $f(x)$中任意点$x$， 在点$x_0$处的次梯度$c$必须满足以下不等式：
$$
f(x)-f(x_0)\ge c(x-x_0)
$$
根据微分中值定理的逆命题，我们可知$c$通常在$[a,b]$之间取值，其中$a,b$是函数$f(x)$在$x_0$处的左右导数，即：
$$
a=\lim_{x\rightarrow x_0^-}\frac{f(x)-f(x_0)}{x-x_0},b=\lim_{x\rightarrow x_0^+}\frac{f(x)-f(x_0)}{x-x_0}
$$
此时，次梯度$c$的集合$[a,b]$被称为次微分，即$\partial f(x_0)$。当$a=b$时，次梯度$c$退化为导数。

次梯度在机器学习领域得到了广泛的应用，特别是在训练支持向量机（SVM）和其他具有非可微损失函数的模型中。
它们还构成了随机次梯度方法的基础，这些方法对于大规模的机器学习问题非常有效。



## 13. 对偶空间

线性泛函（linear form）是指由向量空间$V$到对应标量域$k$的线性映射，满足加法和数乘的性质，即对于任意向量 $x,y\in V$ 和标量 $\alpha\in k$，有：
$$
\begin{aligned}
&f(x+y)=f(x)+f(y)\\
&f(\alpha x)=\alpha f(x)
\end{aligned}
$$
所有$V$到$k$的线性泛函构成的集合被称为$V$的对偶空间（dual space），记为$V^*=Hom_k(V,k)$，对偶空间中的元素被称为对偶向量。



## 15. 勒让德变换

将函数转换为另一种函数，常常可以改变其定义域和属性，从而使问题变得更简单或更易于分析。
其中，勒让德变换（Legendre transform）常用于将一组独立变量转换为另一组独立变量，特别是在经典力学和热力学中。
以下是勒让德变换的基本概念和步骤：

1. **定义函数**：我们有一个凸函数 $f(x)$，其自变量为 $x$。
2. **定义共轭变量**：定义新的变量 $p$，它是原函数 $f(x)$ 的导数，即 $p = \frac{df(x)}{dx}$。
3. **定义共轭函数**：定义新的函数 $g(p)$，其形式为：
   $$g(p) = x \cdot p - f(x)$$
   这里，$x$ 是 $f(x)$ 的自变量，同时也是 $g(p)$ 的隐含变量。

4. **变换关系**：通过勒让德变换，我们从原来的函数 $f(x)$ 得到了新的函数 $g(p)$，这个新的函数 $g(p)$ 依赖于共轭变量 $p$。



## 15. 共轭函数

凸共轭（convex conjugate）是勒让德变换的一种推广，因此也被称为勒让德-芬谢尔变换（Legendre-Fenchel transform）。
通过凸共轭变换，原函数可以转换为凸函数，从而利用凸函数的性质来解决原问题。

形式上，对于函数$f(x)$，其共轭函数$f^*(y)$定义为：
$$
f^*(y)=\sup_{x\in dom(f)}(y^Tx-f(x))
$$
其中，$dom(f)$是函数$f(x)$的定义域。

共轭函数有一些有用的性质，包括：
1. 凸性：函数$f(x)$的共轭函数$f^*(y)$一定是凸函数。证明如下：
$$
\begin{aligned}
f^*(\lambda y_1+(1-\lambda)y_2) &= \sup_{x\in dom(f)}\{x(\lambda y_1+(1-\lambda)y_2)-f(x)\}\\
&\le\lambda\sup_{x\in dom(f)}\{xy_1-f(x)\}+(1-\lambda)\sup_{x\in dom(f)}\{xy_2-f(x)\}\\
&=\lambda f^*(y_1)+(1-\lambda)f^*(y_2)\\
\end{aligned}
$$
其中的不等式缩放利用了本章定理19的内容。

2. 逆序性：对定义域中所有元素$x$，有$f(x)\le g(x)$，那么一定有$f^*(y)\ge g^*(y)$。证明如下：

因为$f(x)\le g(x)$，有$xy-f(x)\ge xy-g(x)$。两边同时取上界，根据定义有：
$$
f^*(y)=sup_{x\in dom(f)}\{xy-f(x)\}\ge sup_{x\in dom(f)}\{xy-g(x)\}=g^*(y)
$$

3. 极值变换：若$f$可微，则对于$\forall y$，则有：
$$
f^*(y)\le f^*(\nabla f(x))=\nabla f^*(x)^Tx-f(x)=-[f(x)+\nabla f(x)^T(0-x)]
$$
此性质即书中的（1.10），此处给出完整证明：

为了在$f^*$的定义中找到上确界，我们对右侧$x$求导，并将其设置为零以找到极大值点：
$$
\frac{d}{dx}(xy−f(x))=y−\nabla f(x)=0
$$
此时有$y=\nabla f(x)$，得证。



##  16. σ-代数

σ-代数（或者σ-域）是数学中测度论和概率论的一个重要概念。σ-代数是一个满足特定封闭性质的集合族，使得我们能够对这些集合定义一致的测度（例如概率）。
具体来说，σ-代数是一个集合族，满足以下三个性质：

1. **包含全集**：如果 $\mathcal{F}$ 是一个 σ-代数，定义在集合 $X$ 上，那么 $X$ 本身属于 $\mathcal{F}$，即 $X \in \mathcal{F}$。
2. **对补集封闭**：如果 $A$ 是 $\mathcal{F}$ 中的一个集合，那么它的补集 $X \setminus A$ 也属于 $\mathcal{F}$，即 $A \in \mathcal{F} \implies X \setminus A \in \mathcal{F}$。
3. **对可数并封闭**：如果 $A_1, A_2, A_3, \ldots$ 是 $\mathcal{F}$ 中的集合，那么它们的可数并集 $\bigcup_{i=1}^{\infty} A_i$ 也属于 $\mathcal{F}$，即 $A_i \in \mathcal{F}$ 对所有 $i \in \mathbb{N}$，则 $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$。

σ-代数的概念在测度论中尤为重要，因为它为定义测度（measure）提供了必要的框架。一个测度是定义在 σ-代数上的集合函数，用于度量集合的“大小”。
在概率论中，σ-代数用于定义事件空间，从而定义概率测度。

### 过滤

σ-代数 $\mathcal{F}$ 是一个固定的集合族，满足特定的封闭性质，表示我们在某一时刻可以知道的所有信息。
而过滤（filtration）则是关于如何随着时间推移而观察信息的一个概念，通常与随机过程（stochastic processes）有关。
具体来说，过滤是一个按时间参数索引的 σ-代数序列 $\{\mathcal{F}_t\}_{t \in T}$，表示随时间变化的可观测事件的集合，并满足以下性质：

1. **每个 $\mathcal{F}_t$ 是一个 σ-代数**：对于每个时刻 $t$，$\mathcal{F}_t$ 是定义在某个固定集合 $X$ 上的一个 σ-代数。
2. **单调性**：对于任意的 $t_1 \leq t_2$，有 $\mathcal{F}_{t_1} \subseteq \mathcal{F}_{t_2}$。这意味着随着时间的推移，所包含的信息只会增加，不会减少。



## 17. 鞅

鞅（Martingale）是概率论中的一个重要概念，用于描述某些类型的随机过程。鞅过程的特点是，它的未来期望值在已知当前信息的条件下等于当前值。

### 形式化定义

设 $\{X_t\}$ 是一个随机过程，$\{\mathcal{F}_t\}$ 是一个随时间 $t$ 变化的过滤（即包含随时间增加的所有信息的 σ-代数的序列）。
当这个随机过程 $\{X_t\}$ 是鞅时，必须满足以下条件：

1. **适应性（Adaptedness）**：对于每一个 $t$，$X_t$ 是 $\mathcal{F}_t$-可测的（即 $X_t$ 的值在时间 $t$ 时刻是已知信息的函数）。
2. **积分性（Integrability）**：对于所有 $t$，$E[|X_t|] < \infty$。
3. **鞅性质（Martingale Property）**：对于所有 $t$ 和 $s \geq t$，有$E[X_s | \mathcal{F}_t] = X_t$。这意味着在已知当前时刻 $t$ 的信息 $\mathcal{F}_t$ 条件下，未来某个时刻 $s$ 的期望值等于当前时刻 $t$ 的值。

### 直观解释
鞅的定义保证了在已知当前信息的条件下，未来值的期望等于当前值，这反映了一种“无偏性”。因此，鞅过程可以被看作是一种“公平游戏”。设想一个赌徒在一个赌场中进行赌博，如果这个赌徒的资金变化形成一个鞅过程，那么在任何时刻，给定当前的资金情况，未来资金的期望值都是当前的资金，这表示没有系统性的赢或输的趋势。

### 举例说明
考虑一个简单的随机游走过程，其中 $X_{t+1} = X_t + Z_{t+1}$，其中 $Z_{t+1}$ 是一个独立同分布的随机变量，取值为 $+1$ 或 $-1$，且概率各为 $50\%$。在这种情况下，如果我们设 $X_0 = 0$，那么 $\{X_t\}$ 是一个鞅，因为每一步的期望值都是零。

### 鞅的类型

除了标准的鞅，还有两个相关的概念：
1. 超鞅（Submartingale）：如果对于所有 $t$ 和 $s \geq t$，有 $E[X_s | \mathcal{F}_t] \geq X_t$，则称 $\{X_t\}$ 为超鞅（或上鞅）。
2. 亚鞅（Supermartingale）：如果对于所有 $t$ 和 $s \geq t$，有 $E[X_s | \mathcal{F}_t] \leq X_t$，则称 $\{X_t\}$ 为亚鞅（或下鞅）。

这里给出一个区分超鞅和亚鞅的记忆方法：“生活是一个超鞅：随着时间的推进，期望降低。”

### 鞅差序列

鞅差 $D_t$ 被定义为为：$D_t = X_t - X_{t-1}$，鞅差序列（Martingale Difference Sequence）$\{D_t\}$ 则满足以下条件：

1. 适应性（Adaptedness）：对于每一个 $t$，$D_t$ 是 $\mathcal{F}_t$-可测的。
2. 零条件期望（Zero Conditional Expectation）：对于所有 $t$，有 $E[D_t | \mathcal{F}_{t-1}] = 0$，即在已知过去的信息 $\mathcal{F}_{t-1}$ 的条件下，$D_t$ 的条件期望为零。这意味着当前的观察值不提供对未来观察值的系统性偏差，即每一步的变化是纯随机的。

虽然鞅差序列中的每个元素的条件期望为零，但这并不意味着这些元素是独立的。相反，它们可以有复杂的依赖关系。鞅差序列的关键性质是每个元素在条件期望下为零，这使得它在分析鞅和集中不等式（如 Bernstein 不等式）中非常有用。



# 常用不等式

记：X 和 Y 表示随机变量，$\mathbb{E}[X]$ 表示 X 的数学期望，$\mathbb{V}[X]$ 表示 X 的方差。 

定理8到18统称为集中不等式 (Concentration Inequalities)，提供了随机变量如何偏离某个值（通常是其期望值）的界限。 
经典概率论的大数定律指出，样本数量越多，则其算术平均值就有越高的机率接近期望值。 

## 定理 1:  Jensen 不等式  
  
对于任意凸函数 $f,$ 则有:  
$$  
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]  
$$  
成立。  
  
  
  
$Proof.$  
  
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
  
  
  
## 定理 2:  Hölder 不等式  
  
$\forall p, q \in \mathbb{R}^{+}, \frac{1}{p}+\frac{1}{q}=1$，则有：  
$$  
\mathbb{E}[|X Y|] \leq\left(\mathbb{E}\left[|X|^p\right]\right)^{\frac{1}{p}}\left(\mathbb{E}\left[|Y|^q\right]\right)^{\frac{1}{q}}  
$$  
成立。  
  
$Proof.$  
  
记 $f(x), g(y)$ 分别为 $X,Y$ 的概率密度函数，  
$$  
M=\frac{|x|}{(\int_X|x|^pf(x)dx)^{\frac{1}{p}}}, N=\frac{|y|}{(\int_Y|y|^qg(y)dy)^{\frac{1}{q}}}
$$  
代入 Young 不等式：  
$$  
MN\leq \frac{1}{p}M^p+\frac{1}{q}N^q  
$$  
对这个不等式两边同时取期望：
$$
\begin{aligned}
\frac{\mathbb{E}[|XY|]}{\left(\mathbb{E}\left[|X|^p\right]\right)^{\frac{1}{p}}\left(\mathbb{E}\left[|Y|^q\right]\right)^{\frac{1}{q}}} &= \frac{\int_{XY}|xy|f(x)g(y)dxdy}{(\int_X|x|^pf(x)dx)^{\frac{1}{p}}(\int_Y|y|^qf(y)dy)^{\frac{1}{q}}}\\
&\leq \frac{\int_X|x|^pf(x)dx}{p\int_X|x|^pf(x)dx} +\frac{\int_Y|y|^qg(y)dy}{q\int_Y|y|^pg(y)dy} \\
&=\frac{1}{p}+\frac{1}{q}\\
&= 1
\end{aligned}
$$  
原不等式得证。  
  
  
  
## 定理 3: Cauchy-Schwarz 不等式  
  
特别的，$p = q = 2$ 时，Hölder不等式退化为 Cauchy-Schwarz 不等式：  
$$  
\mathbb{E}[|X Y|] \leq \sqrt{\mathbb{E}\left[X^{2}\right] \mathbb{E}\left[Y^{2}\right]}  
$$  
  
  
  
## 定理 4: Lyapunov 不等式
  
$\forall 0\lt  r \leq s$，有：  
$$  
\sqrt[r]{\mathbb{E}\left[|X|^{r}\right]} \leq \sqrt[s]{\mathbb{E}\left[|X|^{s}\right]}  
$$  
  
$Proof.$   
由Hölder不等式：  
$\forall p \geq 1:$  
$$  
\begin{aligned}  
\mathbb{E}\left[|X|^{r}\right] &=\mathbb{E}\left[|X \times 1|^{r}\right] \\  
& {\leq}\left(\mathbb{E}\left[\left(|X|^{r}\right)^p\right]\right)^{1 / p} \times 1 \\  
&=\left(\mathbb{E}\left[|X|^{r p}\right]\right)^{1 / p}  
\end{aligned}  
$$  
记 $s=r p \geq r,$ 则 :  
$$  
\mathbb{E}\left[|X|^{r}\right] \leq\left(\mathbb{E}\left[|X|^{s}\right]\right)^{r / s}  
$$  
原不等式得证。  
  
  
  
## 定理 5: Minkowski 不等式  
  
$\forall p \geq 1,$ 有：  
$$  
\sqrt[p]{\mathbb{E}\left[|X+Y|^p\right]} \leq \sqrt[p]{\mathbb{E}\left[|X|^p\right]}+\sqrt[p]{\mathbb{E}\left[|Y|^p\right]}  
$$  
  
  
$Proof.$  
由三角不等式及Hölder不等式：  
$$  
\begin{aligned}  
\mathbb{E}\left[|X+Y|^p\right] & {\leq}\mathbb{E}\left[(|X|+|Y|)|X+Y|^{p-1}\right] \\  
&= \mathbb{E}\left[|X||X+Y|^{p-1}\right]+\mathbb{E}\left[|Y||X+Y|^{p-1}\right] \\  
& {\leq}\left(\mathbb{E}\left[|X|^p\right]\right)^{1 / p}\left(\mathbb{E}\left[|X+Y|^{(p-1) q}\right]\right)^{1 / q}+\left(\mathbb{E}\left[|Y|^p\right]\right)^{1 / p}\left(\mathbb{E}\left[|X+Y|^{(p-1) q}\right]\right)^{1 / q} \\  
&= \left[\left(\mathbb{E}\left[|X|^p\right]\right)^{1 / p}+\left(\mathbb{E}\left[|Y|^p\right]\right)^{1 / p}\right] \frac{\mathbb{E}\left[|X+Y|^p\right]}{\left(\mathbb{E}\left[|X+Y|^p\right]\right)^{1 / p}}  
\end{aligned}  
$$  
化简上式即得证。  



## 定理 6: Bhatia-Davis 不等式

对 $X \in [a,b]$, 则有:  
$$  
\mathbb{V}[X] \leq (b - \mathbb{E}[X])(\mathbb{E}[X] - a) \leq \frac{(b-a)^2}{4}
$$  
成立。 

$Proof.$ 
因为 $a\leq X\leq b$，所以有:
$$  
\begin{aligned}  
0&\leq \mathbb{E}[(b-X)(X-a)] \\
&= -\mathbb{E}[X^2]-ab+(a+b)\mathbb{E}[X]
\end{aligned}
$$
因此，
$$  
\begin{aligned}  
\mathbb{V}[X] &= \mathbb{E}[X^2]-\mathbb{E}[X]^2 \\
&\leq -ab+(a+b)\mathbb{E}[X]-\mathbb{E}[X^2] \\
&=(b-\mathbb{E}[X])(\mathbb{E}[X]-a)
\end{aligned}
$$

考虑 AM-GM 不等式：
$$
xy \leq (\frac{x+y}{2})^2
$$
将$x=b-\mathbb{E}[X]$和$y=\mathbb{E}[X]-a$带入化简即得证。



## 定理 7: Union Bound（Boole's）不等式
$$
P\left(X\cup Y\right) \leq P(X) + P(Y)
$$

$Proof.$

根据概率的加法公式：
$$
P(X \cup Y) = P(X) + P(Y) - P(X \cap Y) \leq P(X) + P(Y)
$$
此处 $P(X \cap Y) \geq 0$.
  


## 定理 8: Markov 不等式  
  
若 $X \geq 0, \forall \varepsilon\gt 0,$ 有：  
$$  
P(X \geq \varepsilon) \leq \frac{\mathbb{E}[X]}{\varepsilon}  
$$   
  
$Proof.$  
$$  
\mathbb{E}[X]=\int_{0}^{\infty} x p(x) d x \geq \int_{\varepsilon}^{\infty} x p(x) d x \geq \int_{\varepsilon}^{\infty} \varepsilon p(x) d x=\varepsilon P(X \geq \varepsilon)  
$$  
  
  
  
## 定理 9: Chebyshev 不等式  
  
$\forall \varepsilon\gt 0,$ 有：  
$$  
P(|X-\mathbb{E}[X]| \geq \varepsilon) \leq \frac{\mathbb{V}[X]}{\varepsilon^{2}}  
$$  

$Proof.$  
  
利用Markov 不等式，有：
$$
P(|X-\mathbb{E}[X]| \geq \varepsilon) = P((X-\mathbb{E}[X])^2 \geq \varepsilon^{2}) \leq \frac{\mathbb{E}[(X-\mathbb{E}[X])^2]}{\varepsilon^{2}} = \frac{\mathbb{V}[X]}{\varepsilon^{2}}
$$
  
  
  
## 定理 10: Cantelli 不等式  
  
$\forall \varepsilon\gt 0,$ 有 :  
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
  
Note: Cantelli 不等式是 Chebyshev 不等式的加强版，也称单边 Chebyshev 不等式。
通过类似的构造，我们可以求得诸多比 Cantelli 不等式更严格的上界。  
  
  
  
## 定理 11: Chernoff 不等式（Chernoff 界）
  
$\forall \lambda\gt 0, \varepsilon\gt 0,$ 有 :  
$$  
P(X \geq \varepsilon) \leq \frac{\mathbb{E}\left[e^{\lambda X}\right]}{e^{\lambda \varepsilon}}  
$$  
$\forall \lambda\lt  0, \varepsilon\gt 0,$ 有 :  
$$  
P(X \leq \varepsilon) \leq \frac{\mathbb{E}\left[e^{\lambda X}\right]}{e^{\lambda \varepsilon}}  
$$  
  
$Proof.$  
应用 Markov 不等式，有：

$$  
P(X \geq \varepsilon)=P\left(e^{\lambda X} \geq e^{\lambda \varepsilon}\right) \leq \frac{\mathbb{E}\left[e^{\lambda X}\right]}{e^{\lambda \varepsilon}}, \lambda\gt 0, \varepsilon\gt 0
$$
$$  
P(X \leq \varepsilon)=P\left(e^{\lambda X} \geq e^{\lambda \varepsilon}\right) \leq \frac{\mathbb{E}\left[e^{\lambda X}\right]}{e^{\lambda \varepsilon}}, \lambda\lt 0, \varepsilon\gt 0
$$  
  

  
## 定理 11: Chernoff 不等式 (乘积形式)

对m个独立同分布的随机变量$x_i \in [0, 1], i \in [m]$，令$X = \sum_{i=1}^m X_i$，$\mu>0$且$r\leq 1$

如果$\mathbb{E}[x_i]\leq \mu$对于所有$i\leq m$都成立，有：
$$  
P(X \geq (1+r)\mu m) \leq e^{-\frac{r^2\mu m}{3}}, r \geq 0
$$  
$$  
P(X \leq (1-r)\mu m) \leq e^{-\frac{r^2\mu m}{2}}, r \geq 0
$$   
  
$Proof.$  
应用 Markov 不等式，有：
$$  
P(X\geq (1+r)\mu m) = P((1+r)^X \geq (1+r)^{(1+r)\mu m}) \leq \frac{\mathbb{E}[(1+r)^X]}{(1+r)^{(1+r)\mu m}}
$$  
根据$x_i$的独立性可知：
$$
\mathbb{E}[(1+r)^X] = \prod_{i=1}^m \mathbb{E}[(1+r)^{x_i}] \leq \prod_{i=1}^m \mathbb{E}[1+rx_i] \leq \prod_{i=1}^m 1+r\mu \leq e^{r\mu m}
$$
第二步用到了$\forall x\in [0,1]$，都有$(1+r)^x\leq 1+rx$

第三步用到了$\forall i\leq m$，都有$\mathbb{E}[x_i]\leq \mu$

第四步用到了$\forall x\in [0,1]$，都有$1+x\leq e^x$

又因为$\forall r\in [0,1]$，有$\frac{e^r}{(1+r)^{1+r}}\leq e^{-\frac{r^2}{3}}$，综上：
$$
P(X\geq (1+r)\mu m) \leq (\frac{e^{r}}{(1+r)^{(1+r)}})^{\mu m} \leq e^{-\frac{r^2\mu m}{3}}
$$

当我们把$r$替换成$-r$，根据之前的推导，且在最后一步利用$\forall r\in [0,1]$，有$\frac{e^r}{(1-r)^{1-r}}\leq e^{-\frac{r^2}{2}}$，我们可以得到第二个不等式的证明



## 定理 12: 最优 Chernoff 界

如果$X$是一个随机变量，且$\mathbb{E}e^{\lambda(X-\mathbb{E}X)} \leq e^{\phi(\lambda)},\forall \lambda \geq 0$
那么，有以下公式成立：
$$
P(X-\mathbb{E}X \geq \varepsilon) \leq e^{-\phi^*(\varepsilon)}, \varepsilon \geq 0
$$
或者
$$
P(X-\mathbb{E}X \leq (\phi^*)^{-1}(ln(1/\delta))) \geq 1 - \delta, \delta \in [0,1]
$$
其中，$\phi^*$是$\phi$的凸共轭函数，即$\phi^*(x) = \sup_{\lambda \geq 0}(\lambda x - \phi(\lambda))$。

$Proof.$

根据 Chernoff 不等式，我们有：
$$
\begin{aligned}
P(X-\mathbb{E}X \geq \varepsilon) &\leq \inf_{\lambda \geq 0} e^{-\lambda \varepsilon} \mathbb{E}[e^{\lambda(X-\mathbb{E}X)}] \\
&\leq \inf_{\lambda \geq 0} e^{\phi(\lambda)-\lambda\varepsilon} \\
&= e^{-\sup_{\lambda \geq 0}(\lambda \varepsilon - \phi(\lambda))} \\
&= e^{-\phi^*(\varepsilon)}
\end{aligned}
$$


  
## 定理 13: Hoeffding 不等式  
  
### 引理1 (Hoeffding 定理)  
若$\mathbb{E}[X] = 0, X\in[a,b]$，则$\forall \lambda \in \mathbb{R}$有：  
$$  
\mathbb{E}[e^{\lambda X}] \leq \exp\left( \frac{\lambda^2(b-a)^2}{8} \right)  
$$  
  
$Proof.$  
由于$e^x$为凸函数，则显然$\forall x\in[a,b]$：  
$$  
   e^{\lambda x} \leq \frac{b-x}{b-a}e^{\lambda a} + \frac{x-a}{b-a}e^{\lambda b}  
$$  
对上式取期望有：  
$$  
\mathbb{E}[e^{\lambda X}] \leq \frac{b-\mathbb{E}[X]}{b-a}e^{\lambda a} + \frac{\mathbb{E}[X]-a}{b-a}e^{\lambda b} = \frac{be^{\lambda a} - ae^{\lambda b}}{b - a}  
$$  
  
记$\theta = -\frac{a}{b-a} \gt  0, h = \lambda(b-a)$，则：  
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
  
$Proof.$  
  
由 Markov 不等式知， $\forall \lambda\gt 0$ :  
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
  
**定义1** (随机变量的次高斯性). 若一个期望为零的随机变量$X$其矩母函数满足，$\forall \lambda \in \mathbb{R}^+$:  
$$  
      \mathbb{E}[e^{\lambda X}] \leq \frac{\sigma^2\lambda^2}{2}  
$$  
则称$X$服从参数为$\sigma$的次高斯分布。  
  
实际上 Hoeffding 引理中的随机变量$X$服从$\frac{(b-a)}{2}$的次高斯分布， Hoeffding 引理也是次高斯分布的直接体现。次高斯性还有一系列等价定义方式，这里不是本笔记讨论的重点。  
  
次高斯分布有一个直接的性质：假设两个独立的随机变量$X_1,X_2$都是次高斯分布的，分别服从参数$\sigma_1,\sigma_2$，那么$X_1+X_2$就是服从参数为$\sqrt{\sigma_1 + \sigma_2}$的次高斯分布。这个结果的证明直接利用定义即可。  
  
  
  
### 随机变量的次指数性  
  
显然，不是所有常见的随机变量都是次高斯的，例如指数分布。为此可以扩大定义：  
**定义2** (随机变量的次指数性). 若非负的随机变量$X$其矩母函数满足，$\forall \lambda \in (0,a)$:  
$$  
      \mathbb{E}[e^{\lambda X}] \leq \frac{a}{a - \lambda}  
$$  
则称$X$服从参数为$(\mathbb{V}[X], 1/a)$的次指数分布。  
  
同样的，次高斯性还有一系列等价定义方式。一种不直观但是更常用的定义方式如下：$\exists (\sigma^2, b)$，s.t.$\forall |s| \lt  1/b$有:  
$$  
      \mathbb{E}[e^{s(X−\mathbb{E}[X])}]\leq \exp \left( \frac{s^2\sigma^2}{2} \right)  
$$  
  
常见的次指数分布包括:指数分布，Gamma 分布，以及**任何的有界随机变量**。  
类似地，次指数分布也对加法是保持的：如果$X_1,X_2$分别是服从$(\sigma_1^2,b_1)$, $(\sigma_2^2,b_2)$的次指数分布，那么$X_1+X_2$是服从$(\sigma_1^2+\sigma_2^2, \max(b_1,b_2))$的次指数分布。  
  
在高维统计的问题中需要利用次高斯分布，次指数分布的尾端控制得到一些重要的结论。
  
  
  
## 定理 14. McDiarmid 不等式  
  
对 $m$ 个独立随机变量 $X_{i} \in \mathcal{X},$ 函数 $f$ 为 差有界的，则 $\forall \varepsilon\gt 0$ 有：  
$$  
P\left(f\left(X_{1}, \cdots, X_{m}\right)-\mathbb{E}\left[f\left(X_{1}, \cdots, X_{m}\right)\right] \geq \varepsilon\right) \leq \exp \left(-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}}\right)  
$$  
  
  
$Proof.$  
  
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
  
  
  
## 定理 15: Bennett 不等式  
  
对 $m$ 个独立随机变量 $X_{i},$ 令 $\bar{X}$ 为 $X_{i}$ 均值, 若 $\exists b\gt 0,$ s.t.$|X-\mathbb{E}[X]|\lt b$  
  
则有，  
$$  
P(\bar{X}-\mathbb{E}[\bar{X}] \geq \varepsilon) \leq \exp \left(-\frac{m \varepsilon^{2}}{2\left(\sum_{i=1}^{m} \mathbb{V}\left[X_{i}\right] / m+b \varepsilon / 3\right)}\right)  
$$  
成立。  
  
Remark: Bernstein 不等式实际是 Hoeffding 不等式的加强版。对于个各随机变量独立的条件可以放宽为弱独立结论仍成立。  
  
上述几个 Bernstein 类集中不等式，更多的是在非渐近观点下看到的大数定律的表现。也即是，这些不等式更多刻画了样本均值如何集中在总体均值的附近。  
  
如果把样本均值看成是样本（数据点的函数），即令 $f\left(X_{1}, \cdots, X_{m}\right)=$ $\sum_{i=1}^{m} X_{i} / m,$ 那么 Bernstein 类不等式刻画了如下的概率：  
$$  
P\left(f\left(X_{1}, \cdots, X_{m}\right)-\mathbb{E}\left[f\left(X_{1}, \cdots, X_{m}\right)\right] \geq \varepsilon\right)  
$$  
为考察在某个泛函上也具有类似 Bernstein 类集中不等式的形式，很显然 f 需要满足一些很好的性质。这类性质有很多，但是我们尝试在一个最常见的约束下进行尝试:  
  
**Definition 3** (差有界). 函数 $f: \mathcal{X}^{m} \rightarrow \mathbb{R}, \forall i, \exists c_{i}\lt \infty,$ s.t.  
$$  
\left|f\left(x_{1}, \cdots, x_{i}, \cdots, x_{m}\right)-f\left(x_{1}, \cdots, x_{i}^{\prime}, \cdots, x_{m}\right)\right| \leq c_{i}  
$$  
则称 f 是差有界的。  
  
为此，需要引入一些新的数学工具。  
  
**Definition 4** (离散鞅). 若离散随机变量序列(随机过程)$Z_m$满足:  
  
1. $\mathbb{E}\left[\left|Z_{i}\right|\right]\lt \infty$  
2. $\mathbb{E}\left[Z_{m+1} \mid Z_{1}, \cdots, Z_{m}\right]=\mathbb{E}\left[Z_{m+1} \mid \mathcal{F}_{m}\right]=Z_{m}$  
  
则称序列 $Z_i$为离散鞅。  
  
**引理 2** (Azuma-Hoeffding 定理). 对于鞅 $Z_{i}, \mathbb{E}\left[Z_{i}\right]=\mu, Z_{1}=\mu_{\circ}$ 作鞅差序列 $X_{i}=Z_{i}-Z_{i-1}, \quad$ 且 $\left|X_{i}\right| \leq c_{i}$ 。 则 $\forall \varepsilon\gt 0$ 有：  
$$  
P\left(Z_{m}-\mu \geq \varepsilon\right)=P\left(\sum_{i=1}^{m} X_{i} \geq \varepsilon\right) \leq \exp \left(-\frac{\varepsilon^{2}}{2 \sum_{i=1}^{m} c_{i}^{2}}\right)  
$$  
  
  
$Proof.$  
  
首先，若 $\mathbb{E}[X \mid Y]=0,$ 则有 $\forall \lambda\gt 0:$  
$$  
\mathbb{E}\left[e^{\lambda X} \mid Y\right] \leq \mathbb{E}\left[e^{\lambda X}\right]  
$$  
于是，由恒等式$\mathbb{E}[\mathbb{E}[X \mid Y]]=\mathbb{E}[X]$及 Chernoff 一般性技巧 $\forall \lambda\gt 0$:
$$  
\begin{aligned}  
P\left(Z_{m}-\mu\geq\varepsilon\right) &\geq e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda\left(Z_{m}-\mu\right)}\right] \\  
& = e^{-\lambda \varepsilon} \mathbb{E}\left[\mathbb{E}\left[e^{\lambda\left(Z_{m}-\mu\right)} \mid \mathcal{F}_{m-1}\right]\right] \\  
& = e^{-\lambda \varepsilon} \mathbb{E}\left[e^{\lambda\left(Z_{m-1}-\mu\right)}\mathbb{E}\left[e^{\lambda (Z_{m}-Z_{m-1})} \mid \mathcal{F}_{m-1}\right]\right]
\end{aligned}  
$$  
  
  
又因为 $\{X_{i}\}$ 为鞅差序列，则 $\mathbb{E}\left[X_{m} \mid \mathcal{F}_{m-1}\right]=0, \mathbb{E}\left[X_{i}\right]=0$ ，再结合不等式$\mathbb{E}\left[e^{\lambda X} \mid Y\right] \leq \mathbb{E}\left[e^{\lambda X}\right]$及 Hoeffding 引理，有：  
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
  
  
  
## 定理 16: Bernstein 不等式  
  
首先定义一下参数为$b \gt 0$的单边 Bernstein 条件（One-sided Bernstein's condition），即随机变量$X$满足：
$$  
\mathbb{E} [e^{\lambda(X−EX)}] \leq \exp(\frac{\mathbb{V}[X]\lambda^2/2}{1 −b\lambda}), \forall \lambda ∈ [0,1/b)
$$  
若独立同分布的随机变量$X_1, \ldots, X_n \sim X$均满足单边 Bernstein 条件，这对于任意$\varepsilon \gt 0,\delta \in [0,1]$，有如下不等式成立：  
$$  
P(\frac{1}{n} \sum_{i=1}^{n}{X_i} - \mathbb{E}[X] \geq \varepsilon) \leq \exp \left(-\frac{n \varepsilon^{2}}{2(\mathbb{V}\left[X] + b \varepsilon\right)}\right)  
$$  
  

  
$Proof.$  
1. 首先，我们先确定 Bernstein 条件下的上尾界（或上尾界限），即：
$$
P(X - \mathbb{E}[X] \geq \varepsilon) \leq \exp(-\frac{\mathbb{V} [X]}{b^2} h(\frac{b\varepsilon}{\mathbb{V} [X]})) \leq \exp(-\frac{\varepsilon^2}{2(\mathbb{V} [X] + b\varepsilon)})
$$
其中$h(x) = 1 + x - \sqrt{1 + 2x}$。
此时，我们有：
$$
P(X - \mathbb{E}[X] \lt b\ln(1/\delta) + \sqrt{2\mathbb{V}[X] \ln(1/\delta)}) \geq 1 - \delta, \delta \in [0,1]
$$

证明：
我们令$\phi(\lambda) = \frac{a\lambda^2}{2(1 - b\lambda)}, \lambda \in [0,1/b), a = \mathbb{V} [X]$。则对于任意$\varepsilon \gt 0$，我们有$\phi(\lambda)$的凸共轭：
$$
\phi^*(\varepsilon) = \sup_{\lambda \geq 0}(\lambda \varepsilon - \phi(\lambda)) = \frac{a}{b^2} h(\frac{b\varepsilon}{a}) \geq \frac{\varepsilon^2}{2(a + b\varepsilon)}
$$
最后一步推导我们利用了不等式$h(x) \geq \frac{x^2}{2(1 + x)}, x \gt 0$，该式可通过对两侧连续求导得证。  
根据最优 Chernoff 界，我们可得上尾界：
$$
e^{-\phi^*(\varepsilon)} = \exp(-\frac{a}{b^2} h(\frac{b\varepsilon}{a})) \leq \exp(-\frac{\varepsilon^2}{2(a + b\varepsilon)})
$$
此时，令$e^{-\phi^*(\varepsilon)} = \delta$，可得$\varepsilon = b\ln(1/\delta) + \sqrt{2\mathbb{V}[X] \ln(1/\delta)})$

2. 接下来，我们证明一个引理：

若$\mathbb{E} e^{\lambda (X - \mathbb{E} X)} \leq e^{\phi(\lambda)}, \lambda \geq 0$，那么对于任意正整数$n$，我们有：
$$
P(\frac{1}{n}\sum_{i=1}^{n} X_i - \mathbb{E} X \geq \varepsilon) \leq e^{-n \phi^*(\varepsilon)}, \varepsilon \geq 0
$$
亦或者：
$$
P(\frac{1}{n}\sum_{i=1}^{n} X_i - \mathbb{E} X \lt (\phi^*)^{-1} (\frac{ln(1/\delta)}{n})) \geq 1 - \delta, \delta \in [0,1]
$$

证明：
$$
\mathbb{E} e^{\frac{\lambda}{n} \sum_{i=1}^{n} ({X_i} - \mathbb{E}[X_i])} = \prod_{i=1}^n \mathbb{E} e^{\frac{\lambda}{n} ({X_i} - \mathbb{E}[X_i])} 
\leq e^{n \phi(\lambda/n)}
= e^{\psi(\lambda)}
$$
其中，我们定义$\psi(\lambda) := n\phi(\lambda/n)$，可得：
$$
\psi^*(\varepsilon) = \sup_{\lambda \geq 0}(\lambda \varepsilon - \psi(\lambda)) = n \sup_{\lambda \geq 0}(\varepsilon \lambda/n - \phi(\lambda/n))
= n \sup_{\lambda \geq 0} (\lambda \varepsilon - \phi(\lambda))= n\phi^*(\varepsilon)
$$
根据最优 Chernoff 界即可得证。

3. 我们考虑 Bernstein 不等式的左边，可知：
$$
\mathbb{E} e^{\frac{\lambda}{n} \sum_{i=1}^{n} ({X_i} - \mathbb{E}[X_i])} \leq \prod_{i=1}^n \mathbb{E} e^{\frac{\lambda}{n} ({X_i} - \mathbb{E}[X_i])}
\leq \prod_{i=1}^n \exp(\frac{\mathbb{V}[X_i] (\lambda/n)^2}{2(1 - b(\lambda/n))}) 
= \exp(\frac{\mathbb{V}[\frac{1}{n} \sum_{i=1}^n X_i] (\lambda/n)^2}{2(1 - b(\lambda/n))})
$$
应用以上引理即可得：
$$
P(\frac{1}{n} \sum_{i=1}^{n}{X_i} - \mathbb{E}[X] \geq \varepsilon) \leq \exp(-\frac{n\mathbb{V} [X]}{b^2} h(\frac{b\varepsilon}{\mathbb{V} [X]})) \leq \exp(-\frac{n\varepsilon^2}{2(\mathbb{V} [X] + b\varepsilon)})
$$


## 定理 17: Azuma（Azuma–Hoeffding） 不等式  
  
对于均值为$Z_0=\mu$的鞅差序列$\{Z_m,m\geq 1\}$，若$|Z_i-Z_{i-1}|\leq c_i$，则$\forall \varepsilon\gt 0$，有
$$
\begin{aligned}
P\left(Z_{m}-\mu\geq\varepsilon\right) &\leq\exp\left(-\frac{\varepsilon^{2}}{2\sum_{i=1}^{m} c_{i}^{2}}\right)\\
P\left(Z_{m}-\mu\leq-\varepsilon\right) &\leq\exp\left(-\frac{\varepsilon^{2}}{2\sum_{i=1}^{m} c_{i}^{2}}\right)\\
\end{aligned}
$$

$Proof.$

注意到Azuma不等式要求在鞍差序列上的对称界限，即$-c_i\leq Z_i-Z_{i-1}\leq c_i$。因此，如果已知的界限是非对称的，即$a_i\leq Z_i-Z_{i-1}\leq b_i$，那么为了使用Azuma不等式，我们需要选择$c_i=\max(|a_i|,|b_i|)$，这可能会浪费关于$X_t - X_{t-1}$的有界性的信息。然而，我们可以通过以下Azuma不等式的一般形式来解决这个问题。

**引理 1** (Doob分解定理)：

不妨让$(\Omega, \mathcal{F}, \mathbb{P})$表示一个概率空间，$I=\{0,1,2,...,N\},N\in\mathbb{N}$是一个索引集合，$(\mathcal{F}_n)_{n \in I}$是F的一个过滤器，$X=(X_n)_{n \in I}$是一个适应的随机过程，且对于任意$n \in I$，$E[|X_n|]\lt \infty$。则存在一个适应的随机过程$M=(M_n)_{n \in I}$和一个$A_0=0$的可积可预测的随机过程$A=(A_n)_{n \in I}$，满足：$X_n=M_n+A_n,n\in I$。

详细证明参考[文章](https://almostsuremath.com/2011/12/30/the-doob-meyer-decomposition/)

根据Doob分解引理，我们可以将超鞍$X_t$分解成$X_t = Y_t + Z_t$，此时$\{Y_t,F_t\}$是鞍差序列，$\{Z_t,F_t\}$是一个非递增的可预测序列。在Azuma不等式的一般性形式中，有$A_t \leq X_t - X_{t-1} \leq B_t$ 且 $B_t - A_t \leq c_t$，此时：
$$
\begin{aligned}
-(Z_t - Z_{t-1}) + A_t \leq Y_t - Y_{t-1} \leq -(Z_t - Z_{t-1}) + B_t 
\end{aligned}
$$

应用 Chernoff 不等式，对于$\forall\varepsilon\gt 0$，有：

$$
\begin{aligned}
P(Y_n-Y_0 \geq \varepsilon)
& \leq \underset{s\gt 0}{\min} \ e^{-s\varepsilon} \mathbb{E} [e^{s (Y_n-Y_0) }] \\
& = \underset{s\gt 0}{\min} \ e^{-s\varepsilon} \mathbb{E} \left[\exp \left( s \sum_{t=1}^{n}(Y_t-Y_{t-1}) \right) \right] \\
& = \underset{s\gt 0}{\min} \ e^{-s\varepsilon} \mathbb{E} \left[\exp \left( s \sum_{t=1}^{n-1}(Y_t-Y_{t-1}) \right) \right] \mathbb{E} \left[\exp \left( s(Y_n-Y_{n-1} ) \mid \mathcal{F}_{n-1} \right) \right]
\end{aligned}
$$

1. $\left\{Y_t\right\}$是鞍差序列，因此$\mathbb{E}[Y_t - Y_{t-1} \mid \mathcal{F}_{t-1}]=0$。
2. $\left\{Z_t\right\}$是一个可预测序列，因此$-(Z_t - Z_{t-1}) + A_t$和$-(Z_t - Z_{t-1}) + B_t$都是$\mathcal{F}_{t-1}$可测量的。

应用 Hoeffding 引理，有：
$$
\begin{aligned}
\mathbb{E} \left[\exp \left(s(Y_t-Y_{t-1}) \mid \mathcal{F}_{t-1} \right) \right] \leq
\exp \left(\frac{s^2 (B_t - A_t)^2}{8} \right)
\leq
\exp \left(\frac{s^2 c_t^2}{8} \right)
\end{aligned}
$$

重复这一步骤，我们可以得到：

$$
\begin{aligned}
\text{P}(Y_n-Y_0 \geq \varepsilon)
\leq
\underset{s\gt 0}{\min} \ e^{-s\varepsilon} \exp \left(\frac{s^2 \sum_{t=1}^{n}c_t^2}{8}\right)
\end{aligned}
$$

当$s = \frac{4 \varepsilon}{\sum_{t=1}^{n}c_t^2}$时，上式右端取得极小值：
$$
\begin{aligned}
\text{P}(Y_n-Y_0 \geq \varepsilon)
\leq
\exp \left(-\frac{2 \varepsilon^2}{\sum_{t=1}^{n}c_t^2}\right)
\end{aligned}
$$

因为$X_n - X_0 = (Y_n - Y_0) + (Z_n - Z_0)$，且由$\{Z_n\}$的非增性得到$Z_n - Z_0 \leq 0$，因此由$\left\{X_n - X_0 \geq \varepsilon\right\}$可推导出$\left\{Y_n - Y_0 \geq \varepsilon\right\}$。

因此，
$$
\begin{aligned}
\text{P}(X_n-X_0 \geq \varepsilon)
\leq
\text{P}(Y_n-Y_0 \geq \varepsilon)
\leq
\exp \left(-\frac{2 \varepsilon^2}{\sum_{t=1}^{n}c_t^2}\right)
\end{aligned}
$$

同理可证得：
$$
\begin{aligned}
\text{P}(X_n-X_0 \leq -\varepsilon)
\leq
\exp \left(-\frac{2 \varepsilon^2}{\sum_{t=1}^{n}c_t^2}\right)
\end{aligned}
$$

当取$A_t = -c_t$，$B_t = c_t$时，退化成Azuma不等式的特殊情况。

定理中涉及到了超鞍（上鞍）序列的概念，该可积随机过程满足：
$$
\begin{aligned}
E[X_{n+1}|X_1,\ldots,X_n] \le X_n,\quad n\in \mathbb N
\end{aligned}
$$
相应地，亚鞍（下鞍）序列满足：
$$
\begin{aligned}
E[X_{n+1}|X_1,\ldots,X_n] \ge X_n,\quad n\in \mathbb N
\end{aligned}
$$
这里给出一个区分下鞅和上鞅的记忆方法：“生活是一个上鞅：随着时间的推进，期望逐渐降低。”



## 定理 18: Slud 不等式  
  
若$X\sim B(m,p)$，则有：  
$$  
      P(\frac{X}{m} \geq \frac{1}{2}) \geq \frac{1}{2}\left[1 - \sqrt{1-\exp\left(-\frac{m\varepsilon^{2}}{1-\varepsilon^{2}}\right)}\right]  
$$  
其中$p = (1-\varepsilon)/2$。  

$Proof.$  
二项随机变量$X$统计在$m$次独立伯努利试验中成功的次数，成功概率为$p$。对于对于大的$m$，二项分布$B(m,p)$可以近似为均值$\mu=mp$和方差$\sigma^2=mp(1-p)$的正态分布：
$$
\begin{aligned}
\mu &= \frac{m(1-\varepsilon)}{2} \\
\sigma^2 &= \frac{m(1-\varepsilon^2)}{4}
\end{aligned}
$$
令$Z=\frac{X-\mu}{\sigma}$，代入$\mu$和$\sigma$，有：
$$
P[\frac{X}{m} \geq \frac{1}{2}] = P[Z \geq \frac{\frac{m}{2}-\mu}{\sigma}] = P[Z \geq \frac{\varepsilon\sqrt{m}}{\sqrt{1-\varepsilon^2}}]
$$
根据正态分布不等式（定理 20），有：
$$
P[Z \geq x] \geq \frac{1}{2}\left[1 - \sqrt{1-\exp\left(-\frac{2x^2}{\pi}\right)}\right] \geq \frac{1}{2}\left[1 - \sqrt{1-\exp\left(-x^2\right)}\right]
$$
代入可得：
$$
P[Z \geq \frac{\varepsilon\sqrt{m}}{\sqrt{1-\varepsilon^2}}] \geq \frac{1}{2}\left[1 - \sqrt{1-\exp\left(-\frac{m\varepsilon^2}{1-\varepsilon^2}\right)}\right]
$$
得证。  

  
  
## 定理 19: Johnson-Lindenstrauss 引理  

JL引理可以非常通俗地表达为：压缩N个向量只需要$O(logN)$维空间，且相对距离的误差可控制在一定范围内。
首先借用上述工具考察一个示例：  
### $\chi_m^2$随机变量的集中度  
若随机变量$Z\sim \chi_m^2$，则$\forall \varepsilon \in (0, 3)$有：  
$$  
P\left((1-\varepsilon) \leq \frac{Z}{m} \leq (1 + \varepsilon)\right) \leq \exp(-\frac{m\varepsilon^2}{6})  
$$  
  
$Proof.$  
若$X\sim N(0,1)$，则显然$\forall \lambda \gt  0$：  
$$  
   \mathbb{E}[e^{-\lambda X^2}] \leq 1 - \lambda\mathbb{E}[X^2] + \frac{\lambda^2}{2}\mathbb{E}[X^4] = 1 - \lambda + \frac{3}{2}\lambda^2  \leq e^{-\lambda + \frac{3}{2}\lambda^2}  
$$  
类似地使用 Chernoff 一般性技巧，在$\lambda = \varepsilon/3$时可以证得左端不等式。  
对于右端不等式，考察矩母函数$\forall \lambda \lt  1/2$：  
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
  
该定理的证明其所需前序知识超出了本笔记的讨论范围，详细证明可参考[论文](https://onlinelibrary.wiley.com/doi/pdf/10.1002/rsa.10073)。  
  
该定理的适用性极其广泛，例如在稀疏感知领域直接导致了约束等距性条件 (RIP条件)，即非凸的 $L_0$范数最小化问题与 $L_1$范数最小化问题等价性条件；在流形学习和优化理论中也有重要的应用。而其在学习理论中最重要的应用是对降维任务的估计。



## 定理 20: 上界不等式之加性公式
若$sup(f)$和$sup(g)$分别为函数$f$和$g$的上界，则有：
$$
sup(f+g)\le sup(f) + sup(g)
$$

$Proof.$

假设 $f,g$ 分别有相同的定义域 $D_f,D_g$。根据上确界的定义，对于每一个 $x \in D_f \cap D_g$，我们有
$$ g(x) \leq \sup_{y \in D_g}g(y),$$
从而
$$f(x)+g(x) \leq f(x)+\sup_{y \in D_g} g(y).$$
因为这对于每一个 $x \in D_f \cap D_g$ 都是成立的，我们可以在不等式的两边取上确界，得到：
$$\sup_{x \in D_f \cap D_g}(f(x)+g(x)) \leq \sup_{x \in D_f \cap D_g}f(x)+\sup_{y \in D_g} g(y)\leq \sup_{z \in D_f}f(z)+\sup_{y \in D_g} g(y).$$
这里我们使用了$\sup_{x \in D_f \cap D_g}f(x) \leq \sup_{z \in D_f}f(z)$ since $D_f \cap D_g \subset D_f$.

值得注意的是，该不等式在（4.33）中利用过两次，且原推导并没有用到Jensen不等式的任何性质。

另外，加性公式有几个常见的变形，例如：
$$
sup(f-g) - sup(f-k)\le sup(k-g)
$$
该不等式在（4.29）中出现过。



## 定理 21: 正态分布不等式
若$X$是一个服从标准正态分布的随机变量，那么对于任意$u\ge 0$，有：
$$\mathbb{P}[X\le u]\le\frac{1}{2}\sqrt{1-e^{-\frac{2}{\pi}u^2}}$$

$Proof.$

令$G(u)=\mathbb{P}[X\le u]$，则有：
$$2G(u)=\int_{-u}^u(2\pi)^{-1/2}e^{-x^2/2}dx=\int_{-u}^u(2\pi)^{-1/2}e^{-y^2/2}dy$$
因此：
$$2\pi[2G(u)]^2=\int_{-u}^u\int_{-u}^ue^{-(x^2+y^2)/2}dxdy$$
让我们考虑更一般的积分形式：
$$2\pi[2G(u)]^2=\underset{R}{\int\int}e^{-(x^2+y^2)/2}dxdy$$
此时$R$为任意面积为$4a^2$的区域，通过反证法易证，只有当$R$为以原点为中心的圆形区域$R_0$时，积分值最大：
$$R_0=\{(x,y):\pi(x^2+y^2)\le 4u^2\}$$
此时，我们有：
$$
\begin{aligned}
2\pi[2G(u)]^2&\le\underset{R_0}{\int\int}e^{-(x^2+y^2)/2}dxdy\\
&=\int_0^{2\pi}\int_0^{2u\pi^{-1/2}}e^{-r^2/2}rdrd\varphi\\
&=2\pi(1-e^{-2u^2/\pi})
\end{aligned}
$$
因此，我们有：
$$G(u)=\mathbb{P}[X\le u]\le\frac{1}{2}\sqrt{1-e^{-\frac{2}{\pi}u^2}}$$
进一步，我们可以得到：
$$\mathbb{P}[X\ge u]\ge\frac{1}{2}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})$$



## 定理 22: AM-GM 不等式

算术平均数和几何平均数的不等式，简称AM-GM不等式。该不等式指出非负实数序列的算术平均数大于等于该序列的几何平均数，当且仅当序列中的每个数相同时，等号成立。
形式上，对于非负实数序列$\{x_n\}$，其算术平均值定义为：
$$
A_n=\frac{1}{n}\sum_{i=1}^nx_i
$$
其几何平均值定义为：
$$
G_n=\sqrt[n]{\prod_{i=1}^nx_i}
$$
则AM-GM 不等式成立：
$$
A_n\ge G_n
$$

$Proof.$

我们可以通过Jensen不等式（1.11）来证明AM-GM不等式。首先，我们考虑函数$f(x)=-\ln x$，该函数是凸函数，因此有：
$$
\frac{1}{n}\sum_{i=1}^n-\ln x_i\ge-\ln\left(\frac{1}{n}\sum_{i=1}^nx_i\right)
$$
即：
$$
\begin{aligned}
&\ln\left(\frac{1}{n}\sum_{i=1}^nx_i\right)\ge\frac{1}{n}\sum_{i=1}^n\ln x_i=\sum_{i=1}^n\ln x_i^{\frac{1}{n}}=\ln\sqrt[n]{\prod_{i=1}^nx_i}\\

\Rightarrow&\frac{1}{n}\sum_{i=1}^nx_i \le \sqrt[n]{\prod_{i=1}^nx_i}
\end{aligned}
$$
当取$x_1=x_2=\cdots=x_n$时，等号成立。
特别地，当$n=2$时，我们有：
$$
\frac{x_1+x_2}{2}\ge\sqrt{x_1x_2}
$$



## 定理 23: Young 不等式

对于任意$a,b\ge 0, p.q\gt 1$，若$\frac{1}{p}+\frac{1}{q}=1$，则有：
$$
ab\le\frac{a^p}{p}+\frac{b^q}{q}
$$
当且仅当$a^p=b^q$时，等号成立。

$Proof.$

我们可以通过Jensen不等式（1.11）来证明Young不等式。
首先，当$ab=0$时，该不等式显然成立。
当$a,b\gt 0$时，我们令$t=1/p,1-t=1/q$，根据$\ln(x)$的凹性，我们有：
$$
\begin{aligned}
\ln(ta^p+(1-t)b^q)&\ge t\ln(a^p)+(1-t)\ln(b^q)\\
&=\ln(a)+\ln(b)\\
&=\ln(ab)
\end{aligned}
$$
当且仅当$a^p=b^q$时，等号成立。



## 定理 24: 分离/支撑超平面定理

超平面（Hyperplane）是指$n$维线性空间中维度为$n-1$的子空间，它可以把线性空间分割成不相交的两部分。
对于一个凸集，支撑超平面（Supporting Hyperplane）是与凸集边界切线的超平面，即它“支撑”了凸集，使得所有的凸集内的点都位于支撑超平面的一侧。
凸集的支撑超平面也称为tac-planes或tac-hyperplanes。

分离超平面定理：如果有两个不相交的非空凸集，则存在一个超平面可以将它们完全分隔开。
形式化来说，若$C,D$为非空凸集，且$C\cap D=\varnothing$，则存在一个超平面$a\neq0,b$，使得$\forall x\in C,a^Tx\le b$且$\forall x\in D,a^Tx\ge b$，即$inf_{x\in D}a^Tx\ge sup_{x\in C}a^Tx$。

支撑超平面定理：对于一个非空的凸集，在凸集的边界上存在至少一个点，该点可以找到一个支撑超平面。
形式化来说，若$C$为非空凸集，则对于$\forall x_0\in\partial C$，则存在一个超平面$\{x|a^Tx=a^Tx_0,a\neq0\}$，使得$\forall x\in C,a^Tx\le a^Tx_0$。

两定理的证明其所需前序知识超出了本笔记的讨论范围，详细证明可参考[教材](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)。