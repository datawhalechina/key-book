# 第7章：收敛率

*Edit: 李一飞*

*Update: 09/03/2020*

---

本章的内容围绕学习理论中的收敛率（convergence rate）展开。具体来说，将会考察确定优化下的收敛率问题，以及随机优化下的收敛率问题，并在最后分析支持向量机的实例。



## 1. 【概念补充】凸函数

回顾第一章中的介绍，**凸函数**是函数曲线围成区域为凸集的函数，即
$$
\forall x_1,x_2\in C\\\theta x_1+(1-\theta x_2)\in C
$$
由于凸函数围成区域的特殊性质（无限区域），我们只需要考虑函数曲线所在的边界，因此条件转化为
$$
f(\theta x + (1-\theta)z) \leq \theta f(x) + (1-\theta) f(z)
$$
由于凸函数的良好性质，我们可以推出凸函数局部最优解就是全局最优：假设 $x^*$ 是局部最优，即 $x^*$ 周围存在一个邻域，
$$
S = B(x^*,\delta),\forall x \in S, f(x)\geq f(x^*)
$$
 因此 
$$
\forall y , f(x^*)\leq f((1-t)x^*+ty) , \,\, \{t\leq \frac{\delta}{|x^*-y|}\}
$$
由凸函数性质:
$$
\forall y , f(x^*)\leq f((1-t)x^*+ty)\leq (1-t)f(x^*)+tf(y)\\\Rightarrow f(x^*)\leq f(y)
$$



## 2.【公式说明】$l$-Lipschitz 

由 $l$-Lipschitz 的定义式 (1.7) 可以推出题目给的梯度条件:
$$
f(z)-f(x) \leq l\cdot|z-x|
\\\Rightarrow\frac{|f(z)-f(x)|}{|z-x|}\leq l
\\\Rightarrow lim_{z-x\rightarrow 0}\frac{|f(z)-f(x)|}{|z-x|}\leq l
\\\Rightarrow||\nabla f(u)||\leq l
$$



## 3. 【概念补充】强凸函数

### 定义

对定义在凸集上的函数 $f: \R^d\rightarrow\R$，若 $\exists \lambda\in\R_+$，使得 $\forall x,z\in\Psi$ 都有
$$
f(\theta x+(1-\theta)z)\leq \theta f(x)+(1-\theta)f(z)-\frac{\lambda}{2}\theta(1-\theta)||x-z||^2 \quad(\forall0\leq\theta\leq 1)
$$

### 关于强凸函数

强凸函数在定义式中可以看出有了一个关于 $\theta$ 和 $||x-z||^2$ 的项，通过简单的化简为 $f(z)\geq f(x)+\nabla f(x)^T(z-x)+\frac{\lambda}{2}||x-z||^2$ 就可以得知，这表明了函数不仅在切线的上方，还保持有二阶的距离，这一点可以从泰勒展开得到更深入地体现。

强凸函数与凸函数的区别为，凸函数只关心二阶 Hessian 矩阵半正定，而强凸函数提出了更高的要求，对 $f$ 的二阶 Hessian 矩阵有 ：$\exists m>0, \nabla^{2} f(x)-m I \succeq 0$    这里的 $A \succeq \theta$ 代表A是半正定矩阵



## 4.【公式推导】公式(7.34)

由 7.25 的假设：
$$
\mathbb{E}[\mathbb{g}_t] = \nabla f(\mathbb{\omega}_t)
$$
随机梯度是真实梯度的无偏估计
$$
\mathbb{E}\big[ \Sigma<\nabla f(\omega_t)-\mathbf{g}_t,\omega_t - \omega> \big]\\=\mathbb{E}\big[ \Sigma<\nabla f(\omega_t),\omega_t - \omega> \big]-\mathbb{E}\big[ \Sigma<\mathbf{g}_t,\omega_t - \omega> \big]\\=\Sigma\big[\mathbb{E}\big[<\nabla f(\omega_t),\omega_t - \omega> \big]-\mathbb{E}\big[<\mathbf{g}_t,\omega_t - \omega>\big]\\=\Sigma\big[\mathbb{E}\big[<\mathbb{E}[\mathbf{g}_t],\omega_t - \omega> \big]-\mathbb{E}\big[<\mathbf{g}_t,\omega_t - \omega>\big]\\=\Sigma\big[\mathbb{E}\big[\mathbf{g}_t,\omega_t - \omega> \big]-\mathbb{E}\big[<\mathbf{g}_t,\omega_t - \omega>\big]\\=0
$$

