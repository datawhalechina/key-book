# 第7章：收敛率

*Edit: 李一飞，赵志民*

---

本章的内容围绕学习理论中的收敛率（convergence rate）展开。具体来说，将会考察确定优化下的收敛率问题，以及随机优化下的收敛率问题，并在最后分析支持向量机的实例。



## 1. 【概念补充】凸函数

回顾第一章中的介绍，**凸函数**是函数曲线围成区域为凸集的函数，即
$$
\forall x_1,x_2\in C\\\theta x_1+(1-\theta x_2)\in C
$$
由于凸函数围成区域具有上述的特殊性质，我们只需要考虑函数曲线所在的边界，因此条件转化为
$$
f(\theta x + (1-\theta)z) \leq \theta f(x) + (1-\theta) f(z)
$$
进一步地，我们可以推出凸函数局部最优解就是全局最优解。

假设$f:\mathbb{R}^d\rightarrow\mathbb{R}$是凸函数，且$x^*$是$f$在凸集合$\mathcal{D}$中的局部最小值。
因为凸集合的性质，对于任意$y$，$y-x^*$都是一个可行的方向。因此，我们总是可以选择一个足够小的$t>0$，满足：
$$
f(x^*)\leq f(x^*+t(y-x^*))
$$
由$f$的凸函数性质可知:
$$
f(x^*+t(y-x^*))=f((1-t)x^*+ty)\leq (1-t)f(x^*)+tf(y)
$$
结合以上两式，我们有：
$$
\begin{aligned}
&f(x^*)\leq (1-t)f(x^*)+tf(y)\\
\Leftrightarrow &f(x^*)\leq f(y)
\end{aligned}
$$
因为$y$是凸集合$\mathcal{D}$中的任意点，所以$x^*$是全局最小解。
对于全局最大解，我们可以通过考虑函数$-f$的局部最小解来得到类似结论。



## 2.【公式说明】$l$-Lipschitz 

由 $l$-Lipschitz 的定义式 (1.7) 可以推出题目给的梯度条件:
$$
\begin{aligned}
&|f(z)-f(x)| \leq l\cdot|z-x|\\
\Leftrightarrow&|\frac{f(z)-f(x)}{z-x}|\leq l\\
\Leftrightarrow&lim_{z\rightarrow x}|\frac{f(z)-f(x)}{z-x}|\leq l\\
\Leftrightarrow&|\nabla f(x)|\leq l
\end{aligned}
$$



## 3. 【概念补充】强凸函数

对定义在凸集上的函数 $f: \R^d\rightarrow\R$，若 $\exists \lambda\in\R_+$，使得 $\forall x,z\in\Psi$且$\theta\in[0,1]$ 都有下式成立：
$$
f(\theta x+(1-\theta)z)\leq \theta f(x)+(1-\theta)f(z)-\frac{\lambda}{2}\theta(1-\theta)||x-z||^2
$$
则称 $f$ 为$\lambda$-强凸函数，其中$\lambda$ 为强凸系数。

强凸函数不仅收敛速度更快，还具备很多优良的性质。比如**P90**中的定理7.2，这里给出证明：

根据强凸函数的定义，我们取$x=w,z=w^*$，然后两边除以$\theta$可得：
$$
\begin{aligned}
&\frac{f(\theta w+(1-\theta)w^*)}{\theta}\leq f(w)+\frac{1-\theta}{\theta}f(w^*)-\frac{\lambda}{2}(1-\theta)||w-w^*||^2\\
\Rightarrow&\frac{\lambda}{2}(1-\theta)||w-w^*||^2\le f(w)-f(w^*)-\frac{f(w^* +(w-w^*)\theta)-f(w^*)}{\theta}
\end{aligned}
$$
令$\theta\rightarrow 0^+$，则有：
$$
\begin{aligned}
&lim_{\theta\rightarrow 0^+}\frac{\lambda}{2}(1-\theta)||w-w^*||^2\le f(w)-f(w^*)+lim_{\theta\rightarrow 0^+}\frac{f(w^* +(w-w^*)\theta)-f(w^*)}{\theta}\\
\Rightarrow&\frac{\lambda}{2}||w-w^*||^2\le f(w)-f(w^*)+lim_{\Delta\rightarrow 0^+}\frac{f(w^* +\Delta)-f(w^*)}{\Delta}(w-w^*)\\
\Rightarrow&\frac{\lambda}{2}||w-w^*||^2\le f(w)-f(w^*)+\nabla f(w^*)^T(w-w^*)
\end{aligned}
$$
其中$\Delta=(w-w^*)\theta$

因为$w^*$为最优解，所以$\nabla f(w^*)=0$，因此有：
$$
f(w)-f(w^*)\ge\frac{\lambda}{2}||w-w^*||^2
$$

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

