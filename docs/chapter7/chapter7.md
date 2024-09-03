# 第7章：收敛率

*编辑：赵志民，李一飞*

------

## 本章前言

本章的内容围绕学习理论中的算法收敛率（convergence rate）展开。具体来说，我们将探讨在确定性优化和随机优化中的收敛率问题，并在最后分析支持向量机的实例。

## 7.1【概念补充】算法收敛率

在算法分析中，收敛率是指迭代算法逼近解或收敛到最优或期望结果的速度，它衡量算法在减少当前解与最优解之间差异的快慢。

设 $\{x_k\}$ 是算法生成的迭代序列，我们可以根据以下公式来衡量算法的收敛率：
$$
\begin{equation}
\lim_{t\rightarrow+\infty}\frac{||x_{t+1} - x^*||}{||x_t - x^*||^p} = C 
\end{equation}
$$
其中，$C$为收敛因子，$p$为收敛阶数，$x^*$ 表示最优解，$||.||$ 表示适当的范数。

根据收敛率的不同情况，我们可以将其分类如下：
1. **超线性收敛**：$p\ge1$，$C=0$，表明每次迭代都会使得误差减小，且减小的速度越来越快。特别地，当$p>1$时，称为$p$阶收敛。例如，$p=2$时称为平方收敛，$p=3$时称为立方收敛。
2. **线性收敛**：$p=1$，$C>0$，表明每次迭代都会使得误差减小（误差呈几何级数下降），但减小的速度是一定的。
3. **次线性收敛**：$p=1$，$C=1$，表明每次迭代都会使得误差减小，但减小的速度越来越慢。

## 7.2【定理补充】凸函数的确定性优化

书中给出的梯度下降算法中，输出的是 $T$ 轮迭代的均值 $\omega$，而不是最后一次迭代的结果 $\omega_T$。这是因为在凸函数的梯度下降过程中，所设定的步长 $\eta$ 是启发式的，因此每次迭代产生的 $\omega'$ 无法保证是局部最优解。

根据定理7.1，$T$ 轮迭代的 $\omega$ 均值具有次线性收敛率，而无法证明最后一次迭代值 $\omega_T$ 也具有相同的收敛率。因此，返回 $\omega$ 的均值虽然会增加计算代价，但可以确保稳定的收敛率。这一思想在7.3.1和7.3.2中梯度下降算法中也有体现。

作为对比，在7.2.2中的强凸函数梯度下降算法中，我们只输出了最后一次迭代值 $\omega_T$。这是因为在强凸函数的条件下，每次迭代的梯度更新都有闭式解 $\omega_{t+1}=\omega_t-\frac{1}{\gamma}\nabla f(\omega_t)$。这种情况下，每次迭代无需启发式算法便可得到该临域的全局最优解，这也是此算法拥有更快收敛率（线性收敛率）的原因。因此，无需返回历史 $\omega$ 的均值。

另外，在 **P139** 定理7.1的（7.12）推导中，利用了第一章补充内容 AM-GM 不等式 $n=2$ 的结论，即对于任意非负实数 $x,y$，有：
$$
\begin{equation}
\sqrt{xy}\le\frac{x+y}{2}
\end{equation}
$$
当且仅当 $x=y$ 时取等号。

因此，只有当 $\frac{\Gamma^2}{2\eta T}=\frac{\eta l^2}{2}$ 时，$\frac{\Gamma^2}{2\eta T}+\frac{\eta l^2}{2}$ 才能取得最小值 $\frac{l\Gamma}{\sqrt T}$，此时步长 $\eta$ 应设置为 $\frac{\Gamma}{l\sqrt T}$。类似的推导可以在（7.35）和（7.39）中找到。

## 7.3【定理补充】强凸函数的确定性优化

**P142** 中，在证明定理7.3时，对于（7.19）的推导补充如下。

首先，如果目标函数满足 $\lambda$-强凸且 $\gamma$-光滑，那么根据第一章补充内容中的结论，我们有 $\gamma\ge\lambda$。这是因为对于任意 $\omega,\omega'$，光滑系数 $\gamma$ 被定义为：
$$
\begin{equation}
f(\omega)\le f(\omega')+\nabla f(\omega')^T(\omega-\omega')+\frac{\gamma}{2}||\omega-\omega'||^2
\end{equation}
$$
而强凸系数 $\lambda$ 被定义为：
$$
\begin{equation}
f(\omega)\ge f(\omega')+\nabla f(\omega')^T(\omega-\omega')+\frac{\lambda}{2}||\omega-\omega'||^2
\end{equation}
$$
光滑系数 $\gamma$ 决定了 $f(\omega)$ 的上界，而强凸系数 $\lambda$ 决定了 $f(\omega)$ 的下界，因此光滑系数 $\gamma$ 不小于强凸系数 $\lambda$。

接着，令 $f(\alpha)=\frac{\gamma-\lambda}{\lambda}\alpha^2-\alpha$，由于 $\frac{\gamma-\lambda}{\lambda}\ge0$，我们可以分成以下两种情况讨论：

1. 当 $\frac{\gamma-\lambda}{\lambda}=0$ 时，（7.19）转化为：
$$
\begin{align}
f(\omega_{t+1})&\le \min_{\alpha\in[0,1]}\{f(\omega_t)-\alpha (f(\omega_t)-f(\omega^*))\} \\
\Rightarrow f(\omega_{t+1})-f(\omega^*)&\le \min_{\alpha\in[0,1]}\{1-\alpha\}(f(\omega_t)-f(\omega^*)) 
\end{align}
$$
因为 $f(\omega_t)-f(\omega^*)\ge0$，所以当且仅当 $\alpha=1$ 时，不等式右侧取得最小值 $0$，此时易知 $f(\omega_{t+1})=f(\omega^*)$。根据凸函数局部最优解等于全局最优解的结论，我们可以得到 $\omega_{t+1}=\omega^*$，即算法在第 $t+1$ 轮迭代中收敛到最优解。

2. 当 $\frac{\gamma-\lambda}{\lambda}>0$ 时，$f(\alpha)$ 为开口向上的二次函数。令 $f'(\alpha)=2\frac{\gamma-\lambda}{\lambda}\alpha-1=0$，得到 $f(\alpha)$ 的对称轴为 $\alpha=\frac{\lambda}{2(\gamma-\lambda)}$。我们可以分成以下两种情况讨论：
    - 当 $\frac{\lambda}{2(\gamma-\lambda)}\ge1$ 时，$f(\alpha)$ 取得最小值只能在 $\alpha=1$ 处，故而得到（7.20）。
    - 当 $0<\frac{\lambda}{2(\gamma-\lambda)}<1$ 时，$f(\alpha)$ 取得最小值在 $\alpha=\frac{\lambda}{2(\gamma-\lambda)}$ 处，故而得到（7.21）。

余下的推导部分与书中相同，此处不再赘述。

## 7.4【定理证明】鞅差序列的 Bernstein 不等式

**P149** 定理7.6 给出了鞅差序列的 Bernstein 不等式，我们在这里给出其证明。

对原文中的条件方差定义进行勘误，$X_n$ 的条件方差定义为：
$$
\begin{equation}
V_n^2 = \sum_{k=1}^n \mathbb{E}[X_k^2|F_{k-1}]
\end{equation}
$$

考虑函数 $f(x) = (e^{\theta x} -\theta x-1)/x^2$，且 $f(0) = \theta^2/2$。

通过对该函数求导，我们知道该函数是非减的。即 $f(x) \leq f(1)$，当 $x \leq 1$ 时：
$$
\begin{equation}e^{\theta x} = 1 + \theta x + x^2f(x) \leq 1+\theta x+x^2f(1) = 1 + \theta x + g(\theta)x^2, \quad x \leq 1\end{equation}
$$

将其用于随机变量 $X_k/K$ 的期望，可得：
$$
\begin{equation}\mathbb{E} \left[\exp \left(\frac{\theta X_k}{K}\right) \bigg| \mathcal{F}_{k-1}\right] \leq 1 + \frac{\theta}{K} \mathbb{E} \left[X_k | \mathcal{F}_{k-1} \right] + \frac{g(\theta)}{K^2} \mathbb{E} \left[X_k^2 | \mathcal{F}_{k-1} \right]\end{equation}
$$

由于 $\{X_k\}$ 是一个鞅差序列，我们有 $\mathbb{E} \left[X_k | \mathcal{F}_{k-1} \right] = 0$，结合 $1+x \leq e^x, x \geq 0$，我们得到：
$$
\begin{equation}
\mathbb{E} \left[\exp \left(\frac{\theta X_k}{K}\right) \bigg| \mathcal{F}_{k-1}\right] = 1 + \frac{g(\theta)}{K^2} \mathbb{E} \left[X_k^2 | \mathcal{F}_{k-1} \right] \leq \exp \left(g(\theta) \frac{\mathbb{E} [X_k^2|\mathcal{F}_{k-1}]}{K^2} \right)
\end{equation}
$$

考虑一个随机过程：
$$
\begin{equation}Q_k = \exp \left(\theta \frac{S_k}{K} - g(\theta) \frac{\Sigma_k^2}{K^2}\right), \quad Q_0 = 1\end{equation}
$$
我们证明这个过程对于滤波 $\mathcal{F}_n$ 是一个超鞅，即 $\mathbb{E} [Q_k|\mathcal{F}_{k-1}] \leq Q_{k-1}$。

证明如下：
$$
\begin{equation}
\begin{align*}
\mathbb{E} [Q_k|\mathcal{F}_{k-1}] &= \mathbb{E} \left[\exp \left(\theta \frac{S_k}{K} - g(\theta) \frac{\Sigma_k^2}{K^2}\right)\bigg|\mathcal{F}_{k-1}\right] \\
&= \mathbb{E} \left[\exp \left(\theta \frac{S_{k-1}}{K} - g(\theta) \frac{\Sigma_{k-1}^2}{K^2} - g(\theta)\frac{\mathbb{E} [X_k^2|\mathcal{F}_{k-1}]}{K^2} + \theta \frac{X_k}{K}\right)\bigg|\mathcal{F}_{k-1}\right] \\
&= \exp \left(\theta \frac{S_{k-1}}{K} - g(\theta) \frac{\Sigma_{k-1}^2}{K^2} - g(\theta)\frac{\mathbb{E} [X_k^2|\mathcal{F}_{k-1}]}{K^2}\right) \mathbb{E} \left[ \exp \left(\theta \frac{X_k}{K}\right)\bigg|\mathcal{F}_{k-1}\right]
\end{align*}
\end{equation}
$$

应用之前证明的不等式，我们得到：
$$
\begin{equation}\mathbb{E} [Q_k|\mathcal{F}_{k-1}] \leq \exp \left(\theta \frac{S_{k-1}}{K} - g(\theta) \frac{\Sigma_{k-1}^2}{K^2}\right) = Q_{k-1}\end{equation}
$$

我们定义 $A = \{k \geq 0: \max_{i=1,\cdots,k} S_i \gt t,\Sigma_k^2 \le v\}$，则有：
$$
\begin{equation}Q_k\geq \exp \left(\theta \frac{t}{K} - g(\theta) \frac{v}{K^2}\right), k \in A\end{equation}
$$

由于 $\{Q_k\}$ 是超鞅，我们有：
$$
\begin{equation}\mathbb{E} [Q_k|\mathcal{F}_{k-1}] \leq \mathbb{E} [Q_{k-1}|\mathcal{F}_{k-2}] \leq \cdots \leq Q_0 = 1\end{equation}
$$

考虑到 $1 \geq \mathbb{P}(A)$，我们有：
$$
\begin{equation}1 \geq \mathbb{E} [Q_k|\mathcal{F}_{k-1}] \geq \exp \left(\theta \frac{t}{K} - g(\theta) \frac{v}{K^2}\right) \mathbb{P}(A), k \in A\end{equation}
$$

因此：
$$
\begin{equation}
\begin{align*}
\mathbb{P}(A) \leq \exp \left(g(\theta) \frac{v}{K^2} -\theta \frac{t}{K} \right)
\end{align*}
\end{equation}
$$

由于上述不等式对任何 $\theta > 0$ 都成立，我们可以写为：
$$
\begin{equation}P(A) \leq \inf_{\theta > 0} \exp \left(g(\theta) \frac{v}{K^2} - \theta \frac{t}{K} \right)\end{equation}
$$
检查不等式右边的一阶导数，我们知道该下确界在 $\theta = \log (1+Kt/v)$ 处取得。

对于指数内部的表达式，我们进行如下变换：
$$
\begin{equation}
\begin{align*}
\theta \frac{t}{K} - g(\theta)\frac{v}{K^2} &= \log \left(1 + \frac{Kt}{v}\right) \frac{t}{K} - \frac{v}{K^2} \left(\frac{Kt}{v} - \log \left(1 + \frac{Kt}{v}\right) \right) \\
&=\frac{v}{K^2} \left( \left(1+\frac{Kt}{v} \right) \log \left(1 + \frac{Kt}{v}\right) - \frac{Kt}{v} \right) \\
&= \frac{v}{K^2} h\left( \frac{Kt}{v} \right)
\end{align*}
\end{equation}
$$
其中 $h(u) = (1+u)\log(1+u) - u$。

通过对表达式求二阶导数的方法，我们也可以证明：
$$
\begin{equation}h(u) \geq \frac{u^2}{2(1 + u/3)},\quad u \geq 0\end{equation}
$$

综上所述，我们有：
$$
\begin{equation}P(A) \leq \exp \left( -\frac{v}{K^2} h \left( \frac{Kt}{v} \right)\right) \leq \exp \left( - \frac{v}{K^2} \frac{K^2t^2}{2v (v+Kt/3)} \right) = \exp\left( -\frac{t^2}{2(v+Kt/3)}\right)\end{equation}
$$

## 7.5【定理补充】Epoch-GD 的收敛率

**P150** 引理7.2给出了Epoch-GD外层循环收敛率的泛化上界，我们对其中部分推导进行必要补充。

首先，（7.60）中第二个不等式的推导利用了Cauchy-Schwarz不等式（1.14），即 $\|x^Ty\|\le\|x\|\|y\|$。这里，我们令 $x=\underbrace{[1,\cdots,1]}_{T}$，$y=\underbrace{[\|\omega_1-w^*\|,\cdots,\|\omega_T-w^*\|]}_{T}$，则有：
$$
\begin{equation}
|x^Ty|=\sum_{t=1}^T\|\omega_t-w^*\|\le \sqrt{T}\sqrt{\sum_{t=1}^T\|\omega_t-w^*\|^2}=|x||y|
\end{equation}
$$

其次，（7.62）中最后两个不等式的推导利用了一些常见的缩放技巧，我们在这里给出完整形式：
$$
\begin{align}
&\sum_{i=1}^m P\left(\sum_{t=1}^T \delta_t \ge 2\sqrt{4l^2A_T\tau}+\frac{2}{3}\frac{4l^2}{\lambda}\tau+\frac{4l^2}{\lambda},V_T^2\le4l^2A_T,A_T\in\left(\frac{4l^2}{\lambda^2T}2^{i-1},\frac{4l^2}{\lambda^2T}2^i\right)\right) \\
\le &\sum_{i=1}^m P\left(\sum_{t=1}^T \delta_t \ge 2\sqrt{4l^2A_T\tau}+\frac{2}{3}\frac{4l^2}{\lambda}\tau,V_T^2\le4l^2A_T,A_T\in\left(\frac{4l^2}{\lambda^2T}2^{i-1},\frac{4l^2}{\lambda^2T}2^i\right)\right) \\
\le &\sum_{i=1}^m P\left(\sum_{t=1}^T \delta_t \ge \sqrt{2\frac{16l^42^i}{\lambda^2T}\tau}+\frac{2}{3}\frac{4l^2}{\lambda}\tau,V_T^2\le\frac{16l^42^i}{\lambda^2T}\right) \\
\le &\sum_{i=1}^m P\left(\max_{j=1,\cdots,T}\underbrace{\sum_{t=1}^j \delta_t}_{S_j} \ge \sqrt{2\underbrace{\frac{16l^42^i}{\lambda^2T}}_{\nu}\tau}+\frac{2}{3}\underbrace{\frac{4l^2}{\lambda}}_{K}\tau,V_T^2\le\underbrace{\frac{16l^42^i}{\lambda^2T}}_{\nu}\right) \\
\le &\sum_{i=1}^m e^{-\tau} \\
= &me^{-\tau}
\end{align}
$$
这里，第一个不等式利用了 $\frac{4l^2}{\lambda} \gt 0$ 的事实对 $\sum_{t=1}^T \delta_t$ 的范围进行概率缩放；
第二个不等式利用了 $A_T$ 的下界和上界分别对 $\sum_{t=1}^T \delta_t$ 和 $V_T^2$ 的范围进行概率缩放；
第三个不等式利用了 $\max_{j=1,\cdots,T}\sum_{t=1}^j \delta_t$ 比 $\sum_{t=1}^T \delta_t$ 更为宽松的事实对 $V_T^2$ 进行概率缩放；
第四个不等式利用了定理7.6的结论。

最后，（7.64）中第二个不等式的推导利用了开口向下的二次函数 $f(x)=ax^2+bx+c,a\lt0$ 拥有最大值点 $x_0=-\frac{b}{2a}$ 的事实。我们令 $x=\sqrt{A_T}$，然后取 $a=-\frac{\lambda}{2},b=2\sqrt{4l^2\ln\frac{m}{\delta}},c=0$，则易知 $f(x)$ 的最大值为 $\frac{8l^2}{\lambda}\ln\frac{m}{\delta}$，于是得到了（7.64）中的结论。

进一步地，**P152**引理7.3利用数学归纳法给出了特定步长和迭代次数下Epoch-GD外层循环收敛率的泛化上界，这为**P154**定理7.7中Epoch-GD的收敛率奠定了基础。我们对后者的部分推导进行必要补充。

首先，观察（7.75）可以发现，Epoch-GD外层的迭代次数 $k$ 需要满足 $\frac{\alpha}{2}(2^k-1) \le T$，即 $k=\lfloor \log_2(\frac{2T}{\alpha}+1)\rfloor$，因此构造了（7.66）中的 $k^{\dagger}$。

其次，（7.77）的推导利用了函数 $f(x)=(1-\frac{1}{x})^x$ 在 $x=\frac{k^{\dagger}}{\delta}\gt1$ 时单调递增的事实，以下是更严格的证明。

对函数 $f(x)$ 两边取对数，得到：
$$
\begin{equation}
\ln f(x)=x\ln(1-\frac{1}{x})
\end{equation}
$$
接着对两边分别求导，可得：
$$
\begin{equation}
\frac{f'(x)}{f(x)}=\ln(1-\frac{1}{x})+\frac{1}{x-1}
\end{equation}
$$
易知当 $x\gt1$ 时，$f(x)\gt0$，因此我们只需要关注等式右边在 $x\gt1$ 时的符号。
令 $g(x)=\ln(1-\frac{1}{x})+\frac{1}{x-1}$，则有：
$$
\begin{equation}
g'(x)=\frac{1}{x(x-1)^2}
\end{equation}
$$
易知当 $x\gt1$ 时，$g'(x)\lt0$，因此：
$$
\begin{equation}
g(x)\gt\lim_{x\rightarrow+\infty}g(x)=\lim_{x\rightarrow+\infty}\ln(1-\frac{1}{x})+\lim_{x\rightarrow+\infty}\frac{1}{x-1}=0
\end{equation}
$$
综上，当 $x\gt1$ 时，$\frac{f'(x)}{f(x)}=g(x)\gt0$，即 $f'(x)\gt0$，因此 $f(x)$ 在 $x\gt1$ 时单调递增。
