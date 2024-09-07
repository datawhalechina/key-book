# 第8章：遗憾界

*编辑：赵志民，Hao ZHAN*

---

## 本章前言

本章的内容围绕学习理论中的遗憾（regret）概念展开（有的教材里也翻译为“悔”）。通常，我们使用超额风险（excess risk）来评估批量学习的分类器性能，而用遗憾来评估在线学习的分类器性能。二者的不同在于，前者衡量的是整个学习过程结束后所得到的分类器性能，可以理解为学习算法**最终输出的模型**与假设空间内**最优模型**的风险之差；而后者衡量的是算法运行过程中，所产生的**模型**与假设空间内**最优模型**的损失之差的**和**。

## 8.1 【概念补充】超额风险与遗憾的区别

8.1介绍了遗憾这一评估指标的基本概念，我们在此基础上梳理一下其与超额风险这一评估指标的区别。

超额风险这一评估指标被定义为：
$$
\begin{equation}
ER = \mathbb{E}_{(x,y)\sim D}[l(w_{T+1},(x,y))] - \min_{w \in W} \mathbb{E}_{(x,y)\sim D}[l(w,(x,y))]
\end{equation}
$$
其中，$ER$ 指的是excess risk，等式右边的前半部分 $\mathbb{E}_{(x,y)\sim D}[l(w_{T+1},(x,y))]$ 指的是模型 $w_{T+1}$ 的风险，等式右边的后半部分 $\min_{w \in W} \mathbb{E}_{(x,y)\sim D}[l(w,(x,y))]$ 指的是假设空间内的最优模型的风险。值得注意的是，这里的评估是在整个数据集上进行的，也正是因为如此，我们必须要引入期望的操作。

而遗憾这一评估指标，被定义为：
$$
\begin{equation}
regret = \sum^{T}_{t=1}f_t(w_t)-\min_{w\in W}\sum^{T}_{t=1}f_t(w)
\end{equation}
$$
其中，$f_t(w_t)$ 指的是：
$$
\begin{equation}
\sum^{T}_{t=1}l(w_t,(x_t,y_t)) - \min_{w \in W}\sum^{T}_{t=1}l(w,(x_t,y_t))
\end{equation}
$$
由于$w_t$的计算过程与样本$(x_t,y_t)$ 无关，而是与$(x_1,y_1),...,(x_{t-1},y_{t-1})$ 有关，因此可以直接使用 $l(w,(x_t,y_t))$ 来衡量性能。

由此，我们可以总结出二者之间的两个主要区别：首先，超额风险引入了**期望**，而遗憾没有；其次，超额风险是在所有数据上进行的一次性计算，而遗憾是对多次损失的一个**求和**。同时，由于在线学习不依赖于任何分布假设，因此适用于非独立同分布样本或固定分布的情形。

## 8.2 【案例补充】Maler 算法

在8.2.3节的**P170**末尾，作者提到了Maler算法（multiple sub-algorithms and learning rates）（详细证明参考：[Adaptivity and Optimality: A Universal Algorithm for Online Convex Optimization](http://proceedings.mlr.press/v115/wang20e/wang20e.pdf)），这是一个能够自适应选择最优专家的在线学习算法，并在不同类型的损失函数上实现最优的遗憾界限：
- **一般凸函数**：$R(T) \leq O\sqrt{T})$
- **指数凹函数**：$R(T) \leq O(d\log T)$
- **强凸函数**：$R(T) \leq O(\log T)$
这里$T$表示时间总步数，$d$表示特征空间的维度。

下面，我们简要补充Maler算法的原理和实现。

### 假设和定义

1. **假设 1（梯度有界性）**：所有损失函数 $f_t(x)$ 的梯度被 $G$ 所有界：
   $$
   \forall t > 0, \quad \max_{x \in D} \|\nabla f_t(x)\| \leq G
   $$

2. **假设 2（行动集的直径有界性）**：行动集 $D$ 的直径被 $D$ 所有界：
   $$
   \max_{x_1, x_2 \in D} \|x_1 - x_2\| \leq D
   $$

3. **定义 1（凸函数）**：函数 $f : D arrow \mathbb{R}$ 是凸的，如果：
   $$
   f(x_1) \geq f(x_2) + \nabla f(x_2)^\top (x_1 - x_2), \quad \forall x_1, x_2 \in D
   $$

4. **定义 2（强凸函数）**：函数 $f : D arrow \mathbb{R}$ 是 $\lambda$-强凸的，如果：
   $$
   f(x_1) \geq f(x_2) + \nabla f(x_2)^\top (x_1 - x_2) + \frac{\lambda}{2} \|x_1 - x_2\|^2, \quad \forall x_1, x_2 \in D
   $$

5. **定义 3（指数凹函数）**：函数 $f : D arrow \mathbb{R}$ 是 $\alpha$-指数凹的（简称 $\alpha$-exp-concave），如果：
   $$
   \exp(-\alpha f(x)) \text{ 是凹的}
   $$

### 元算法（Maler）

**输入**：学习率 $\eta^c, \eta_1, \eta_2, \dots$，专家的先验权重 $\pi_1^c, \pi_1^{\eta_1,s}, \pi_1^{\eta_2,s} \dots$，以及 $\pi_1^{\eta_1,l}, \pi_1^{\eta_2,l}, \dots$。

1. **对于每个回合 $t = 1, \dots, T$：**
   - 从凸专家算法（专家 1）获取预测 $x^c_t$，从指数凹专家算法（专家 2）和强凸专家算法（专家 3）分别获取 $x^{\eta, l}_t$ 和 $x^{\eta, s}_t$。
   - 执行：
     $$
     x_t = \frac{\pi^c_t \eta^c x^c_t + \sum_{\eta} (\pi^{\eta,s}_t \eta x^{\eta,s}_t + \pi^{\eta,l}_t \eta x^{\eta,l}_t)}{\pi^c_t \eta^c + \sum_{\eta} (\pi^{\eta,s}_t \eta + \pi^{\eta,l}_t \eta)}
     $$
   - 观察梯度 $g_t$ 并发送给所有专家算法。
   - 对所有的 $\eta$ 更新权重：
     $$
     \pi^c_{t+1} = \frac{\pi^c_t e^{-c_t(x^c_t)}}{\Phi_t}, \quad \pi^{\eta,s}_{t+1} = \frac{\pi^{\eta,s}_t e^{-s^{\eta}_t(x^{\eta,s}_t)}}{\Phi_t}, \quad \pi^{\eta,l}_{t+1} = \frac{\pi^{\eta,l}_t e^{-l^{\eta}_t(x^{\eta,l}_t)}}{\Phi_t}
     $$
     其中
     $$
     \Phi_t = \sum_{\eta} (\pi^{\eta,s}_t e^{-s^{\eta}_t(x^{\eta,s}_t)} + \pi^{\eta,l}_t e^{-l^{\eta}_t(x^{\eta,l}_t)} ) + \pi^c_t e^{-c_t(x^c_t)}
     $$

### 凸专家算法（专家 1）

1. $x^c_1 = 0$
2. **对于每个回合 $t = 1, \dots, T$：**
   - 将 $x^c_t$ 发送给元算法
   - 从元算法接收梯度 $g_t$
   - 更新：
     $$
     x^c_{t+1} = \Pi^{I_d}_D (x^c_t - \frac{D}{\eta^c G \sqrt{t}} \nabla c_t(x^c_t))
     $$
     其中 $\nabla c_t(x^c_t) = \eta^c g_t$

### 指数凹专家算法（专家 2）

1. **输入**：学习率 $\eta$
2. $x^{\eta,l}_1 = 0, \beta = \frac{1}{2} \min\{\frac{1}{4G^l D}, 1\}, G^l = \frac{7}{25D}, \Sigma_1 = \frac{1}{\beta^2 D^2}I_d$
3. **对于每个回合 $t = 1, \dots, T$：**
   - 将 $x^{\eta,l}_t$ 发送给元算法
   - 从元算法接收梯度 $g_t$
   - 更新：
     $$
     \Sigma_{t+1} = \Sigma_t + \nabla l^{\eta}_t(x^{\eta,l}_t) \nabla l^{\eta}_t(x^{\eta,l}_t)^\top
     $$
     $$
     x^{\eta,l}_{t+1} = \Pi^{\Sigma_{t+1}}_D (x^{\eta,l}_t - \frac{1}{\beta} \Sigma_{t+1}^{-1} \nabla l^{\eta}_t(x^{\eta,l}_t))
     $$
     其中 $\nabla l^{\eta}_t(x^{\eta,l}_t) = \eta g_t + 2 \eta^2 g_t g_t^\top (x^{\eta,l}_t - x_t)$

### 强凸专家算法（专家 3）

1. **输入**：学习率 $\eta$
2. $x^{\eta,s}_1 = 0$
3. **对于每个回合 $t = 1, \dots, T$：**
   - 将 $x^{\eta,s}_t$ 发送给元算法
   - 从元算法接收梯度 $g_t$
   - 更新：
     $$
     x^{\eta,s}_{t+1} = \Pi^{I_d}_D (x^{\eta,s}_t - \frac{1}{2\eta^2 G^2 t} \nabla s^{\eta}_t(x^{\eta,s}_t))
     $$
     其中 $\nabla s^{\eta}_t(x^{\eta,s}_t) = \eta g_t + 2 \eta^2 G^2 (x^{\eta,s}_t - x_t)$



## 8.3 【定理补充】随机多臂赌博机遗憾界

**P172**中定理8.3给出了随机多臂赌博机的遗憾界，我们在此基础上对部分证明过程进行补充。

首先，（8.42）给出当$\overline{\mu}_*(p)+\sqrt{\frac{2\ln t}{p}}\le\overline{\mu}_i(q)+\sqrt{\frac{2\ln t}{q}}$成立时，必然有三种可能情况中的一种成立。但这三种情况并不是互斥的，因此显得不直观，这里将第二种情况做了细微调整，即：
$$
\begin{equation}
\overline{\mu}_*(p)+\sqrt{\frac{2\ln t}{p}}\le\mu_*,\mu_*\le\overline{\mu}_i(q)+\sqrt{\frac{2\ln t}{q}},\overline{\mu}_i(q)+\sqrt{\frac{2\ln t}{q}}\le\overline{\mu}_i(p)
\end{equation}
$$
此时，构造（8.44）和（8.45）的逻辑更加顺畅。我们令$l=\lceil(2\ln T)/\Delta_i^2\rceil$，则（8.45）转化为：
$$
\begin{equation}
P(\mu_*\le\mu_i+\sqrt{\frac{2\ln t}{q}})=0,q\ge l
\end{equation}
$$
代入（8.44），可得：
$$
\begin{align}
\mathbb{E}[n_i^T]&\le\lceil\frac{2\ln T}{\Delta_i^2}\rceil+2\sum_{t=1}^{T-1}\sum_{p=1}^{t-1}\sum_{q=l}^{t-1}t^{-4} \\
&\le\frac{2\ln T}{\Delta_i^2}+1+2\sum_{t=1}^{T-1}\sum_{p=1}^{t}\sum_{q=1}^{t}t^{-4} \\
&\le\frac{2\ln T}{\Delta_i^2}+1+2\lim_{Tarrow+\infty}\sum_{t=1}^{T-1}t^{-2} 
\end{align}
$$
根据$p$-级数判别法，当$p=2\gt1$时，级数收敛，因此$\lim_{Tarrow+\infty}\sum_{t=1}^{T-1}t^{-2}$是有界的。至于该级数的具体值，对定理的结论没有影响，因此我们可以将其视为一个常数，然后带入后续推导中。为了证明的完整性，我们对此进行简要说明。

$\lim_{Tarrow+\infty}\sum_{t=1}^{T-1}t^{-2}$的取值在数学界被称为Basel问题，推导过程涉及诸多前置定理，感兴趣的读者可以查看这个[讲义](https://www.math.cmu.edu/~bwsulliv/basel-problem.pdf)。此处提供另一种在微积分变换中常见的缩放方法：
$$
\begin{align}
\sum_{t=1}^{T-1}t^{-2}&\le1+\int_{1}^{T-1}\frac{1}{x^2}dx \\
&=1+(-\frac{1}{x})|_1^{T-1} \\
&=2-\frac{1}{T}
\end{align}
$$
对不等式两边同时取极限，可得：
$$
\begin{equation}
\lim_{Tarrow+\infty}\sum_{t=1}^{T-1}t^{-2}\le2
\end{equation}
$$
代入（8.46），同样可得类似（8.47）的结论。

这里继续沿用书中给出的$\lim_{Tarrow+\infty}\sum_{t=1}^{T}t^{-2}=\frac{\pi^2}{6}$，代入（8.46）得到遗憾界（8.47）：
$$
\begin{equation}
\mathbb{E}[regret]\le\sum_{i=1}^{K}\frac{2\ln T}{\Delta_i^2}+O(1)
\end{equation}
$$

此时（8.46）变为：
$$
\begin{equation}
\mathbb{E}[n_i^T]\le\sum_{i\neq*}^K\frac{2\ln T}{\Delta_i}+(1+\frac{\pi^2}{3}){\Delta_i}=O(K\log T)
\end{equation}
$$
观察（8.47）可知，求和公式中的每一项符合对钩函数的构造，即：
$$
\begin{equation}
f(x)=Ax+\frac{B}{x},x\gt0,A\gt0,B\gt0
\end{equation}
$$
这里$x=\Delta_i,A=1+\frac{\pi^2}{3},B=2\ln T$，因此无论$\Delta_i$过大或过小时，都会导致遗憾界的上界变大。另外，遗憾界跟摇臂的个数$K$呈线性关系，当$K$越大时，遗憾界也越大。

## 8.4 【证明补充】Sherman-Morrison-Woodbury (或 Woodbury) 公式

**P177**页的 Sherman-Morrison-Woodbury 公式变种是矩阵求逆中的一个重要工具，它可以通过已知矩阵的逆来快速计算被低秩修正的矩阵的逆。该公式如下所示：
$$
\begin{equation}
(A + UCV)^{-1} = A^{-1} - A^{-1}U (C^{-1} + VA^{-1}U)^{-1} VA^{-1}
\end{equation}
$$

其中，A 是一个 $n \times n$ 的矩阵，C 是 $k \times k$ 的矩阵，U 和 V 是 $n \times k$ 的矩阵，（8.54）中$C$为单位矩阵。

### 证明

该公式可以通过验证 $A + UCV$ 与其假设的逆（公式右侧）的乘积是否为单位矩阵来证明。我们对以下乘积进行计算：

$$
\begin{equation}
(A + UCV) [ A^{-1} - A^{-1}U (C^{-1} + VA^{-1}U )^{-1} VA^{-1} ]
\end{equation}
$$

逐步推导如下：
$$
\begin{align}
=& \{ I + UCVA^{-1} \} - \{ U (C^{-1} + VA^{-1}U )^{-1}VA^{-1} + UCVA^{-1}U (C^{-1} + VA^{-1}U )^{-1} VA^{-1} \} \\
=& I + UCVA^{-1} - (U + UCVA^{-1}U ) (C^{-1} + VA^{-1}U )^{-1}VA^{-1} \\
=& I + UCVA^{-1} - UC (C^{-1} + VA^{-1}U) (C^{-1} + VA^{-1}U)^{-1}VA^{-1} \\
=& I + UCVA^{-1} - UCVA^{-1} \\
=& I
\end{align}
$$
$\square$

## 8.5 【概念补充】线性赌博机

**P176**的8.3.2节介绍了线性赌博机的概念，我们在此基础上对参数估计部分进行补充。为了估计线性赌博机的参数，我们将原问题转化为岭回归问题，即（8.52）：
$$
\begin{equation}
f(w)=(Y-w^T X)^T(Y-w^T X)+\lambda w^T w
\end{equation}
$$
为了求得最优解$w^*$，我们令$f'(w)=0$，可推导出（8.53）：
$$
\begin{align}
&\frac{\partial f(w)}{\partial w}=-2X^T(Y-w^T X)+2\lambda w = 0 \\
arrow&X^TY = (X^TX + \lambda I)w \\
arrow&w^* = (X^TX + \lambda I)^{-1}X^TY
\end{align}
$$
相比于每次传入新数据$(x_t,y_t)$时从头计算$w_t$，这里巧妙地利用了 Sherman-Morrison-Woodbury 公式将任何形如$(A+uv^T)^{-1}$的矩阵逆转化为可逆矩阵$A$和列向量$u,v$之间的运算，在$O(d^2)$的时间复杂度内完成参数的更新。

## 8.6 【定理补充】凸赌博机的遗憾界

**P182**中定理8.5给出了凸赌博机的遗憾界，在证明开始时，作者对$\eta,\alpha,\delta$的取值进行了限定。我们可以发现这些取值不是很直观，证明给出的解释也较为分散，部分取值与证明略有出入，因此我们在此进行补充。

对于步长$\eta$，在缩放（8.87）中 $\mathbb{E}[\sum_{t=1}^T\hat f_t(z_t)]-\min_{w\in(1-\alpha)\mathcal{W}}\sum_{t=1}^T\hat f_t(w)$ 时，为使用引理8.3创造条件，因此采用步长$\eta=\frac{\Lambda}{l'\sqrt{T}}$。根据（8.89）的推导，我们可令$\Lambda=\Lambda_2$且$l'=\frac{dc}{\delta}$，此时，将$\eta=\frac{\Lambda_2}{(dc/\delta)\sqrt T}$带入到更新公式（8.76）中即可得到（8.88）。

对于缩减系数$\alpha$与扰动系数$\delta$，可以一同考虑这两个系数的取值。观察（8.91）第一个不等式的形式，我们发现这是一个关于$\delta$的对钩函数：
$$
\begin{equation}
f(\delta)=A\delta+\frac{B}{\delta}+C
\end{equation}
$$
假设$\alpha$的取值与$\delta$无关，那么：
$$
\begin{equation}
A=3lT,B=dc\Lambda_2\sqrt T,C=2\alpha cT
\end{equation}
$$
令$f'(\delta)=0$，可得：
$$
\begin{equation}
\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_2}{3l}}
\end{equation}
$$
此时，$f(\delta)$的最小值为：
$$
\begin{equation}
f(\delta^*)=O(T^{3/4})
\end{equation}
$$
如果我们想加速收敛，则可将$\alpha$的取值与$\delta$相关联。根据上面的结论，当迭代次数$T$足够大时，必然有$\deltaarrow0$。因此，不妨取$\alpha=\frac{\delta}{\Lambda_1}$，代入（8.91）中并利用对钩函数$f(\delta)$的性质，得到：
$$
\begin{align}
&\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_1\Lambda_2}{3(l\Lambda_1+c)}} \\
&f(\delta^*)=O(T^{3/4})
\end{align}
$$
进一步地，可以发现，$\delta^*$的取值并不唯一，这是因为（8.91）的第二个不等式缩放并非必需。如果取$\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_1\Lambda_2}{3l\Lambda_1+2c}}$，同样可以得到更紧致的遗憾界，并保证定理的结论不变。
