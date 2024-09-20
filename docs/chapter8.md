# 第8章：遗憾界

*编辑：赵志民，Hao ZHAN*

---

## 本章前言

本章的内容围绕学习理论中的遗憾（regret）概念展开（有的教材里也翻译为“悔”）。通常，我们使用超额风险（excess risk）来评估批量学习的分类器性能，而用遗憾来评估在线学习的分类器性能。二者的不同在于，前者衡量的是整个学习过程结束后所得到的分类器性能，可以理解为学习算法**最终输出的模型**与假设空间内**最优模型**的风险之差；而后者衡量的是算法运行过程中，所产生的**模型**与假设空间内**最优模型**的损失之差的**和**。

## 8.1 【概念解释】超额风险与遗憾的区别

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

## 8.2 【案例分享】Maler 算法

在8.2.3节的**170页**末尾，作者提到了Maler算法（multiple sub-algorithms and learning rates）（详细证明参考：[Adaptivity and Optimality: A Universal Algorithm for Online Convex Optimization](http://proceedings.mlr.press/v115/wang20e.html)），这是一个能够自适应选择最优专家的在线学习算法，并在不同类型的损失函数上实现最优的遗憾界限：
- **一般凸函数**：$R(T) \leq O\sqrt{T})$
- **指数凹函数**：$R(T) \leq O(d\log T)$
- **强凸函数**：$R(T) \leq O(\log T)$
这里$T$表示时间总步数，$d$表示特征空间的维度。

下面，我们简要补充Maler算法的原理和实现。

### 假设和定义

1. **假设 1（梯度有界性）**：所有损失函数 $f_t(x)$ 的梯度被 $G$ 所有界：
   $$
   \begin{equation}
   \forall t \gt 0, \quad \max_{x \in D} \|\nabla f_t(x)\| \leq G
   \end{equation}
   $$

2. **假设 2（行动集的直径有界性）**：行动集 $D$ 的直径被 $D$ 所有界：
   $$
   \begin{equation}
   \max_{x_1, x_2 \in D} \|x_1 - x_2\| \leq D
   \end{equation}
   $$

3. **定义 1（凸函数）**：函数 $f : D \rightarrow \mathbb{R}$ 是凸的，如果：
   $$
   \begin{equation}
   f(x_1) \geq f(x_2) + \nabla f(x_2)^\top (x_1 - x_2), \quad \forall x_1, x_2 \in D
   \end{equation}
   $$

4. **定义 2（强凸函数）**：函数 $f : D \rightarrow \mathbb{R}$ 是 $\lambda$-强凸的，如果：
   $$
   \begin{equation}
   f(x_1) \geq f(x_2) + \nabla f(x_2)^\top (x_1 - x_2) + \frac{\lambda}{2} \|x_1 - x_2\|^2, \quad \forall x_1, x_2 \in D
   \end{equation}
   $$

5. **定义 3（指数凹函数）**：函数 $f : D \rightarrow \mathbb{R}$ 是 $\alpha$-指数凹的（简称 $\alpha$-exp-concave），如果：
   $$
   \begin{equation}
   \exp(-\alpha f(x)) \text{是凹的}
   \end{equation}
   $$

### 元算法（Maler）

**输入**：学习率 $\eta^c, \eta_1, \eta_2, \dots$，专家的先验权重 $\pi_1^c, \pi_1^{\eta_1,s}, \pi_1^{\eta_2,s} \dots$，以及 $\pi_1^{\eta_1,l}, \pi_1^{\eta_2,l}, \dots$。

1. **对于每个回合 $t = 1, \dots, T$：**
   - 从凸专家算法（专家 1）获取预测 $x^c_t$，从指数凹专家算法（专家 2）和强凸专家算法（专家 3）分别获取 $x^{\eta, l}_t$ 和 $x^{\eta, s}_t$。
   - 执行：
     $$
     \begin{equation}
     x_t = \frac{\pi^c_t \eta^c x^c_t + \sum_{\eta} (\pi^{\eta,s}_t \eta x^{\eta,s}_t + \pi^{\eta,l}_t \eta x^{\eta,l}_t)}{\pi^c_t \eta^c + \sum_{\eta} (\pi^{\eta,s}_t \eta + \pi^{\eta,l}_t \eta)}
     \end{equation}
     $$
   - 观察梯度 $g_t$ 并发送给所有专家算法。
   - 对所有的 $\eta$ 更新权重：
     $$
     \begin{equation}
     \pi^c_{t+1} = \frac{\pi^c_t e^{-c_t(x^c_t)}}{\Phi_t}, \quad \pi^{\eta,s}_{t+1} = \frac{\pi^{\eta,s}_t e^{-s^{\eta}_t(x^{\eta,s}_t)}}{\Phi_t}, \quad \pi^{\eta,l}_{t+1} = \frac{\pi^{\eta,l}_t e^{-l^{\eta}_t(x^{\eta,l}_t)}}{\Phi_t}
     \end{equation}
     $$

     其中：
     $$
     \begin{equation}
     \Phi_t = \sum_{\eta} (\pi^{\eta,s}_t e^{-s^{\eta}_t(x^{\eta,s}_t)} + \pi^{\eta,l}_t e^{-l^{\eta}_t(x^{\eta,l}_t)} ) + \pi^c_t e^{-c_t(x^c_t)}
     \end{equation}
     $$

### 凸专家算法（专家 1）

1. $x^c_1 = 0$
2. **对于每个回合 $t = 1, \dots, T$：**
   - 将 $x^c_t$ 发送给元算法
   - 从元算法接收梯度 $g_t$
   - 更新：
     $$
     \begin{equation}
     x^c_{t+1} = \Pi^{I_d}_D (x^c_t - \frac{D}{\eta^c G \sqrt{t}} \nabla c_t(x^c_t))
     \end{equation}
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
     \begin{equation}
\begin{align*}
     \Sigma_{t+1} &= \Sigma_t + \nabla l^{\eta}_t(x^{\eta,l}_t) \nabla l^{\eta}_t(x^{\eta,l}_t)^\top \\
     x^{\eta,l}_{t+1} &= \Pi^{\Sigma_{t+1}}_D (x^{\eta,l}_t - \frac{1}{\beta} \Sigma_{t+1}^{-1} \nabla l^{\eta}_t(x^{\eta,l}_t))
     \end{align*}
\end{equation}
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
     \begin{equation}
     x^{\eta,s}_{t+1} = \Pi^{I_d}_D (x^{\eta,s}_t - \frac{1}{2\eta^2 G^2 t} \nabla s^{\eta}_t(x^{\eta,s}_t))
     \end{equation}
     $$
     其中 $\nabla s^{\eta}_t(x^{\eta,s}_t) = \eta g_t + 2 \eta^2 G^2 (x^{\eta,s}_t - x_t)$



## 8.3 【证明补充】随机多臂赌博机的遗憾界

**172页**中定理8.3给出了随机多臂赌博机的遗憾界，我们在此基础上对公式（8.42）至（8.47）证明过程进行补充。

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
\begin{equation}
\begin{align*}
\mathbb{E}[n_i^T]&\le\lceil\frac{2\ln T}{\Delta_i^2}\rceil+2\sum_{t=1}^{T-1}\sum_{p=1}^{t-1}\sum_{q=l}^{t-1}t^{-4} \\
&\le\frac{2\ln T}{\Delta_i^2}+1+2\sum_{t=1}^{T-1}\sum_{p=1}^{t}\sum_{q=1}^{t}t^{-4} \\
&\le\frac{2\ln T}{\Delta_i^2}+1+2\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2} 
\end{align*}
\end{equation}
$$
根据$p$-级数判别法，当$p=2\gt1$时，级数收敛，因此$\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2}$是有界的。至于该级数的具体值，对定理的结论没有影响，因此我们可以将其视为一个常数，然后带入后续推导中。为了证明的完整性，我们对此进行简要说明。

$\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2}$的取值在数学界被称为Basel问题，推导过程涉及诸多前置定理，感兴趣的读者可以查看这个讲义：[The Basel Problem - Numerous Proofs](https://www.math.cmu.edu/~bwsulliv/basel-problem.pdf)。此处提供另一种在微积分变换中常见的缩放方法：
$$
\begin{equation}
\begin{align*}
\sum_{t=1}^{T-1}t^{-2}&\le1+\int_{1}^{T-1}\frac{1}{x^2}dx \\
&=1+(-\frac{1}{x})|_1^{T-1} \\
&=2-\frac{1}{T}
\end{align*}
\end{equation}
$$
对不等式两边同时取极限，可得：
$$
\begin{equation}
\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2}\le2
\end{equation}
$$
代入（8.46），同样可得类似（8.47）的结论。

这里继续沿用书中给出的$\lim_{T\rightarrow+\infty}\sum_{t=1}^{T}t^{-2}=\frac{\pi^2}{6}$，代入（8.46）得到遗憾界（8.47）：
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



## 8.4 【概念解释】线性赌博机

**176页**的8.3.2节介绍了线性赌博机的概念，我们在此基础上对参数估计部分进行补充。

为了估计线性赌博机的参数，我们将原问题转化为岭回归问题，即（8.52）：
$$
\begin{equation}
f(w)=(Y-w^T X)^T(Y-w^T X)+\lambda w^T w
\end{equation}
$$
为了求得最优解$w^*$，我们令$f'(w)=0$，可推导出（8.53）：
$$
\begin{equation}
\begin{align*}
\frac{\partial f(w)}{\partial w} = -2X^T(Y-w^T X)+2\lambda w &= 0 \\
\rightarrow X^TY &= (X^TX + \lambda I)w \\
\rightarrow w^* &= (X^TX + \lambda I)^{-1}X^TY
\end{align*}
\end{equation}
$$
相比于每次传入新数据$(x_t,y_t)$时从头计算$w_t$，这里巧妙地利用了 Sherman-Morrison-Woodbury 公式将任何形如$(A+uv^T)^{-1}$的矩阵逆转化为可逆矩阵$A$和列向量$u,v$之间的运算，在$O(d^2)$的时间复杂度内完成参数的更新。



## 8.5 【证明补充】Sherman-Morrison-Woodbury (或 Woodbury) 公式

**177页**的 Sherman-Morrison-Woodbury 公式变种是矩阵求逆中的一个重要工具，它可以通过已知矩阵的逆来快速计算被低秩修正的矩阵的逆。

该公式如下所示：
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
\begin{equation}
\begin{align*}
=& \{ I + UCVA^{-1} \} - \{ U (C^{-1} + VA^{-1}U )^{-1}VA^{-1} + UCVA^{-1}U (C^{-1} + VA^{-1}U )^{-1} VA^{-1} \} \\
=& I + UCVA^{-1} - (U + UCVA^{-1}U ) (C^{-1} + VA^{-1}U )^{-1}VA^{-1} \\
=& I + UCVA^{-1} - UC (C^{-1} + VA^{-1}U) (C^{-1} + VA^{-1}U)^{-1}VA^{-1} \\
=& I + UCVA^{-1} - UCVA^{-1} \\
=& I
\end{align*}
\end{equation}
$$



## 8.6 【证明补充】单样本的近似梯度

**第181页**的引理8.2给出了单样本条件下的梯度近似公式，本节将提供该引理的完整证明过程。

$$
\begin{equation}
\mathbb{E}_{u \in \mathbb{S}}[f(x+\delta u)u] = \frac{\delta}{d}\nabla \mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)]
\end{equation}
$$

其中：
- $d$ 为空间的维数；
- $\delta$ 为任意正数；
- $\mathbb{B}$ 为单位球的空间，即 $\mathbb{B} = \{v \in \mathbb{R}^d \mid \|v\| \leq 1\}$；
- $\mathbb{S}$ 为单位球的表面，即 $\mathbb{S} = \{u \in \mathbb{R}^d \mid \|u\| = 1\}$。

### 证明

为了证明上述等式，我们将分三个步骤进行推导。

#### 1. 表达左边的期望

首先，考虑左边的期望：

$$
\begin{equation}
\mathbb{E}_{u \in \mathbb{S}}[f(x+\delta u)u] = \frac{1}{\text{Vol}_{d-1}(\mathbb{S})} \int_{\mathbb{S}} f(x + \delta u) u \, dS(u)
\end{equation}
$$

其中，$\text{Vol}_{d-1}(\mathbb{S})$ 表示 $(d-1)$ 维单位球面的体积，$dS(u)$ 为球面上的微分面积元素。

进行变量替换，令 $w = \delta u$。此时：
- 当 $u \in \mathbb{S}$ 时，$w \in \delta \mathbb{S}$；
- 球面上的微分面积元素变化为 $dS(u) = \frac{dS(w)}{\delta^{d-1}}$，因为每个维度按 $\delta$ 缩放，$(d-1)$ 维体积按 $\delta^{d-1}$ 缩放。

将变量替换代入期望的表达式：

$$
\begin{equation}
\mathbb{E}_{u \in \mathbb{S}}[f(x+\delta u)u] = \frac{1}{\text{Vol}_{d-1}(\mathbb{S})} \int_{\mathbb{S}} f(x + \delta u) u \, dS(u) = \frac{1}{\text{Vol}_{d-1}(\mathbb{S}) \cdot \delta^{d-1}} \int_{\delta \mathbb{S}} f(x + w) \frac{w}{\delta} \, dS(w)
\end{equation}
$$

简化后得到：

$$
\begin{equation}
\mathbb{E}_{u \in \mathbb{S}}[f(x+\delta u)u] = \frac{1}{\text{Vol}_{d-1}(\delta \mathbb{S})} \int_{\delta \mathbb{S}} f(x + w) \frac{w}{\|w\|} \, dS(w)
\end{equation}
$$

#### 2. 表达右边的期望及其梯度

接下来，考虑右边的期望：

$$
\begin{equation}
\mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)] = \frac{1}{\text{Vol}_d(\mathbb{B})} \int_{\mathbb{B}} f(x + \delta v) \, dv
\end{equation}
$$

其中，$\text{Vol}_d(\mathbb{B})$ 表示 $d$ 维单位球的体积，$dv$ 为体积上的微分元素。

同样进行变量替换，令 $w = \delta v$。则：
- 当 $v \in \mathbb{B}$ 时，$w \in \delta \mathbb{B}$；
- 微分体积元素变化为 $dv = \frac{dw}{\delta^d}$，因为每个维度按 $\delta$ 缩放，体积按 $\delta^d$ 缩放。

代入后得到：

$$
\begin{equation}
\mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)] = \frac{1}{\text{Vol}_d(\mathbb{B}) \cdot \delta^d} \int_{\delta \mathbb{B}} f(x + w) \, dw = \frac{1}{\text{Vol}_d(\delta \mathbb{B})} \int_{\delta \mathbb{B}} f(x + w) \, dw
\end{equation}
$$

为了计算 $\nabla \mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)]$，令：

$$
\begin{equation}
F(x) = \mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)] = \frac{1}{\text{Vol}_d(\delta \mathbb{B})} \int_{\delta \mathbb{B}} f(x + w) \, dw
\end{equation}
$$

梯度作用在积分上，由于 $x$ 和 $w$ 是独立变量，可以将梯度算子移入积分内部：

$$
\begin{equation}
\nabla F(x) = \frac{1}{\text{Vol}_d(\delta \mathbb{B})} \int_{\delta \mathbb{B}} \nabla_x f(x + w) \, dw
\end{equation}
$$

注意到：

$$
\begin{equation}
\nabla_x f(x + w) = \nabla_w f(x + w)
\end{equation}
$$

这是因为 $x$ 和 $w$ 的关系是通过相加连接的，故梯度对 $x$ 的作用等同于对 $w$ 的作用。

根据散度定理，有：

$$
\begin{equation}
\int_{\delta \mathbb{B}} \nabla_w f(x + w) \, dw = \int_{\delta \mathbb{S}} f(x + w) n(w) \, dS(w)
\end{equation}
$$

其中，$\delta \mathbb{S}$ 是半径为 $\delta$ 的球面，$n(w)$ 为点 $w$ 处的单位外法向量。因此：

$$
\begin{equation}
\nabla F(x) = \frac{1}{\text{Vol}_d(\delta \mathbb{B})} \int_{\delta \mathbb{S}} f(x + w) \frac{w}{\|w\|} \, dS(w)
\end{equation}
$$

#### 3. 关联两边的表达式

将步骤 1 和步骤 2 的结果进行对比，可以得到：

$$
\begin{equation}
\mathbb{E}_{u \in \mathbb{S}}[f(x+\delta u)u] = \frac{\text{Vol}_d(\delta \mathbb{B})}{\text{Vol}_{d-1}(\delta \mathbb{S})} \nabla \mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)]
\end{equation}
$$

为了确定系数，我们需要利用 $d$ 维球的体积与表面积之间的关系。

$d$ 维球的体积与半径 $\delta$ 的关系为：

$$
\begin{equation}
\text{Vol}_d(\delta \mathbb{B}) = \delta^d \cdot \text{Vol}_d(\mathbb{B})
\end{equation}
$$

而球面的表面积与半径 $\delta$ 的关系为：

$$
\begin{equation}
\text{Vol}_{d-1}(\delta \mathbb{S}) = \delta^{d-1} \cdot \text{Vol}_{d-1}(\mathbb{S})
\end{equation}
$$

结合这两个关系，可以得到：

$$
\begin{equation}
\text{Vol}_d(\delta \mathbb{B}) = \int_0^{\delta} \text{Vol}_{d-1}(\mathbb{rS}) \, dr = \int_0^{\delta} \text{Vol}_{d-1}(\mathbb{S}) \, r^{d-1} \, dr = \frac{\text{Vol}_{d-1}(\mathbb{S}) \cdot \delta^{d}}{d} = \frac{\delta}{d} \cdot \text{Vol}_{d-1}(\delta \mathbb{S})
\end{equation}
$$

带入上述等式中，得证：

$$
\begin{equation}
\mathbb{E}_{u \in \mathbb{S}}[f(x+\delta u)u] = \frac{\delta}{d}\nabla \mathbb{E}_{v \in \mathbb{B}}[f(x + \delta v)]
\end{equation}
$$



## 8.7 【证明补充】凸赌博机的在线梯度下降


**182页**中引理8.3给出了凸赌博机的随机版本在线梯度下降，我们在此给出完整的证明过程。

设 $f_1, f_2, \dots, f_T: W \to \mathbb{R}$ 为一列凸且可微的函数，$\omega_1, \omega_2, \dots, \omega_T \in W$ 的定义满足 $\omega_1$ 为任意选取的点，且 $\omega_{t+1} = \Pi_W(\omega_t − \eta g_t)$，其中 $\eta \gt 0$，且 $g_1, \dots, g_T$ 是满足 $\mathbb{E}[g_t|\omega_t] = \nabla f_t(\omega_t)$ 的随机向量变量，且 $\|g_t\| \leq l$，其中 $l \gt 0$。则当 $\eta = \frac{\Lambda}{l\sqrt{T}}$ 时，有：

$$
\begin{equation}
\sum_{t=1}^{T} \mathbb{E}[f_t(\omega_t)] - \min_{\omega \in W} \sum_{t=1}^{T} f_t(\omega) \le l\Lambda \sqrt{T}
\end{equation}
$$

**证明:**  
设 $\omega^\star$ 为在 $W$ 中使 $\sum_{t=1}^{T} f_t(\omega)$ 最小化的点。由于 $f_t$ 是凸且可微的，我们可以使用梯度界定 $f_t(\omega_t)$ 和 $f_t(\omega^\star)$ 之间的差异：

$$
\begin{equation}
f_t(\omega^\star) - f_t(\omega_t) \ge \nabla f_t(\omega_t)^\top (\omega^\star − \omega_t) = \mathbb{E}[g_t|\omega_t]^\top (\omega^\star − \omega_t)
\end{equation}
$$

对该不等式取期望，得到：

$$
\begin{equation}
\mathbb{E}[f_t(\omega_t) − f_t(\omega^\star)] \leq \mathbb{E}[g_t^\top (\omega_t − \omega^\star)]
\end{equation}
$$

我们使用 $\|\omega_t − \omega^\star\|^2$ 作为潜在函数。注意到 $\|\Pi_W(\omega) − \omega^\star\| \leq \|\omega − \omega^\star\|$，因此：

$$
\begin{equation}
\begin{align*}
\|\omega_{t+1} − \omega^\star\|^2 &= \|\Pi_W(\omega_t − \eta g_t) − \omega^\star\|^2 \\
&\leq \|\omega_t − \eta g_t − \omega^\star\|^2 \\
&= \|\omega_t − \omega^\star\|^2 + \eta^2 \|g_t\|^2 − 2\eta (\omega_t − \omega^\star)^\top g_t \\
&\leq \|\omega_t − \omega^\star\|^2 + \eta^2 l^2 − 2\eta (\omega_t − \omega^\star)^\top g_t
\end{align*}
\end{equation}
$$

整理后得到：

$$
\begin{equation}
g_t^\top (\omega_t − \omega^\star) \leq \frac{\|\omega_t − \omega^\star\|^2 − \|\omega_{t+1} − \omega^\star\|^2 + \eta^2 l^2}{2\eta}
\end{equation}
$$

因此，我们有：

$$
\begin{equation}
\begin{align*}
\sum_{t=1}^{T} \mathbb{E}[f_t(\omega_t)] − \sum_{t=1}^{T} f_t(\omega^\star) &= \sum_{t=1}^{T} \mathbb{E}[f_t(\omega_t) − f_t(\omega^\star)] \\
&\leq \sum_{t=1}^{T} \mathbb{E}[g_t^\top (\omega_t − \omega^\star)] \\
&\leq \sum_{t=1}^{T} \mathbb{E} \left[\frac{\|\omega_t − \omega^\star\|^2 − \|\omega_{t+1} − \omega^\star\|^2 + \eta^2 l^2}{2\eta}\right] \\
&= \frac{\mathbb{E}[\|\omega_1 − \omega^\star\|^2] - \mathbb{E}[\|\omega_{T+1} − \omega^\star\|^2]}{2\eta} + \frac{T \eta l^2}{2} \\
&\le \frac{\mathbb{E}[\|\omega_1 − \omega^\star\|^2]}{2\eta} + \frac{T \eta l^2}{2} \\
&\le \frac{\Lambda^2}{2\eta} + \frac{T \eta l^2}{2}
\end{align*}
\end{equation}
$$

代入 $\eta = \frac{\Lambda}{l\sqrt{T}}$ 可得最终结果。



## 8.8 【证明补充】凸赌博机的缩减投影误差

**182页**中引理8.4给出了凸赌博机的缩减投影误差，我们在此给出完整的证明过程。

设 $f_1, f_2, \dots, f_T: W \to \mathbb{R}$ 为一列凸且可微的函数且 $\forall \omega \in W,i \in [T]$ 满足 $|f_i(\omega)| \le c$，有：

$$
\begin{equation}
\min_{\omega \in (1−\alpha)W} \sum_{t=1}^T f_t(\omega) - \min_{\omega \in W} \sum_{t=1}^T f_t(\omega) \leq 2\alpha cT
\end{equation}
$$

### 证明
  
显然，$(1−\alpha)W \subseteq W$。因此，有：

$$
\begin{equation}
\min_{\omega \in (1−\alpha)W} \sum_{t=1}^T f_t(\omega) = \min_{\omega \in W} \sum_{t=1}^T f_t((1−\alpha)\omega)
\end{equation}
$$

由于每个$f_t$是凸函数，且$0 \in W$，则我们有：

$$
\begin{equation}
\begin{align*}
\min_{\omega \in W} \sum_{t=1}^T f_t((1−\alpha)\omega) &\leq \min_{\omega \in W} \sum_{t=1}^T \alpha f_t(0) + (1−\alpha) f_t(\omega) \\
&= \min_{\omega \in W} \sum_{t=1}^T \alpha (f_t(0) − f_t(\omega)) + f_t(\omega)
\end{align*}
\end{equation}
$$

最后，由于对于任意$\omega \in W$和$t \in \{1, \dots, T\}$，我们有$|f_t(\omega)| \leq c$，因此可以得出：

$$
\begin{equation}
\begin{align*}
\sum_{t=1}^{T} \min_{\omega \in W} \alpha (f_t(0) − f_t(\omega)) + f_t(\omega) &\leq \min_{\omega \in W}\sum_{t=1}^{T} 2\alpha c + f_t(\omega) \\
&= 2\alpha cT + \min_{\omega \in W} \sum_{t=1}^{T} f_t(\omega)
\end{align*}
\end{equation}
$$

进行适当移项即可得原不等式。



## 8.9 【证明补充】凸赌博机的遗憾界

**182页**中定理8.5给出了凸赌博机的遗憾界，在证明开始时，作者对$\eta,\alpha,\delta$的取值进行了限定。我们可以发现这些取值不是很直观，证明给出的解释也较为分散，部分取值与证明略有出入，因此我们在此进行补充。

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
如果我们想加速收敛，则可将$\alpha$的取值与$\delta$相关联。根据上面的结论，当迭代次数$T$足够大时，必然有$\delta\rightarrow0$。因此，不妨取$\alpha=\frac{\delta}{\Lambda_1}$，代入（8.91）中并利用对钩函数$f(\delta)$的性质，得到：
$$
\begin{equation}
\begin{align*}
&\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_1\Lambda_2}{3(l\Lambda_1+c)}} \\
&f(\delta^*)=O(T^{3/4})
\end{align*}
\end{equation}
$$
进一步地，可以发现，$\delta^*$的取值并不唯一，这是因为（8.91）的第二个不等式缩放并非必需。如果取$\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_1\Lambda_2}{3l\Lambda_1+2c}}$，同样可以得到更紧致的遗憾界，并保证定理的结论不变。
