# 第4章：泛化界

*Edit: 李一飞，王茂霖，Hao ZHAN*

*Update: 07/07/2020*

---

## 1. 【概念补充】可分情形中的“等效”假设

**P61**中的「可分情形」部分提到了“等效假设”的概念。这其实是我们在面对模型选择时所需要处理的问题。机器学习的任务实际上就是从样本空间（sample space）或者属性空间（attribute space）中选择一个最符合实际的模型假设。在一种最为理想的状态下，我们当然希望能够排除掉那些不可能的情况，直接选择出唯一可能的模型。然而这明显是不现实的，因为我们的训练数据不可能覆盖所有可能的情况，这些数据只是部分经验片段的记录而已。这使得机器学习成为了一个不适定问题（ill-posed problem）。

通常而言，不适定问题是指不满足以下任一条件的问题：

（1）Having a solution（有一个解）

（2）Having a unique solution（有一个唯一解）

（3）Having a solution that depends continuously on the parameters or input data.（解连续依赖于定解条件）

在这里，由于我们没有办法仅依靠输入数据本身来找到唯一解，因此这使得学习问题成为了一个不适定问题。也就是说，在这个地方之所以说学习问题是一个不适定问题，主要是因为违反了条件（2）。然而，在更多的时候，我们说机器学习是不适定的，主要是指其违反了条件（3），在那种情况下，我们通常会用正则化等方式来进行解决。



## 2.【概念补充】定理4.1与定理2.1、定理2.2的关系

**P61**中的**定理4.1**，其实与**定理2.1**和**定理2.2**之间存在密切的联系。

**定理2.1**指出一个学习算法  $\mathfrak{L}$  能从假设空间 $\mathcal{H}$ 中PAC辨识概念类 $\mathcal{C}$ ，需要满足：
$$
P(E(h) \leqslant \epsilon) \geqslant 1-\delta
$$
其中， 0 < $\epsilon$ , $\delta<1$，所有 $c \in \mathcal{C}$， $h \in \mathcal{H}$ 。

**定理2.2**指出，所谓PAC可学，是指对于任何 $m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$ ，学习算法  $\mathfrak{L}$  能从假设空间 $\mathcal{H}$ 中PAC辨识概念类 $\mathcal{C}$ 。

而在**定理4.1**中，则是利用了**定理2.1**和**定理2.2**，假设学习算法算法  $\mathfrak{L}$  能够从假设空间 $\mathcal{H}$ 中 PAC 辨识概念类 $\mathcal{C}$ ，且这一过程是依赖于大小为 m 的训练集 D 的，其中 $m \geqslant \frac{1}{\epsilon} \left( ln \left| \mathcal{H} \right| + ln \left| \frac{1}{\delta} \right| \right)$ ，也就是说，满足了
$$
m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))
$$


的条件，从而得到
$$
P(E(h) \leqslant \epsilon) \geqslant 1-\delta
$$
因此，**定理4.1**实际上就是逆向使用了**定理2.1**和**定理2.2**而已。



## 3.【概念补充】可分情形下收敛率为何是 $O(1/m)$

**P61**中提出，随着训练集中样本数目的逐渐增加，泛化误差的上界逐渐趋于0，收敛率为 $O(1/m)$ 。

由于 *$m \geq \frac{1}{\epsilon}(ln|\mathcal{H}|+ln\frac{1}{\delta})$ 时，有 $P(E(h)\leq\epsilon)\geq 1-\delta$* ，我们可以看出在给定 $\delta$ 与 $\mathcal{H}$ 之后，我们考虑泛化误差上界关于样本数目增加时的收敛情况是在取等条件下考虑问题，事实上我们有， $\forall m_1, m_2，\ m_1>m_2\geq \frac{1}{\epsilon}(ln|\mathcal{H}|+ln\frac{1}{\delta})$ ，$\exist \epsilon_1<\epsilon,，s.t.m1\geq\frac{1}{\epsilon_1}(ln|\mathcal{H}|+ln\frac{1}{\delta}) >m_2$。因此，总有样本数量增大时误差上界减小。根据等式： 
$$
m = \frac{1}{\epsilon}(ln|\mathcal{H}|+ln\frac{1}{\delta})\Rightarrow \epsilon = \frac{1}{m}(ln|\mathcal{H}|+ln\frac{1}{\delta})
$$
我们得到在其他因素确定的情况下，误差上界 $\epsilon$ 与 $\frac{1}{m}$ 成正比，因此收敛速度为 $o(1/m)$。



## 4.【证明补充】不可分情形下收敛率为何是 $O(1/\sqrt{m})$

**P62**中提出，在有限不可分的情形下，泛化误差的收敛率为 $O(1/m)$ 

不可分情形下我们无法得到类似于可分情形下的定理，这是因为不可分情形下无法保证做到经验误差为0，因此只能给出泛化误差关于经验误差的上界。即定理中的 $P(E(h))\leq \hat E+\sqrt{\frac{ln|\mathcal{H}|+ln(2/\delta)}{2m}}\geq1-\delta$ 。可以看到，同可分情形下一样，不可分情形下同样是按照取等条件分析，因此可以得到样本数量增大时误差上界减小。根据等式： 
$$
\epsilon = \sqrt{\frac{1}{2m}(ln|\mathcal{H}|+ln\frac{2}{\delta})}
$$
 我们得到在其他因素确定的情况下，误差上界 $\epsilon$ 与 $\sqrt{\frac{1}{m}}$ 成正比，因此收敛速度为 $o(1/\sqrt{m})$。



## 5.【证明补充】定理4.2补充

**P63**中，在证明定理4.2时，省略了式4.6到式4.7的推导过程。在这一过程中，主要用到了**P28**中式2.7的内容。

根据式4.6，有
$$
\begin{aligned}
& P(\exists h \in \mathcal{H}:|\widehat{E}(h)-E(h)|>\epsilon) \\
=& P\left(\left(\left|\widehat{E}\left(h_{1}\right)-E\left(h_{1}\right)\right|>\epsilon\right) \vee \cdots \vee\left(\left|\widehat{E}\left(h_{|\mathcal{H}|}\right)-E\left(h_{|\mathcal{H}|}\right)\right|>\epsilon\right)\right) \\
\leqslant & \sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-E(h)|>\epsilon)
\end{aligned}
$$
引理2.1提出，若训练集 D 包含 $m$ 个从分布 D 上独立同分布采样而得的样本, $0<\epsilon<1,$ 则对任意 $h \in \mathcal{H},$ 有
$$
\begin{aligned}
P(\widehat{E}(h)-E(h) \geqslant \epsilon) & \leqslant \exp \left(-2 m \epsilon^{2}\right) \\
P(E(h)-\widehat{E}(h) \geqslant \epsilon) & \leqslant \exp \left(-2 m \epsilon^{2}\right) \\
P(|E(h)-\widehat{E}(h)| \geqslant \epsilon) & \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)
\end{aligned}
$$
使用第三个式子，即，
$$
P(|E(h)-\widehat{E}(h)| \geqslant \epsilon) \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)
$$
将其带入式4.6，则有，
$$
\begin{array}{l}
\sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-E(h)|>\epsilon) \leqslant \sum_{h \in \mathcal{H}} 2 \exp \left(-2 m \epsilon^{2}\right)
\end{array}
$$
令 $2 \exp \left(-2 m \epsilon^{2}\right)=\delta /|\mathcal{H}|$，则有，
$$
\begin{array}{l}
\sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-E(h)|>\epsilon) \leqslant \sum_{h \in \mathcal{H}} \delta /|\mathcal{H}| \leqslant|\mathcal{H}| \cdot \delta /|\mathcal{H}|=\delta
\end{array}
$$
从而得到式4.7。



## 6.【证明补充】引理4.1的证明思路

**P63**中，引入了引理4.1及其相关的证明。但是由于证明过程较长，可能不利于理解。因此在这里对其证明思路进行梳理和分析。

#### （1）证明简述

当我们想要证明这个定理的时候，我们需要首先回忆一下增长函数的定义：

> 对于 $m \in\mathbb{N}$ ,假设空间 $\mathcal{H}$ 的**增长函数** (growth function) $\Pi_{\mathcal{H}}(m)$ 表示为
> $$
> \Pi_{\mathcal{H}}(m)=max_{\{\mathbf{x}_1,...,\mathbf{x}_m\}\subset \mathcal{X}}|\{(h(\mathbf{x}_1),...,h(\mathbf{x}_m))|h\subset \mathcal{H}\}|
> $$

并且考虑到泛化误差在实际过程中难以评估，所以要想办法通过别的方式转化，但此时是无限维空间，并不能像之前一样简单的假设 $|\mathcal{H}|$ 。因此**我们考虑证明时，首先要想办法将泛化误差和经验误差的差距放缩为经验误差之间的差距**，之后再进行证明，因此得到了我们证明中的第一步：
$$
\frac{1}{2}P(sup_{h\in\mathcal{H}}|E(h)-\hat E_{D'}(h)|>\epsilon)\leq P(sup_{h\in\mathcal{H}}|\hat E_D(h)-\hat E_{D'}(h)|\geq\frac{1}{2}\epsilon)
$$
证明的过程中，中心思想是将概率问题转化为示性函数的期望问题。通过概率与期望之间的转化，我们将问题进一步转化，并通过上确界的定义给出了一个具体的概念 $h_0$ 从而用三角不等式将经验误差与泛化误差之间的差距放缩到了经验误差之间。再使用 Chebyshev 不等式中概率与分布函数积分的关系，将三角不等式拆分中的后一半概率转化为了期望，得到了前一半概率（即经验误差与泛化误差之间的差距）与经验误差之间的不等式。

第二步则是考虑将经验误差之间的差距进一步转化放缩至增长函数，即得到了第二个要证明的式子：
$$
P(sup_{h\in\mathcal{H}}|\hat E_D(h)-\hat E_{D'}(h)|\geq\frac{1}{2}\epsilon)\leq 2|\mathcal{H}_{|D+D'|}exp(-\frac{\epsilon^2m}{8})|
$$


同样的考虑将概率问题转化为示性函数的期望问题。这一步中使用了式 4.16，通过给出任意置换下的情况将期望问题转化成了级数求和，自然的放缩成了有关指数函数的式子：
$$
\frac{1}{2m}\Sigma^{(2m)!}_{i=1}\mathbb{I}(|\hat E_{T_iD}(h)-\hat E_{T_iD'}(h)||)=\Sigma_{k\in[l]\\s.t.|2k/m-l/m|\geq\epsilon/2}\frac{\tbinom{l}{k}\tbinom{2m-l}{m-k}}{\tbinom{2m}{m}}
$$


再通过进一步放缩，得到了最后的放缩式。再将前后结合便能够证明引理。

#### （2）思路分析

这里引理构造思路分为了几块：

1. 由于无限假设空间无法使用 $|\mathcal{H}|$ 分析，因此，这里要考虑第三章提到的 VC 维，使用它来分析无限假设空间的问题，而在 3.1 章中我们介绍了 VC 维和增长函数的关系。由于增长函数的定义与假设密切相关，因此考虑先得到关于增长函数的不等式也就不足为奇了。
2. 由于泛化误差难以直接得到评估，因此我们自然地会想到考虑放缩至经验误差之间的关系，再考虑到泛化误差难以整体评估（事实上我们正是在考虑它的上界来避免直接分析它），因此我们考虑使用某个具体的假设 $h_0$ ，再由数学证明中非常重要的三角不等式来分解。
3. 想办法将经验误差转化到增长函数的过程中非常重要的一点就在于考虑增长函数一定大于等于任何一个 $|\mathcal{H}_{|D+D'|}|$ 因此，我们在之前将概率问题转化到期望问题的基础上，考虑对每一个 $h$ 进行分析放缩。因此这里使用了式 4.16 来考虑每一个情况。



## 7.【证明补充】二分类问题支持向量机泛化误差界

**P67**定理4.3的证明过程中提到将式（4.24）带入引理4.1，该证明过程补充如下。

定理4.3 为
$$
P(|E(h)-\widehat{E}(h)| \leqslant \sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \geqslant 1-\delta
$$
可以将其等价转化为
$$
P(|E(h)-\widehat{E}(h)| > \sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant \delta
$$

而将（4.24）带入引理4.1可得
$$
P(|E(h)-\widehat{E}(h)|>\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right)
$$

从3.1可得：
$$
4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right) \leqslant 4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right)
$$

所以引理4.1可以转化为：
$$
P(|E(h)-\widehat{E}(h)|>\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right) \leqslant 4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right)
$$

令 $4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right) = \delta$

由此，可得到 
$$
P(|E(h)-\widehat{E}(h)|>\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant \delta
$$
从而得到了定理4.3的结论。

定理4.3其实表明了期望误差和经验误差之间的差异程度以概率的形式限定到了一定的区域范围之内，虽然这并不完全代表其误差一定会在$\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}$这个范围之内,但在这其中的的概率达到了 $1-\delta$ 。并且我们从中可以发现其差异程度的控制范围和样本量以及其维度之间的关系，我们可以发现，当 $\frac{m}{d}$ 较大的时候（其代表样本量大，而 VC 维较低，这是数据在空间中分布较为密集的情况）由于 $ln(x)$ 相对于 $x$ 增加较慢，所以其差异可以控制的越小，反之亦然。



## 8.【概念补充】回顾 Rademacher 复杂度

**P68**谈论了基于 Rademacher 的泛化误差界，在此对 Rademacher 复杂度进行一下回顾。

对于 Rademacher 复杂度而言，由于 VC 维和数据分布无关，未考虑数据的特定分布情况，所以其得到的结论往往是“松”的，但由于其未考虑分布，所以其对应的会更为具有“普适性”，而 Rademacher 复杂度是基于数据分布所以考虑的，这在牺牲掉一定“普适性”下，可以得到更为“紧”的结论。

首先我们来说一下复杂度这个概念其实是人为定义的一套量化复杂度程度的概念。而对应 Rademacher 复杂度而言，其复杂度越高，代表假设空间中有表示能力越强的函数，回到P46-47页，如果 $\mathbb{E}_{\boldsymbol{\sigma}}\left[\sup _{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^{m} \sigma_{i} h\left(\boldsymbol{x}_{i}\right)\right]=1$ ,也就是对于 $x$ 的任意标签分布情况都能打散（特别注意这里针对的是这个特定的 $x$ 数据，这也是 Rademacher 复杂度和数据分布相关的原因，我们只有知道数据的具体分布情况，才能求解其 Rademacher 复杂度）。并且由3.27可等价得到3.29的经验 Rademacher 复杂度。

而对于 Rademacher 复杂度的定义，我们进一步将具体的数据样本点，转化为数据的样本空间分布，在经验误差的形式外面在套一层期望，从而得到了一般化的 Rademacher 复杂度的定义。经验 Rademacher 复杂度和 Rademacher 复杂度的关系就如同概率论中掷硬币的观测序列和将其视为一个先验分布的随机变量序列一样。



## 9.【概念补充】$\rho$-间隔损失函数的 Lipschitz 性

**P79**提到由经验损失（4.72）可知 $\Phi_\rho$ 最多是 $\frac{1}{\rho} - Lipschitz$。

考虑由 Lipschitz 的定义证明，由拉格朗日中值定理我们得到$|\Phi_\rho(x_1)-\Phi_\rho(x_2)|\leq|\Phi_\rho'(\xi)||x_1-x_2|$ ，由于 $\Phi_\rho$ 表达式已经给出，我们可以直接计算其导数，得到$|\Phi_\rho'(\xi)|\leq\frac{1}{\rho}$，因此直接根据定义我们可以得到 $\rho$-间隔损失函数是 $\frac{1}{\rho}-Lipschitz$ 函数



## 10.【证明补充】二分类问题支持向量机泛化误差界

**P79**的定理4.8给出了机遇间隔损失函数的而分类问题SVM的泛化误差界。

此处存在一个小的错误：证明中 4.80 式前的 **“代入 (4.96)”** 应为 **“代入 (4.76)”**。

通过观察要证明的公式我们可以发现这是关于 Rademacher 复杂度的泛化上界推理，那么自然地回顾一下 Rademacher 复杂度：

> 现实任务中样本标记有时候会受到噪声的影响，因此我们与其在假设空间 $\mathcal{H}$ 中选择训练集上表现最好的假设，不如选择 $\mathcal{H}$ 中事先已经考虑了随机噪声影响的假设。

我们在这里直接考虑利用前面讲到的关于实值假设空间中的期望与 Rademacher 复杂度的不等式，并通过前面 4.73 讲到的关于间隔函数的经验间隔损失的式子可以带入得到大体的形式。

由于前面引理提到的关于 Lipschitz 函数的性质，同时我们说明了$\rho$-间隔损失函数的 Lipschitz 性，因此在简单改写复杂度之后便能得到要证明的定理。

第二个定理同理可证。