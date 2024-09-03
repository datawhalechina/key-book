# 第4章：泛化界

*Edit: 李一飞，王茂霖，Hao ZHAN，赵志民*

------

## 本章前言

在机器学习中，泛化能力是衡量模型性能的核心标准之一。如何从有限的训练数据中获得能够在未见数据上表现良好的模型，始终是研究者关注的重要问题。本章将深入探讨与泛化界相关的理论基础和定理，通过对关键概念的补充说明和定理的详细推导，帮助读者更好地理解泛化误差的收敛性质以及不同假设空间下的泛化能力。本章还将介绍与泛化界密切相关的Rademacher复杂度及其在实际应用中的意义，为进一步的研究提供理论支持。

## 4.1 【概念补充】可分情形中的“等效”假设

**P61**中的「可分情形」部分提到了“等效假设”的概念。这其实是我们在面对模型选择时需要处理的问题。机器学习的任务实际上是从样本空间或属性空间中选择一个最符合实际的模型假设。在理想状态下，我们希望能排除不可能的情况，直接选择唯一可能的模型。然而，这是不现实的，因为训练数据无法覆盖所有可能的情况，这些数据仅是部分经验片段的记录。因此，机器学习成为了一个不适定问题（ill-posed problem）。

通常而言，不适定问题是指不满足以下任一条件的问题：

1. **存在解**（Having a solution）
2. **唯一解**（Having a unique solution）
3. **解连续依赖于定解条件**（Having a solution that depends continuously on the parameters or input data）

在这里，由于我们无法仅依靠输入数据找到唯一解，这使得学习问题成为一个不适定问题，主要违反了条件2。而在更多时候，我们说机器学习是不适定的，主要是指其违反了条件3，在那种情况下，我们通常会用正则化等方式来解决。

## 4.2【概念补充】定理4.1与定理2.1、定理2.2的关系

**P61**中的**定理4.1**与**定理2.1**和**定理2.2**之间存在密切联系。

**定理2.1**指出一个学习算法 $\mathfrak{L}$ 能从假设空间 $\mathcal{H}$ 中PAC辨识概念类 $\mathcal{C}$ ，需要满足：
$$
\begin{equation}
P(E(h) \leqslant \epsilon) \geqslant 1-\delta
\end{equation}
$$
其中， $0 < \epsilon, \delta < 1$，所有 $c \in \mathcal{C}$， $h \in \mathcal{H}$ 。

**定理2.2**指出，所谓PAC可学，是指对于任何 $m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$ ，学习算法 $\mathfrak{L}$ 能从假设空间 $\mathcal{H}$ 中PAC辨识概念类 $\mathcal{C}$ 。

在**定理4.1**中，假设学习算法 $\mathfrak{L}$ 能从假设空间 $\mathcal{H}$ 中 PAC 辨识概念类 $\mathcal{C}$，且这一过程依赖于大小为 $m$ 的训练集 $D$ ，其中 $m \geqslant \frac{1}{\epsilon} \left( \ln \left| \mathcal{H} \right| + \ln \frac{1}{\delta} \right)$，满足
$$
\begin{equation}
m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))
\end{equation}
$$
的条件，从而得到
$$
\begin{equation}
P(E(h) \leqslant \epsilon) \geqslant 1-\delta
\end{equation}
$$
因此，**定理4.1**实际上就是逆向使用了**定理2.1**和**定理2.2**。

## 4.3【概念补充】可分情形下收敛率为何是 $O(\frac{1}{m})$

**P61**中提出，随着训练集中样本数目的增加，泛化误差的上界逐渐趋于0，收敛率为 $O(\frac{1}{m})$ 。

由于 $m \geq \frac{1}{\epsilon}( \ln|\mathcal{H}|+\ln\frac{1}{\delta})$ 时，有 $P(E(h)\leq\epsilon)\geq 1-\delta$，我们可以看出在给定 $\delta$ 与 $\mathcal{H}$ 后，考虑泛化误差上界关于样本数目的收敛情况是在取等条件下进行的。事实上，$\forall m_1, m_2，\ m_1>m_2\geq \frac{1}{\epsilon}( \ln|\mathcal{H}|+\ln\frac{1}{\delta})$ ，$\exist \epsilon_1<\epsilon,，s.t.m1\geq\frac{1}{\epsilon_1}( \ln|\mathcal{H}|+ln\frac{1}{\delta}) >m_2$。因此，总有样本数量增大时误差上界减小。根据等式： 
$$
\begin{equation}
m = \frac{1}{\epsilon}( \ln|\mathcal{H}|+\ln\frac{1}{\delta})\Rightarrow \epsilon = \frac{1}{m}( \ln|\mathcal{H}|+ \ln\frac{1}{\delta})
\end{equation}
$$
我们得到在其他因素确定的情况下，误差上界 $\epsilon$ 与 $\frac{1}{m}$ 成正比，因此收敛速度为 $o(1/m)$。

## 4.4【证明补充】不可分情形下收敛率为何是 $O(1/\sqrt{m})$

**P62**中提出，在有限不可分的情形下，泛化误差的收敛率为 $O(1/m)$ 。

不可分情形下我们无法得到类似于可分情形下的定理，因为不可分情形下无法保证做到经验误差为0，因此只能给出泛化误差关于经验误差的上界。即定理中的 $P(E(h))\leq \hat E+\sqrt{\frac{ \ln|\mathcal{H}|+\ln(2/\delta)}{2m}}\geq1-\delta$ 。同样可以看到，不可分情形下是按照取等条件分析，因此可以得到样本数量增大时误差上界减小。根据等式： 
$$
\begin{equation}
\epsilon = \sqrt{\frac{1}{2m}( \ln|\mathcal{H}|+\ln\frac{2}{\delta})}
\end{equation}
$$
 我们得到在其他因素确定的情况下，误差上界 $\epsilon$ 与 $\sqrt{\frac{1}{m}}$ 成正比，因此收敛速度为 $o(1/\sqrt{m})$。

## 4.5【证明补充】定理4.2补充

**P63**中，在证明定理4.2时，省略了从式4.6到式4.7的推导过程。在这一过程中，主要用到了**P28**中式2.7的内容。

根据式4.6，有
$$
\begin{align}
& P(\exists h \in \mathcal{H}:|\widehat{E}(h)-E(h)|>\epsilon) \\
=& P\left(\left(\left|\widehat{E}\left(h_{1}\right)-E\left(h_{1}\right)\right|>\epsilon\right) \vee \cdots \vee\left(\left|\widehat{E}\left(h_{|\mathcal{H}|}\right)-E\left(h_{|\mathcal{H}|}\right)\right|>\epsilon\right)\right) \\
\leqslant & \sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-E(h)|>\epsilon)
\end{align}
$$
引理2.1提出，若训练集 D 包含 $m$ 个从分布 D 上独立同分布采样而得的样本, $0<\epsilon<1,$ 则对任意 $h \in \mathcal{H},$ 有
$$
\begin{align}
P(\widehat{E}(h)-E(h) \geqslant \epsilon) & \leqslant \exp \left(-2 m \epsilon^{2}\right) \\
P(E(h)-\widehat{E}(h) \geqslant \epsilon) & \leqslant \exp \left(-2 m \epsilon^{2}\right) \\
P(|E(h)-\widehat{E}(h)| \geqslant \epsilon) & \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)
\end{align}
$$
使用第三个式子，即，
$$
\begin{equation}
P(|E(h)-\widehat{E}(h)| \geqslant \epsilon) \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)
\end{equation}
$$
将其带入式4.6，则有，
$$
\begin{equation}
\begin{array}{l}
\sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-E(h)|>\epsilon) \leqslant \sum_{h \in \mathcal{H}} 2 \exp \left(-2 m \epsilon^{2}\right)
\end{array}
\end{equation}
$$
令 $2 \exp \left(-2 m \epsilon^{2}\right)=\delta /|\mathcal{H}|$，则有，
$$
\begin{equation}
\begin{array}{l}
\sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-E(h)|>\epsilon) \leqslant \sum_{h \in \mathcal{H}} \delta /|\mathcal{H}| \leqslant|\mathcal{H}| \cdot \delta /|\mathcal{H}|=\delta
\end{array}
\end{equation}
$$
从而得到式4.7。

## 4.6【证明补充】引理4.1的证明思路

**P63**中，引入了引理4.1及其相关的证明。由于证明过程较长，这里对其思路进行梳理和分析。

### 4.6.1 证明简述

当我们要证明这个定理时，需要首先回忆增长函数的定义：

> 对于 $m \in\mathbb{N}$, 假设空间 $\mathcal{H}$ 的**增长函数** (growth function) $\Pi_{\mathcal{H}}(m)$ 表示为
> $$
\begin{equation}
> \Pi_{\mathcal{H}}(m)=\max_{\{\mathbf{x}_1,...,\mathbf{x}_m\}\subset \mathcal{X}}|\{(h(\mathbf{x}_1),...,h(\mathbf{x}_m))|h\subset \mathcal{H}\}|
> \end{equation}
$$

由于泛化误差在实际过程中难以评估，证明中首先将泛化误差和经验误差的差距缩放为经验误差之间的差距。通过概率与期望之间的转化，我们将问题进一步转化，并通过上确界的定义给出一个具体的概念 $h_0$ ，用三角不等式将经验误差与泛化误差之间的差距缩放至经验误差之间。再使用 Chebyshev 不等式中的概率与分布函数积分关系，拆分三角不等式，得出前一半概率（即经验误差与泛化误差之间的差距）与经验误差之间的不等式。

第二步则是将经验误差之间的差距进一步转化为增长函数的差距，即证明了第二个公式：
$$
\begin{equation}
P(sup_{h\in\mathcal{H}}|\hat E_D(h)-\hat E_{D'}(h)|\geq\frac{1}{2}\epsilon)\leq 2|\mathcal{H}_{|D+D'|}| \exp(-\frac{\epsilon^2m}{8})|
\end{equation}
$$

在这个过程中，使用了式 4.16，通过给出任意置换下的情况，将期望问题转化为级数求和，进一步缩放成有关指数函数的公式：
$$
\begin{equation}
\frac{1}{2m}\sum_{i=1}^{(2m)!}\mathbb{I}(|\hat E_{T_iD}(h)-\hat E_{T_iD'}(h)||)=\sum_{k\in[l]\\s.t.|2k/m-l/m|\geq\epsilon/2}\frac{\tbinom{l}{k}\tbinom{2m-l}{m-k}}{\tbinom{2m}{m}}
\end{equation}
$$

再通过进一步缩放，得到最后的缩放公式（4.19）。注意，原不等式中的上界 $2\exp(-\frac{\epsilon^2l}{8})$ 可以通过 Hoeffding 不等式推导出。

结合前述推导可证明引理。

即使将原不等式中的 $2\exp(-\frac{ε^2l}{8})$ 替换为 $2\exp(-\frac{ε^2l}{4})$，原不等关系依然成立。此结论亦可推广到定理4.3的结论，但即便如此，泛化误差的收敛率依旧为 $O(\sqrt\frac{ln(m/d)}{m/d})$。

### 4.6.2 思路分析

引理的构造思路分为以下几个部分：

1. 由于无限假设空间无法使用 $|\mathcal{H}|$ 进行分析，因此这里要考虑第三章提到的 VC 维，使用它来分析无限假设空间的问题。在 3.1 章中我们介绍了 VC 维和增长函数的关系。由于增长函数的定义与假设密切相关，因此先得到关于增长函数的不等式是自然的选择。
2. 由于泛化误差难以直接评估，因此考虑将其缩放至经验误差之间的关系。考虑到泛化误差难以整体评估（实际上我们正是在考虑它的上界来避免直接分析它），因此通过引入具体的假设 $h_0$ ，再使用三角不等式进行分解。
3. 通过放缩将经验误差转化为增长函数。关键在于增长函数一定大于等于任何一个 $|\mathcal{H}_{|D+D'|}|$，因此在将概率问题转化为期望问题的基础上，对每一个 $h$ 进行分析放缩，从而得到最终结论。

## 4.7【证明补充】引理4.2补充

为了证明**P74**的引理4.2，我们设想两枚硬币 $x_A$ 和 $x_B$。两枚硬币都稍有不均匀，即 $P[x_A = 0] = 1/2−\alpha/2$ 和 $P[x_B = 0] = 1/2+\alpha/2$，其中 $0<\alpha<1$。0 表示正面，1 表示反面。假设我们随机从口袋里拿出一枚硬币 $x \in \{x_A,x_B\}$，抛 $m$ 次，得到的 0 和 1 的序列即为引理中构造的随机变量 $\alpha_{\sigma}$。如果我们想通过序列推测是哪一枚硬币被抛出，即选取并求得最佳决策函数 $f:\{0,1\}^m\rightarrow\{x_A,x_B\}$，则该实验假设的泛化误差可表示为 $error(f)=\mathbb{E}[\mathbb{P}_{S\sim\mathcal{D}_\alpha^m}(f(S)\neq x)]$。

用 $f$ 代表任意决策函数，用 $F_A$ 代表满足 $f(S)=x_A$ 的样本集合，用 $F_B$ 代表满足 $f(S)=x_B$ 的样本集合，用 $N(S)$ 表示样本 $S$ 中出现 0 的个数，根据泛化误差的定义，有：
$$
\begin{align}
error(f)&=\sum_{S\in F_A}\mathbb{P}[S\wedge x_B]+\sum_{S\in F_B}\mathbb{P}[S\wedge x_A]\\
&=\frac{1}{2}\sum_{S\in F_A}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_B}\mathbb{P}[S|x_A]\\
&=\frac{1}{2}\sum_{S\in F_A\atop N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_A\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_B]
+\frac{1}{2}\sum_{S\in F_B\atop N(S)\lt \lceil m/2\rceil}\mathbb{P}[S|x_A]+\frac{1}{2}\sum_{S\in F_B\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
\end{align}
$$
如果 $N(S)\ge \lceil m/2\rceil$，易证 $\mathbb{P}[S|x_B]\ge\mathbb{P}[S|x_A]$。类似地，如果 $N(S)\lt \lceil m/2\rceil$，易证 $\mathbb{P}[S|x_A]\ge\mathbb{P}[S|x_B]$。因此，我们可以得到：
$$
\begin{align}
error(f) &\ge\frac{1}{2}\sum_{S\in F_A\atop N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_A\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]
+\frac{1}{2}\sum_{S\in F_B\atop N(S)\lt \lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_B\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
&=\frac{1}{2}\sum_{S:N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S:N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
&=error(f_o)
\end{align}
$$
因此，当我们选取 $f_o$ 为决策函数时，泛化误差取得最小值，即当且仅当 $N(S)\lt \lceil m/2\rceil$ 时，我们认为被抛的硬币是 $f_o(S)=x_A$。

注意到 $\mathbb{P}[N(S)\ge \lceil m/2\rceil|x=x_A]=\mathbb{P}[B(2\lceil m/2\rceil,p)\ge k]$，且 $p=1/2-\alpha /2,k=\lceil m/2\rceil$，因此 $2\lceil m/2\rceil p\le k\le 2\lceil m/2\rceil(1-p)$。

根据 Slud 不等式，我们有：
$$
\begin{align}
error(f_o) &\ge \frac{1}{2}\mathbb{P}[N\ge\frac{\lceil m/2\rceil\alpha}{\sqrt{1/2(1-\alpha^2)\lceil m/2\rceil}}]=\frac{1}{2}\mathbb{P}[N\ge\sqrt{\frac{2\lceil m/2\rceil}{1-\alpha^2}}\alpha]
\end{align}
$$

根据第一章补充内容中的正态分布不等式推论，我们有：
$$
\begin{equation}error(f_o)\ge\frac{1}{4}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})\ge\frac{1}{4}(1-\sqrt{1-e^{-u^2}})\end{equation}
$$
此处 $u=\sqrt{\frac{2\lceil m/2\rceil}{1-\alpha^2}}\alpha$

事实上，根据上面的推导，我们可以进一步提升泛化误差的下界，即：
$$
\begin{equation}\mathbb{E}[\mathbb{P}_{S\sim\mathcal{D}_\alpha^m}(f(S)\neq x)]\ge\frac{1}{4}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})\end{equation}
$$

在引理末尾处，提到了至少需要 $\Omega(\frac{1}{\alpha^2})$ 次采样才能准确估计 $\sigma_i$ 的取值，其推理过程如下：
令泛化误差下界至多为 $error(f_o)=\delta>0$，则有：
$$
\begin{equation}\frac{1}{4}(1-\sqrt{1-e^{-u^2}})\le\delta\Leftrightarrow m\ge 2\lceil \frac{1-\epsilon^2}{2\epsilon^2} \ln\frac{1}{8\delta(1-2\delta)} \rceil\end{equation}
$$
此时，我们发现 $m$ 至少为 $\Omega(\frac{1}{\alpha^2})$ 时，才能以 $1-\delta$ 的概率确定 $\sigma$ 的取值。

## 4.8【证明补充】定理4.3补充

**P67**定理4.3的证明过程中提到将式（4.24）带入引理4.1，具体推导如下。

定理4.3 表示为：
$$
\begin{equation}
P(|E(h)-\widehat{E}(h)| \leqslant \sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \geqslant 1-\delta
\end{equation}
$$
可以将其等价转化为：
$$
\begin{equation}
P(|E(h)-\widehat{E}(h)| > \sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant \delta
\end{equation}
$$

将（4.24）带入引理4.1可得：
$$
\begin{equation}
P(|E(h)-\widehat{E}(h)|>\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right)
\end{equation}
$$

根据 3.1 可得：
$$
\begin{equation}
4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right) \leqslant 4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right)
\end{equation}
$$

所以引理4.1可以转化为：
$$
\begin{equation}
P(|E(h)-\widehat{E}(h)|>\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right) \leqslant 4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right)
\end{equation}
$$

令 $4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right) = \delta$，由此可得：
$$
\begin{equation}
P(|E(h)-\widehat{E}(h)|>\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant \delta
\end{equation}
$$
从而得到了定理4.3的结论。

定理4.3 说明了期望误差和经验误差之间的差异程度，以概率形式限定在一定的区域范围内，虽然这并不完全代表误差一定会在 $\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}$ 这个范围内，但在此范围内的概率达到了 $1-\delta$。我们可以发现其差异程度的控制范围和样本量及维度之间的关系。当 $\frac{m}{d}$ 较大时（即样本量大，而 VC 维较低），由于 $ln(x)$ 相对于 $x$ 增加较慢，所以其差异可以控制得越小，反之亦然。

## 4.9【概念补充】回顾 Rademacher 复杂度

**P68**谈论了基于 Rademacher 的泛化误差界，这里对 Rademacher 复杂度进行回顾。

由于 VC 维和数据分布无关，未考虑数据的特定分布情况，其得到的结论往往是“松”的。Rademacher 复杂度则是基于数据分布的考虑，在牺牲了一定“普适性”的情况下，得到更为“紧”的结论。

复杂度是人为定义的一套量化复杂度程度的概念。对应 Rademacher 复杂度，假设空间中表示能力越强的函数，其复杂度越高。回到**P46-47**，如果 $\mathbb{E}_{\boldsymbol{\sigma}}\left[\sup _{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^{m} \sigma_{i} h\left(\boldsymbol{x}_{i}\right)\right]=1$ ，即对于 $x$ 的任意标签分布情况都能打散（特别注意这里针对的是这个特定的 $x$ 数据，这也是 Rademacher 复杂度和数据分布相关的原因，我们只有知道数据的具体分布情况，才能求解其 Rademacher 复杂度）。由 3.27 可等价得到 3.29 的经验 Rademacher 复杂度。

对于 Rademacher 复杂度的定义，我们进一步将具体的数据样本点转化为数据的样本空间分布，在经验误差的形式外面套一层期望，从而得到了一般化的 Rademacher 复杂度的定义。经验 Rademacher 复杂度和 Rademacher 复杂度的关系就如同概率论中掷硬币的观测序列和将其视为一个先验分布的随机变量序列一样。

## 4.10【证明补充】引理4.6的证明解析

**P71**的定理4.6给出了泛化误差界的下界。

不等式右边的数字$\frac{1}{100}$取得有些随意，但这里作者想表达的是：对于任意学习算法，总是存在某种分布和目标概念，使得输出的假设以较高的概率产生错误。事实上，根据（4.50）的结果，只要我们选取任意小于$\frac{1-e^{-\frac{d-1}{12}}}{7}$的数字，原不等式都是成立的。当$d=2$时，这个数字刚好为$0.0114$左右，此时选取$\frac{1}{100}$是较为恰当的。进一步我们可以发现，随着$d$的增加，这个数字会逐渐增大，并不断逼近$\frac{1}{7}$这个极限。但注意这并不意味着对于任何分布和目标概念所能训练得到的泛化误差下界不会超过$\frac{1}{7}$，这一切只是因为定理证明时假设的数据分布是（4.42）。

至于$32$这个数字，更是证明需要的产物。根据（4.50）的推导，我们可以发现，想要套用（4.49）的结论，就只能令$\epsilon=\frac{d-1}{16(1+r)}$。此时取$r=1$，分母部分自然得到$32$。

## 4.11【概念补充】$\rho$-间隔损失函数的 Lipschitz 性

**P79**提到，由经验损失（4.72）可知 $\Phi_\rho$ 最多是 $\frac{1}{\rho} - Lipschitz$。

考虑由 Lipschitz 的定义证明。由拉格朗日中值定理我们得到 $|\Phi_\rho(x_1)-\Phi_\rho(x_2)|\leq|\Phi_\rho'(\xi)||x_1-x_2|$ ，由于 $\Phi_\rho$ 的表达式已经给出，我们可以直接计算其导数，得到 $|\Phi_\rho'(\xi)|\leq\frac{1}{\rho}$，因此根据定义我们可以得到 $\rho$-间隔损失函数是 $\frac{1}{\rho}-Lipschitz$ 函数。

## 4.12【证明补充】引理4.7的补充

**P75**的定理4.7主要表达的是：无论算法有多强，在不可分的情况下，总会有某种“坏”分布使得输出假设的泛化误差以常数概率为$O(\sqrt\frac{d}{m})$。其中（4.61）中第二步变形用到了以下等式：
$$
\begin{equation}
\sum_{x_i\in S}(\mathbb{I}(h(x_i)\neq h_{\mathcal{D}_{\sigma}^*}(x_i))+\mathbb{I}(h(x_i) = h_{\mathcal{D}_{\sigma}^*}(x_i))) = d
\end{equation}
$$
另外，（4.63）的第三步为何不直接利用引理4.2进行推导呢？这是考虑到函数$\Phi(·,\alpha)$为减函数，即由$m/d+1\le2\lceil m/2\rceil$可知$\Phi(m/d+1,\alpha)\ge\Phi(2\lceil m/2\rceil,\alpha)$。可见后者并不是一个特别紧致的下界，因此我们转而考虑按照$|Z|_x$的取值进行拆分。

在**P76**左下角的最后一个脚注中，提到了$m/d$为变量$|Z|_x$的期望值，如何得到这个结论呢？根据（4.58）和（4.59）以及$\mathcal{U}$为$\{-1,+1\}^d$均匀分布的性质，我们可以得到从分布中抽取给定点$x$的期望概率为$1/d$。
当我们从 $D_σ$ 中独立抽取 $m$ 个样本的情况下，$S$ 中点 $x$ 出现的次数的期望值为 $m/d$。

此外，（4.65）中用到了引理4.3。令 $Z'=\frac{1}{\alpha}(E(h_Z)-E(h_{\mathcal{D}_{\sigma^*}^{m}}^{*}))$，根据（4.62）可知 $0\le Z'\le1$。
令 $\gamma'=\gamma u$，因为 $\Phi(·,\alpha)$为减函数，易知其最大值为$1/4$，因此有$\gamma'\in[0,1/4)\subseteq[0,1)$。此时带入引理4.3可得：
$$
\begin{equation}
P(Z'\gt\gamma')\ge E[Z']-\gamma' \ge u-u\gamma = (1-\gamma)u
\end{equation}
$$

同时，（4.69）到（4.70）的推导中体现了充分条件的思想。由（4.69）可知：
$$
\begin{equation}
\frac{m}{d}\le \frac{A}{\epsilon^2}+B
\end{equation}
$$
其中 $A=(\frac{7}{64})^2 \ln \frac{4}{3}$，$B=-\ln \frac{4}{3}-1$。

我们希望能推导出更为简洁的 $\frac{m}{d}$ 与 $\frac{1}{\epsilon^2}$ 之间的关系，因此考虑寻找充分条件使以下不等式成立：
$$
\begin{equation}
\frac{m}{d}\le \frac{A}{\epsilon^2}+B\le\frac{\omega}{\epsilon^2}
\end{equation}
$$
即使得 $\omega\ge B\epsilon^2+A$ 成立。当 $\epsilon\le 1/64$ 时，很容易得到 $\omega$ 的最小值（4.70）。

值得注意的是，整个证明过程共进行了四次启发式限制，分别为 $\gamma=1-8\delta$，$\alpha=8\epsilon/(1-8\epsilon)$，$\delta\le1/64$ 和 $\epsilon\le1/64$。这些启发式限制构造出来都是为了使得最终的不等式成立，实际上我们亦可根据实际需要进行调整，继而得到该定理的不同变种。

## 4.13【证明补充】定理4.8补充

**P79**的定理4.8给出了关于间隔损失函数的分类问题SVM的泛化误差界。

此处存在一个小的错误：证明中 4.80 式前的 **“代入 (4.96)”** 应为 **“代入 (4.76)”**。

观察要证明的公式，我们发现这是关于 Rademacher 复杂度的泛化上界推理，自然地回顾一下 Rademacher 复杂度：

> 现实任务中样本标记有时会受到噪声影响，因此我们与其在假设空间 $\mathcal{H}$ 中选择训练集上表现最好的假设，不如选择 $\mathcal{H}$ 中事先已经考虑了随机噪声影响的假设。

在此直接考虑利用前面讲到的关于实值假设空间中的期望与 Rademacher 复杂度的不等式。通过前面 4.73 讲到的关于间隔函数的经验间隔损失的式子，可以带入得到大体形式。

由于前面引理提到的关于 Lipschitz 函数的性质，结合 $\rho$-间隔损失函数的 Lipschitz 性，在简单改写复杂度之后便能得到要证明的定理。

第二个定理同理可证。
