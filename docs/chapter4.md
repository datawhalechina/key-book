# 第4章：泛化界

*Edit: 赵志民，李一飞，王茂霖，詹好*

------

## 本章前言

在机器学习中，泛化能力是衡量模型性能的核心标准之一。如何从有限的训练数据中获得能够在未见数据上表现良好的模型，始终是研究者关注的重要问题。本章将深入探讨与泛化界相关的理论基础和定理，通过对关键概念的补充说明和定理的详细推导，帮助读者更好地理解泛化误差的收敛性质以及不同假设空间下的泛化能力。本章还将介绍与泛化界密切相关的Rademacher复杂度及其在实际应用中的意义，为进一步的研究提供理论支持。

## 4.1 【概念解释】可分情形中的“等效”假设

**61页**中的「可分情形」部分提到了“等效假设”的概念。这其实是我们在面对模型选择时需要处理的问题。机器学习的任务实际上是从样本空间或属性空间中选择一个最符合实际的模型假设。在理想状态下，我们希望能排除不可能的情况，直接选择唯一可能的模型。然而，这是不现实的，因为训练数据无法覆盖所有可能的情况，这些数据仅是部分经验片段的记录。因此，机器学习成为了一个不适定问题（ill-posed problem）。

通常而言，不适定问题是指不满足以下任一条件的问题：

1. **存在解**：对于给定的问题，至少存在一个解，即这个问题是可以解决的。
2. **唯一解**：对于给定的问题，解是唯一的，没有其他可能的解。
3. **解连续依赖于定解条件**：解会随着初始条件或参数的变化而连续变化，不会出现突然跳跃或不连续的情况

在这里，由于我们无法仅依靠输入数据找到唯一解，这使得学习问题成为一个不适定问题，主要违反了条件2。而在更多时候，我们说机器学习是不适定的，主要是指其违反了条件3，在那种情况下，我们通常会用正则化等方式来解决。

## 4.2 【概念解释】定理4.1与定理2.1、定理2.2的关系

**61页**中的**定理4.1**与**定理2.1**和**定理2.2**之间存在密切联系。

**定理2.1**指出一个学习算法 $\mathfrak{L}$ 能从假设空间 $\mathcal{H}$ 中PAC辨识概念类 $\mathcal{C}$ ，需要满足：
$$
\begin{equation}
P(\mathbb{E}(h) \leqslant \epsilon) \geqslant 1-\delta
\end{equation}
$$
其中， $0 \lt \epsilon, \delta \lt 1$，所有 $c \in \mathcal{C}$， $h \in \mathcal{H}$ 。

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
P(\mathbb{E}(h) \leqslant \epsilon) \geqslant 1-\delta
\end{equation}
$$
因此，**定理4.1**实际上就是逆向使用了**定理2.1**和**定理2.2**。

## 4.3 【证明补充】定理4.2补充

**63页**中，在证明定理4.2时，省略了从式4.6到式4.7的推导过程。在这一过程中，主要用到了**28页**中式2.7的内容。

根据式4.6，有
$$
\begin{equation}
\begin{align*}
& P(\exists h \in \mathcal{H}:|\widehat{E}(h)-\mathbb{E}(h)|\gt\epsilon) \\
=& P\left(\left(\left|\widehat{E}\left(h_{1}\right)-E\left(h_{1}\right)\right|\gt\epsilon\right) \vee \cdots \vee\left(\left|\widehat{E}\left(h_{|\mathcal{H}|}\right)-E\left(h_{|\mathcal{H}|}\right)\right|\gt\epsilon\right)\right) \\
\leqslant & \sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-\mathbb{E}(h)|\gt\epsilon)
\end{align*}
\end{equation}
$$
引理2.1提出，若训练集 D 包含 $m$ 个从分布 D 上独立同分布采样而得的样本, $0\lt\epsilon\lt1$ 则对任意 $h \in \mathcal{H},$ 有
$$
\begin{equation}
\begin{align*}
P(\widehat{E}(h)-\mathbb{E}(h) \geqslant \epsilon) & \leqslant \exp \left(-2 m \epsilon^{2}\right) \\
P(\mathbb{E}(h)-\widehat{E}(h) \geqslant \epsilon) & \leqslant \exp \left(-2 m \epsilon^{2}\right) \\
P(|\mathbb{E}(h)-\widehat{E}(h)| \geqslant \epsilon) & \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)
\end{align*}
\end{equation}
$$
使用第三个式子，即，
$$
\begin{equation}
P(|\mathbb{E}(h)-\widehat{E}(h)| \geqslant \epsilon) \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)
\end{equation}
$$
将其带入式4.6，则有，
$$
\begin{equation}
\begin{array}{l}
\sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-\mathbb{E}(h)|\gt\epsilon) \leqslant \sum_{h \in \mathcal{H}} 2 \exp \left(-2 m \epsilon^{2}\right)
\end{array}
\end{equation}
$$
令 $2 \exp \left(-2 m \epsilon^{2}\right)=\delta /|\mathcal{H}|$，则有，
$$
\begin{equation}
\begin{array}{l}
\sum_{h \in \mathcal{H}} P(|\widehat{E}(h)-\mathbb{E}(h)|\gt\epsilon) \leqslant \sum_{h \in \mathcal{H}} \delta /|\mathcal{H}| \leqslant|\mathcal{H}| \cdot \delta /|\mathcal{H}|=\delta
\end{array}
\end{equation}
$$
从而得到式4.7。

## 4.4 【证明补充】引理4.1的证明思路

**63页**中，引入了引理4.1及其相关的证明。由于证明过程较长，这里对其思路进行梳理和分析。

对于假设空间 $\mathcal{H}, h \in \mathcal{H}, m \in \mathbb{N}, \epsilon \in (0,1)$，当 $m \ge 2/\epsilon^2$ 时有：

$$
\begin{equation}
P(|\mathbb{E}(h)-\hat{E}| \gt \epsilon) \le 4\Pi_{\mathcal{H}}(2m)\exp(-\frac{m\epsilon^2}{8})
\end{equation}
$$

### 证明简述

当我们要证明这个定理时，需要首先回忆增长函数的定义：对于 $m \in\mathbb{N}$, 假设空间 $\mathcal{H}$ 的**增长函数** (growth function) $\Pi_{\mathcal{H}}(m)$ 表示为
$$
\begin{equation}
\Pi_{\mathcal{H}}(m)=\max_{\{\mathbf{x}_1,...,\mathbf{x}_m\}\subset \mathcal{X}}|\{(h(\mathbf{x}_1),...,h(\mathbf{x}_m))|h\subset \mathcal{H}\}|
\end{equation}
$$

由于泛化误差在实际过程中难以评估，证明中首先将泛化误差和经验误差的差距缩放为经验误差之间的差距。通过概率与期望之间的转化，我们将问题进一步转化，并通过上确界的定义给出一个具体的概念 $h_0$ ，用三角不等式将经验误差与泛化误差之间的差距缩放至经验误差之间。再使用 Chebyshev 不等式中的概率与分布函数积分关系，拆分三角不等式，得出前一半概率（即经验误差与泛化误差之间的差距）与经验误差之间的不等式。

第二步则是将经验误差之间的差距进一步转化为增长函数的差距，即证明了第二个公式：
$$
\begin{equation}
P(\sup_{h\in\mathcal{H}}|\hat E_D(h)-\hat E_{D'}(h)|\geq\frac{1}{2}\epsilon)\leq 2|\mathcal{H}_{|D+D'|}| \exp(-\frac{\epsilon^2m}{8})|
\end{equation}
$$

在这个过程中，使用了式 4.16，通过给出任意置换下的情况，将期望问题转化为级数求和，进一步缩放成有关指数函数的公式：
$$
\begin{equation}
\frac{1}{2m}\sum_{i=1}^{(2m)!}\mathbb{I}(|\hat E_{T_iD}(h)-\hat E_{T_iD'}(h)\|)=\sum_{k\in[l]\\s.t.|2k/m-l/m|\geq\epsilon/2}\frac{\tbinom{l}{k}\tbinom{2m-l}{m-k}}{\tbinom{2m}{m}}
\end{equation}
$$

注意，原不等式中的上界 $2\exp(-\frac{\epsilon^2l}{8})$ 可以通过 Hoeffding 不等式推导出。

再通过进一步缩放，得到最后的缩放公式（4.19）。此时，结合前述推导可证明引理。

即使将原不等式中的 $2\exp(-\frac{ε^2l}{8})$ 替换为 $2\exp(-\frac{ε^2l}{4})$，原不等关系依然成立。此结论亦可推广到定理4.3的结论，但即便如此，泛化误差的收敛率依旧为 $O(\sqrt\frac{ln(m/d)}{m/d})$。



## 4.5 【证明补充】定理4.3补充

**67页**中提到将式（4.24）带入引理4.1，即可证明定理4.3，具体推导如下：

定理4.3 表示为：
$$
\begin{equation}
P(|\mathbb{E}(h)-\widehat{E}(h)| \leqslant \sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \geqslant 1-\delta
\end{equation}
$$

可以将其等价转化为：
$$
\begin{equation}
P(|\mathbb{E}(h)-\widehat{E}(h)| \gt \sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant \delta
\end{equation}
$$

将（4.24）带入引理4.1可得：
$$
\begin{equation}
P(|\mathbb{E}(h)-\widehat{E}(h)|\gt\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right)
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
P(|\mathbb{E}(h)-\widehat{E}(h)|\gt\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right) \leqslant 4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right)
\end{equation}
$$

令 $4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right) = \delta$，由此可得：
$$
\begin{equation}
P(|\mathbb{E}(h)-\widehat{E}(h)|\gt\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}) \leqslant \delta
\end{equation}
$$
从而得到了定理4.3的结论。

定理4.3 说明了期望误差和经验误差之间的差异程度，以概率形式限定在一定的区域范围内，虽然这并不完全代表误差一定会在 $\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}$ 这个范围内，但在此范围内的概率达到了 $1-\delta$。我们可以发现其差异程度的控制范围和样本量及维度之间的关系。当 $\frac{m}{d}$ 较大时（即样本量大，而 VC 维较低），由于 $ln(x)$ 相对于 $x$ 增加较慢，所以其差异可以控制得越小，反之亦然。

## 4.6 【概念解释】回顾 Rademacher 复杂度

**68页**谈论了基于 Rademacher 的泛化误差界，这里对 Rademacher 复杂度进行回顾。

由于 VC 维和数据分布无关，未考虑数据的特定分布情况，其得到的结论往往是“松”的。Rademacher 复杂度则是基于数据分布的考虑，在牺牲了一定“普适性”的情况下，得到更为“紧”的结论。

复杂度是人为定义的一套量化复杂度程度的概念。对应 Rademacher 复杂度，假设空间中表示能力越强的函数，其复杂度越高。回到**46-47页**，如果 $\mathbb{E}_{\boldsymbol{\sigma}}\left[\sup _{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^{m} \sigma_{i} h\left(\boldsymbol{x}_{i}\right)\right]=1$ ，即对于 $x$ 的任意标签分布情况都能打散（特别注意这里针对的是这个特定的 $x$ 数据，这也是 Rademacher 复杂度和数据分布相关的原因，我们只有知道数据的具体分布情况，才能求解其 Rademacher 复杂度）。由 3.27 可等价得到 3.29 的经验 Rademacher 复杂度。

对于 Rademacher 复杂度的定义，我们进一步将具体的数据样本点转化为数据的样本空间分布，在经验误差的形式外面套一层期望，从而得到了一般化的 Rademacher 复杂度的定义。经验 Rademacher 复杂度和 Rademacher 复杂度的关系就如同概率论中掷硬币的观测序列和将其视为一个先验分布的随机变量序列一样。

## 4.7 【证明补充】引理4.6的证明解析

**71页**的定理4.6给出了泛化误差下界的形式化表述：

$$
\begin{equation}
P\left(\mathbb{E}(h_D, c) > \frac{d-1}{32m}\right) \ge \frac{1}{100}
\end{equation}
$$

虽然不等式右边的常数 $\frac{1}{100}$ 看似有些随意，但作者意在表明：对于任意学习算法，总是存在某种分布和目标概念，使得学习算法输出的假设在较高概率下产生显著错误。

事实上，根据公式（4.50）的推导，只要选择一个小于 $\frac{1 - e^{-\frac{d-1}{12}}}{7}$ 的常数，原不等式仍然成立。以 $d=2$ 为例，此时该常数约为 $0.0114$，因此取 $\frac{1}{100}$ 是较为合理的选择。

进一步分析发现，随着维度 $d$ 的增加，这个常数会逐渐增大，最终逼近 $\frac{1}{7}$。然而，这并不意味着在任何数据分布和目标概念下，泛化误差下界都不会超过 $\frac{1}{7}$。这一限制是由定理证明过程中所假设的数据分布（公式4.42）导致的。

至于常数 $32$，则是证明过程中产生的结果。通过公式（4.50）的推导，可以看到为了套用公式（4.49）的结论，需要将 $\epsilon$ 设为 $\frac{d-1}{16(1+r)}$。在取 $r=1$ 的情况下，分母部分自然得到 $32$。



## 4.8 【证明补充】引理4.2补充

**74页**提出了引理4.2，这里给出完整的证明过程。

令 $\sigma$ 为服从 $\{-1,+1\}$上均匀分布的随机变量，对于 $0\lt\alpha\lt1$构造随机变量 $\alpha_{\sigma} = 1/2 - \alpha
\sigma/2$，基于 $\sigma$ 构造 $X \sim D_{\sigma}$，其中 $D_{\sigma}$ 为伯努利分布 $Bernoulli(\alpha_{\sigma})$，即 $P(X=1)=\alpha_{\sigma}$。
令 $S=\{X_1,\cdots,X_m\}$ 表示从分布 $D_{\alpha}^m$ 独立同分布采样得到的大小为 $m$ 的集合，即 $S \sim D_{\alpha}^m$，这对于函数 $f:X^m \rightarrow \{-1,+1\}$ 有：

$$
\begin{equation}
\mathbb{E}_{\sigma}[P_{S \sim D_{\alpha}^m}(f(S) \neq \sigma)] \ge \Phi (2\lceil m/2 \rceil, \alpha)
\end{equation}
$$

其中 $\Phi (m, \alpha) = \frac{1}{4} (1 - \sqrt{1 - \exp(-\frac{m\alpha^2}{1 - \alpha^2})})$

### 证明

我们设想两枚硬币 $x_A$ 和 $x_B$。两枚硬币都稍有不均匀，即 $P[x_A = 0] = 1/2−\alpha/2$ 和 $P[x_B = 0] = 1/2+\alpha/2$，其中 $0\lt\alpha\lt1$。0 表示正面，1 表示反面。假设我们随机从口袋里拿出一枚硬币 $x \in \{x_A,x_B\}$，抛 $m$ 次，得到的 0 和 1 的序列即为引理中构造的随机变量 $\alpha_{\sigma}$。如果我们想通过序列推测是哪一枚硬币被抛出，即选取并求得最佳决策函数 $f:\{0,1\}^m\rightarrow\{x_A,x_B\}$，则该实验假设的泛化误差可表示为 $error(f)=\mathbb{E}[\mathbb{P}_{S\sim\mathcal{D}_\alpha^m}(f(S)\neq x)]$。

用 $f$ 代表任意决策函数，用 $F_A$ 代表满足 $f(S)=x_A$ 的样本集合，用 $F_B$ 代表满足 $f(S)=x_B$ 的样本集合，用 $N(S)$ 表示样本 $S$ 中出现 0 的个数，根据泛化误差的定义，有：

$$
\begin{equation}
\begin{align*}
error(f)&=\sum_{S\in F_A}\mathbb{P}[S\wedge x_B]+\sum_{S\in F_B}\mathbb{P}[S\wedge x_A]\\
&=\frac{1}{2}\sum_{S\in F_A}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_B}\mathbb{P}[S|x_A]\\
&=\frac{1}{2}\sum_{S\in F_A\atop N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_A\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_B]
+\frac{1}{2}\sum_{S\in F_B\atop N(S)\lt \lceil m/2\rceil}\mathbb{P}[S|x_A]+\frac{1}{2}\sum_{S\in F_B\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
\end{align*}
\end{equation}
$$

如果 $N(S)\ge \lceil m/2\rceil$，易证 $\mathbb{P}[S|x_B]\ge\mathbb{P}[S|x_A]$。类似地，如果 $N(S)\lt \lceil m/2\rceil$，易证 $\mathbb{P}[S|x_A]\ge\mathbb{P}[S|x_B]$。因此，我们可以得到：

$$
\begin{equation}
\begin{align*}
error(f) &\ge\frac{1}{2}\sum_{S\in F_A\atop N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_A\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]
+\frac{1}{2}\sum_{S\in F_B\atop N(S)\lt \lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S\in F_B\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
&=\frac{1}{2}\sum_{S:N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\sum_{S:N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
&=error(f_o)
\end{align*}
\end{equation}
$$

因此，当我们选取 $f_o$ 为决策函数时，泛化误差取得最小值，即当且仅当 $N(S)\lt \lceil m/2\rceil$ 时，我们认为被抛的硬币是 $f_o(S)=x_A$。

注意到 $\mathbb{P}[N(S)\ge \lceil m/2\rceil|x=x_A]=\mathbb{P}[B(2\lceil m/2\rceil,p)\ge k]$，且 $p=1/2-\alpha /2,k=\lceil m/2\rceil$，因此 $2\lceil m/2\rceil p\le k\le 2\lceil m/2\rceil(1-p)$。

根据 Slud 不等式，我们有：
$$
\begin{equation}
error(f_o) \ge \frac{1}{2}\mathbb{P}[N\ge\frac{\lceil m/2\rceil\alpha}{\sqrt{1/2(1-\alpha^2)\lceil m/2\rceil}}]=\frac{1}{2}\mathbb{P}[N\ge\sqrt{\frac{2\lceil m/2\rceil}{1-\alpha^2}}\alpha]
\end{equation}
$$

根据第一章补充内容中的正态分布不等式推论，我们有：
$$
\begin{equation}
error(f_o)\ge\frac{1}{4}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})\ge\frac{1}{4}(1-\sqrt{1-e^{-u^2}})
\end{equation}
$$
此处 $u=\sqrt{\frac{2\lceil m/2\rceil}{1-\alpha^2}}\alpha$

事实上，根据上面的推导，我们可以进一步提升泛化误差的下界，即：
$$
\begin{equation}
\mathbb{E}[\mathbb{P}_{S\sim\mathcal{D}_\alpha^m}(f(S)\neq x)]\ge\frac{1}{4}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})
\end{equation}
$$

在引理末尾处，提到了至少需要 $\Omega(\frac{1}{\alpha^2})$ 次采样才能准确估计 $\sigma_i$ 的取值，其推理过程如下：
令泛化误差下界至多为 $error(f_o)=\delta\gt0$，则有：
$$
\begin{equation}
\frac{1}{4}(1-\sqrt{1-e^{-u^2}})\le\delta\Leftrightarrow m\ge 2\lceil \frac{1-\epsilon^2}{2\epsilon^2} \ln\frac{1}{8\delta(1-2\delta)} \rceil
\end{equation}
$$
此时，我们发现 $m$ 至少为 $\Omega(\frac{1}{\alpha^2})$ 时，才能以 $1-\delta$ 的概率确定 $\sigma$ 的取值。



## 4.9 【证明补充】引理4.7的补充

**75页**的定理4.7主要表达的是：无论算法有多强，在不可分的情况下，总会有某种“坏”分布使得输出假设的泛化误差以常数概率为$O(\sqrt\frac{d}{m})$。其中（4.61）中第二步变形用到了以下等式：
$$
\begin{equation}
\sum_{x_i\in S}(\mathbb{I}(h(x_i)\neq h_{\mathcal{D}_{\sigma}^*}(x_i))+\mathbb{I}(h(x_i) = h_{\mathcal{D}_{\sigma}^*}(x_i))) = d
\end{equation}
$$
另外，（4.63）的第三步为何不直接利用引理4.2进行推导呢？这是考虑到函数$\Phi(·,\alpha)$为减函数，即由$m/d+1\le2\lceil m/2\rceil$可知$\Phi(m/d+1,\alpha)\ge\Phi(2\lceil m/2\rceil,\alpha)$。可见后者并不是一个特别紧致的下界，因此我们转而考虑按照$|Z|_x$的取值进行拆分。

在**76页**左下角的最后一个脚注中，提到了$m/d$为变量$|Z|_x$的期望值，如何得到这个结论呢？根据（4.58）和（4.59）以及$\mathcal{U}$为$\{-1,+1\}^d$均匀分布的性质，我们可以得到从分布中抽取给定点$x$的期望概率为$1/d$。
当我们从 $D_σ$ 中独立抽取 $m$ 个样本的情况下，$S$ 中点 $x$ 出现的次数的期望值为 $m/d$。

此外，（4.65）中用到了引理4.3。令 $Z'=\frac{1}{\alpha}(\mathbb{E}(h_Z)-\mathbb{E}(h_{\mathcal{D}_{\sigma^*}^{m}}^{*}))$，根据（4.62）可知 $0\le Z'\le1$。
令 $\gamma'=\gamma u$，因为 $\Phi(·,\alpha)$为减函数，易知其最大值为$1/4$，因此有$\gamma'\in[0,1/4)\subseteq[0,1)$。此时带入引理4.3可得：
$$
\begin{equation}
P(Z'\gt\gamma')\ge \mathbb{E}[Z']-\gamma' \ge u-u\gamma = (1-\gamma)u
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



## 4.10 【概念解释】$\rho$-间隔损失函数的 Lipschitz 性

**79页**提到，由经验损失（公式4.72）可知 $\Phi_\rho$ 最多是 $\frac{1}{\rho}$-Lipschitz。对此进行详细解读如下：

根据Lipschitz连续性的定义，我们可以通过拉格朗日中值定理来证明这一点。具体来说，由拉格朗日中值定理可得：

$$
\begin{equation}
|\Phi_\rho(x_1)-\Phi_\rho(x_2)|\leq|\Phi_\rho'(\xi)||x_1-x_2|
\end{equation}
$$

其中 $\xi$ 是 $x_1$ 和 $x_2$ 之间的某一点。

已知 $\Phi_\rho$ 的具体表达式，因此可以直接计算其导数 $\Phi_\rho'(\xi)$。通过计算，我们可以得到：

$$
\begin{equation}
|\Phi_\rho'(\xi)| \leq \frac{1}{\rho}
\end{equation}
$$

因此，根据Lipschitz条件的定义，$\rho$-间隔损失函数是 $\frac{1}{\rho}$-Lipschitz 函数。



## 4.11 【证明补充】定理4.8补充

**79页**的定理4.8给出了关于间隔损失函数的分类问题SVM的泛化误差界。

此处存在一个小的错误：公式4.80前的 **“代入 (4.96)”** 应为 **“代入 (4.76)”**。

观察要证明的公式，我们发现这是关于 Rademacher 复杂度的泛化上界推理，自然地回顾一下 Rademacher 复杂度。

现实任务中样本标记有时会受到噪声影响，因此我们与其在假设空间 $\mathcal{H}$ 中选择训练集上表现最好的假设，不如选择 $\mathcal{H}$ 中事先已经考虑了随机噪声影响的假设。

在此直接考虑利用前面讲到的关于实值假设空间中的期望与 Rademacher 复杂度的不等式。通过前面 4.73 讲到的关于间隔函数的经验间隔损失的式子，可以带入得到大体形式。

由于前面引理提到的关于 Lipschitz 函数的性质，结合 $\rho$-间隔损失函数的 Lipschitz 性，在简单改写复杂度之后便能得到要证明的定理。
