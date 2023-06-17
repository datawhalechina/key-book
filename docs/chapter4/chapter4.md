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
其中， 0 < $\epsilon$ , $\delta < 1$，所有 $c \in \mathcal{C}$， $h \in \mathcal{H}$ 。

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



## 3.【概念补充】可分情形下收敛率为何是 $O(\frac{1}{m})$

**P61**中提出，随着训练集中样本数目的逐渐增加，泛化误差的上界逐渐趋于0，收敛率为 $O(\frac{1}{m})$ 。

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

再通过进一步放缩，得到了最后的放缩式（4.19）。注意，该上界在原论文中并未给出证明，但是我们可以通过Hoeffding不等式来得到：

> $X=2k\in[0,2m]$ 可以被理解为$2m$个伯努利随机变量的和，这些随机变量的$p = 0.5$，并且是独立同分布的。
> $\Gamma$可以被看作是所有$X$偏离其期望值$\mathbb{E}(X)=m$超过$\frac{εl}{2}$的结果的总和。应用霍夫丁不等式和$m\le2l$h这个约束，我们有：
> $$
> \begin{aligned}
> P(|X - m| \ge \frac{εl}{2})&\le2exp(-\frac{2(\frac{εl}{2})^2}{\sum_{i=1}^m(1-0)^2})\\
> &=2exp(-\frac{ε^2l^2}{2m})\le2exp(-\frac{ε^2l^2}{4l})\\
> &=2exp(-\frac{ε^2l}{4})\le2exp(-\frac{ε^2l}{8})
> \end{aligned}
> $$

再将前后结合便能够证明引理。

注意，根据上面的推导，我们发现，即使把原不等式中$2exp(-\frac{ε^2l}{8})$替换成$2exp(-\frac{ε^2l}{4})$，原不等关系依然成立。
此结论亦可推广到定理4.3的结论，但即便如此，泛化误差的收敛率依旧为$O(\sqrt\frac{ln(m/d)}{m/d})$，因此我们在这里不再赘述。

#### （2）思路分析

这里引理构造思路分为了几块：

1. 由于无限假设空间无法使用 $|\mathcal{H}|$ 分析，因此，这里要考虑第三章提到的 VC 维，使用它来分析无限假设空间的问题，而在 3.1 章中我们介绍了 VC 维和增长函数的关系。由于增长函数的定义与假设密切相关，因此考虑先得到关于增长函数的不等式也就不足为奇了。
2. 由于泛化误差难以直接得到评估，因此我们自然地会想到考虑放缩至经验误差之间的关系，再考虑到泛化误差难以整体评估（事实上我们正是在考虑它的上界来避免直接分析它），因此我们考虑使用某个具体的假设 $h_0$ ，再由数学证明中非常重要的三角不等式来分解。
3. 想办法将经验误差转化到增长函数的过程中非常重要的一点就在于考虑增长函数一定大于等于任何一个 $|\mathcal{H}_{|D+D'|}|$ 因此，我们在之前将概率问题转化到期望问题的基础上，考虑对每一个 $h$ 进行分析放缩。因此这里使用了式 4.16 来考虑每一个情况。



## 7.【证明补充】引理4.2补充

为了证明**P74**的引理4.2，我们设想两枚硬币$x_A$和硬币$x_B$。两枚硬币都稍微有些不均匀，即$P[x_A = 0] = 1/2−\alpha/2$ 和 $P[x_B = 0] = 1/2+\alpha/2$，其中 $0\lt\alpha\lt1$，0 表示正面，1 表示反面。
假设我们随机从口袋里拿出一枚硬币 $x\in\{x_A,x_B\}$，抛 $m$ 次，所得到的 0 和 1 的序列即为引理中构造的随机变量$\alpha_{\sigma}$。
如果我们想通过序列来推测出是哪一枚硬币被抛出，即选取并求得最佳决策函数$f:\{0,1\}^m\rightarrow\{x_A,x_B\}$，则该实验假设的泛化误差可表示为$error(f)=\mathbb{E}[\mathbb{P}_{S\sim\mathcal{D}_\alpha^m}(f(S)\neq x)]$。

我们用$f$代表任意决策函数，用$F_A$代表满足$f(S)=x_A$的样本集合，用$F_B$代表满足$f(S)=x_B$的样本集合，用$N(S)$表示样本$S$中出现0的个数，根据泛化误差的定义，有：
$$
\begin{aligned}
error(f)&=\displaystyle\sum_{S\in F_A}\mathbb{P}[S\wedge x_B]+\displaystyle\sum_{S\in F_B}\mathbb{P}[S\wedge x_A]\\
&=\frac{1}{2}\displaystyle\sum_{S\in F_A}\mathbb{P}[S|x_B]+\frac{1}{2}\displaystyle\sum_{S\in F_B}\mathbb{P}[S|x_A]\\
&=\frac{1}{2}\displaystyle\sum_{S\in F_A\atop N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\displaystyle\sum_{S\in F_A\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_B]
+\frac{1}{2}\displaystyle\sum_{S\in F_B\atop N(S)\lt \lceil m/2\rceil}\mathbb{P}[S|x_A]+\frac{1}{2}\displaystyle\sum_{S\in F_B\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
\end{aligned}
$$
如果$N(S)\ge \lceil m/2\rceil$，易证$\mathbb{P}[S|x_B]\ge\mathbb{P}[S|x_A]$。类似地，如果$N(S)\lt \lceil m/2\rceil$，易证$\mathbb{P}[S|x_A]\ge\mathbb{P}[S|x_B]$。因此，我们可以得到：
$$
\begin{aligned}
error(f) &\ge\frac{1}{2}\displaystyle\sum_{S\in F_A\atop N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\displaystyle\sum_{S\in F_A\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]
+\frac{1}{2}\displaystyle\sum_{S\in F_B\atop N(S)\lt \lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\displaystyle\sum_{S\in F_B\atop N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
&=\frac{1}{2}\displaystyle\sum_{S:N(S)\lt\lceil m/2\rceil}\mathbb{P}[S|x_B]+\frac{1}{2}\displaystyle\sum_{S:N(S)\ge \lceil m/2\rceil}\mathbb{P}[S|x_A]\\
&=error(f_o)
\end{aligned}
$$
因此，当我们选取$f_o$为决策函数时，泛化误差取得最小值，即当且仅当$N(S)\lt \lceil m/2\rceil$时，我们认为被抛的硬币是$f_o(S)=x_A$。

注意到$\mathbb{P}[N(S)\ge \lceil m/2\rceil|x=x_A]=\mathbb{P}[B(2\lceil m/2\rceil,p)\ge k]$，且$p=1/2-\alpha /2,k=\lceil m/2\rceil$，因此$2\lceil m/2\rceil p\le k\le 2\lceil m/2\rceil(1-p)$。

根据Slud不等式，我们有：
$$
\begin{aligned}
error(f_o) &\ge \frac{1}{2}\mathbb{P}[N\ge\frac{\lceil m/2\rceil\alpha}{\sqrt{1/2(1-\alpha^2)\lceil m/2\rceil}}]=\frac{1}{2}\mathbb{P}[N\ge\sqrt{\frac{2\lceil m/2\rceil}{1-\alpha^2}}\alpha]
\end{aligned}
$$

根据第一章补充内容中的正态分布不等式推论，我们有：
$$error(f_o)\ge\frac{1}{4}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})\ge\frac{1}{4}(1-\sqrt{1-e^{-u^2}})$$
此处$u=\sqrt{\frac{2\lceil m/2\rceil}{1-\alpha^2}}\alpha$

事实上，根据上面的推导，我们可以进一步提升泛化误差的下界，即：
$$\mathbb{E}[\mathbb{P}_{S\sim\mathcal{D}_\alpha^m}(f(S)\neq x)]\ge\frac{1}{4}(1-\sqrt{1-e^{-\frac{2}{\pi}u^2}})$$

在引理末尾处，提到了至少需要 $\Omega(\frac{1}{\alpha^2})$ 次采样才能个样本才能准确估计$\sigma_i$的取值，其推理过程如下：
令泛化误差下界至多为$error(f_o)=\delta>0$，则有：
$$\frac{1}{4}(1-\sqrt{1-e^{-u^2}})\le\delta\Leftrightarrow m\ge 2\lceil \frac{1-\epsilon^2}{2\epsilon^2}ln\frac{1}{8\delta(1-2\delta)} \rceil$$
此时，我们发现m至少为$\Omega(\frac{1}{\alpha^2})$时，我们才能以$1-\delta$的概率确定$\sigma$的取值。



## 8.【证明补充】定理4.3补充

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

定理4.3其实表明了期望误差和经验误差之间的差异程度以概率的形式限定到了一定的区域范围之内，虽然这并不完全代表其误差一定会在$\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}$这个范围之内，但在这其中的的概率达到了 $1-\delta$ 。并且我们从中可以发现其差异程度的控制范围和样本量以及其维度之间的关系，我们可以发现，当 $\frac{m}{d}$ 较大的时候（其代表样本量大，而 VC 维较低，这是数据在空间中分布较为密集的情况）由于 $ln(x)$ 相对于 $x$ 增加较慢，所以其差异可以控制的越小，反之亦然。



## 9.【概念补充】回顾 Rademacher 复杂度

**P68**谈论了基于 Rademacher 的泛化误差界，在此对 Rademacher 复杂度进行一下回顾。

对于 Rademacher 复杂度而言，由于 VC 维和数据分布无关，未考虑数据的特定分布情况，所以其得到的结论往往是“松”的，但由于其未考虑分布，所以其对应的会更为具有“普适性”，而 Rademacher 复杂度是基于数据分布所以考虑的，这在牺牲掉一定“普适性”下，可以得到更为“紧”的结论。

首先我们来说一下复杂度这个概念其实是人为定义的一套量化复杂度程度的概念。而对应 Rademacher 复杂度而言，其复杂度越高，代表假设空间中有表示能力越强的函数，回到P46-47页，如果 $\mathbb{E}_{\boldsymbol{\sigma}}\left[\sup _{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^{m} \sigma_{i} h\left(\boldsymbol{x}_{i}\right)\right]=1$ ,也就是对于 $x$ 的任意标签分布情况都能打散（特别注意这里针对的是这个特定的 $x$ 数据，这也是 Rademacher 复杂度和数据分布相关的原因，我们只有知道数据的具体分布情况，才能求解其 Rademacher 复杂度）。并且由3.27可等价得到3.29的经验 Rademacher 复杂度。

而对于 Rademacher 复杂度的定义，我们进一步将具体的数据样本点，转化为数据的样本空间分布，在经验误差的形式外面在套一层期望，从而得到了一般化的 Rademacher 复杂度的定义。经验 Rademacher 复杂度和 Rademacher 复杂度的关系就如同概率论中掷硬币的观测序列和将其视为一个先验分布的随机变量序列一样。



## 10.【证明补充】引理4.6的证明解析

**P71**的定理4.6给出了泛化误差界的下界。

不等式右边的数字$\frac{1}{100}$取得有些随意，但这里作者想表达的是：对于任意学习算法，总是存在某种分布和目标概念，使得输出的假设以较高的概率产生错误。事实上，根据（4.50）的结果，只要我们选取任意小于$\frac{1-e^{-\frac{d-1}{12}}}{7}$的数字，原不等式都是成立的。当$d=2$时，这个数字刚好为$0.0114$左右，此时选取$\frac{1}{100}$是较为恰当的。进一步我们可以发现，随着$d$的增加，这个数字会逐渐增大，并不断逼近$\frac{1}{7}$这个极限。但注意这并不意味对于任何分布和目标概念所能训练得到的泛化误差下界不会超过$\frac{1}{7}$，这一切只是因为定理证明时假设的数据分布是（4.42）。

至于$32$这个数字，更是证明需要的产物。根据（4.50）的推导，我们可以发现，想要套用（4.49）的结论，就只能令$\epsilon=\frac{d-1}{16(1+r)}$。此时取$r=1$，分母部分自然得到$32$。
## 10.【概念补充】$\rho$-间隔损失函数的 Lipschitz 性

**P79**提到由经验损失（4.72）可知 $\Phi_\rho$ 最多是 $\frac{1}{\rho} - Lipschitz$。

考虑由 Lipschitz 的定义证明，由拉格朗日中值定理我们得到$|\Phi_\rho(x_1)-\Phi_\rho(x_2)|\leq|\Phi_\rho'(\xi)||x_1-x_2|$ ，由于 $\Phi_\rho$ 表达式已经给出，我们可以直接计算其导数，得到$|\Phi_\rho'(\xi)|\leq\frac{1}{\rho}$，因此直接根据定义我们可以得到 $\rho$-间隔损失函数是 $\frac{1}{\rho}-Lipschitz$ 函数



## 11.【证明补充】引理4.7的补充

**P75**的定理4.7主要想表达无论算法有多强，在不可分的情况下，总会有某种“坏”分布使得输出假设的泛化误差以常数概率为$O(\sqrt\frac{d}{m})$，其中（4.61）中第二步变形用到了以下等式：
$$
\sum_{x_i\in S}(\mathbb{I}(h(x_i)\neq h_{\mathcal{D}_{\sigma}^*}(x_i))+\mathbb{I}(h(x_i) = h_{\mathcal{D}_{\sigma}^*}(x_i))) = d
$$
另外，（4.63）的第三步为何不直接利用引理4.2进行推导呢？这要是考虑到函数$\Phi(·,\alpha)$为减函数，即由$m/d+1\le2\lceil m/2\rceil$可知$\Phi(m/d+1,\alpha)\ge\Phi(2\lceil m/2\rceil,\alpha)$。可见后者并不是一个特别紧致的下界，因此我们转而考虑按照$|Z|_x$的取值进行拆分。

在**P76**左下角的最后一个脚注中，提到了$m/d$为变量$|Z|_x$的期望值，如何得到这个结论呢？根据（4.58）和（4.59）以及$\mathcal{U}$为$\{-1,+1\}^d$均匀分布的性质，我们可以得到从分布中抽取给定点$x$的期望概率为$1/d$。
当我们从 $D_σ$ 中独立抽取 $m$ 个样本的情况下，$S$ 中点 $x$ 出现的次数的期望值为 $m/d$。

此外，（4.65）中用到了引理4.3，我们令$Z'=\frac{1}{\alpha}(E(h_Z)-E(h_{\mathcal{D}_{\sigma^*}^{m}}^{*}))$，根据（4.62）可知$0\le Z'\le1$。
令$\gamma'=\gamma u$，因为$\Phi(·,\alpha)$为减函数，易知其最大值为$1/4$，因此有$\gamma'\in[0,1/4)\subseteq[0,1)$，此时带入引理4.3可得：
$$
P(Z'\gt\gamma')\ge E[Z']-\gamma' \ge u-u\gamma = (1-\gamma)u
$$

同时，（4.69）到（4.70）的推导中体现了充分条件的思想，由（4.69）可知：
$$
\frac{m}{d}\le \frac{A}{\epsilon^2}+B
$$
其中$A=(\frac{7}{64})^2ln\frac{4}{3}$，$B=-ln\frac{4}{3}-1$。

我们希望能够推导出更为简洁的$\frac{m}{d}$与$\frac{1}{\epsilon^2}$之间关系，因此我们考虑寻找充分条件使以下不等式成立：
$$
\frac{m}{d}\le \frac{A}{\epsilon^2}+B\le\frac{\omega}{\epsilon^2}
$$
即使得$\omega\ge B\epsilon^2+A$成立，当$\epsilon\le 1/64$时，很容易得到$\omega$的最小值（4.70）。

值得注意的是，整个证明过程共进行了四次启发式限制，分别为$\gamma=1-8\delta$，$\alpha=8\epsilon/(1-8\epsilon)$，$\delta\le1/64$和$\epsilon\le1/64$。
这些启发式限制构造出来都是为了使得最终的不等式成立，其实我们亦可根据实际需要进行调整，继而得到该定理的不同变种。



## 12.【证明补充】定理4.8补充

**P79**的定理4.8给出了机遇间隔损失函数的而分类问题SVM的泛化误差界。

此处存在一个小的错误：证明中 4.80 式前的 **“代入 (4.96)”** 应为 **“代入 (4.76)”**。

通过观察要证明的公式我们可以发现这是关于 Rademacher 复杂度的泛化上界推理，那么自然地回顾一下 Rademacher 复杂度：

> 现实任务中样本标记有时候会受到噪声的影响，因此我们与其在假设空间 $\mathcal{H}$ 中选择训练集上表现最好的假设，不如选择 $\mathcal{H}$ 中事先已经考虑了随机噪声影响的假设。

我们在这里直接考虑利用前面讲到的关于实值假设空间中的期望与 Rademacher 复杂度的不等式，并通过前面 4.73 讲到的关于间隔函数的经验间隔损失的式子可以带入得到大体的形式。

由于前面引理提到的关于 Lipschitz 函数的性质，同时我们说明了$\rho$-间隔损失函数的 Lipschitz 性，因此在简单改写复杂度之后便能得到要证明的定理。

第二个定理同理可证。