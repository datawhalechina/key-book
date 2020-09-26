# 第5章：稳定性

*Edit: 李一飞，王茂霖，Hao ZHAN*

*Update: 06/12/2020*

---

本章的内容围绕学习理论中的稳定性而展开。在第四章中，我们给予不同的复杂度度量方式来给出了一些泛化界，这些泛化界与特定的算法无关。因此，可能就有人会问，对特定算法的性质进行分析能否得到更好的学习保障？这种对某一算法的分析能否扩展到其他有相似性质的学习算法上？

本章针对于这些问题进行了解答，应用算法稳定性来推导出了依赖于算法的学习保证。



## 1.【概念补充】留一风险

**P90**提到留一风险（leave-one-out risk）。

所谓留一风险，就是依次计算从数据集中剔除某一数据后所训练出来的模型与该被剔除出的数据之间的误差。它的本质就是保证用于误差测试的数据不会被加入到训练数据之中。类似于模型选择时的留一验证。



## 2.【证明补充】均匀稳定性与泛化上界

**P92**中，定义5.1讨论了**均匀稳定性**与泛化性的关系。这里重新梳理一下均匀稳定性和泛化性究竟在哪一步证明过程中得以关联。

### （1）证明简述

根据读者几章阅读的经验，想必大家对这些不等式有些熟门熟路了，见到题目中有 $ln$ 有根号等等便能够意识到这里又是关于指数函数的不等式然后反解误差 $\epsilon$ 了。这里是希望通过样本的稳定性推出关于误差的泛化性，因此我们要证明的时候必须想办法将误差之间的差距转化为损失函数之间的误差。

由于定理中给出的替换样本 $\beta$-均匀稳定性和移除样本 $\gamma$-均匀稳定性是非常强的条件，对于任意的数据集 D 和任意的样本 **z** 成立，因此我们很容易可以得到关于*经验风险和泛化风险差距*（即 $\Phi(D)$ ) 的估计式。下面我们主要讨论替换样本 $\beta$-均匀稳定性，而移除样本 $\gamma$-均匀稳定性可以类似得到结果。

我们通过对损失函数的差求和平均即是风险 (Risk) 的差距，因为替换样本 $\beta$-均匀稳定性满足对任意 **z** 成立，因此我们得到 (5.22) 与 (5.23) 式。使用 McDiarmid 不等式便可以得到关于*经验风险与泛化风险的差距*（即 $\Phi(D)$ ) 超过其平均值至少 $\epsilon$ 的概率。即：
$$
P(\phi(D)\geq\mathbb{E}[\Phi(D)]+\epsilon)\leq exp(\frac{-2m\epsilon^2}{(2m\beta+M)^2})
$$
之后进行简单的放缩估计便可以得到最终的式子
$$
P(R(\mathcal{L_D})-\hat R(\mathcal{L_D})\geq\beta+\epsilon)\leq exp(\frac{-2m\epsilon^2}{(2m\beta+M)^2})
$$

### （2）均匀稳定性与泛化性的关系

在这个证明过程中，有多处涉及到了损失函数作差放缩，即替换样本 $\beta$-均匀稳定性，但实际上多数情况下使用只是为了放缩得到更简单的式子，只有在 (5.24) 与 (5.25) 处主要体现了稳定性与泛化性的关系。

在 (5.24) 处通过替换样本的稳定性，我们可以得到*经验误差与泛化误差的差距*（即 $\Phi(D)$ ) 在替换样本前后的误差能够被上界 $2\beta+M/m$ 控制住，根据 McDiarmid 不等式的描述，如果实值函数关于变量的替换如果具有较好的稳定性，那么该实值函数总有与期望的差距同样被上界控制住。（简单描述就是如果实值函数替换一个变量之后误差不大，那么无论怎么替换误差都不会过大，因此实值函数取值总会在一定范围内，从而总和均值（即期望）相差不大。）。因此在 (5.25) 处我们能够得到*经验误差与泛化误差的差距*（即 $\Phi(D)$ ) 同样有了上界，自然我们通过简单的放缩就可以获得一个常数上界，得到泛化风险的上界。



## 3.【证明补充】假设稳定性与泛化上界

**P94**中，定义5.2讨论了**假设稳定性**与泛化性的关系。这里重新梳理一下假设稳定性和泛化性究竟在哪一步证明过程中得以关联。

### （1）证明简述

证明是关于 $R(\mathcal{L_D})-\hat R(\mathcal{L_D})$ 的平方平均，这是因为假设稳定性是比较弱的条件，只能保证误差的期望被上界控制，因此这里只能得到关于期望的不等式。同样的，因为不涉及到概率与置信度，因此我们并不需要复杂的不等式，只需要使用简单的放缩便能够得到答案。

单纯从证明的角度来讲，里面可能最不容易理解的一点是关于 (5.30) 式的放缩，实质上是有关于期望的线性性的等式化简：
$$
\begin{aligned}
&\frac{1}{m^2}\Sigma_{i\neq j}\mathbb{E}_D
[R(\mathcal{L_D})-\ell(\mathcal{L_D},\mathbf{z}_i))(R(\mathcal{L_D})-\ell(\mathcal{L}_D,\mathbf{z}_j))]\\
&=2\cdot C_m^2\cdot\frac{1}{m^2}\mathbb{E}_D
[R(\mathcal{L_D})-\ell(\mathcal{L_D},\mathbf{z}_i))(R(\mathcal{L_D})-\ell(\mathcal{L}_D,\mathbf{z}_j))]\\
&=\frac{m-1}{m}\mathbb{E}_D
[R(\mathcal{L_D})-\ell(\mathcal{L_D},\mathbf{z}_1))(R(\mathcal{L_D})-\ell(\mathcal{L}_D,\mathbf{z}_2))]\\[1mm]
&\leq \mathbb{E}_D
[R(\mathcal{L_D})-\ell(\mathcal{L_D},\mathbf{z}_1))(R(\mathcal{L_D})-\ell(\mathcal{L}_D,\mathbf{z}_2))]\\[2mm]
&=\mathbb{E}_D
[\mathbb{E}_{z\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z})]-\ell(\mathcal{L_D},\mathbf{z}_1))(\mathbb{E}_{z'\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z'})]-\ell(\mathcal{L}_D,\mathbf{z}_2))]\\[2mm]
&=(
\mathbb{E}_{z\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z})]-\mathbb{E}_D[\ell(\mathcal{L_D},\mathbf{z}_1)])(\mathbb{E}_{z'\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z'})]-\mathbb{E}_D[\ell(\mathcal{L_D},\mathbf{z}_2)])\\[2mm]
&=\mathbb{E}_{z\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z})]\cdot\mathbb{E}_{z'\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z'})]-\mathbb{E}_{z\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z})]\cdot\mathbb{E}_D[\ell(\mathcal{L_D},\mathbf{z}_2)]\\[2mm]
&\qquad\quad+\mathbb{E}_D[\ell(\mathcal{L_D},\mathbf{z}_1)]\cdot\mathbb{E}_D[\ell(\mathcal{L_D},\mathbf{z}_2)]-\mathbb{E}_{z'\sim \mathcal{D}}[\ell(\mathcal{L}_D,\mathbf{z'})]\cdot\mathbb{E}_D[\ell(\mathcal{L_D},\mathbf{z}_1)]\\[2mm]
&=\mathbb{E}_{z',z, \mathcal{D}}[\ell(\mathcal{L_D},\mathbf{z})\ell(\mathcal{L_D},\mathbf{z}')-\ell(\mathcal{L_D},\mathbf{z}_1)\ell(\mathcal{L_D},\mathbf{z}_2)\\[2mm]
&\qquad\quad+\ell(\mathcal{L_D},\mathbf{z}_1)\ell(\mathcal{L_D},\mathbf{z}_2)-\ell(\mathcal{L_D},\mathbf{z}')\ell(\mathcal{L_D},\mathbf{z}_1)]
\end{aligned}
$$
之后的部分在 (5.32) 式第二步实际是添加了一项 $(-\ell(\mathcal{L_{D^{1,\mathbb{z}}}},\mathbf{z}')\ell(\mathcal{L_D},\mathbf{z}')+\ell(\mathcal{L_{D^{1,\mathbb{z}}}},\mathbf{z}')\ell(\mathcal{L_D},\mathbf{z}'))$ 之后简单的三角不等式放缩即得到结果。

其他地方的证明都非常朴素的放缩和化简，至此，证明的难点部分就讲述完成了。

### （2）泛化性与假设稳定性

这里实际上并不是简单的泛化界的关系，实际上给出了经验误差与泛化误差的差距的平方平均的界，这是因为假设稳定性并不是非常强的结论，事实上假设稳定性就是为了放松均匀稳定性这个较强的条件得到的。

两个性质联系的关键点在 (5.32) 式，通过插项的三角不等式将两个数据集下 $\mathcal{D},\mathcal{D}^{1,z}$ 的损失函数复杂乘法关系转化成简单的加减法，再通过假设稳定性的上界得到最终需要的上界。



## 4.【证明补充】稳定性与可学性

**P97**中，定理5.4涉及了稳定性与可学性之间的转换。这里通过对定理5.4进行梳理，分析稳定性和可学性在哪一步证明过程中得以关联。

### （1）证明简述

这里要回顾一下不可知 PAC 可学的概念：

对所有分布 $\mathcal{D}$ ，若存在学习算法 $\mathcal{L}$ 与多项式函数 $poly(\cdot,\cdot,\cdot,\cdot)$ ，使得对于任何 $m\geq poly(1/\epsilon,1/\delta,size(\mathbf{x}),size(c))$ $\mathcal{L}$ 输出的假设能满足
$$
   	P\big(E(h)-min_{h'\in\mathcal{H}}E(h')\leq\epsilon\big)\geq1-\delta
$$

这里的证明实际上就是利用经验误差与泛化误差之间的差距界将两个函数的泛化风险差距用三角不等式放缩控制住。即：
$$
\begin{aligned}
|R(&\mathcal{L}_D)-R(h^*)|\leq\\&|R(\mathcal{L}_D)-\hat R(\mathcal{L}_D)|+|\hat R(\mathcal{L}_D)-\hat R(h^*)|+|\hat R(h^*)-R(h^*)|
\end{aligned}
$$
三个部分分别使用三个上界（均匀稳定性带来的泛化上界，$h^*$ 定义带来的界，最小泛化风险函数的泛化上界）将其控制住。

到这里，定理证明的简单描述就结束了，但是这一章的几个实例又一次带我们回顾了前面几章的各个概念，足以体现本书结构优秀。

### （2）稳定性与可学习性

这里之所以只能到达不可知 PAC 可学是因为泛化界只能够以概率达到，不能保证在任何的函数空间都达到上界以下，因此只能得到稳定性与不可知 PAC 可学性的关系。

事实上这里的稳定性与可学习性的关系类似于第四章我们讲到的泛化界与可学习性的关系，即在拥有一直上界之后通过 ERM 算法得到最小经验风险函数，便得到了更为细致的界。事实上这里稳定性与可学习性联系的时候便是使用了均匀稳定性带来的**泛化上界**与后面 ERM 的结果联合作用得到了可学习性。



