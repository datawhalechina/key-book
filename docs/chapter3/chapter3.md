# 第3章：复杂性分析

*Edit: 王茂霖，李一飞，Hao ZHAN，赵志民*

---

在机器学习理论中，复杂性分析与计算理论中的算法复杂度类似，是衡量模型和假设空间能力的关键指标。复杂性越高，模型的表达能力越强，但同时也意味着过拟合的风险增加。因此，研究假设空间的复杂性有助于理解模型的泛化能力。

## 3.1【概念补充】VC维

VC维（Vapnik-Chervonenkis 维度）是衡量假设空间$\mathcal H$复杂性的重要工具。它表示假设空间能够打散的最大样本集的大小，是描述二元分类问题下假设空间复杂度的核心指标。

VC维的定义如下：
$$
\begin{equation}
VC(\mathcal H)=\max\{m:\Pi_{\mathcal H}(m)=2^m\}
\end{equation}
$$
其中，$\Pi_{\mathcal H}(m)$是假设空间$\mathcal H$对大小为$m$的样本集的增长函数。VC维可以理解为模型在二元分类问题中有效的自由度。

**例子：**对于假设空间$sign(wx+b)$（即线性分类器），其在二维空间$R^2$中的VC维为3。这意味着，线性分类器能够打散最多三个点，但无法打散四个点。

## 3.2【概念补充】Natarajan维

在多分类问题中，我们使用Natarajan维来描述假设空间的复杂性。Natarajan维是能被假设空间$\mathcal H$打散的最大样本集的大小。

当类别数$K=2$时，Natarajan维与VC维相同：
$$
\begin{equation}
VC(\mathcal H)=Natarajan(\mathcal H)
\end{equation}
$$
对于更一般的$K$分类问题，Natarajan维的增长函数上界为：
$$
\begin{equation}
\Pi_{\mathcal H}(m)\leqslant m^dK^{2d}
\end{equation}
$$
随着样本数$m$和分类数$K$的增加，Natarajan维的复杂度呈指数级增长。

## 3.3【概念补充】Rademacher复杂度

VC维和Natarajan维均未考虑数据分布的影响，而Rademacher复杂度则引入了数据分布因素。它通过考察数据的几何结构和信噪比等特性，提供了更紧的泛化误差界。

函数空间$\mathcal F$关于$\mathcal Z$在分布$\mathcal D$上的Rademacher复杂度定义如下：
$$
\begin{equation}
\Re_{\mathcal Z}(\mathcal F)=E_{Z\subset\mathcal Z:|Z|=m}\left[E_{\sigma}\left[\underset{f\in\mathcal F}{\sup}\frac{1}{m} \sum_{i=1}^m \sigma_i f(z_i)\right]\right]
\end{equation}
$$
其中$\sigma_i$是服从均匀分布的随机变量。假设空间$\mathcal H$的Rademacher复杂度上界为：
$$
\begin{equation}
\Re_m(\mathcal H)\leqslant\sqrt{\frac{2\ln\Pi_{\mathcal H}(m)}{m}}
\end{equation}
$$

## 3.4【概念补充】shattering 概念的可视化

**Shattering**是指假设空间能够实现样本集上所有对分的能力。以下通过二维空间$R^2$中的线性分类器示例来说明。

**示例：**对于二维空间$R^2$中的三个点，线性分类器$sign(wx+b)$可以实现三点的所有对分，但无法实现四点的所有对分，如下图所示：

![shattering](../images/shattering.jpg)

因此，线性分类器在$R^2$中的VC维为3。
