# 第6章：一致性

*Edit: 王茂霖，Hao ZHAN*

*Update: 09/03/2020*

---

本章的内容围绕学习理论中的一致性（consistency）展开。一致性关注的是随着训练数据的增加，通过学习算法所学习到的分类器是否趋于贝叶斯最优分类器。具体内容包括一致性的定义、参数方法下的一致性分析和非参方法下的一致性分析，以及随机森林的一致性分析案例。



### 1. 【公式证明】公式(6.25)是泛化误差的无偏估计

P117 中，公式（6.25）中给出了分类器的经验风险$R_{exp}$，并提到了其为泛化风险$R$的无偏估计。这里对于这一概念进行说明。
首先，我们需要理解经验风险$R_{exp}$和泛化风险$R$这两个概念。经验风险是根据模型的预测结果和真实结果的比较从而计算出的风险指标，是可以量化的。泛化风险是在模型给定的情况下，基于数据-标签联合分布的样本（视为随机变量）的预测结果和真实值的比较的期望值，而这一指标在数据-标签联合分布未知的情况下是不可能求得的（在实际情况下，数据的分布往往是未知的），且如果在数据标签联合分布已知的情况下，我们其实也不需要去学习模型了（数据分布本身就是最好的生成模型）。所以泛化风险$R$这一概念可以说是一个理论化的概念。
其次，我们这里对于无偏估计进行说明，当我们说$y$是$x$的无偏估计的时候，就代表 $\mathbb{E}{x}=y$。


泛化风险定义：
$$
\begin{aligned}
R(f) &=\mathbb{E}_{(\boldsymbol{x}, y) \sim \mathcal{D}}[\mathbb{I}(y f(\boldsymbol{x}) \leqslant 0)] \\
&=\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}_{\mathcal{X}}[\eta(\boldsymbol{x}) \mathbb{I}(f(\boldsymbol{x}) \leqslant 0)+(1-\eta(\boldsymbol{x})) \mathbb{I}(f(\boldsymbol{x}) \geqslant 0)]}
\end{aligned}
$$

经验风险定义：
$$
R_{exp}(f)  = \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(y_{i} f\left(\boldsymbol{x}_{i}\right) \leqslant 0\right)
$$

现在我们来证明经验风险是泛化误差的无偏估计：
首先这需要一些先验假设，我们假设所有的样本都是从一个未知的样本-标签空间$D$中独立同分布（I.i.d）采样的。由此，对于经验风险求期望：
$$
\begin{aligned}
\mathbb{E}({R_{exp}(f)}) &=\mathbb{E}_{(\boldsymbol{x_i}, y_i) \sim \mathcal{D}}[{ \frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(y_{i} f\left(\boldsymbol{x}_{i}\right) \leqslant 0\right)}] \\
&= \frac{1}{m} \sum_{i=1}^{m} \mathbb{E}_{(\boldsymbol{x_i}, y_i) \sim \mathcal{D}}[{ \mathbb{I}\left(y_{i} f\left(\boldsymbol{x}_{i}\right) \leqslant 0\right)}]\\
&= \frac{1}{m} \sum_{i=1}^{m} \mathbb{E}_{(\boldsymbol{x}, y) \sim \mathcal{D}}[{ \mathbb{I}\left(y f\left(\boldsymbol{x}\right) \leqslant 0\right)}]\\
&= \frac{1}{m} \sum_{i=1}^{m} R(f)\\
&= R(f)
\end{aligned} 
$$
得证。



### 2. 【定理补充】P120，对于定理6.1中，关于具有一致性的替代函数的性质进行说明以及思路解释。

P120, 定理6.1是对于替代一致性给出了一个充分性的条件，也就是说在满足这一条件下的替代函数是满足替代一致性的，（但我们需要注意到满足替代一致性的函数，不一定满足这一条件，这条件是充分性条件）。
对于这个定理的证明，文中是先给出了函数的泛化风险和贝叶斯风险差值的不等式，根据一致性的定义，我们需要证明，${R_{\phi}\left(\hat{f}_{m}\right) \rightarrow R_{\phi}^{*}} $ 时 $R\left(\hat{f}_{m}\right) \rightarrow R^{*}$。所以我们需要进一步构造关于 ${R_{\phi}\left(\hat{f}_{m}\right) - R_{\phi}^{*}} $ 的不等式。然后利用两个不等式的关联性，最终得到 $R\left(\hat{f}_{m}\right)-R^{*} \leqslant 2 c \sqrt[s]{R_{\phi}\left(\hat{f}_{m}\right)-R_{\phi}^{*}}$ ，从而证明 ${R_{\phi}\left(\hat{f}_{m}\right) \rightarrow R_{\phi}^{*}} $ 时 $R\left(\hat{f}_{m}\right) \rightarrow R^{*}$。其中关于 不等式（6.40），其中包含着一定的构造技巧，然后再利用定理中的条件从而得到了 不等式（6.43）。再利用构造的凸函数性质，从而证明了最终的结论。



### 3. 【概念补充】划分机制方法

P122，介绍一种将样本空间划分成多个互不相容的区域，然后对各区域中对正例和反例分别计数，以多数的类别作为区域中样本的标记的方法。这种方法在本质上与参数方法不同，它并非是在参数空间中进行搜索，从而对样本空间上构建一个划分超平面的方法，而是直接在泛函空间上进行搜索。

一种典型的方案就是我们熟悉的决策树模型：

<center><img src="./chapter6/imgs/1.png" width= "800"/></center>

每当构造出一个决策树的节点时，就等同于在样本空间上做出了一次划分。这一洞察方式同样可解释剪枝操作，例如剪枝前的样本空间为：

<center><img src="./chapter6/imgs/2.png" width= "400"/></center>



而剪枝之后则为：

<center><img src="./chapter6/imgs/3.png" width= "400"/></center>

此即所谓的划分机制。



### 4. 【概念解释】P124 对 定理6.2 的依照概率成立 进行解释说明（什么是依概率成立）

对于定理6.2 此处文中提到了一个定义 叫做依概率成立，这是概率论与数理统计中的知识点，可用如下的公式表达：$\lim _{n \rightarrow \infty} P((Diam(\Omega)-0) \geq \epsilon)=0 $ 和对于所有 $N>0$, $\lim _{n \rightarrow \infty} P((N(x)>N)=1$。其代表着当$n$趋于无穷的时候，几乎处处的$Diam(\Omega)$都在$0$的$\epsilon$邻域。$N(x)$的极限几乎处处都为无穷。依照概率成立是比极限更弱的一种情况（可以忽略概率趋于$0$处的情况）。



### 5. 【定理补充】P124，对于定理6.2中，关于划分机制一致性的说明（尤其是6.63之后）

对于定理6.2的划分一致性，这里进行详细说明，首先定义了 $ \Omega(\boldsymbol{x})$这样一个划分区域的条件概率的极大似然估计量：
$$
\hat{\eta}(\boldsymbol{x})=\sum_{\boldsymbol{x}_{i} \in \Omega(\boldsymbol{x})} \frac{\mathbb{I}\left(y_{i}=+1\right)}{N(\boldsymbol{x})}
$$
再由估计量根据划分机制构造出了分类器（输出函数） $h_{m}(\boldsymbol{x})=2 \mathbb{I}\left(\hat{\eta}(\boldsymbol{x}) \geqslant \frac{1}{2}\right)-1$。根据划分机制的一致性，我们此处就需要证明，其输出函数的泛化误差在$m$趋于无穷时，趋于贝叶斯风险。而此处利用的为基于条件概率估计的插值法，所以基于引理 6.2我们会得到其输出函数的泛化误差和贝叶斯风险差值的不等式，在对于不等式右侧的期望利用三角不等式进行放缩可得到（6.62），由于$\eta(x)$的连续性这代表着邻域内的近似，由此：
当$m \rightarrow  \infty$时，$\lim _{m \rightarrow \infty} P((Diam(\Omega)-0) \geq \epsilon) =  \lim _{m \rightarrow \infty} P(( \sup _{\boldsymbol{x}, \boldsymbol{x}^{\prime} \in \Omega}\left\|\boldsymbol{x}-\boldsymbol{x}^{\prime}\right\| -0) \geq \epsilon)=0 $
此时由于，$\bar{\eta}(\boldsymbol{x})=\mathbb{E}\left[\eta\left(\boldsymbol{x}^{\prime}\right) \mid \boldsymbol{x}^{\prime} \in \Omega(\boldsymbol{x})\right]$，所以
$$
\begin{aligned}
\mathbb{E}[|\bar{\eta}(\boldsymbol{x})-\eta(\boldsymbol{x})|] \rightarrow 0
\end{aligned}
$$
这是由于${x}^{\prime}$被依概率限制到了一个$\epsilon$邻域内，而期望可以不考虑概率趋于0的点，由此$\bar{\eta}(\boldsymbol{x})$由于其$\eta(x)$的连续性也被限制到了一个$\eta(x)$的$\epsilon$邻域内。从而期望的极限得证。
接下来，再对于三角不等式右式的前半部分，将其拆分为了$N(x)=0$和$N(x)>0$两部分：
$$
\begin{array}{c}
\mathbb{E}\left[|\hat{\eta}(\boldsymbol{x})-\bar{\eta}(\boldsymbol{x})| \mid \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right] = 
\mathbb{E}\left[|\hat{\eta}(\boldsymbol{x})-\bar{\eta}(\boldsymbol{x})|\mid N(x)=0 , \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right] \\
+\mathbb{E}\left[\left|\sum_{\boldsymbol{x}_{i} \in \Omega(\boldsymbol{x})} \frac{\mathbb{I}\left(y_{i}=+1\right)-\bar{\eta}(\boldsymbol{x})}{N(\boldsymbol{x})}\right| N(\boldsymbol{x})>0, \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right]\\
\leqslant P\left(N(\boldsymbol{x})=0 \mid \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right) + \mathbb{E}\left[\left|\sum_{\boldsymbol{x}_{i} \in \Omega(\boldsymbol{x})} \frac{\mathbb{I}\left(y_{i}=+1\right)-\bar{\eta}(\boldsymbol{x})}{N(\boldsymbol{x})}\right| N(\boldsymbol{x})>0, \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right]
\end{array}
$$
然后对于不等式右侧的第二部分，再利用引理6.3的不等式我们可以得到：
$$
\begin{array}{l}
\mathbb{E}\left[\left|\sum_{\boldsymbol{x}_{i} \in \Omega(\boldsymbol{x})} \frac{\mathbb{I}\left(y_{i}=+1\right)-\bar{\eta}(\boldsymbol{x})}{N(\boldsymbol{x})}\right| N(\boldsymbol{x})>0, \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right] \\
\leqslant \mathbb{E}\left[\sqrt{\frac{\bar{\eta}(\boldsymbol{x})(1-\bar{\eta}(\boldsymbol{x}))}{N(\boldsymbol{x})}} \mathbb{I}(N(\boldsymbol{x})>0) \mid \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right]
\end{array}
$$
对于此不等式的右侧，再进行放缩，对于任意$k \geq 3$，当 $N(x) \leqslant k$时，$\sqrt{\frac{\bar{\eta}(\boldsymbol{x})(1-\bar{\eta}(\boldsymbol{x}))}{N(\boldsymbol{x})}} \leqslant \frac12 $, 当 $N(x) > k$时, $\sqrt{\frac{\bar{\eta}(\boldsymbol{x})(1-\bar{\eta}(\boldsymbol{x}))}{N(\boldsymbol{x})}} \leqslant \frac{1}{2\sqrt(k)} $, 从而得到不等式右侧的进一步放缩：

$$
\leqslant \frac{1}{2} P\left(N(\boldsymbol{x}) \leqslant k \mid \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right)+\frac{1}{2 \sqrt{k}} P\left(N(\boldsymbol{x}) > k \mid \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right)\\
\leqslant \frac{1}{2} P\left(N(\boldsymbol{x}) \leqslant k \mid \boldsymbol{x}, \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right)+\frac{1}{2 \sqrt{k}}
$$

再结合前面的结果，我们可以得到：
$$
\mathbb{E}[|\hat{\eta}(x)-\bar{\eta}(x)|] \leqslant \frac{1}{2} P(N(x) \leqslant k)+\frac{1}{2 \sqrt{k}}+P(N(x)=0)
$$

而根据$N(x) \rightarrow  \infty$依概率成立，则，当 $m \rightarrow  \infty$时候，$P(N(x) \leqslant k) \rightarrow  0$, $P(N(x) = 0) \rightarrow  0$, 并且当取 $k=\sqrt{N(x)}$时，$\frac{1}{2 \sqrt{k}} \rightarrow  0$依概率成立的，从而得到结论：
$$
\mathbb{E}[|\hat{\eta}(\boldsymbol{x})-\bar{\eta}(\boldsymbol{x})|] \rightarrow 0
$$
最终证明，其输出函数的泛化误差在$m$趋于无穷时，趋于贝叶斯风险：
$$
R\left(h_{m}\right)-R^{*} \leqslant 2 \mathbb{E}[|\hat{\eta}(\boldsymbol{x})-\eta(\boldsymbol{x})|]  \rightarrow 0
$$
得证。

