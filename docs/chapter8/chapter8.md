# 第8章：遗憾界

*Edit: 赵志民，Hao ZHAN*

---

本章的内容围绕学习理论中的遗憾（regret）概念展开（有的教材里也翻译为“悔” ）。
通常，我们使用超额风险（excess risk）来评估批量学习的分类器性能，而用遗憾来评估在线学习的分类器性能。
二者的不同在于，前者衡量的是整个学习过程结束后所得到的分类器性能，可以理解为学习算法**最终输出的模型**与假设空间内**最优模型**的风险之差；
而后者衡量的是算法运行过程中，所产生的**模型**与假设空间内**最优模型**的损失之差的**和**。



## 1. 【概念补充】超额风险与遗憾的区别

8.1介绍了遗憾这一评估指标的基本概念，我们在此基础上梳理一下其与超额风险这一评估指标的区别。

超额风险这一评估指标被定义为，
$$
ER = \mathbb{E}_{(x,y)\sim D[l(w_{T+1},(x,y))]} - min_{w \in W} \mathbb{E}_{(x,y)\sim D[l(w,(x,y))]}
$$
其中，$ER$ 指的是excess risk，等式右边的前半部分 $\mathbb{E}_{(x,y)\sim D[l(w_{T+1},(x,y))]}$ 指的是模型 $w_{T+1}$ 的风险，等式右边的后半部分 $min_{w \in W} \mathbb{E}_{(x,y)\sim D[l(w,(x,y))]}$ 指的是假设空间内的最优模型的风险。值得注意的是，这里的评估是在整个数据集上进行的，也正是因为如此，我们必须要引入期望的操作。

而遗憾这一评估指标，被定义为，
$$
regret = \sum^{T}_{t=1}f_t(w_t)-min_{w\in W}\sum^{T}_{t=1}f_t(w)
$$
其中，$f_t(w_t)$ 指的是，
$$
\sum^{T}_{t=1}l(w_t,(x_t,y_t)) - min_{w \in W}\sum^{T}_{t=1}l(w,(x_t,y_t))
$$
由于$w_t$的计算过程与样本$(x_t,y_t)$ 无关，而是与$(x_1,y_1)...(x_{t-1},y_{t-1})$ 有关，因此可以直接使用 $l(w,(x_t,y_t))$ 来衡量性能。

由此，我们可以总结出二者之间的两个主要区别。一是超额风险引入了**期望**而遗憾没有；二是超额风险计算是一次性在所有数据上进行的计算，而遗憾是对多次损失的一个**求和**。同时，由于在线学习不依赖于任何分布假设，因此其适用于一系列非独立同分布样本，或者才样子固定分布的情形。



## 2. 【定理补充】随机多臂赌博机遗憾界

**P172**中定理8.3给出了随机多臂赌博机的遗憾界，我们在此基础上对部分证明过程进行补充。

首先，（8.42）给出当$\overline{\mu}_*(p)+\sqrt{\frac{2\ln t}{p}}\le\overline{\mu}_i(q)+\sqrt{\frac{2\ln t}{q}}$成立时，必然有一个成立的三种可能情况。
但是这三种情况并不是互斥的，因此显得很不直观，这里把第二种情况做了细微调整，即：
$$
\overline{\mu}_*(p)+\sqrt{\frac{2\ln t}{p}}\le\mu_*,\mu_*\le\overline{\mu}_i(p)+\sqrt{\frac{2\ln t}{q}},\overline{\mu}_i(p)+\sqrt{\frac{2\ln t}{q}}\le\overline{\mu}_i(q)
$$
此时，构造（8.44）和（8.45）的逻辑就显得更为顺畅。
我们令$\ell=\lceil(2\ln T)/\Delta_i^2\rceil$，则（8.45）转化为：
$$
P(\mu_*\le\mu_i+\sqrt{\frac{2\ln t}{q}})=0,q\ge\ell
$$
代入（8.44），可得：
$$
\begin{aligned}
\mathbb{E}[n_i^T]&\le\lceil\frac{2\ln T}{\Delta_i^2}\rceil+2\sum_{t=1}^{T-1}\sum_{p=1}^{t-1}\sum_{q=\ell}^{t-1}t^{-4} \\
&\le\frac{2\ln T}{\Delta_i^2}+1+2\sum_{t=1}^{T-1}\sum_{p=1}^{t}\sum_{q=1}^{t}t^{-4} \\
&\le\frac{2\ln T}{\Delta_i^2}+1+2\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2} 
\end{aligned}
$$
根据$p$-级数判别法，当$p=2\gt1$时，级数收敛，因此$\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2}$是有界的。
至于该级数的具体值，对定理的结论并没有影响，因此我们可以直接将其视为一个常数，然后带入后续的推导过程中。
不过这里出于证明完整性的考虑，我们对此进行简要说明。

$\lim_{T\rightarrow+\infty}\sum_{t=1}^{T}t^{-2}$的取值在数学界被称为Basel问题，推导过程涉及诸多前置定理，感兴趣的同学可以查看这个[讲义](https://www.math.cmu.edu/~bwsulliv/basel-problem.pdf)。
此处给出另一种在微积分变换中更为常见的缩放方法，即：
$$
\begin{aligned}
\sum_{t=1}^{T-1}t^{-2}&\le1+\int_{1}^{T-1}\frac{1}{x^2}dx \\
&=1+(-\frac{1}{x})|_1^{T-1} \\
&=2-\frac{1}{T}
\end{aligned}
$$
对不等式两边同时取极限，可得：
$$
\lim_{T\rightarrow+\infty}\sum_{t=1}^{T-1}t^{-2}\le2
$$
代入（8.46），一样可以得到类似（8.47）的结论。

这里依旧沿用书中给出的$\lim_{T\rightarrow+\infty}\sum_{t=1}^{T}t^{-2}=\frac{\pi^2}{6}$，代入（8.46）得到遗憾界（8.47），即：
$$
\mathbb{E}[regret]\le\sum_{i=1}^{K}\frac{2\ln T}{\Delta_i^2}+O(1)
$$

此时（8.46）变成：
$$
\mathbb{E}[n_i^T]\le\sum_{i\neq*}^K\frac{2\ln T}{\Delta_i}+(1+\frac{\pi^2}{3}){\Delta_i}=O(K\log T)
$$
观察（8.47）可知，求和公式中的每一项符合对钩函数的构造，即：
$$
f(x)=Ax+\frac{B}{x},x\gt0,A\gt0,B\gt0
$$
这里$x=\Delta_i,A=1+\frac{\pi^2}{3},B=2\ln T$，因此无论$\Delta_i$过大或过小时，都会导致遗憾界的上界变大。
另外，遗憾界跟摇臂的个数$K$呈线形关系，当$K$越大时，遗憾界也会越大。



## 3. 【概念补充】线性赌博机

**P176**的8.3.2节介绍了线性赌博机的概念，我们在此基础上对参数估计部分进行补充。
为了估计线性赌博机的参数，我们把原问题转化为了岭回归问题，即（8.52）：
$$
f(w)=(Y-w^T X)^T(Y-w^T X)+\lambda w^T w
$$
为了求得最优解$w^*$，我们令$f'(w)=0$，可推导出（8.53）：
$$
\begin{aligned}
&\frac{\partial f(w)}{\partial w}=-2X^T(Y-w^T X)+2\lambda w = 0 \\
\Rightarrow&X^TY = (X^TX + \lambda I)w \\
\Rightarrow&w^* = (X^TX + \lambda I)^{-1}X^TY
\end{aligned}
$$
相比于每次传入新数据$(x_t,y_t)$时从头计算$w_t$，这里巧妙地利用了Sherman-Morrison-Woodbury公式，继而在$O(d^2)$的时间复杂度内完成参数的更新，即（8.55）至（8.57）。
值得注意的是，Sherman-Morrison-Woodbury公式可以将任何形如$(A+uv^T)^{-1}$的矩阵逆转化为可逆矩阵$A$和列向量$u,v$之间的运算，有效地降低了计算量。



## 4. 【定理补充】凸赌博机的遗憾界

**P182**中定理8.5给出了凸赌博机的遗憾界，在证明的开始，作者就对$\eta,\alpha,\delta$的取值进行了限定。
我们可以发现这些取值不是很直观，证明给出的解释也比较分散，特别地，部分取值跟证明略有出入，因此我们在此对其进行补充。

对于步长$\eta$，我们在缩放（8.87）中 $\mathbb{E}[\sum_{t=1}^T\hat f_t(z_t)]-\min_{w\in(1-\alpha)\mathcal{W}}\sum_{t=1}^T\hat f_t(w)$ 时，想要为使用引理8.3创造条件，因此采用步长$\eta=\frac{\Lambda}{l'\sqrt{T}}$。
根据（8.89）的推导，我们可令$\Lambda=\Lambda_2$且$l'=\frac{dc}{\delta}$，此时，将$\eta=\frac{\Lambda_2}{(dc/\delta)\sqrt T}$带入到更新公式（8.76）中便可得到（8.88）。

对于缩减系数$\alpha$与扰动系数$\delta$，我们可以一同考虑这两个系数的取值。
观察（8.91）第一个不等式的形式，我们发现这是一个关于$\delta$的对钩函数：
$$
f(\delta)=A\delta+\frac{B}{\delta}+C
$$
假设$\alpha$的取值与$\delta$无关，那么：
$$
A=3lT,B=dc\Lambda_2\sqrt T,C=2\alpha cT
$$
令$f'(\delta)=0$，可得：
$$
\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_2}{3l}}
$$
此时，可得到$f(\delta)$的最小值：
$$
f(\delta^*)=O(T^{3/4})
$$
如果我们想要加速收敛，那么我们可以将$\alpha$的取值与$\delta$相关联。
从上面的结论可知，当迭代次数$T$足够大时，必然有$\delta\rightarrow0$。
因此，我们不妨取$\alpha=\frac{\delta}{\Lambda_1}$，代入（8.91）中并利用对钩函数$f(\delta)$的性质，可得：
$$
\begin{aligned}
&\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_1\Lambda_2}{3(l\Lambda_1+c)}} \\
&f(\delta^*)=O(T^{3/4})
\end{aligned}
$$
进一步地，我们可以发现，$\delta*$的取值并不唯一，这是因为（8.91）的第二个不等式缩放并非必需。
如果我们取$\delta^*=T^{-1/4}\sqrt{\frac{dc\Lambda_1\Lambda_2}{3l\Lambda_1+2c}}$，亦可得到更紧致的遗憾界，并保证定理的结论不变。
