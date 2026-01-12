接下里的内容是值函数近似
**首先为什么需要值函数近似？** 因为当我们的state space $|\mathcal{S}|$ 非常大的时候（不再是Grid world，可能是真实世界，存在上百万个state 和 action），我们的计算机内存无法储存这么多的state value以及action value。*这时候就需要一个节省储存空间的思想。*
*另外一个优点是泛化能力，后续会详说。*
值函数近似的核心思想就是，不再储存每个state value（tabular形式），而是储存一个曲线的参数，让这个曲线去拟合state value的probability distribution。
比如这样：![[Pasted image 20250218105159.png]]图中有10个点，使用了一个5次方程去拟合，这样存储的数据量就减小了一倍（从10到5）。
使用值函数近似，state value就可以这样计算得来：$$\hat{v}(s, w) = as + b = \underbrace{[s, 1]}_{\phi^T(s)} \underbrace{\begin{bmatrix} a \\ b \end{bmatrix}}_{w} = \phi^T(s)w$$这个公式中体现了一个线性函数，也就是使用一个线性函数去拟合所有的state value。我们可以把它写成matrix-vector form，也就是 $\phi^T(s)$ 乘 $w$，其中 $\phi^T(s)$ 是feature vector，特征向量，也就是储存着*拟合曲线的维度信息*，一维（线性）就是 $[s,1]$，二维（抛物线）就是$[s^2,s,1]$ 等等。这里面的 $s$ 就是当前state的信息，也就是自变量。$w$ 是parameter vector参数向量，储存着*拟合曲线的参数信息*，*也就是每一项的系数。*
这样确确实实减少了计算机内存的负担，但是也是有代价的。那就是它的拟合精度肯定没有tabular form好。为了提高精度可以选择增加拟合曲线的维度，但是维度通常非常难以选择，即使是在对目标任务有着透彻理解的境况下。
还有一种方法就是不要人工设定维度，*让神经网络去拟合这个方程。* 这也是现在的主流做法。

为什么说它的泛化能力也是优势呢？因为我们现在更新的不再是state value这个值，而是参数 $w$，也就是每一次更新改变的是曲线的整体的状态，更新一次每一个曲线上的state value都会或多或少的变化。