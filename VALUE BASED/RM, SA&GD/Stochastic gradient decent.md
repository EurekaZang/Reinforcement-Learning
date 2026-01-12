我们即将介绍的SGD是一种特殊的Robbins-Monro算法，刚刚介绍的 $\text{mean estimation}$ 算法也是一种特殊的SGD算法。
**Motivating example：**
假设我们要优化以下问题：$$\underset{w}{\min} \quad J(w)=\mathbb{E}[f(w,X)]$$$w$ 是需要被优化的参数。$X$ 是随机变量，具有固定的概率分布。他俩可以是标量或者向量。
我们有多种方法来求解：
1. 梯度下降 (**GD**)：$$w_{k+1} = w_k - \alpha_k \nabla_w \mathbb{E}[f(w_k, X)] = w_k - \alpha_k \mathbb{E}[\nabla_w.f(w_k, X)]$$第二个等号什么意思呢？因为是对函数的期望的梯度，所以可以把梯度放到期望里面去。两者是等效的。
	- ***Why it works***：梯度下降的基本思想是*沿着梯度的相反方向更新参数*，==因为在梯度的方向上，目标函数的值上升最快==，而沿反方向更新能够让我们朝着*函数值最小化*的方向移动。
	>这个方法能够最小化参数的关键原因在于**目标函数的局部线性化**。在每一步更新时，GD通过线性近似目标函数来调整参数。假设目标函数是连续且可微的，梯度下降方法在每一步都会使得目标函数的值朝着*局部最小值*方向移动。随着迭代次数的增加，参数逐渐接近最优解，特别是在一些优化条件下，如目标函数是凸的时，梯度下降能够保证收敛到全局最小值。
	
2. 批量梯度下降 (**BGD**)：求全局期望的效率实在是太低了，可以使用局部的期望来代替全局期望，这样能够提高效率。$$
\mathbb{E}[\nabla_w f(w_k, X)] \approx \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i).
$$$$
w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i).
$$
3. 随机梯度下降 (**SGD**)：$$
w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k),
$$
* Compared to the **GD**: 替换 the true gradient $\mathbb{E}[\nabla_w f(w_k, X)]$ by the stochastic gradient $\nabla_w f(w_k, x_k)$.
* Compared to the batch gradient descent method: let $n = 1$.
	*有人说这样精度不行，但实际上效率比前两种都高。*


# 收敛性证明
证明 SGD 的收敛性：证明他是一个Robbins-Monro算法
The aim of SGD is to minimize$$J(w) = \mathbb{E}[f(w, X)]$$This problem can be converted to a root-finding problem (***这个转化恒成立***):$$\nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)] = 0$$Let$$g(w) = \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)]$$Then, the aim of SGD is to find the root of $g(w) = 0$.
那就太简单了，已知 $g(w) = \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)]$
观测值 $\tilde{g}(w)$ 就是 $\tilde{g}(w)=\nabla_w f(w, X)$
代入公式得到：$$w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k) = w_k - a_k \nabla_w f(w_k, x_k).$$证毕


# 收敛行为分析
先说结论，当 $w$ 距离 $w^*$ 较远的时候，SGD算法相似于GD，也就是随机性不大，能够较为精准的下降。
当 $w$ 距离 $w^*$ 较近时，SGD才体现出它的随机性。
SGD的这个特性是很好的，能够有效保障收敛。
$$\delta_k \doteq \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{|\mathbb{E}[\nabla_w f(w_k, X)]|}.$$
Since $\mathbb{E}[\nabla_w f(w^*, X)] = 0$, we further have
$$\delta_k = \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{|\mathbb{E}[\nabla_w f(w_k, X)] - \mathbb{E}[\nabla_w f(w^*, X)]|} = \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{|\mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)(w_k - w^*)]|}.$$
where the last equality is due to the mean value theorem and $\tilde{w}_k \in [w_k, w^*]$. 
>这一步用到的是中值定理，也就是$$f(x_1)-f(x_2)=f'(x_3)(x_1-x_2)$$

继续对式子做变换：假设 $f$ 是严格凸的，such that
$$\nabla_w^2 f \geq c > 0$$
for all $w, X$, where $c$ is a positive bound.
Then, the denominator of $\delta_k$ becomes
$$\begin{aligned}|\mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)(w_k - w^*)]| &= |\mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)](w_k - w^*)| \\&= |\mathbb{E}[\nabla_w^2 f(\tilde{w}_k, X)]||(w_k - w^*)| \geq c|w_k - w^*|.\end{aligned}$$
Substituting the above inequality to $\delta_k$ gives
$$\delta_k \leq \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{c|w_k - w^*|}.$$
这下，刚刚给出的结论就很直观了，denominator中差越大（离得越远）

# 比较BGD，mBGD和SGD
Suppose we would like to minimize $J(w) = \mathbb{E}[f(w, X)]$ given a set of random samples $\{x_i\}_{i=1}^n$ of $X$. The BGD, MBGD, SGD algorithms solving this problem are, respectively,$$w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i),$$(BGD)$$w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in \mathcal{I}_k} \nabla_w f(w_k, x_j),$$(MBGD)$$w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k).$$(SGD)
如果 $n=m$，MBGD变成了BGD；
如果 $n=1$，MBGD不完全是 BGD。因为BGD用的是全部采样里的n个采样（非重复），而MBGD用的是n个采样里的n个采样，意思就是在n个采样中随机抽取，有可能有些采样抽到了好多次，有可能有些采样一次也没用上。（有可能重复）
下面附上一张三种算法的可视化：改图是根据给定的线性方程 `y = 2x + 3` 生成数据，并添加一些随机噪声（`randn`），让三种算法去拟合这个线性方程。==很明显，SGD下降的最快，效率最高；BGD下降的最平稳；MBGD介于两者之间。==![[pics.png]]该图是在 (10, -10) 间随机生成点，让三种算法去逼近数据的均值。从这张图中可以看出SGD在离True Mean较远时，呈现出的性质和有batch的GD差不多，但是离True Mean较近时却呈现出较大的随机性。![[pics2.png]]