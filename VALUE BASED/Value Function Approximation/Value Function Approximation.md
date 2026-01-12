下面正式的介绍这个算法。
我们刚刚说要用一个曲线来储存所有的state value，也就是这其实是在做policy evaluation，因为我们在试图求解最优的state value $v_\pi(s)$。
我们刚刚又说了，由于feature vector $\phi^T(s)$ 是不变的（给定的），我们的值函数近似算法只需要更新parameter vector $w$。目标就是优化一个方程。
分为两步：
1. 正式的定义这个目标方程（优化目标）
2. 讨论优化方法
首先来看这个目标函数：$$J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S, w))^2]$$这就是大名鼎鼎的均方误差 **Mean Squared Error, MSE**，就是真实值和预测值的平方误差的均值。
一般情况下均方误差展开是长这样：$$J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S, w))^2] = \frac{1}{|S|} \sum_{s \in S} (v_\pi(s) - \hat{v}(s, w))^2$$但是这样默认所有在state space $\mathcal{S}$ 中的所有state的权重是相同的。这很明显是不对的，设想agent从一个state出发要到达target state，离target state近的state明显比离target state很远的state更重要。为了体现重要性，我们要给他们不同的权重。也就是改写这个probability distribution。
那我们怎么知道这些state的重要性呢？这里引入一个long-run思想。也就是让agent在environment中进行马尔可夫行为，当episode很大的时候概率分布会趋于稳定，这个稳定的概率分布我们叫做***stationary distribution***也就是***静态分布***。从stationary distribution中可以知道哪些state被访问的次数/频率最高。越高的次数/频率就代表着应得的更高的权重。
数学定义：
Let $\{d_\pi(s)\}_{s \in S}$ denote the stationary distribution of the Markov process under policy $\pi$. By definition, $d_\pi(s) \geq 0$ and $\sum_{s \in S} d_\pi(s) = 1$.
那么目标函数就可以展开成这样：$$J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S, w))^2] = \sum_{s \in S} d_\pi(s) (v_\pi(s) - \hat{v}(s, w))^2$$
用一个例子来更好的理解stationary distribution：![[Pasted image 20250218114111.png]]我们让agent在如图的网格世界（固定策略 $\pi$）中跑很多个episode，然后给出Percentage each state visited计算方式如下$$d_\pi(s) \approx \frac{n_\pi(s)}{\sum_{s' \in S} n_\pi(s')}$$也就是$$\frac{\text{该state的访问次数}}{总步数}$$
回到刚刚的目标函数
只要minimize这个均方误差，我们就能实现优化 $w$ 的目标。因为预测值向真实值逐渐靠近。
说白了就是最小化这个方程。提到最小化，我们首先想到的就是梯度下降 GD 算法。$$w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)$$化简一下：$$\begin{aligned}
\nabla_w J(w) &= \nabla_w \mathbb{E}[(v_\pi(S) - \hat{v}(S, w))^2] \\
&= \mathbb{E}[\nabla_w (v_\pi(S) - \hat{v}(S, w))^2] \\
&= 2\mathbb{E}[(v_\pi(S) - \hat{v}(S, w))(-\nabla_w \hat{v}(S, w))] \\
&= -2\mathbb{E}[(v_\pi(S) - \hat{v}(S, w))\nabla_w \hat{v}(S, w)]
\end{aligned}$$可以看到这里有个期望，那不如直接上SGD：$$w_{t+1} = w_t + \alpha_t (v_\pi(s_t) - \hat{v}(s_t, w_t)) \nabla_w \hat{v}(s_t, w_t),$$其中 $2\alpha_t$ 简写成了 $\alpha_t$ 。将所有的期望都换成了随机采样。欸但是我们👉发现，方程中有一个 $v_\pi(s_t)$ 这个我们不知道，甚至就是我们要求的。怎么办？首先可以用蒙特卡罗方法，在当前state开启很多个episode，然后对他们的return求均值：$$w_{t+1} = w_t + \alpha_t (g_t - \hat{v}(s_t, w_t))\nabla_w\hat{v}(s_t, w_t).$$既然可以用MC，那么TD也自然能用：$$w_{t+1} = w_t + \alpha_t \overbrace{\underbrace{[r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)}_{\text{TD target}} - \hat{v}(s_t, w_t)]}^{\text{TD error}} \nabla_w \hat{v}(s_t, w_t).$$
伪代码：
**Initialization:** A function $\hat{v}(s, w)$ that is a differentiable in $w$. Initial parameter $w_0$.
**Aim:** Approximate the true state values of a given policy $\pi$.
For each episode generated following the policy $\pi$, do
  For each step $(s_t, r_{t+1}, s_{t+1})$, do
    In the general case,
    $w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t)] \nabla_w \hat{v}(s_t, w_t)$
    In the linear case,
    $w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \phi^T(s_{t+1})w_t - \phi^T(s_t)w_t]\phi(s_t)$

下面来介绍如何选取so-called $\hat{v}(s_t,w_t)$，有两种主流思路：
1. 使用线性函数：$$\hat{v}(s_t,w_t)=\phi^T(s)w$$也就是feature vector乘parameter vector，是一个线性的关系。
2. 使用神经网络：$$\hat{v}(s_t,w_t)=f^w(s)$$这里的神经网络就是一个非线性的函数。

如果选择线性函数，那么：$$\nabla_w \hat{v}(s, w) = \phi(s).$$可以直接将 $\phi(s)$ 带入TD方程中。也就诞生了TD-Linear。

最后还有一个小特性，那就是Tabular形式的TD算法可以和值函数近似的算法进行统一。怎么做到呢？那就是选择线性，然后让feature vector变成一个独热向量。这里就不展开了。

 