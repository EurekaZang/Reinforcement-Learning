# Motivating example
区别于之前讲的值迭代和策略迭代 (**model-based RL**)，MC-basic是一种**model-free**的方法，也就是完全依靠于数据。
MC方法的基本思想：
1. 做很多次实验，得到很多采样，求它们的平均值
2. 根据大数定理 (The Law of Large Number)，当样本量增加，它们的平均值将趋近于它们的理论期望。
具体来说：
通过大量采样得到了一组数据： $\{x_1,x_2,x_3,x_4,...,x_\mathbf{N}\}$
它们的期望可以被近似为：$$\mathbb{E}[X]\approx \bar{x}=\frac{1}{N}\sum^{N}_{j=1}x_j$$在蒙特卡罗方法中，为什么要关注变量的期望值？因为我们 RL 中的state value 和 action value都是期望值。以上阐明了MC方法的核心思想，怎么使用且听下回分解。
想要知道MC方法如何工作，先从主流算法**Policy Iteration**中切入：**Policy iteration** has two steps in each iteration:$$\left\{
\begin{aligned}
    & \textbf{Policy evaluation}: v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k} \\
    & \textbf{Policy improvement}: \pi_{k+1} = \arg \max_{\pi}(r_{\pi} + \gamma P_{\pi}v_{\pi_k})
\end{aligned}
\right.$$The elementwise form of the policy improvement step is:
$$
\begin{aligned}
\pi_{k+1}(s) &= \arg \max_{\pi} \sum_a \pi(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s') \right] \\
&= \arg \max_{\pi} \sum_a \pi(a|s)q_{\pi_k}(s, a), \quad s \in S
\end{aligned}
$$
The key is $q_{\pi_k}(s, a)$ ! 怎么求q呢？按照传统的model-based方法应该按照model计算，也就是$$\color{violet}\sum_r p(r|s, a),\sum_{s'} p(s'|s, a)$$但同时，我们也要想到q值的原始的表达式：
$$q_{\pi_k}(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$而这个从表达式中就能看出来，它不依赖于模型（概率）。而且它是一个discounted return的期望，这就给了MC方法操作空间。
使用MC方法的具体的计算方式：
*   Starting from $(s, a)$, following policy $\pi_k$, generate an episode.
*   The (discounted) return of this episode is $g(s, a)$
*   $g(s, a)$ 是 $G_t$ 的一个采样$$q_{\pi_k}(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$
*   Suppose we have a set of policy and hence a set of episodes and hence $\{g^{(i)}(s,a)\}$. Then,$$q_{\pi_k}(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a] \approx \frac{1}{N} \sum_{i=1}^{N} g^{(i)}(s, a)$$
***Fundamental idea: When model is unavailable, we can use data.***
可见，MC方法实际上就是policy iteration，只是在第一步PE计算每一个 $(s,a)$ 对的所有 $q_\pi(s,a)$ 的时候没有用模型，而是通过样本（episode的discounted return）的均值去近似它的期望。
伪代码：
**Initialization:** Initial guess $\pi_0$.
**Aim:** Search for an optimal policy.

While the value estimate has not converged, for the $k$th iteration, do
  For every state $s \in \mathcal{S}$, do
    For every action $a \in \mathcal{A}(s)$, do
      Collect sufficiently many episodes starting from $(s, a)$ following $\pi_k$
      *MC-based policy evaluation step:*
      $q_{\pi_k}(s, a) =$ average return of all the episodes starting from $(s, a)$
    **Policy improvement step:**
    $a_k^*(s) = \arg \max_a q_{\pi_k}(s, a)$
    $\pi_{k+1}(a|s) = \begin{cases} 1 & \text{if } a = a_k^* \\ 0 & \text{otherwise} \end{cases}$

到这里MC-basic就介绍完毕了。但是在生产实践中，我们不适用MC-basic，因为它的效率极低。这也就是为什么我们后面要介绍其他MC方法的原因。

# The END