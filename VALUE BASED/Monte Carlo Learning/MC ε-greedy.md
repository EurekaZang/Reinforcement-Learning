再重复一遍，MC ε-greedy是MC Exploring Starts的改进版，也就是把Exploring Starts的条件去掉，不再需要agent从每个 (s,a) 出发生成episodes。**也就是让visit发挥作用，只要visit到当前 (s,a) 就能估计这个 (s,a) 的 q(s,a) 值。**
先讲清楚一个概念：
1. *deterministic的策略都是greedy策略*，因为agent贪婪的选择最大（最优，最好）的action，***丝毫不会考虑其他action。***
2. *stochastic的策略是soft的*，也就是策略选择最优action的概率并不为1，而是0.99或者0.95等等。***有少许的概率选择其他的action。***
即将要介绍的ε-greedy policy就是一个stochastic策略
加入少许概率让agent选择其他action有啥好处呢？可以去掉exploring starts这个条件。为什么？
想象一次训练，agent从一个 (s,a) 出发，按照一个策略在grid-world中运动，**它不一定一直在最优action trajectory上运动，有可能选择非最优action从而visit不在最优策略上的 (s,a)。** 这样，只要我们的episode length足够大，***agent就一定能访问到所有的 (s,a)。*** 也就能够去掉exploring starts这个条件———我们不再需要从每一个 (s,a) 出发，而是从一个或者少数 (s,a) 出发就能计算所有 (s,a) 的 q(s,a)了。
策略具体计算方式如下：
$$
\pi(a|s) = 
\begin{cases}
1 - \frac{\epsilon}{|A(s)|}(|A(s)| - 1), & \text{for the greedy action,} \\
\frac{\epsilon}{|A(s)|}, & \text{for the other } |A(s)| - 1 \text{ actions.}
\end{cases}
$$

where $\epsilon \in [0, 1]$ and $|A(s)|$ is the number of actions for $s$.
其中，选择最优action的概率永远比选择其他action的概率更大，具体证明如下：
$$1 - \frac{\epsilon}{|A(s)|}(|A(s)| - 1) = 1 - \epsilon + \frac{\epsilon}{|A(s)|} \geq \frac{\epsilon}{|A(s)|}.$$
**为什么要使用 ε-greddy?** 因为它更好的平衡了exploitation 剥削和exploration 探索
1. exploitation指的是充分利用，比如说现在有一个最优action，agent保证自己会去采取这个最优action，后面能获得更多的reward
2. exploration指的是探索性，也就是寻找最优策略，我现在知道了一个最优action，但是我现在的信息是不完备的，我可以去探索其他的action，去发现有着更高g的trajectory。

1. 如果 ε=0，那么ε-greedy就变成了greedy策略，也就是deterministic策略。
2. 如果 ε=1，那么选择所有action的概率就一样了，也就是变成了均匀分布，探索性会变得很强。

接下来我们来讲一下 epsilon-greedy策略怎么用于MC方法中。在MC basic和 MC exploring starts中，我们要求解一个方程（PE）：$$
\pi_{k+1}(s) = \underset{\pi \in \color{violet}\Pi}{\text{arg max}} \sum_a \pi(a|s)q_{\pi_k}(s, a).
$$其中 $\color{violet}{\Pi}$ 是全部可能策略的集合。
策略会直接选择具有最大的q的action。数学表达：$$
\pi_{k+1}(a|s) = 
\begin{cases}
1, & a = a_k^*, \\
0, & a \neq a_k^*.
\end{cases}
$$但是在ε-greedy中，这个方程变成了这样：$$
\pi_{k+1}(s) = \underset{\pi \in \color{violet}\Pi_\epsilon}{\text{arg max}} \sum_a \pi(a|s)q_{\pi_k}(s, a)
$$其中 $\color{violet}\Pi_\epsilon$ 是全部 ε-greedy策略的集合。
具体选择方法如下：
- **随机选择:** 有 ϵ 的概率会随机选择一个 action，这保证了对环境的探索，避免陷入局部最优。
- **贪婪选择:** 有 (1 - ϵ) 的概率会选择当前认为最优的 action，利用已有的知识。
数学表达：$$
\pi_{k+1}(a|s) = 
\begin{cases}
1 - \frac{|A(s)|-1}{|A(s)|}\epsilon, & a = a_k^*, \\
\frac{1}{|A(s)|}\epsilon, & a \neq a_k^*.
\end{cases}
$$
MC epsilon-greedy方法的伪代码表示如下：
**Initialization**: Initial guess $\pi_0$ and the value of $\epsilon \in [0, 1]$
**Aim**: Search for an optimal policy.

For each episode, do
  *Episode generation*: Randomly select a starting state-action pair $(s_0, a_0)$. Following the current policy, generate an episode of length $T$: $s_0, a_0, r_1, \dots, s_{T-1}, a_{T-1}, r_T$.
  **Policy evaluation** and **policy improvement**:
    *Initialization*: $g \leftarrow 0$
    For each step of the episode, $t = T - 1, T - 2, \dots, 0$, do
      $g \leftarrow \gamma g + r_{t+1}$
      Use the **every-visit** method:
        $\text{Returns}(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) + g$
        $q(s_t, a_t) = \text{average}(\text{Returns}(s_t, a_t))$
      Let $a^* = \arg \max_a q(s_t, a)$ and $\pi(a|s_t) =\begin{cases}1 - \frac{|\mathcal{A}(s_t)|-1}{|\mathcal{A}(s_t)|}\epsilon, & a = a^*, \\\frac{1}{|\mathcal{A}(s_t)|}\epsilon, & a \neq a^*.\end{cases}$
  可见，相比MC exploring statrs，MC ε-greedy只是将策略选择部分从greedy换成了ε-greedy，且 **First-visit** 变成了 **every-visit** ，因为我们会有一个非常长的episode，如果只是使用第一次visit计算该 (s,a) 的q，会很浪费。

下面是我是实现的**MC optimal ε-greedy policy**：
![[Pasted image 20250203131422.png]]