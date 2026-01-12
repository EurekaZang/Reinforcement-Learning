它是一个基于sarsa改进的算法，长这样:$$\begin{aligned} q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + \gamma \mathbb{E}_{a'\sim\pi}[q_t(s_{t+1}, A)])], \\ q_{t+1}(s, a) &= q_t(s, a), \quad \forall (s, a) \neq (s_t, a_t), \end{aligned}$$where $$\mathbb{E}[q_t(s_{t+1}, A)]) = \sum_a \pi_t(a|s_{t+1}) q_t(s_{t+1}, a) \doteq v_t(s_{t+1})$$is the expected value of $q_t(s_{t+1}, a)$ under policy $\pi_t$.

tips: Action-value Function的定义：在状态 $s$ 下采取动作 $a$ 后，遵循策略 $\pi$ 所能获得的期望回报。*更具体地说，它是从状态 $s$ 采取动作 $a$ 开始，然后遵循策略 $\pi$ 的期望累积折扣奖励。*

相比sarsa算法，TD target改变了：$$r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})\Rightarrow r_{t+1}+\gamma \mathbb{E}[q_t(s_{t+1},A)]$$因为求了期望，所以计算量肉眼可见的增加了。
*但是随机性变得更小了*，因为我们不再需要对 $a_{t+1}$ 随机采样，我们需要的数据从 $\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\}$ 变成了$\{s_t,a_t,r_{t+1},s_{t+1}\}$。

Expected sarsa求解的数学方程是什么呢？是另外一种形式的Bellman Expectation Equation$$q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a]$$如果说sarsa求解的是Bellman Expectation Equation的迭代式，使用5个变量 ($s_t,a_t,r_{t+1},s_{t+1},a_{t+1}$) 的随机采样，那么Expected sarsa求解的也是Bellman Expectation Equation的迭代式，但是只使用了4个变量的随机采样。Expected sarsa 的 Bellman Expectation Equation长这样：$$Q_\pi(s, a) = \mathbb{E}_\pi[r(s, a) + \gamma \mathbb{E}_{a' \sim \pi} Q_\pi(s', a')]$$作为对比，sarsa的Bellman Expectation Equation长这样：$$Q_\pi(s, a) = \mathbb{E}_\pi[r(s, a) + \gamma Q_\pi(s', a')]$$***他俩都是名副其实的Bellman Expectation Equation，只不过一个是采样估计，一个是精确期望。***
