这篇笔记将统一sarsa算法和Monte Carlo算法。
我们来看这样的推导：
The definition of action value is
$$
\begin{equation}
q_\pi(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a].
\end{equation}
$$
The discounted return $G_t$ can be written in different forms as:$$
\begin{aligned}
\text{Sarsa} \leftarrow \quad G_t^{(1)} &= R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}), \\
G_t^{(2)} &= R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, A_{t+2}), \\
& \vdots \\
\text{n-step Sarsa} \leftarrow \quad G_t^{(n)} &= R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_\pi(S_{t+n}, A_{t+n}), \\
& \vdots \\
\text{MC} \leftarrow \quad G_t^{(\infty)} &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
\end{aligned}
$$
It should be noted that $G_t = G_t^{(1)} = G_t^{(2)} = G_t^{(n)} = G_t^{(\infty)}$, where the superscripts merely indicate the different decomposition structures of $G_t$.
可以见得，sarsa和MC都是n-steps sarsa的极端情况。sarsa只使用仅下一步的reward和后续的return的估计来更新q值，而MC使用后续所有的reward来更新q值，不采用任何估计。

- Sarsa aims to solve
$$
\begin{aligned}
q_\pi(s, a) &= \mathbb{E}[G_t^{(1)} | s, a] \\
&= \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | s, a].
\end{aligned}
$$
- MC learning aims to solve
$$
\begin{aligned}
q_\pi(s, a) &= \mathbb{E}[G_t^{(\infty)} | s, a] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots | s, a].
\end{aligned}
$$
- An intermediate algorithm called n-step Sarsa aims to solve
$$
\begin{aligned}
q_\pi(s, a) &= \mathbb{E}[G_t^{(n)} | s, a] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) | s, a].
\end{aligned}
$$
- The algorithm of n-step Sarsa is
$$
\begin{aligned}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, a_t) [q_t(s_t, a_t) - (r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n q_t(s_{t+n}, a_{t+n}))].
\end{aligned}
$$
n-step Sarsa is more general because it becomes the (one-step) Sarsa algorithm when $n = 1$ and the MC learning algorithm when $n = \infty$.
*当n很大的时候，它的表现更类似于MC learning，它会有较大的variance和较小的bias。*
*当n很小的时候，它的表现类似于sarsa，会有相对来更小的variance和更大的bias。*
之前讲MC learning的时候就知道，MC方法需要等，要等agent将episodes走完后才能更新策略（学习），所以它是一个offline方法。sarsa完全不用等，一拿到一个 $(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$ 就能更新策略，所以它是一个Online方法。
n-steps sarsa不是严格意义上的Online方法，因为它也需要等待，等agent走n个steps才能更新。

代码实现：
**不展示了，可以在GridWorld.py中找到。**

# The END