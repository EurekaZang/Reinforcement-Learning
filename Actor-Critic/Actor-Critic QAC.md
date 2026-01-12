也就是AC方法，算法由两部分组成：
1. Actor：也就是迭代策略，更新参数 $\theta$ 的部分，具体来说是这一整个公式 $$\nabla_{\theta} J(\theta) = \nabla_{\theta} \ln \pi(a_t|s_t, \theta) \, q_{\pi}(s_t, a_t)$$
2. Critic：也就是求解（估计，近似） $q_\pi$ 的值的部分，例如：REINFORCE中使用的是蒙特卡罗方法。这节课要讲的AC方法就是用TD方法去估计 $q_\pi$ 的值。
伪代码：
**Aim**: Search for an optimal policy by maximizing $J(\theta)$.
At time step $t$ in each episode, do:
	Generate $a_t$ following $\pi(a|s_t, \theta_t)$, observe $r_{t+1}$, $s_{t+1}$, and then generate $a_{t+1}$ following $\pi(a|s_{t+1}, \theta_t)$.
Critic (value update):
$$
w_{t+1} = w_t + \alpha_w \left[ r_{t+1} + \gamma q(s_{t+1}, a_{t+1}, w_t) - q(s_t, a_t, w_t) \right] \nabla_w q(s_t, a_t, w_t)
$$

Actor (policy update):
$$
\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \cdot q(s_t, a_t, w_{t+1})
$$
这里的第一个式子会很眼熟，因为他就是一个Sarsa Learning。那么收集的数据自然也是sarsa需要的数据，也就是：$\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\}$ 。
然后第二步正常的进行actor的工作，在拿到 $q(s_t,a_t,w_{t+1})$ 之后。
这个AC算法是On-policy的，且是Online的。
既然是On-policy，要不要做成 $\epsilon$-greedy给予它一定的探索性呢？其实是不用的，因为我们的 $\pi$ 具有恒大于零小于一的性质（$\text{softmax}$）。
上面的这个算法叫做 $\text{Q Actor-Critic}$。也就是使用sarsa来估计 $q$ 值。

# The END