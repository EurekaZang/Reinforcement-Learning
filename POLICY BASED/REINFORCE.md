在Policy gradient中我们已经得到了梯度上升的公式：$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \ln \pi(a_t|s_t, \theta) \, q_{\pi}(s_t, a_t)$$可以看到式子中有一个 $q_\pi(s_t,a_t)$ 这个是我们不知道的，但是我们可以使用一些算法来估计它。常见的估计算法有蒙特卡罗方法，或者sarsa。
这里使用MC方法来使用 $q_t$ 估计 $q_\pi$ 。
大致思路：在一个state产生很多条episodes，然后求折扣回报的均值：$G_t$ 带回公式得到：$$\nabla_{\theta} J(\theta) = \alpha\,G_t\nabla_{\theta} \ln \pi(a_t|s_t, \theta)$$
这个算法很简单，直接给出伪代码：$$\begin{aligned} &\text{Initialization: A parameterized function } \pi(a|s, \theta),\ \gamma \in (0,1),\ \text{and } \alpha > 0. \\ &\text{Aim: Search for an optimal policy maximizing } J(\theta). \\ \\ &\text{For the } k^{\text{th}} \text{ iteration, do} \\ &\quad \text{Select } s_0 \text{ and generate an episode following } \pi(\theta_k). \text{ Suppose the episode is} \\ &\quad \{s_0, a_0, r_1, \dots, s_{T-1}, a_{T-1}, r_T\}. \\ &\quad \text{For } t = 0, 1, \dots, T-1 \text{, do} \\ &\qquad \text{Value update: } q_t(s_t, a_t) = \sum_{k=t+1}^T \gamma^{k-t-1} r_k \\ &\qquad \text{Policy update: } \theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t, \theta_t) \cdot q_t(s_t, a_t) \\ &\quad \theta_k = \theta_T \end{aligned}$$
可以看到，首先这是一个On-policy算法，其次，它是一个offline算法。

# The END
