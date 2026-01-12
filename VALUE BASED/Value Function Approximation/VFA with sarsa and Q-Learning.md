个人认为线性的值函数近似不如后面的DQN。所以这里将这两个算法合在一起记录。
#### 首先看saras with VFA：
$$w_{t+1} = w_t + \alpha_t \Big[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \Big] \nabla_w \hat{q}(s_t, a_t, w_t).$$还记得吗，从TD到sarsa算法只需要将 $v(s_t)$ 改成 $q(s_t,a_t)$，这里也是一样。
这样就是直接估计每个state 的q值来更新参数了。
刚刚的TD with VFA还是在做policy evaluation，但现在的sarsa with VFA还可以加入policy improvement的部分，直接看伪代码：
**Aim:** Search a policy that can lead the agent to the target from an initial state-action pair $(s_0, a_0)$.

For each episode, do:
*   If the current $s_t$ is not the target state, do:
    *   Take action $a_t$ following $\pi_t(s_t)$, generate $r_{t+1}, s_{t+1}$, and then take action $a_{t+1}$ following $\pi_t(s_{t+1})$.
    *   **Value update (parameter update):**
        $$w_{t+1} = w_t + \alpha_t \Big[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \Big] \nabla_w \hat{q}(s_t, a_t, w_t)$$
    *   **Policy update:**
        $$\pi_{t+1}(a|s_t) = \begin{cases}
        1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1) & \text{if } a = \arg \max_{a \in \mathcal{A}(s_t)} \hat{q}(s_t, a, w_{t+1}) \\
        \frac{\epsilon}{|\mathcal{A}(s)|} & \text{otherwise}
        \end{cases}$$
这里首先使用了VFA with sarsa更新参数，然后使用ε-greedy做policy improvement。
#### 然后是Q-Learning with VFA：
方程：$$w_{t+1} = w_t + \alpha_t \Big[ r_{t+1} + \gamma \max_{a \in \mathcal{A}(s_{t+1})} \hat{q}(s_{t+1}, a, w_t) - \hat{q}(s_t, a_t, w_t) \Big] \nabla_w \hat{q}(s_t, a_t, w_t),$$which is the same as Sarsa except that $\hat{q}(s_{t+1}, a_{t+1}, w_t)$ is replaced by $\max_{a \in \mathcal{A}(s_{t+1})} \hat{q}(s_{t+1}, a, w_t)$.
也没什么可以讲的，直接把Q-Learning的公式套进来就完了。
下面看一下Q-Learning with VFA的伪代码，但这是一个On-policy的版本：
**Initialization:** Initial parameter vector $w_0$. Initial policy $\pi_0$. Small $\epsilon > 0$.
**Aim:** Search a good policy that can lead the agent to the target from an initial state-action pair $(s_0, a_0)$.

For each episode, do:
*   If the current $s_t$ is not the target state, do:
    *   Take action $a_t$ following $\pi_t(s_t)$, and generate $r_{t+1}, s_{t+1}$.
    *   **Value update (parameter update):**
        $$w_{t+1} = w_t + \alpha_t \Big[ r_{t+1} + \gamma \max_{a \in \mathcal{A}(s_{t+1})} \hat{q}(s_{t+1}, a, w_t) - \hat{q}(s_t, a_t, w_t) \Big] \nabla_w \hat{q}(s_t, a_t, w_t)$$
    *   **Policy update:**
        $$\pi_{t+1}(a|s_t) = \begin{cases}
        1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1) & \text{if } a = \arg \max_{a \in \mathcal{A}(s_t)} \hat{q}(s_t, a, w_{t+1}) \\
        \frac{\epsilon}{|\mathcal{A}(s)|} & \text{otherwise}
        \end{cases}$$
跟sarsa 差不多，先用QL优化参数，然后ε-greedy policy improvement。

