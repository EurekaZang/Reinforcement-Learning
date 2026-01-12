刚刚介绍的TD算法是一种policy evaluation的算法，不能够policy improvement。
下面讲的**SARSA**就是一种集policy evaluation和policy improvement为一身的算法。*不估计state value，而是直接估计action value。*
直接给公式：$$
\begin{aligned}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[q_t(s_t, a_t) - \left[r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})\right]\right], \\
q_{t+1}(s, a) &= q_t(s, a), \quad \forall (s, a) \neq (s_t, a_t), \\
\text{where } t &= 0, 1, 2, \dots
\end{aligned}
$$可以看出来，这个sarsa就是把之前讲的TD算法的state value $v(s_t)$ 换成了 $q_t(s_t,a_t)$ 而已，其他都没区别。
$q_t(s_t, a_t)$ is an estimate of $q_\pi(s_t, a_t)$;
$\alpha_t(s_t, a_t)$ is the learning rate depending on $s_t, a_t$.

为什么叫sarsa呢？因为sarsa使用的数据是 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ 。

下面来看sarsa的伪代码，理解sarsa算法的policy evaluation和policy improvement如何运作：
Algorithm: **SARSA**
For each episode, do
    If the current $s_t$ is not the target state, do
        Collect the experience $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$: In particular, take action $a_t$ following $\pi_t(s_t)$, generate $r_{t+1}, s_{t+1}$, and then take action $a_{t+1}$ following $\pi_t(s_{t+1})$.
        Update q-value:$$
        q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[q_t(s_t, a_t) - \left[r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})\right]\right]
        $$
        Update policy:$$
        \pi_{t+1}(a|s_t) = \begin{cases} 1 - \frac{\epsilon}{|\mathcal{A}|}(|\mathcal{A}| - 1) & \text{if } a = \arg \max_a q_{t+1}(s_t, a) \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}
        $$
也就是直接更新action value，然后采用epsilon-greedy更新策略。
好像有点似懂非懂？为什么能用 $q_t(s_t,a_t)$ 直接替换 $v_t(s_t,a_t)$ ？我们回想一下TD算法求解的是什么数学问题：$$V^\pi(s) = \mathbb{E}_\pi \left[ r(s, a) + \gamma V^\pi(s') \right]$$是State-value Function的Bellman Expectation Equation，使用 $r(s,a)+\gamma v_\pi(s')$ 去近似 $\mathbb{E}_\pi \left[ r(s, a) + \gamma V^\pi(s') \right]$。那么相似的，sarsa求解的就是Action-value Function的Bellman Expectation Equation，即: $$Q^\pi(s, a) = \mathbb{E}_\pi \left[ r(s, a) + \gamma Q^\pi(s', a') \right]$$你可以发现这俩value function只有 $Q$ 和 $V$ 的区别，说明他俩使用Robbins-Monro推导成TD算法的方法也是一模一样，自然最后推导出来的公式也就一模一样（只有 $q$ 和 $v$ 的区别）。

这里是用python实现的Sarsa算法：
```Python
class SarsaAgent:
    """
    Sarsa算法智能体
    """
    def __init__(self, env, epsilon=0.15, gamma=0.99, alpha=0.1):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha  # 学习率
        self.q_values = {}
        # 初始化Q表
        for row in range(env.rows):
            for col in range(env.cols):
                state = (row, col)
                for action in env.action_space:
                    self.q_values[(state, action)] = 0.0

    def choose_action(self, state):
        """使用epsilon-greedy策略选择动作"""
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            # 找出当前状态下Q值最大的动作
            possible_actions = self.env.action_space
            q_values = [self.q_values[(state, a)] for a in possible_actions]
            max_q = max(q_values)
            # 随机选择具有最大Q值的动作之一
            best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def train(self, num_episodes):
        """使用Sarsa算法训练智能体"""
        for _ in tqdm(range(num_episodes), desc="SARSA Training", unit="episode"):
            state = self.env.reset()
            action = self.choose_action(state)
            steps = 0
            while steps < self.env.episode_length:
                # 执行动作，得到下一个状态和奖励
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(next_state, action)
                # 选择下一个动作（如果下一个状态是终止则无动作）
                if self.env.is_terminal_state(next_state):
                    next_action = None
                else:
                    next_action = self.choose_action(next_state)
                # 计算当前Q值的更新
                current_q = self.q_values[(state, action)]
                if next_action is not None:
                    next_q = self.q_values[(next_state, next_action)]
                    target = reward + self.gamma * next_q
                else:
                    target = reward  # 终止状态后的Q值为0
                # 更新Q值
                self.q_values[(state, action)] += self.alpha * (target - current_q)
                # 移动到下一个状态和动作
                state = next_state
                action = next_action
                steps += 1
                # 检查是否到达终止状态
                if self.env.is_terminal_state(state):
                    break

```


# 详细解析：代码如何实现Sarsa算法
### Sarsa更新公式
$$
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha \left[ q_t(s_t, a_t) - \left( r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}) \right) \right]
$$代码实现中实际用到的方程是这个：$$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) + \alpha \left[ \left( r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}) \right) - q_t(s_t, a_t) \right]$$简单的变号。
### 代码实现解析
#### **1. 初始化Q表**
```python
for row in range(env.rows):
    for col in range(env.cols):
        state = (row, col)
        for action in env.action_space:
            self.q_values[(state, action)] = 0.0
```
- 为所有状态-动作对初始化Q值，对应公式中的 $q(s,a)$。

#### **2. 动作选择（On-Policy）**
```python
def choose_action(self, state):
    if random.random() < self.epsilon:
        return random.choice(self.env.action_space)  # 探索
    else:
        # 利用：选择当前Q值最大的动作
        best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
        return random.choice(best_actions)
```
- 使用 **epsilon-greedy 策略**，符合Sarsa的 **on-policy** 特性。

#### 3. 核心训练逻辑
##### **a. 执行动作并观察结果**
```python
next_state = self.env.get_next_state(state, action)
reward = self.env.get_reward(next_state, action)
```
获取 $s_{t+1}$ 和 $r_{t+1}$。

##### **b. 选择下一动作 $a_{t+1}$**
```python
if self.env.is_terminal_state(next_state):
    next_action = None  # 终止状态无动作
else:
    next_action = self.choose_action(next_state)  # 按策略选择下一动作
```
通过 `choose_action` 选择 $a_{t+1}$，体现 **on-policy** 特性。

##### **c. 计算TD目标**
```python
if next_action is not None:
    next_q = self.q_values[(next_state, next_action)]
    target = reward + self.gamma * next_q  # 非终止状态
else:
    target = reward  # 终止状态后Q值为0
```
- **非终止状态**：目标为 $r_{t+1} + \gamma q(s_{t+1}, a_{t+1})$。
- **终止状态**：目标为 $r_{t+1}$。

##### **d. 更新Q值**
```python
current_q = self.q_values[(state, action)]
self.q_values[(state, action)] += self.alpha * (target - current_q)
```
对应公式：
$$
q(s_t,a_t) \leftarrow q(s_t,a_t) + \alpha \left[ \text{目标} - q(s_t,a_t) \right]
$$
##### **e. 转移到下一状态**
```python
state = next_state
action = next_action
```
推进到 $s_{t+1}$ 和 $a_{t+1}$，准备下一步迭代。

### 关键点总结
1. **On-Policy动作选择**：每一步的 $a_{t+1}$ 由当前策略生成。
2. **TD目标计算**：使用实际选择的下一动作的Q值（与Q-learning不同）。
3. **终止状态处理**：目标仅含即时奖励。
4. **更新步骤**：直接应用公式中的 $\alpha$ 和 $\gamma$。

### 与Q-learning的区别
- **Q-learning**：使用最大Q值（off-policy）：
  ```python
  target = reward + self.gamma * max_q_value(next_state)
  ```
- **Sarsa**：使用实际选择的下一动作的Q值（on-policy）：
  ```python
  target = reward + self.gamma * self.q_values[(next_state, next_action)]
  ```
