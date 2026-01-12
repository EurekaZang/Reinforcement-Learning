QL跟sarsa和MC的主要区别在哪呢？他是直接估计optimal action value，所以它不需要policy evaluation。
直接给公式：$$
\begin{aligned}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[ q_t(s_t, a_t) - \left[ r_{t+1} + \gamma \max_{a \in \mathcal{A}} q_t(s_{t+1}, a) \right] \right], \\
q_{t+1}(s, a)     &= q_t(s, a), \quad \forall (s, a) \neq (s_t, a_t),
\end{aligned}
$$Q-learning is very similar to Sarsa. They are different only in terms of the TD target:
*   The TD target in Q-learning is $r_{t+1} + \gamma \max_{a \in \mathcal{A}} q_t(s_{t+1}, a)$
*   The TD target in Sarsa is $r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$.
数学上，sarsa求解的是一个Bellman Expectation Equation，但是Q-Learning求解的是一个**Bellman Optimality Equation**。
它得到的不是某一个q值，而是最优的q值，是最优策略的q值。
它求解的bellman optimality equation长这样：$$
q(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a} q(S_{t+1}, a) \mid S_t = s, A_t = a \right], \quad \forall s, a.
$$跟action-value Function 的 bellman expectation equation的区别就是在q前面加了个 $\max$ 最优化。
Algorithm: **Q-Learning**
Initialize $q(s, a), \forall s \in \mathcal{S}, a \in \mathcal{A}(s)$, arbitrarily (e.g., to 0), and ensure $q(\text{terminal-state}, \cdot) = 0$.

For each episode, do:
1.  Initialize $s_t$ as the starting state of the episode.
2.  While $s_t$ is not a terminal state, do:
    1.  **Choose Action:**
        Choose action $a_t$ from state $s_t$ using a policy derived from the current q-values. This is the **behavior policy**, which is typically $\epsilon$-greedy to ensure exploration.
        $$
        a_t \leftarrow \begin{cases} \arg \max_{a' \in \mathcal{A}(s_t)} q(s_t, a') & \text{with probability } 1 - \epsilon \\ \text{a random action } a' \in \mathcal{A}(s_t) & \text{with probability } \epsilon \end{cases}
        $$
    2.  **Take Action & Observe:**
        Take action $a_t$, and observe the resulting reward $r_{t+1}$ and the next state $s_{t+1}$. This generates the experience tuple $(s_t, a_t, r_{t+1}, s_{t+1})$.
        *Note: Unlike Sarsa, we do not need to choose the next action $a_{t+1}$ at this stage.*
    3.  **Update Q-value:**
        Update the q-value for the state-action pair $(s_t, a_t)$ using the Q-Learning TD target. The target uses the maximum possible q-value for the next state, representing the optimal **target policy**.
        $$
        q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a' \in \mathcal{A}} q(s_{t+1}, a') - q(s_t, a_t) \right]
        $$
    4.  **Update State:**
        Move to the next state for the next iteration.
        $$
        s_t \leftarrow s_{t+1}
        $$
下面简单讲一下什么是Off-policy和On-policy。
在强化学习中存在两个策略：
1. **行为策略 (Behavior Policy)** 是*实际控制智能体与环境互动的策略*，负责在训练过程中：
	- *做出具体的动作选择*（如用ε-greedy策略平衡探索与利用）
	- 生成训练所需的状态-动作样本
	- 决定与环境交互时的实时行为模式 示例：在Q-learning中，通常用带随机性的ε-greedy策略作为行为策略

2. **目标策略 (Target Policy)** 是我们*通过训练想要得到的最优策略*，它的作用是：
	- *定义*我们要逼近的理想决策方式（如完全贪婪策略）
	- 作为价值函数更新的参考目标
	- 最终用于实际应用的决策 ***(更新Q值)*** 示例：在Q-learning中，目标策略是完全贪婪策略，每次选择Q值最大的动作

==比喻理解： 想象学骑自行车时：==
- Behavior policy  = 你实际尝试的骑行方式（可能会故意摇晃车把来试探平衡）
- Target policy = 教练希望达到的标准骑行姿势（身体直立，平稳踩踏） 你通过不断尝试（行为策略）收集经验，逐步修正动作向理想姿势（目标策略）靠拢

怎么判断一个算法是On-policy还是Off-policy呢？
当behavior policy和target policy相同时，就是On-policy
当它俩不同时，就是Off-policy。
前面讲的MC，sarsa是On-policy，Q-Learning是Off-policy。

Off-policy有什么好处呢？***能够充分的利用experience。***
比如我有一个探索性非常强的behavior policy，那么我的target policy就能利用这些experience来得到尽可能多的state-action pair的估计。
但是如果是On-policy，我的behavior policy和target policy是同一个，那么探索性就不强（可能是greedy的或者ε-greedy的），很难探索到所有的s和a。

下面使用sarsa和QL来阐明怎么判断On-policy&Off-policy：

### Sarsa (On-policy) vs Q-Learning (Off-policy) 三时间步操作对比 
#### 环境设定： 
- 状态：`s0 → s1 → s2 → s3`（目标状态，奖励+1） 
- 动作：{左, 右, 上, 下}（部分状态边界限制动作） 
- 参数：`α=0.1`, `γ=0.9`, `ε=0.1` 
- 初始Q值：所有`Q(s,a)=0`

| 时间步    | 状态  | Behavior policy选动作 | 执行结果     | Target policy选下一动作 | Q值更新公式                                                |
| ------ | --- | ------------------ | -------- | ------------------ | ----------------------------------------------------- |
| **t0** | s0  | ε-greedy选右（90%概率）  | 到达s1，r=0 | ε-greedy选上（探索）     | `Q(s0,右) += 0.1*(0 + 0.9*Q(s1,上) - 0)` → 0→0.09       |
| **t1** | s1  | ε-greedy选上（实际执行）   | 到达s2，r=0 | ε-greedy选右（利用）     | `Q(s1,上) += 0.1*(0 + 0.9*Q(s2,右) - 0.09)` → 0.09→0.09 |
| **t2** | s2  | ε-greedy选右（利用）     | 到达s3，r=1 | 终止状态无动作            | `Q(s2,右) += 0.1*(1 + 0 - 0)` → 0→0.1                  |

**关键特征**：
- 所有动作选择均使用ε-greedy（策略一致）
- 更新时使用实际执行的下一动作（如t0的"上"）
- 探索动作直接影响后续更新（如t0的探索影响t1的Q值）

| 时间步    | 状态  | Behavior Policy选动作 | 执行结果     | Target policy选下一动作 | Q值更新公式                                              |
| ------ | --- | ------------------ | -------- | ------------------ | --------------------------------------------------- |
| **t0** | s0  | ε-greedy选右（探索）     | 到达s1，r=0 | 贪婪选右（max Q）        | `Q(s0,右) += 0.1*(0 + 0.9*maxQ(s1,右) - 0)` → 0→0.09  |
| **t1** | s1  | ε-greedy选下（探索）     | 返回s0，r=0 | 贪婪选右（max Q）        | `Q(s1,下) += 0.1*(0 + 0.9*maxQ(s0,右) - 0)` → 0→0.09  |
| **t2** | s0  | ε-greedy选右（利用）     | 到达s1，r=0 | 贪婪选右（max Q）        | `Q(s0,右) += 0.1*(0 + 0.9*0.09 - 0.09)` → 0.09→0.088 |

**关键特征**：
- 行为策略（ε-greedy）与目标策略（贪婪）分离
- 更新时使用理论最优动作（如t0的"右"未实际执行）
- 实际探索动作（如t1的"下"）不影响目标策略的更新

| 区别维度           | Sarsa (On-policy) | Q-Learning (Off-policy) |
| -------------- | ----------------- | ----------------------- |
| **策略一致性**      | 始终使用ε-greedy      | ε-greedy生成数据，贪婪策略更新     |
| **探索影响**       | 直接影响后续更新（保守）      | 仅影响数据收集，不污染目标更新         |
| **t1操作结果**     | 实际执行探索动作"上"→s2    | 实际执行探索动作"下"→s0          |
| **Q(s0,右)最终值** | 0.1（受后续实际奖励影响）    | 0.088（仅依赖理论最大回报）        |
| **数据效率**       | 需要持续新数据（无法复用历史数据） | 可复用旧数据（如t1的探索数据）        |
 