By Eureka 25-1-19 5:25 

1. 智能体（$\text{Agent}$）与环境（$\text{Environment}$）
2. 状态（State, $S$）
3. 状态空间（State Space, $\mathcal{S}$）ex. $\mathcal{S}=\{s_i\}^9_{i=1}$
4. 动作（Action, $A$）ex. $a_1$: move upwards; $a_2$: move rightwards
5. **状态转移（State Transition）** ex.$$s_1 \xrightarrow{a_2} s_2 \text{ or }s_1 \xrightarrow{a_1} s_1$$
	- 从一个State采取一个Action，跳到另外一个State
	- 有可能撞到边界，State不会发生改变
	1. State Transition 可以用 tabular 的形式表达，**但是局限于deterministic的情况**。
	2. 更多情况下使用 **State Transition Probability**:
		- Intuition：At state $s_1$, if we choose action $a_2$, the next state is $s_2$.
		- Math：$$\begin{aligned}p(s_2 | s_1, a_2) &= 1 \\p(s_i | s_1, a_2) &= 0 \quad \forall i \neq 2\end{aligned}$$
6. **策略（Policy, $π$）**: Tell the agent what action to take at a state
	- ![[Pasted image 20250119035833.png]]
	- 箭头代表着 Policy
	- Mathematical representation: 仍然使用条件概率 ex.$$
	\begin{aligned}
	\pi(a_1|s_1)&=0 \\
	\pi(a_2|s_1)&=1 \\
	\pi(a_3|s_1)&=0 \\
	\pi(a_4|s_1)&=0 \\
	\pi(a_5|s_1)&=0
	\end{aligned}
	$$
	- 这里展示的是一个 deterministic policy 即一个确定性的情况，下面是一个 stochastic policy，即不确定性的情况：
	- ![[Pasted image 20250119040456.png]]
	- In this case, for $s_1$: $$
	\begin{aligned}
	\pi(a_1|s_1)&=0 \\
	\pi(a_2|s_1)&=0.5 \\
	\pi(a_3|s_1)&=0.5 \\
	\pi(a_4|s_1)&=0 \\
	\pi(a_5|s_1)&=0
	\end{aligned}
	$$
	- policy 也有tabular的表示形式，即一个$\{a_1, a_2, a_3, ..., a_n\}$与$\{s_1, s_2, ..., s_n\}$的table，实际programming中就是这种办法
7. **奖励（Reward, $R$）**
	- 既有正数也有负数，*正数代表对特定 actions 的 encouragement，负数代表对特定 actions 的 punishment*
	- ex. $r_{\text{bound}}=-1 \text{ and } r_{\text{fobid}}=-1$
	- Mathematical description：仍然是条件概率，下面展示一种 deterministic 的情况$$p(r=-1|s_1,a_1)=1\text{ and }p(r\neq-1|s_1,a_1)=0$$
	- 参数方面，reward依赖于**当前的state**和**采取的action**，并不是取决于下一个state
8. **轨迹（Trajectory）**
	- 轨迹Trajectory是一个 ***state-action-reward*** chain，也就是一个***状态-动作-奖励***链
	- ex.$$s_1 \xrightarrow[r_1=0]{a_2} s_2 \xrightarrow[r_2=0]{a_3} s_5 \xrightarrow[r_3=0]{a_3} s_8 \xrightarrow[r_4=1]{a_2} s_9$$
9. **回报（return, $G$）**：回报return是一个trajectory中收集到的所有reward：
	- 上面的轨迹的return可表示为：$$\text{return} = 0+0+0+1=1$$
	- 一些return的作用：
		1. 用于衡量一个policy的好坏：一个policy会产生一个trajectory，该trajectory的return用于衡量该轨迹的好坏，也就是policy的好坏。比如 $\text{return}=1$ 就明显比 $\text{return}=-1$ 要好
10. **折扣回报（discounted return）**：如果trajectory是无限长的怎么办？
	- 那么return就是 $+\infty$，多条无限长的trajectory就无法比较好坏，所以引入discounted return
	- **Discount rate / Discount factor $γ\in[0,1)$ 折扣因子、折扣比率**
	- Mathematical description：$$\begin{aligned} \text{discounted return}&=0+γ0+γ^20+γ^31+γ^41+...\\&=γ^3(1+γ+γ^2+...)=γ^3\frac{1}{1-γ}\end{aligned}
	$$
	- 处理无穷项数，最后的多项1使用*几何级数的无穷求和公式*处理
		1. 如果$γ$趋向于0，则折扣回报更关注更近的未来（near future），更加近视
		2. 如果$γ$趋向于1，则折扣汇报更关注更远的未来（far future），更加远视
11. 回合（Episode）：即一个有限步的，拥有**Terminal state**（agent不再执行action的state）的trajectory。tasks with episodes are called *episodic tasks*.
	- 当然，没有terminal state的tasks被称为*continuing tasks*, which will never end。
	- 为了抽象出统一的数学表示，***我们可以将episodic tasks转换为continuing tasks***：
		1. Option 1：让Target state变成absorbing state，让该state只拥有一个“停留在原地”的action，且$\text{reward}=0$
		2. Option 2：将Target state变成一种normal state，拥有健全的5个action（grid-world）和正常的reward，agent可以自由进出。
12. **马尔可夫决策过程（Markov Decision Process, $\text{MDP}$）它有以下特征组合而成：**
	1. 集合 Sets：
		1. State: the set of state $\mathcal{S}$
		2. Action: the set of actions $\mathcal{A}(s)$ is associated for state $s\in\mathcal{S}$
		3. Reward: the set of rewards $\mathcal{R}(s,a)$
	2. 概率分布 Probability distribution: 
		1. *状态转移概率：在状态 $s$，采取action $a$，转移到相应状态 $s'$ 的概率是 $p(s'|s,a)$*
		2. *奖励概率：在状态 $s$，采取action $a$，得到reward的概率是 $p(r|s,a)$*
	3. 策略 policy：在状态 $s$，选择action $a$ 的概率是 $\pi(a|s)$
	4. 马尔可夫性质：Memoryless property:$$
	\begin{aligned}
	p(s_{t+1}|a_{t+1},s_t,...,a_1,s_0)&=p(s_{t+1},s_t)\\
	p(r_{t+1}|a_{t+1},s_t,...,a_1,s_0)&=p(r_{t+1},s_t)
	\end{aligned}
	$$
	5. 一个 grid-world 例子，描述的是一个Markov process，因为policy是stochastic的（存在$\text{prob}=0.5$）：![[Pasted image 20250119051410.png]]每一个圆代表一个state，每个arrow代表当前state下的policy以及它的probability
		- ***Markov process 和 Markov decision process是什么关系呢？当policy固定之后，也就是不再stochastic之后，Markov process 成为了Markov decision process。***


### **Policy-based vs. Value-based** 和 **Model-based vs. Model-free** 是两个完全**不同维度**的分类标准。

把强化学习算法想象成一个大家族，这两组概念就是给家族成员贴的两种不同类型的标签：

*   **Model-Free vs. Model-Based**：这个标签描述的是“**如何看待世界**”。
    *   **Model-Free (无模型)**：不理解世界的规则，是个“经验主义者”。
    *   **Model-Based (有模型)**：试图理解世界的规则，是个“理论物理学家”。
*   **Value-Based vs. Policy-Based**：这个标签描述的是“**如何做出决策**”。
    *   **Value-Based (基于价值)**：通过给每个选择“打分”，然后选分最高的。
    *   **Policy-Based (基于策略)**：直接学习在某种情况下“应该做什么”。

一个算法可以同时拥有这两个维度的标签。比如，最著名的 Q-Learning 算法，它既是 **Model-Free** 的，又是 **Value-Based** 的。

---

### 复习第一维度：Value-Based vs. Policy-Based (如何决策)

这个维度关注的是智能体最终**直接学习的目标是什么**。

#### 1. Value-Based (基于价值)

*   **核心思想**：学习一个“价值函数”（Value Function），这个函数用来给“状态”或“状态-动作对”打分，评估其好坏程度。智能体不直接学习策略。
*   **决策方式**：策略是**隐式 (implicit)** 的。在做决策时，智能体查看当前状态下所有可能动作的“得分”（Q值），然后选择得分最高的那个动作去执行（即贪心策略）。
*   **好比**：一个美食评论家想在一家新餐厅点菜。他不会直接背下来“在这家店就该点宫保鸡丁”。而是先品尝（或预测）菜单上每道菜的“美味度评分”（价值），然后在点菜时，选择评分最高的那道菜。
*   **学习目标**：学习价值函数，通常是 Q 函数 `Q(s, a)`，它表示在状态 `s` 下执行动作 `a` 的长期回报有多大。
*   **优点**：
    *   通常样本效率和稳定性在早期更好。
    *   学习过程相对简单直观。
*   **缺点**：
    *   很难处理**连续动作空间**（你无法在一个无限的动作集合中取 argmax）。
    *   只能学出**确定性策略**（总是选最好的），无法处理需要随机性的最优策略（比如“石头剪刀布”）。
*   **代表算法**：Q-Learning, SARSA, **DQN** (Deep Q-Network)。

#### 2. Policy-Based (基于策略)

*   **核心思想**：不通过中间的价值函数，直接学习一个“策略”（Policy）。这个策略本身就是一个函数，直接告诉你在某个状态下应该执行什么动作。
*   **决策方式**：策略是**显式 (explicit)** 的。决策时，直接将当前状态输入策略函数，得到要执行的动作（或动作的概率分布）。
*   **好比**：一个老司机开车。他不是在脑中计算“方向盘左打5度的价值”和“左打6度的价值”然后比较，而是凭借经验和直觉，形成一个从“看到弯道”到“打方向盘”的直接映射。
*   **学习目标**：学习策略函数 `π(a|s)`，它表示在状态 `s` 下执行动作 `a` 的概率。目标是优化这个策略，使得累积回报最大化。
*   **优点**：
    *   能很自然地处理**连续动作空间**。
    *   可以学习**随机性策略**。
    *   在某些情况下收敛性更好。
*   **缺点**：
    *   容易收敛到局部最优。
    *   策略梯度的方差很大，导致训练不稳定、收敛慢。
*   **代表算法**：REINFORCE, TRPO, **PPO** (Proximal Policy Optimization)。

#### 3. Actor-Critic (演员-评论家) —— 两者的结合

现代强化学习的王道。它结合了以上两者的优点。

*   **核心思想**：同时学习一个策略（Actor）和一个价值函数（Critic）。
    *   **Actor (演员, Policy-based)**：负责做出动作决策。
    *   **Critic (评论家, Value-based)**：负责评估 Actor 做的动作好不好，并指导 Actor 进行更新。
*   **工作流程**：Actor 做出动作，Critic 根据这个动作给出评分（比如 TD 误差）。Actor 根据 Critic 的评分来调整自己的策略，尽量多做能获得高分的动作。
*   **优点**：结合了两者的长处。Critic 提供的低方差的梯度指导，使得 Actor 的学习比纯 Policy-based 方法更稳定、更高效。
*   **代表算法**：A2C/A3C, **DDPG**, **TD3**, **SAC**。几乎所有当前SOTA的算法都是 Actor-Critic 架构。

---

### “大局观”：两个维度的四象限图

现在，我们可以把这两个维度结合起来，形成一个更完整的强化学习算法地图。

| | **Value-Based (学打分)** | **Policy-Based (学动作)** | **Actor-Critic (边学边评)** |
| :--- | :--- | :--- | :--- |
| **Model-Free**<br>(经验主义) | **Q-Learning, DQN**<br>通过试错，学习在不同情况下做不同选择的分数。 | **REINFORCE**<br>通过试错，直接学习在不同情况下的行动指南。 | **A3C, PPO, DDPG, TD3**<br>（**现代RL的主力军**）通过试错，同时学习行动指南和评分标准。 |
| **Model-Based**<br>(理论主义) | **动态规划 (DP), 价值迭代**<br>在**已知的**完美地图上，计算每个位置的价值。 | **策略迭代, Policy Search**<br>在**学习到的**地图模型里，规划出一条最好的行动路线。 | **AlphaGo, MuZero, World Models**<br>学习一个世界模型（地图），然后在这个虚拟世界里用 Actor-Critic 方法进行推演和训练。 |

**总结一下：**

*   **Model-Free/Based** 回答的是：“我们是否需要一张地图？”
*   **Value/Policy-Based** 回答的是：“有了（或没有）地图后，我们是记住每个路口的价值，还是直接记住该怎么走？”

所以，当你分析一个算法时（比如 TD3），可以给它贴上两个标签：
*   从“世界观”维度看，它是 **Model-Free** 的，因为它不学习环境模型。
*   从“决策方法”维度看，它是 **Actor-Critic** 的，因为它既有策略网络（Actor）又有价值网络（Critic）。

## The END
