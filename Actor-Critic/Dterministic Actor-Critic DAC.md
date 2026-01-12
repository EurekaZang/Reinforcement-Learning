我们为什么要关心Deterministic的策略？因为stochastic的策略它的输出是有限的，神经网络它不能有无限个输出，它不能显式的将每一个 $\pi(a_t|s_t,\theta_t)$ 都输出出来。如果一个任务中有无限个action，就需要deterministic的策略。
如果是deterministic的，那么 $\pi(a|s,\theta)$ 就等于一，也就是在一个state，一个策略选择的action是唯一的。也就能够写成：$$a=\mu(s,\theta)=\mu(s)$$其中 $\mu$ 是一个从 $\mathcal{S}$ 到 $\mathcal{A}$ 的映射，从状态空间到动作空间。
***也就是说这个 $\mu$ 是一个神经网络，输入一个 $s$ ，输出一个 $a$。*** 
==（ 如果将 $\mu$ 选取为神经网络，那么这就是 $\text{DDPG}$，如果是指选择线性的函数，那么就是 $\text{DPG}$。 ）==
不过既然能用神经网络，为什么不用呢是吧。
在写它时可以把参数 $\theta$ 隐藏掉。
AC方法其实已经讲的很透彻了，所以这个方法直接开始介绍它的梯度怎么计算：
首先我们选用 average state value作为 metrix，得到目标函数 $$J(\theta)=\mathbb{E}[v_\mu(s)]=\sum_{s\in\mathcal{S}}d_0(s)v_\mu(s)$$
其中 $d$ 满足求和等于 1.
这里 $d_0$ 是与 $\mu$ 无关的，它的选取有两种情况：
1. 从某个state 出发，想要最大化该state的长期回报，那么就可以选择 $d_0(s_0)=1$ and $d_0(s\neq s_0)$ 。 
2. 还有一种情况就是 $d_0$ 服从一个behavior policy的概率分布。这也就是为什么DAC是天然的离策略，不需要用重要性采样把它转换为离策略。

直接给出来梯度，推导太复杂了。
$$\begin{aligned} \nabla_\theta J(\theta) &= \sum_{s \in S} \rho_\mu(s) \nabla_\theta \mu(s) \left( \nabla_a q_\mu(s, a) \right) |_{a=\mu(s)} \\ &= \mathbb{E}_{S \sim \rho_\mu} \left[ \nabla_\theta \mu(S) \left( \nabla_a q_\mu(S, a) \right) |_{a=\mu(S)} \right]\end{aligned}$$这个公式活光光了，用了一堆没见过的符号
暂且不理解无所谓，可以看后面的伪代码逻辑。

带入梯度上升公式中得到：$$\theta_{t+1} = \theta_t + \alpha_\theta \mathbb{E}_{S \sim \rho_\mu} \left[ \nabla_\theta \mu(S) \left( \nabla_a q_\mu(S, a) \right) |_{a=\mu(S)} \right]$$有期望，直接引入随即近似$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t) \left( \nabla_a q_\mu(s_t, a) \right) |_{a=\mu(s_t)}$$
这是老师课件中给出的伪代码：
Initialization: A given behavior policy $\beta(a|s)$. A deterministic target policy $\mu(s, \theta_0)$ where $\theta_0$ is the initial parameter vector. A value function $v(s, w_0)$ where $w_0$ is the initial parameter vector.  
Aim: Search for an optimal policy by maximizing $J(\theta)$.  
At time step $t$ in each episode, do  
*   Generate $a_t$ following $\beta$ and then observe $r_{t+1}, s_{t+1}$.  
*   TD error:  
    $$  
    \delta_t = r_{t+1} + \gamma q(s_{t+1}, \mu(s_{t+1}, \theta_t), w_t) - q(s_t, a_t, w_t)  
    $$  
*   Critic (value update):  
    $$  
    w_{t+1} = w_t + \alpha_w \delta_t \nabla_w q(s_t, a_t, w_t)  
    $$  
*   Actor (policy update):  
    $$  
    \theta_{t+1} = \theta_t + \alpha_\theta \cdot \nabla_\theta \mu(s_t, \theta_t) \cdot \nabla_a\left( \nabla_a q(s_t, a, w_{t+1}) \right) \Big|_{a = \mu(s_t)}  
    $$

可以看出来还是使用了两个神经网络，一个Actor（策略网络），一个Critic（Q网络）。
1. Actor（策略）网络：**输入**：状态 $s$，**输出**：确定性动作 $a=\mu(s,\theta)$ ，**作用**：生成动作，直接控制智能体的行为。
2. Critic（Q值）网络：**输入**：状态 $s$ + 动作 $a$，**输出**：Q值 $q(s,a,w)$ ，**作用**：评估当前状态-动作对的长期收益，指导Actor如何改进策略。

下面是具体的代码实现步骤，注意其中 $\theta$ 是一个字典，储存着神经网络 ($\mu$) 各层的*权重矩阵*和*偏置向量*。
对 $\mu$ 求导就是反向传播的一步，但是要注意，这里的反向传播更新神经网络参数根本不是监督学习中的做法，两者完全不一样。
这里是通过***梯度上升方法更新参数***，而不是传统监督学习中使用的***最小化预测误差***（依靠标注数据）。
但是在实际的代码实现中，*还是要调用库来实现反向传播*，例如PyTorch中：
```Python
# 定义策略网络（Actor）和Q网络（Critic）
policy_net = DeterministicPolicy(state_dim, action_dim)
q_net = QNetwork(state_dim, action_dim)

# 优化器（注意：Actor使用梯度上升，Critic使用梯度下降）
optimizer_actor = torch.optim.Adam(policy_net.parameters(), lr=α_θ)

# 假设当前状态为s
state = torch.tensor(s, dtype=torch.float32)

# 前向传播生成动作
a = policy_net(state)

# 计算Q值（Critic评估）
q_value = q_net(state, a)

# 清空Actor的梯度
optimizer_actor.zero_grad()

# 计算梯度（最大化Q等价于最小化负Q）
# 注意：这里通过负号将梯度上升转换为梯度下降框架
policy_loss = -q_value.mean()  # 目标是最小化负Q值（即最大化Q值）

# 反向传播计算梯度
policy_loss.backward()

# 更新Actor参数（实际执行梯度上升）
optimizer_actor.step()
```
在代码实践中可以直接调用现成的轮子来计算 $θ ← θ + α_θ * (∇θ_μ ⊙ ∇a_q)$
```python
optimizer_actor = torch.optim.Adam(policy_net.parameters(), lr=α_θ)
```
所以本质上在这里反向传播更新网络参数和显式的使用公式计算是等价的，只不过要手动实现两个梯度的求导，没有直接用轮子方便。


下面是真正意义上的伪代码：
```Python
Initialize:
    θ ← θ₀        # 初始化策略参数
    w ← w₀        # 初始化值函数参数
    α_θ, α_w, γ   # 设置学习率和折扣因子

for each episode do:
    Initialize state s
    while s is not terminal do:
        # 1. 根据行为策略选择动作
        a_t ∼ β(·|s)
        
        # 2. 执行动作，获得奖励和下一状态
        Take action a_t, observe r_{t+1}, s_{t+1}
        
        # 3. 计算TD误差 (使用当前参数θ和w)
        q_current = q(s, a_t, w)  # 当前状态-动作值
        a_next = μ(s_{t+1}, θ)    # 目标策略的下一动作
        q_next = q(s_{t+1}, a_next, w)  # 下一状态-动作值
        δ = r_{t+1} + γ * q_next - q_current
        
        # 4. Critic更新：更新值函数参数w
        ∇w_q = ∇_w q(s, a_t, w)   # q对w的梯度
        w ← w + α_w * δ * ∇w_q    # 梯度上升
        
        # 5. Actor更新：更新策略参数θ
        a_μ = μ(s, θ)             # 目标策略当前动作
        ∇θ_μ = ∇_θ μ(s, θ)        # 策略对θ的梯度
        ∇a_q = ∇_a q(s, a, w) evaluated at a = a_μ  # q对动作的梯度
        θ ← θ + α_θ * (∇θ_μ ⊙ ∇a_q)  # 梯度上升 (⊙表示逐元素相乘)
        
        # 6. 转移到下一状态
        s ← s_{t+1}
    end while
end for

```