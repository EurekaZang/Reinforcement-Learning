为什么说迄今为止的Actor-Critic算法都是On-policy的呢？原因是目标函数的梯度中使用的采样的分布是基于策略的。
$$\nabla_\theta J(\theta)=\mathbb{E}_{S\sim n,A \sim\pi}[*]$$
同时我们要更新的targte 策略也是这个策略。所以它是On-policy的。
为什么要转换成Off-policy？因为我们想要复用之前的数据，让这些数据不要被浪费了。
那么怎么做呢？答案是通关重要性采样。 (**Important sampling**)。
而且这个技术也不是只有Off-policy的AC能用。*实际上任何求期望的算法都能用重要性采样*。
比如说蒙特卡罗方法和TD方法。
先回顾一下讲蒙塔卡罗方法时候提到的expectation的近似求法，也就是如果每个state的权重一样，那么就可以用求均值的方法去近似期望。也就是：$$\bar{x}=\frac{1}{n}\sum^n_{i=1}x_i \to \mathbb{E}[X]$$
现在来关注另外一种情况：也就是每个state的权重不一样，例如：$$p_1(X=+1)=0.8 \quad p_1(X=-1)=0.2$$
那么正常情况期望就可以这样求出来：$$\mathbb{E}_{X\sim p1}[X]=(+1)\cdot 0.8+(-1)\cdot0.2=0.6$$能看出来，这种情况下之前的近似方法就不奏效了。
我们想用Off-policy的方法，就免不了要用两个不同的policy，一个是***Behavior policy***，也就是生成数据的policy，另外一个是***Target policy***，也就是需要被更新的policy。那么就存在一个情况：使用概率分布 $P_1$ 生成的采样数据，拿给概率分布 $P_2$ 求均值。这很明显就属于第二种情况。所以我们需要以下方法：$$\begin{align*} E_{X \sim p_0}[X] = \sum_{x} p_0(x)x  = \sum_{x} p_1(x) \underbrace{\left( \frac{p_0(x)}{p_1(x)} \right)}_{f(x)}x = E_{X \sim p_1}[f(X)] \end{align*}$$这个式子有什么用？第一项是根据 $p_0$ 分别得到的均值，经过变换得到了服从于 $p_1$ 的数据，也就能求 $p_1$ 分布下的expectation了。其中，使用了 $f$ 来纠正bias。
**intuition**：其实理解起来很直观，已知一个分布的期望，已知两个分布的概率，将已知期望除以该分布的概率，乘以目标分布的概率就是目标分布的期望了。

然后就是离策略的AC算法本体了：
先讨论怎么获取目标函数的梯度$$
J(\theta) = \sum_{s \in \mathcal{S}} d_\beta(s) \upsilon_\pi(s) = \mathbb{E}_{S \sim d_\beta} [\upsilon_\pi(S)]
$$这是需要被优化的目标函数，
	其中 $\beta$ 是behavior policy，$d_\beta$ 就是我们刚刚提到的 $p_1$ 分布，也就是behavior policy的distribution。
	$\theta$ 是target policy $\pi$ 的参数。
求出目标函数的梯度：$$ \nabla_{\theta} J(\theta) = \mathbb{E}_{S \sim \rho, A \sim \beta} \left[ \frac{\pi(A|S,\theta)}{\beta(A|S)} \nabla_{\theta} \ln \pi(A|S,\theta) \, q_{\pi}(S,A) \right] $$引入baseline (使用 $v_\pi(S)$) 来创建优势函数：$$\begin{aligned} \nabla_{\theta} J(\theta) &= \mathbb{E}_{S \sim p, A \sim \beta} \left[ \frac{\pi(A|S,\theta)}{\beta(A|S)} \nabla_{\theta} \ln \pi(A|S,\theta) \, (q_{\pi}(S,A) - b(S)) \right]\\ &=\mathbb{E}_{S \sim p, A \sim \beta} \left[\,··· \, (q_{\pi}(S,A) - v_\pi(S)) \right] \end{aligned}$$带入随机梯度上升算法：：$$ \theta_{t+1} = \theta_{t} + \alpha_{\theta} \frac{\pi(a_{t}|s_{t}, \theta_{t})}{\beta(a_{t}|s_{t})} \nabla_{\theta} \ln \pi(a_{t}|s_{t}, \theta_{t}) \left( q_{t}(s_{t}, a_{t}) - v_{t}(s_{t}) \right) $$使用 **A2C** 中的思想，取消用用于估计 $q$ 值的神经网络，使用奖励和状态值来估计 $q$ 值$$ q_t(s_t, a_t) - V_t(s_t) \approx r_{t+1} + \gamma v_t(s_{t+1}) - v_t(s_t) = \delta_t(s_t, a_t) $$带入原式：$$\begin{aligned} \theta_{t+1} &= \theta_{t} + \alpha_{\theta} \frac{\pi(a_{t}|s_{t}, \theta_{t})}{\beta(a_{t}|s_{t})} \nabla_{\theta} \ln \pi(a_{t}|s_{t}, \theta_{t}) \left( r_{t+1} + \gamma v_t(s_{t+1}) - v_t(s_t) \right)\\ &=\theta_{t} + \alpha_{\theta} \frac{\pi(a_{t}|s_{t}, \theta_{t})}{\beta(a_{t}|s_{t})} \nabla_{\theta} \ln \pi(a_{t}|s_{t}, \theta_{t}) \left( \delta_t(s_t,a_t) \right) \end{aligned}$$
我们进行最后的化简，也就是把log的导数拆开：$$\theta_{t+1} = \theta_t + \alpha_\theta \left( \frac{\delta_t(s_t, a_t)}{\beta(a_t|s_t)} \right) \nabla_\theta \pi(a_t|s_t, \theta_t)$$这样就得到了离策略的AC方法了。 
伪代码：
```python
# ds define
class Policy:
    def __init__(self, θ, α):
        self.θ = θ      # 策略参数向量
        self.α = α      # 学习率
        # 包含方法：概率计算、梯度计算等

class ValueFunction:
    def __init__(self, w, α):
        self.w = w      # 价值函数参数向量 
        self.α = α      # 学习率
        # 包含方法：价值估计、梯度计算等

class Experience:
    def __init__(self):
        self.s = None   # 当前状态
        self.a = None   # 采取动作
        self.r = None   # 即时奖励
        self.s_next = None  # 下一状态

# init
def initialize():
    β = Policy(θ=..., α=0)    # 固定行为策略（学习率为0）
    π = Policy(θ=θ0, α=α_θ)   # 待优化的目标策略
    v = ValueFunction(w=w0, α=α_w)
    return β, π, v

# main
def actor_critic_algorithm(num_episodes, γ):
    β, π, v = initialize()
    
    for episode in range(num_episodes):
        exp = Experience()
        exp.s = env.reset()  # 初始化状态
        
        while not done:
            # 生成动作
            exp.a = β.sample_action(exp.s)  # 从行为策略采样
            
            # 与环境交互
            exp.r, exp.s_next, done = env.step(exp.a)
            
            # 计算TD误差
            δ = exp.r + γ * v.estimate(exp.s_next) - v.estimate(exp.s)
            
            # Critic更新 (值函数参数)
            ratio = π.probability(exp.a, exp.s) / β.probability(exp.a, exp.s) # 重要性采样
            v.w += v.α * ratio * δ * v.gradient(exp.s)
            
            # Actor更新 (策略参数)
            π.θ += π.α * ratio * δ * π.log_gradient(exp.a, exp.s)
            
            # 转移到下一状态
            exp.s = exp.s_next

    return π  # 返回优化后的策略
```
