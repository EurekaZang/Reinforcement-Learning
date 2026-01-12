它和value based的区别：
1. value based 的state space和action space都是tabular的，但是policy gradient用的是函数形式，也就是一个映射（最广泛的就是神经网络）。最重要的是，Policy based算法直接学习一个策略，而Value based学习的是每一个状态的值。
	- value based中参数使用 $w$ 来表示，但是policy based中参数使用 $\theta$ 来表示
	- **WHY?** 因为policy based的方法常用于处理一些拥有很大的state / action space的任务，比如现实世界中的机器人 & 无人机控制等。如果使用tabular的形式，会对内存要求造成很大的负担。所以使用函数映射，只需要储存函数的参数即可（神经网络的权重）。
2. tabular中，有 $v_{\pi}^* > v_{\text{others}}$。函数情况下不同，我们定义一个scalar的目标函数，我们要去优化这个目标函数。
	- 怎么拿到一个action的probability？我们要将 $s$ 送进神经网络进行一次前向传播，得到每个action的probability。
3. 怎么update 一个 policy？
	- tabular直接用索引访问matrix，更改state value。但是函数形式更改的是参数

定义目标函数：
1. 最优策略 $$\mathcal{J}(\theta)=\left\{\begin{aligned} & \bar{v}_\pi \quad \text{average state value} \\ & \bar{r}_\pi \quad \text{average reward}\end{aligned}\right.$$
2. 以梯度为基础（*梯度下降上升*）的最优算法用于寻找最优策略：$$\theta_{t+1}=\theta_t+\alpha\Delta_{\theta} \mathcal{J}(\theta)$$
### 所以policy gradient的思路就是最大化目标函数（一个episode中的奖励）
下面要讨论怎么定义这个optimal policy，也就是需要一个metric (标准)。
#### 第一种matric是 **average state value**，也就是状态值加权平均。
$$\begin{aligned}\bar{v}_\pi&=\sum_{s\in \mathcal{S}}d(s)v_\pi(s)\\&=d^Tv_\pi \text{  向量内积形式}\end{aligned}$$这个 $\bar{v}_\pi$ 是一个策略的函数，不同的策略对应的 $\bar{v}_\pi$ 不同。所以就可以去优化，找到一个策略让这个值达到最大。
Where：
1. $\sum_{s\in\mathcal{S}}d(s)=1$。，他是一个概率分布。
2. $v_\pi=[...,v_\pi(s),...]^T\in\mathbb{E}^{|\mathcal{S}|}$
3. $d=[...,d(s),...]^T\in\mathbb{E}^{|\mathcal{S}|}$
在论文中，也会经常见到它的另一种形式：$$\mathcal{J}(\theta)=\bar{v}_\pi=\mathbb{E}\left[\sum^\infty_{t=0}\gamma^t R_{t+1}\right]$$懒得证了，很简单,可以自己试试。:D
那么怎么选择这个 $d$ （概率分布）?
我们先考虑 $d$ 跟策略没关系的情况。 $d$ is **independent** of the policy $\pi$.
为了表示清晰，将没有关系的 $d$ 写成 $d_0$ 。将有关系的 $d$ 写成 $d_\pi$ 。
- 首先就是每个状态的权重都是 $\frac{1}{n}$，也就是每一个状态都是相同重要的。这种情况下$$d_0(s)=\frac{1}{|\mathcal{S}|}=\frac{1}{n}$$
- 或者有一些状态很重要，比如说一个游戏开始的时候总是从一个位置开始，那么这个位置就对应着 $s_0$ ，从那出发我希望得到的reward越大越好。要给他们分配更大的权重：$$d_0(s_0)=1, \quad d_0(s \neq s_))=0$$ ==（极端情况，只考虑 $s_0$）==
现在来看一下当 $d$ 和 $\pi$ 有关系的情况。
设想一个agent，它在一个环境根据一个策略中不停的交互，当它执行策略很多次之后，就可以获得agent在每个状态的概率是多少。那个概率分布就叫做 *stationary distribution*。可以直接通过以下方程得到： $$d^T_\pi P_\pi=d^T_\pi$$其中 $P_\pi$ 是状态转移矩阵。当 $P_\pi$ 知道了之后，可以直接求出 $d^T_\pi$。
上面这个式子什么意思呢？意思就是*这个分布 $d$ 在经过一次转移之后分布保持不变*（这也是马尔可夫的平稳分布的定义）下面涉及一点点的原理知识：

> 当马尔科夫链满足 **不可约性** （所有状态互通）和 **非周期性** 时，根据 *Perron-Frobenius* 定理，存在唯一的稳定分布 $d_\pi$ 。此时，$d_\pi$ 可以通求解以上方程得到，等价于求解 $P^T_\pi d_\pi=d_\pi$

也就是访问多的状态权重大，访问少的状态权重小。
#### 第二种metric是 **average one-step reward** 或者 **average reward**
直接：$$\bar{r}_\pi=\sum_{s\in \mathcal{S}}d_\pi(s)r_\pi(s)=\mathbb{E}[r_\pi(S)]$$其中 $S$ 服从 $d_\pi$ 分布（也是一个*stationary reward*）。
其中 $$r_\pi(s)=\sum_{a\in \mathcal{A}}\pi(a|s)r(s,a)$$也就是state $s$ 下的所有action 的immediate reward 的加权和。
理顺一下思路：$$r(s,a)\underset{\text{加权平均}}{\Rightarrow}r_\pi(s)\underset{\text{加权平均}}{\Rightarrow}\bar{r}_\pi$$这个metric也可以写成第二种形式，也是在书籍和论文中更长常见的形式：$$\begin{aligned}\bar{r}_\pi=&\underset{n\to\infty}{\lim}\frac{1}{n}\mathbb{E}\left[R_{t+1}+R_{t+2}+...+R_{t+n}|S_t=s_0\right]\\=&\underset{n\to\infty}{\lim}\frac{1}{n}\mathbb{E}\left[\sum^n_{k=1}R_{t+k}|S_t=s_0\right]\\=&\underset{n\to\infty}{\lim}\frac{1}{n}\mathbb{E}\left[\sum^n_{k=1}R_{t+k}\right]\quad (S_t=s_0\text{的影响在极限中消失})\end{aligned}$$这个是该metric的*时间平均*形式，而之前那个是*空间平均*形式。

> 根据遍历定理 (**Erogodic Theorem**) 对于不可约，非周期，正常返的马尔科夫链，时间平均等于空间平均: $$\underset{n\to\infty}{\lim}\frac{1}{n}\mathbb{E}\sum^n_{k=1}R_{t+k}=\mathbb{E}_{S\sim d_\pi}[r_\pi(S)]\quad \text{a.s.}$$不可约性（Irreducible）：所有状态互通。
   非周期性（Aperiodic）：状态的周期性为 1。
   正常返（Positive Recurrent）：状态被访问的期望时间有限。

无论是第一个metric还是第二个，都是策略的函数，也就能做优化，找到最大的policy。
这两个metric也是有关系的，实际上我们能证明$$\bar{r}_\pi=(1-\gamma)\bar{v}_\pi$$
#### 有了metric接下来干什么呢？求 $\mathcal{J}(\theta)$ 的梯度然后使用gradient-based方法
下面来看一下怎么求 $\Delta_\theta \mathcal{J}(\theta)$ 
详细推导过程是很复杂，这里先给出结论，推导过程将放在附录中。$$\nabla_\theta\mathcal{J}(\theta)=\sum_{s\in\mathcal{S}}\eta(s)\sum_{a\in \mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)$$
- 其中，$\mathcal{J}(\theta)$ 可以是 $\bar{v}_\pi$, $\bar{r}_\pi$, or $\bar{v}_\pi^0$ 。
- "$=$" 或许不是直接相等，而有可能是约等，成正比。
- $\eta$ 是state的分布，或权重。 
这个公式又可以进一步做变换：$$
\begin{align}
\nabla_{\theta} J(\theta) 
&= \sum_{s \in S} \eta(s) \sum_{a \in A} \nabla_{\theta} \pi(a|s, \theta) \, q_{\pi}(s, a) \\
\nabla_{\theta} \pi(a|s, \theta) 
&= \pi(a|s, \theta) \nabla_{\theta} \ln \pi(a|s, \theta) \\
\nabla_{\theta} J(\theta) 
&= \sum_{s \in S} \eta(s) \sum_{a \in A} \pi(a|s, \theta) \nabla_{\theta} \ln \pi(a|s, \theta) \, q_{\pi}(s, a) \\
\sum_{a \in A} \pi(a|s, \theta) \nabla_{\theta} \ln \pi(a|s, \theta) \, q_{\pi}(s, a) 
&= \mathbb{E}_{a \sim \pi(\cdot|s, \theta)} \left[ \nabla_{\theta} \ln \pi(a|s, \theta) \, q_{\pi}(s, a) \right] \\
\sum_{s \in S} \eta(s) \mathbb{E}_{a \sim \pi} [\cdot] 
&= \mathbb{E}_{s \sim \eta(\cdot)} \mathbb{E}_{a \sim \pi(\cdot|s, \theta)} \left[ \nabla_{\theta} \ln \pi(a|s, \theta) \, q_{\pi}(s, a) \right] \\
\nabla_{\theta} J(\theta) 
&= \mathbb{E}_{s \sim \eta, a \sim \pi} \left[ \nabla_{\theta} \ln \pi(a|s, \theta) \, q_{\pi}(s, a) \right] \\
&= \mathbb{E} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) \, q_{\pi}(S, A) \right]
\end{align}
$$其中，如果你忘记了第二个公式如何推导：here is log trick
$$\begin{align}
\nabla_{\theta} \pi(a|s, \theta)
&= \pi(a|s, \theta) \cdot \frac{\nabla_{\theta} \pi(a|s, \theta)}{\pi(a|s, \theta)} \\
&= \pi(a|s, \theta) \nabla_{\theta} \ln \pi(a|s, \theta)
\end{align}
$$至于为什么要引入对数，是因为要将目标函数梯度的求和形式转换成期望形式（*在复杂高维连续空间中，求和显然不可能。而期望是可以被估计的，比如使用蒙特卡罗或者时序差分方法。*）
这样就得到了一个简单的期望形式的梯度表达式。
还有一点要补充。在推导过程中出现了 $\ln \pi(a|s, \theta)$ ，也就是我们要保证对于每一个 $a$ ，$\pi$ 都要大于零。所以我们需要人为定义 $\pi$ 这个函数。
自然就想到了 $\text{softmax}$ 函数，他将 $\{-\infty,\infty\}$ 映射到 $(0,1)$ 之间。
对于任意向量 $v=[x_1,x_2,...,x_n]^T$ ，$\text{softmax}$ 长这样：$$z_i=\frac{e^{x_i}}{\sum^n_{j=1}e^{x_j}}$$所以 $\pi$ 就写成这样：$$\pi(a|s, \theta) = \frac{e^{h(s,a,\theta)}}{\sum_{a' \in \mathcal{A}} e^{h(s,a',\theta)}}$$
其中 $h(s,a,\theta)$ 是特征向量，由神经网络选取。
但是我们不喜欢求期望，其实是我们没有方法求期望（无法获得所有的状态和动作）。所以我们需要把期望拿掉，换成随即近似的方法。
也就是变成：$$\begin{aligned}\nabla_{\theta} J(\theta) &= \nabla_{\theta} \ln \pi(a_t|s_t, \theta) \, q_{\pi}(s_t, a_t)\\&=\left(\frac{\nabla_\theta\pi(a_t|s_t,\theta_t)}{\pi(a_t|s_t,\theta_t)}\right)\cdot q_{\pi}(s_t, a_t)\text{  (Log Trick)}\\&=\underbrace{\left(\frac{q_{\pi}(s_t, a_t)}{\pi(a_t|s_t,\theta_t)}\right)}_{\color{lime}\beta_t}\cdot \nabla_\theta\pi(a_t|s_t,\theta_t)\\&=\beta_t\nabla_\theta\pi(a_t|s_t,\theta_t)\end{aligned}$$带入梯度上升公式，我们得到：
$$
\theta_{t+1} = \theta_{t} + \alpha \beta_{t} \nabla_{\theta} \pi(a_{t} | s_{t}, \theta_{t})
$$
- $\theta_{t+1}$: 时间步 $t+1$ 时的参数值（更新后的参数）
- $\theta_{t}$: 时间步 $t$ 时的参数值（当前参数）
- $\alpha$: **学习率**，控制更新步长的超参数（通常为标量常数）
- $\beta_{t}$: **权重系数**，依赖于时间步 $t$ 和环境反馈（如奖励信号）
- $\nabla_{\theta} \pi(a_{t} | s_{t}, \theta_{t})$:  
  策略函数 $\pi$ 对参数 $\theta$ 的梯度，表示在状态 $s_t$ 下选择动作 $a_t$ 的概率随 $\theta$ 变化的方向和强度
参数更新逻辑：  
	可以直观地看出来，这个公式是在最优化 $\pi(a_t|s_t,\theta_t)$ ，也就是如果 $\beta$ 是正值，那么梯度上升，$\pi$ 变大，选择该动作的概率变大，反之亦然。
	通过梯度 $\nabla_{\theta} \pi(a_{t} | s_{t}, \theta_{t})$ 的方向调整 $\theta$，使策略 $\pi$ 更倾向于选择高 $\beta_t$ 的动作。
$\beta_t$ 的关键作用：  
   - 若 $\beta_t > 0$，梯度上升会**增大**动作 $a_t$ 的概率  
   - 若 $\beta_t < 0$，梯度上升会**减小**动作 $a_t$ 的概率 （变成了梯度下降）
   - 在强化学习中，$\beta_t$ 通常与**回报信号**相关（如 $G_t$, $Q(s,a)$, 或优势函数 $A(s,a)$），但这里我们给出的算法是广义上的策略梯度算法，既不是REINFORCE也不是AC。所以我们直接使用 $\frac{q_\pi(a_t,s_t)}{\pi(a_t|s_t,\theta_t)}$ 作为 $\beta$
在这里 $\beta$ 的选择其实很精妙，巧妙地平衡了 exploration和exploitation，当分子 $q_\pi(s_t,a_t)$ 比较大的时候，$\beta$ 也会比较大，那么该动作的 $\pi$ 也会增长较快。分母 $\pi(a_t|s_t,\theta_t)$ 较小的时候（也就是虽然这个动作很好但是概率较小的时候），由于分母恒小于一，$\beta$ 也会比较大，那么该动作的 $\pi$ 也会较快的增长。


# The END 









#### Appendix 1:
### 1. 目标函数定义
目标函数 $\mathcal{J}(\theta)$ 的一般形式为：
$$
\mathcal{J}(\theta) = \sum_{s \in \mathcal{S}} \eta(s) v_\pi(s),
$$
其中 $\eta(s)$ 是状态的权重分布：
- 对于 $\bar{v}_\pi^0$（初始状态价值），$\eta(s) = \mathbb{P}(s_0 = s)$。
- 对于 $\bar{r}_\pi$（平均奖励），$\eta(s)$ 是策略的稳态分布。

### 2. 对目标函数求梯度
目标函数的梯度为：
$$
\begin{align}
\nabla_\theta \mathcal{J}(\theta) 
&= \nabla_\theta \left[ \sum_{s \in \mathcal{S}} \eta(s) v_\pi(s) \right] \\
&= \sum_{s \in \mathcal{S}} \nabla_\theta \left[ \eta(s) v_\pi(s) \right] \\
&= \underbrace{\sum_{s \in \mathcal{S}} \nabla_\theta \eta(s) \cdot v_\pi(s)}_{\text{(Term 1)}} + \underbrace{\sum_{s \in \mathcal{S}} \eta(s) \cdot \nabla_\theta v_\pi(s)}_{\text{(Term 2)}}.
\end{align}
$$
### 3. 处理 Term 1：状态分布梯度
**关键结论**：$\text{Term 1} = 0$。  
**推导**：
- 状态分布 $\eta(s)$ 满足归一化条件：$\sum_{s \in \mathcal{S}} \eta(s) = 1$。
- 对 $\theta$ 求梯度得：
  $$
  \sum_{s \in \mathcal{S}} \nabla_\theta \eta(s) = \nabla_\theta \left[ \sum_{s \in \mathcal{S}} \eta(s) \right] = \nabla_\theta 1 = 0.
  $$
- 结合价值函数的稳态性质（详见补充推导），可证明：
  $$
  \sum_{s \in \mathcal{S}} \nabla_\theta \eta(s) \cdot v_\pi(s) = 0.
  $$

### 4. 展开 Term 2：状态价值梯度
状态价值函数定义为：
$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s,\theta) q_\pi(s,a).
$$
对其求梯度：
$$
\begin{align}
\nabla_\theta v_\pi(s) 
&= \sum_{a \in \mathcal{A}} \nabla_\theta \left[ \pi(a|s,\theta) q_\pi(s,a) \right] \\
&= \sum_{a \in \mathcal{A}} \left[ \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a) + \pi(a|s,\theta) \cdot \nabla_\theta q_\pi(s,a) \right].
\end{align}
$$

### 5. 处理动作价值梯度项
动作价值函数 $q_\pi(s,a)$ 的梯度可递归展开为：
$$
\nabla_\theta q_\pi(s,a) = \nabla_\theta \left[ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) v_\pi(s') \right] = \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \nabla_\theta v_\pi(s').
$$
将 $\nabla_\theta q_\pi(s,a)$ 代入 Term 2：
$$
\begin{align}
\text{Term 2} 
&= \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \left[ \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a) + \pi(a|s,\theta) \cdot \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \nabla_\theta v_\pi(s') \right] \\
&= \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a) + \gamma \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \pi(a|s,\theta) \sum_{s' \in \mathcal{S}} P(s'|s,a) \nabla_\theta v_\pi(s') \\
&= \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a) + \gamma \sum_{s' \in \mathcal{S}} \left[ \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \pi(a|s,\theta) P(s'|s,a) \right] \nabla_\theta v_\pi(s').
\end{align}
$$

### 6. 递归消去梯度项
定义**折扣状态分布** $\eta'(s')$：
$$
\eta'(s') = \gamma \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \pi(a|s,\theta) P(s'|s,a).
$$
代入 Term 2 的第二部分：
$$
\gamma \sum_{s' \in \mathcal{S}} \eta'(s') \nabla_\theta v_\pi(s') = \gamma \cdot \text{Term 2}'.
$$
递归展开后，所有高阶梯度项 $\nabla_\theta v_\pi(s')$ 将形成无限级数，最终收敛到零：
$$
\text{Term 2} = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a) + \gamma \cdot \text{Term 2}' \quad \Rightarrow \quad \text{高阶项总和为 } 0.
$$

### 7. 最终梯度表达式
综合上述步骤，目标函数的梯度为：
$$
\boxed{
\nabla_\theta \mathcal{J}(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a)
}
$$

### 补充说明
1. 比例关系  
   实际应用中，$\eta(s)$ 可能包含折扣因子 $\gamma^t$，导致梯度表达式为：
   $$
   \nabla_\theta \mathcal{J}(\theta) \propto \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) \cdot q_\pi(s,a).
   $$

2. 状态分布 $\eta(s)$ 的具体形式  
   - 对于**平均奖励**（$\bar{r}_\pi$），$\eta(s)$ 是策略的稳态分布。
   - 对于**折扣回报**（$\bar{v}_\pi$），$\eta(s) = \sum_{t=0}^\infty \gamma^t \mathbb{P}(s_t = s)$。
$\textbf{Q.E.D}$
