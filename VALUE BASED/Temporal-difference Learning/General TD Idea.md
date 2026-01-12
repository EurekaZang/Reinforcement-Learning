by Eureka 11/2/25
介于我们已经掌握了Robbins-Monro算法，下面将用一个更为复杂的RM例子来引入TD算法：$$w = \mathbb{E}[R + \gamma v(X)],$$其中 $R$ 和 $X$ 都是random variable，目标是求解mean。
利用RM算法：$$
\begin{aligned}
g(w) &= w - \mathbb{E}[R + \gamma v(X)], \\
\tilde{g}(w, \eta) &= w - [r + \gamma v(x)] \\
&= (w - \mathbb{E}[R + \gamma v(X)]) + (\mathbb{E}[R + \gamma v(X)] - [r + \gamma v(x)]) \\
&= g(w) + \eta.
\end{aligned}
$$代入公式：$$w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k) = \color{lime}w_k - \alpha_k [w_k - (r_k + \gamma v(x_k))]$$其中 $r_k$ 和 $x_k$ 是 $R$ 和 $X$ 的随机采样。
*接下来我们要讲的TD算法是一种求解state value的算法*，指的不是业界内所谓的一大类算法。
我们来看如何利用上述RM例子来求解state value ($\pi$ 固定的情况下)
首先，TD算法是一种model-free算法，也就是不依赖于模型，而依赖于数据。是一个data-based算法。
需要的数据就是一个trajectory，形如$(s_0, r_1, s_1, ..., s_t, r_{t+1}, s_{t+1}, ...)$ or $\{ (s_t, r_{t+1}, s_{t+1}) \}_t$，由一个固定的策略 $\pi$ 产生。
这是TD的数学表达形式：$$
\begin{aligned}
v_{t+1}(s_t) &= v_t(s_t) - \alpha_t(s_t) [v_t(s_t) - [r_{t+1} + \gamma v_t(s_{t+1})]], \quad \text{(1)} \\
v_{t+1}(s) &= v_t(s), \quad \forall s \neq s_t, \quad \text{(2)}
\end{aligned}
$$可以清晰地看出这个算法是在计算state value，也就是做policy evaluation。
其中 $t$ 代表的是时间步，比如 $v_{t+1}(s_t)$ 就是在第 $t+1$ 时间步中 $s_t$ 状态的state value
拆解第一个式子：$$
\underbrace{v_{t+1}(s_t)}_{\text{new estimate}} = \underbrace{v_t(s_t)}_{\text{current estimate}} - \alpha_t(s_t) [ \overbrace{v_t(s_t) - \underbrace{[r_{t+1} + \gamma v_t(s_{t+1})]}_{\text{TD target }\bar{v}_t}}^{\text{TD error } \delta_t} ], \quad \text{(3)}
$$TD error是当先时间步 ($t$) 下$s_t$ 和 TD target的差，而这整个式子就是想通过迭代来缩小 $s_t$ 和 TD target 的差值。
下面来看一下这个式子是怎么来的：$$\begin{aligned} \delta_t &= v(s_t) - [r_{t+1} + \gamma v(s_{t+1})]\\ &\Downarrow\\ 同时求&期望得到\\&\Downarrow\\ \mathbb{E}[\delta_{t} | S_t = s_t] &= v(s_t) - \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s_t] \end{aligned}$$在等式的右边，我们得到了一个贝尔曼公式。根据Bellman eqn，有：(如果这点有疑问可以回去看Bellman Equation的推导，这里用到的是一个比较原始的式子(*未展开式，没有把求和符号显式写出来。*))$$\mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s_t] = v_\pi(s_t)$$也就是$$\mathbb{E}[\delta_{t} | S_t = s_t]=0$$**这就是要用在RM算法中的等式。** 我们令 $${g}(s_t)=v(s_t) - \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s_t]$$该函数的Observation就是 $$\tilde{g}(s_t)=v(s_t)-[r_{t+1}+\gamma v(s_{t+1})]$$带入RM公式得到：$$v_{t+1}(s_t) = v_t(s_t) - \alpha [v_t(s_t) - [r_{t+1} + \gamma v_t(s_{t+1})]]$$当 $t \rightarrow \infty$，有 $v_t(s_t) \rightarrow \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t = s_t]=v_\pi(s_t)$
讲完啦。
这种TD算法使用了RM (SA) 的思想来计算state value，通过多次迭代来逼近state value的准确值。

如果你还记得的话，我们在最开始的时候讲解如何求解BE时也有一种Iterative的解法：
Iterative solution：$$v_{k+1}=r_\pi+\gamma P_\pi v_k$$同为迭代算法，同求解state value，但它们完全不一样。
上面这个Iterative solution是model-based的，基于模型的算法。其中$P_\pi \in \mathbb{R}^{n \times n}, \text{ where } [P_\pi]_{ij} = p_\pi(s_j|s_i)$，它是一个概率矩阵。
而我们刚刚讲的是model-free的，不需要这个概率矩阵 (model)。它只需要 $s_t$，$r_t$ 和 $s_{t+1}$ (数据)。

| Type        | ***MC Learning***                                                            | ***TD (Sarsa) Learning***                                                                                               |
| ----------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **更新时机**    | 必须等待一个完整的episode结束后更新                                                        | 每一步（单步更新）后立即更新                                                                                                          |
| **更新公式**    | 使用实际回报（Return）更新值函数：<br>  $V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t))$ | 使用TD目标（自举）更新值函数：<br> $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))$ |
| **偏差与方差**   | **无偏**（基于真实回报），**高方差**（依赖完整轨迹）                                               | **有偏**（基于估计值），**低方差**（仅依赖单步）                                                                                            |
| **收敛性**     | 在足够探索下收敛，但可能较慢（高方差影响）                                                        | 通常收敛更快（低方差），但可能受初始值影响                                                                                                   |
| **适用场景**    | 仅适用于分幕式（episodic）任务                                                          | 适用于连续任务和分幕式任务                                                                                                           |
| **在线/离线学习** | Online（需存储完整轨迹）                                                              | Offline（可实时更新）                                                                                                          |
| **计算效率**    | 需存储整个episode的数据，内存消耗大                                                        | 单步更新，内存效率高                                                                                                              |
| **探索策略**    | 通常为on-policy（依赖当前策略生成轨迹）                                                     | on-policy（Sarsa使用当前策略选择下一步动作）                                                                                           |
| **对非平稳性适应** | 较差（需完整轨迹，无法动态调整）                                                             | 较好（实时更新适应环境变化）                                                                                                          |

# The END








