
值迭代算法其实已经在上一篇笔记中讲过了，但这次以一个正式的角度重讲一遍，以便引出策略迭代。

# 值迭代 value iteration
值迭代可以分成两个步骤：
1. Policy Update (PU)：这个步骤求解一个方程$$\pi_{k+1}=\arg \max_\pi(r_\pi+\gamma P_\pi v_k)$$其中 $v_k$ 是一个给定值
	- 由于最优策略是greedy的，所以贝尔曼最优公式的求和中实际上只有一项。是 $a^*$ 则概率为1，不是则概率为0
	- $$\pi_{k+1}(s) =\arg \max_{\pi} \sum_a \pi(a|s) \left( \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v_k(s') \right), \quad \forall s \in \mathcal{S}$$
1. Value Update (VU)：上一步求解出 $\pi_k$ 之后，这个步骤将 $v_k$ 带入方程求解一个更接近最优解的 $v_{k+1}$ $$v_{k+1}=r_\pi+\gamma P_{\pi}v_k$$这个公式是matrix-vector form，它的elementwise form是：$$v_{k+1}(s) = \max_{\pi} \sum_a \pi(a|s) \left( \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v_k(s') \right), \quad \forall s \in \mathcal{S}$$由于策略的greedy性质，$v_{k+1}$ 就等于action value中最大的那个数。最终公式会变成：$$v_{k+1}(s)=\max_aq_k(a,s)$$
这里的 $v_k$ 或者 $v_{k+1}$ 都不是state value，因为上面这个式子不是贝尔曼公式。
一次Value Iteration的procedure：$$v_k(s)\Rightarrow q_k(s,a)\Rightarrow \text{greedy policy }\pi_{k+1}(a|s)\Rightarrow \text{new value } v_{k+1}(s)$$
# 策略迭代 policy iteration
策略迭代其实是把值迭代的步骤反过来，分别称为：
1. policy evaluation (PE)
	- 这一步给定一个策略 $v_\pi$，求解所有状态的state value：$$v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$$
2. policy improvement (PI)
	- 然后根据上一步求解出的state value计算新一轮的q value，更新策略，即选择拥有最大的q value的action：$$\pi_{k+1} = \underset{\pi}{\arg\max}(r_{\pi} + \gamma P_{\pi} v_{\pi_k})$$
整个迭代方法总结下来就是：$$\pi_0 \xrightarrow{PE} v_{\pi_0} \xrightarrow{PI} \pi_1 \xrightarrow{PE} v_{\pi_1} \xrightarrow{PI} \pi_2 \xrightarrow{PE} v_{\pi_2} \dots$$其中策略 $\pi$ 和state value $v_\pi$ 交替更新。
**第一步 (PE) 具体怎么进行呢？** 其实就是求解一个贝尔曼公式。前文提到有两种方法求解bellman eqn，分别是matrix-vector通过矩阵运算求解，和使用fixed point通过迭代算法求解。而且前文提到，***实际操作中一般使用迭代算法***（矩阵求逆运算量太大）。
也就是：$$v_{\pi_k}^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(j)}, \quad j = 0, 1, 2, ...$$
迭代到什么时候停止呢？***在实际的agent训练中，当一次迭代前后两个state value 相差足够小时迭代就可停止，将此时的结果视为state value。***
Stop when $j \rightarrow \infty$ or $j$ is sufficiently large or $||v_{\pi_k}^{(j+1)} - v_{\pi_k}^{(j)}||$ is sufficiently small. 前者很明显是不可能的，后者是实践中常用的方法。
**第二步 (PI) 怎么进行的？** 方法和value iteration中一样，都是求解一个bellman optimal eqn，详解见前面的笔记 **（如何求解bellman optimal eqn）**
### 下面是策略迭代的伪代码：
***Pseudocode*: Policy iteration algorithm**
**Initialization:** The probability model $p(r|s,a)$ and $p(s'|s, a)$ for all $(s, a)$ are known. Initial guess $\pi_0$.
**Aim:** Search for the optimal state value and an optimal policy.

While the policy has not converged, for the $k$th iteration, do:
    **Policy evaluation:**
    Initialization: an arbitrary initial guess $v_{\pi_k}^{(0)}$
    While $v_{\pi_k}^{(j)}$ has not converged, for the $j$th iteration, do:
        For every state $s \in S$, do:
            $v_{\pi_k}^{(j+1)}(s) = \sum_a \pi_k(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}^{(j)}(s') \right]$
    **Policy improvement:**
    For every state $s \in S$, do:
        For every action $a \in A(s)$, do:
            $q_{\pi_k}(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s')$
        $a_k^*(s) = \arg \max_a q_{\pi_k}(s, a)$
        $\pi_{k+1}(a|s) = 1$ if $a = a_k^*$, and $\pi_{k+1}(a|s) = 0$ otherwise

# 接下来就是 Truncated policy iteration 截断策略迭代
> 上文提到的policy iteration和value iteration其实非常相似，The two algorithms are very similar:

**Policy iteration**: $\pi_0 \xrightarrow{PE} v_{\pi_0} \xrightarrow{PI} \pi_1 \xrightarrow{PE} v_{\pi_1} \xrightarrow{PI} \pi_2 \xrightarrow{PE} v_{\pi_2} \xrightarrow{PI} ...$
**Value iteration**: $\quad \text{    } \quad \quad u_0 \xrightarrow{PU} \pi'_1 \xrightarrow{VU} u_1 \xrightarrow{PU} \pi'_2 \xrightarrow{VU} u_2 \xrightarrow{PU} ...$
下图是tabular形式的两种算法的description，*可以清晰的看出来两种算法的steps基本一样。* 啊但是！实际上除了标蓝的部分 (4) value，其他部分都是一摸一样的，下面我们具体来分析 (4) value 中出现了什么样的区别![[Pasted image 20250129203402.png]]考虑Policy Iteration algorithm的 (4) value 步骤，我们需要求解一个bellman eqn： $$\color{violet}v_{\pi_1}=r_{\pi_1}+\gamma P_{\pi_1}v_{\pi_1}$$使用迭代算法：$$
\begin{aligned}
v_{\pi_1}^{(0)} &= v_0 \\
v_{\pi_1}^{(1)} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(0)} \\
v_{\pi_1}^{(2)} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(1)} \\
&\vdots \\
v_{\pi_1}^{(j)} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(j-1)} \\
&\vdots \\
v_{\pi_1}^{(\infty)} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(\infty)}
\end{aligned}
$$很明显，这个计算process是不合理的，我们不能计算无限多步。再在这个process中考虑Value Iteration 的算法。 $$\begin{aligned}
v_{\pi_1}^{(0)}(v_{0}) &= v_0 \\
v_{\pi_1}^{(1)}(v_{1}) &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(0)}(v_{0})
\end{aligned}
$$以上是value iteration的 (4) value 的计算过程，也就是policy iteration的process进行到第二步的时候就停止了，将得到的 $v_1$ 作为下一个循环的input。***相当于“迭代”计算只进行了1次迭代。精度也是可想而知的。*** 
policy iteration是迭代无穷步，value iteration 是迭代1步，我们很自然地想：有没有一种algorithm计算有限步，仍然具有不错的精度呢？比如迭代500步，或者2000步。
就这样，**Truncated Policy Iteration** 截断策略迭代 诞生了。
也就是policy iteration在 (4) value步骤进行 $\mathbf{j}$ 次迭代，具体如下：$$\begin{aligned}
v_{\pi_1}^{(0)} &= v_0 \\
v_{\pi_1}^{(1)} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(0)} \\
v_{\pi_1}^{(2)} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(1)} \\
&\vdots \\
v_{\pi_1}^{(\mathbf{j})} &= r_{\pi_1} + \gamma P_{\pi_1} v_{\pi_1}^{(\mathbf{j-1})} \\
\end{aligned}$$伪代码实现如下：
**Initialization:** The probability model $p(r|s, a)$ and $p(s'|s, a)$ for all $(s, a)$ are known. Initial guess $\pi_0$.
**Aim:** Search for the optimal state value and an optimal policy.

While the policy has not converged, for the $k$th iteration, do
  **Policy evaluation:**
    Initialization: select the initial guess as $v_k^{(0)} = v_{k-1}$. The maximum iteration is set to be $j_{\text{truncate}}$.
    While $j < j_{\text{truncate}}$, do
      For every state $s \in \mathcal{S}$, do
        $v_k^{(j+1)}(s) = \sum_a \pi_k(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a) v_k^{(j)}(s') \right]$
    Set $v_k = v_k^{(j_{\text{truncate}})}$
  **Policy improvement:**
    For every state $s \in \mathcal{S}$, do
      For every action $a \in \mathcal{A}(s)$, do
        $q_k(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a) v_k(s')$
      $a_k^*(s) = \arg \max_a q_k(s, a)$
      $\pi_{k+1}(a|s) = \begin{cases} 1 & \text{if } a = a_k^* \\ 0 & \text{otherwise} \end{cases}$

以下是policy iteration，value iteration和truncated policy iteration的收敛速度对比：
![[Pasted image 20250129211311.png]]

# The END

