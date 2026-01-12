By Eureka 25-1-20 10:12

you will get:
1. A core concept: **State Value**
2. A fundamental tool: **The Bellman Equation**

Catalog:
[[#这一部分讲State value]]
[[#接下来讲Bellman Equation]]
[[#下面是Action Value]]

# 这一部分讲State value
先来做一些计算return的练习：
![[Pasted image 20250119053259.png]]
img1:
$$
\begin{aligned}
\text{return}_1&=0+\gamma1+\gamma^21+...\\
&=\frac{\gamma}{1-\gamma}.
\end{aligned}
$$
img2:$$
\begin{aligned}
\text{return}_2&=-1+\gamma1+\gamma^21+\gamma^31+...\\
&=-1+\frac{\gamma}{1-\gamma}.
\end{aligned}
$$
img3:$$
\begin{aligned}
\text{return}_3&=0.5×\text{return}_{\text{3a}}+0.5×\text{return}_{\text{3b}}\\
&=0.5(-1+\frac{\gamma}{1-\gamma})+0.5(\frac{\gamma}{1-\gamma})\\
&=-0.5+\frac{\gamma}{1-\gamma}.
\end{aligned}
$$
Tips：return的对象是trajectory。若有多个trajectory要求总return，就将它们的return相加
summary: 
$$\text{return}_1>\text{return}_3>\text{return}_2$$
**接下来介绍一个更好的方法计算return：**
引例：![[Pasted image 20250119054451.png]]
定义 $v_i$ 为从 $s_i$ 出发的trajectory得到的return, $(i=1,2,3,4)$
$$
\begin{aligned}
v_1&=r_1+\gamma r_2+\gamma^2 r_3+...\\
v_2&=r_2+\gamma r_3+\gamma^2 r_4+...\\
v_3&=r_3+\gamma r_4+\gamma^2 r_1+...\\
v_4&=r_4+\gamma r_1+\gamma^2 r_2+...
\end{aligned}
$$
注意到：
$$
\begin{aligned}
v_1&=r_1+\gamma (r_2+\gamma r_3+...)=r_1+\gamma v_2\\
v_2&=r_2+\gamma (r_3+\gamma r_4+...)=r_1+\gamma v_3\\
...
\end{aligned}
$$
不同状态出发的trajectory得到的return (ex.$v_1$)，依赖于从其他state出发得到的return (ex.$v_2$)，Boostrapping!
接下来，怎么求解 $v$ 呢？将上式写成矩阵的格式：
$$
\underbrace{\begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ v_4 \end{bmatrix}}_{\mathbf{v}} = \begin{bmatrix} r_1 \\ r_2 \\ r_3 \\ r_4 \end{bmatrix} + \begin{bmatrix} \gamma v_2 \\ \gamma v_3 \\ \gamma v_4 \\ \gamma v_1 \end{bmatrix} = \underbrace{\begin{bmatrix} r_1 \\ r_2 \\ r_3 \\ r_4 \end{bmatrix}}_{\mathbf{r}} + \gamma \underbrace{\begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \end{bmatrix}}_{\mathbf{P}} \underbrace{\begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ v_4 \end{bmatrix}}_{\mathbf{v}}
$$
which can be rewritten and solve as:
$$
\begin{aligned}
\mathbf{v}&=\mathbf{r}+\gamma\mathbf{Pv}\\
(\mathbf{I}-\gamma\mathbf{P})\mathbf{v}&=\mathbf{r}\\
\mathbf{v}&=(\mathbf{I}-\gamma\mathbf{P})^{-1}\mathbf{r}
\end{aligned}
$$
所以，$$\mathbf{v}=\mathbf{r}+\gamma\mathbf{Pv}$$
就是我们的 Bellman Equation 贝尔曼公式，但是 **only for specific deternimistic problem**
这个例子给出的idea：
1. 一个 state value实际上依赖于其他的state value
2. 给出了一个matrix-vector form来求解state value

 小练笔：![[Pasted image 20250119061041.png]]$$
 \begin{aligned}
 v_1&=0+\gamma v_3\\
 v_3&=1+\gamma v_4\\
 v_4&=1+\gamma v_4\\
 v_2&=1+\gamma v_4
 \end{aligned}
 $$只要知道 $\gamma$ 计算就很ez了，$\gamma$ 根据情况一般是 $\text{0.1 or 0.01}$
 下面正式的介绍 State Value：
$$
S_t \xrightarrow{A_t} R_{t+1}, S_{t+1}
$$
1. $S_t$ 当前的state
2. $A_t$ 采取的action
3. $R_{t+1}$ 得到的reward，有时候也会写成 $R_t$，因为得到的reward是action $A_t$带来的
4. $S_{t+1}$ 跳到的下一个state

这些steps都是由probability来决定的，假设我们知道model **(i.e., the probability distribution)**：
*   $S_t \rightarrow A_t$ is governed by $\pi(A_t = a | S_t = s)$
*   $S_t, A_t \rightarrow R_{t+1}$ is governed by $p(R_{t+1} = r | S_t = s, A_t = a)$
*   $S_t, A_t \rightarrow S_{t+1}$ is governed by $p(S_{t+1} = s' | S_t = s, A_t = a)$

这样，我们可以得到一个multi-step trajectory:
$$
S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} R_{t+2}, S_{t+2} \xrightarrow{A_{t+2}} R_{t+3}, ...
$$
The discounted return is:
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...
$$
- $\gamma \in[0,1)$ is discount rate
- $G_t$ is also a random variable，因为 $R_{t+1},R_{t+2}$ 都是random variable

***Definition of state value***: $G_t$ 的期望，或者说均值，被称为state-value function or just state value。
$$v_\pi(s)=\mathbb{E}[G_t|S_t=s]$$
其中 $s$ 是一个特定的 $s$，比如 $s_1$。
$\pi$ 也得是一个特定的策略，因为不同的策略对应不同的trajectory，也就有不同的return。
而 $v_\pi$ 就代表state value。

 那么，return和state value 有什么区别？
 **return针对单个trajectory，而state value是对多个trajectory的return求平均值。**
 这时候我们再看之前做的练习：![[Pasted image 20250119053259.png]]
 可以发现三张图分别是三个策略，那么state value就可求解：
$$
\begin{aligned}
v_{\pi_1}(s_1)&=0+\gamma1+\gamma^21+...\\
&=\frac{\gamma}{1-\gamma}\\
v_{\pi_2}(s_2)&=-1+\gamma1+\gamma^21+\gamma^31+...\\
&=-1+\frac{\gamma}{1-\gamma}\\
v_{\pi_3}(s_3)&=0.5(-1+\frac{\gamma}{1-\gamma})+0.5(\frac{\gamma}{1-\gamma})\\
&=-0.5+\frac{\gamma}{1-\gamma}.
\end{aligned}
$$
在 $v_{\pi_3}(s_3)$ 的求解中，就体现了expectation的作用，即把对应的probability乘在前面
越大的state value越好！

# 接下来讲Bellman Equation
像之前的 $v_n$ 一样，$G_t$ 也可以同样表示：
$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ..., \\
&= R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + ...), \\
&= R_{t+1} + \gamma G_{t+1},
\end{aligned}
$$
带入 $v_{\pi}(s)$ 的表达式：
$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}[G_t | S_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \mathbb{E}[R_{t+1} | S_t = s] + \gamma \mathbb{E}[G_{t+1} | S_t = s]
\end{aligned}
$$
接下来，我们可以进一步展开 $\mathbb{E}[R_{t+1} | S_t = s]$ 和 $\mathbb{E}[G_{t+1} | S_t = s]$
1. First, calculate the first term: $$
\begin{aligned}
\mathbb{E}[R_{t+1} | S_t = s] &= \sum_a \pi(a|s) \mathbb{E}[R_{t+1} | S_t = s, A_t = a] \\
&= \sum_a \pi(a|s) \sum_r p(r|s, a) r
\end{aligned}
$$
	- 第一项是什么呢？是mean of ***immediate rewards***
2. Then, calculate the sec:$$
\begin{aligned}
\mathbb{E}[G_{t+1} | S_t = s] &= \sum_{s'} \mathbb{E}[G_{t+1} | S_t = s, S_{t+1} = s'] p(s'|s) \\
&= \sum_{s'} \mathbb{E}[G_{t+1} | S_{t+1} = s'] p(s'|s) \\
&= \sum_{s'} v_\pi(s') p(s'|s) \\
&= \sum_{s'} v_\pi(s') \sum_a p(s'|s, a) \pi(a|s)
\end{aligned}
$$
	- tips：
		1. $s'$ 被称为 s prime
		2. 第一个等式为什么成立？***全期望公式 (Law of Total Expectation)***：*一个随机变量的期望值可以分解为在不同条件下该随机变量的条件期望值的加权和，权重是这些条件的概率。*
		3. 还是第一个式子，为什么没有考虑 $S_{t+2},S_{t+3},...$ 当成condition来考虑？不是没有考虑，而是包含在那些 $S_{t+1}$ 中了。
		4. 第一个等式为什么等于第二个等式？因为Markov property，无记忆性使得不需要考虑历史。
带入两项得：
$$
\begin{align*}
v_\pi(s) &= \mathbb{E}[R_{t+1} | S_t = s] + \gamma \mathbb{E}[G_{t+1} | S_t = s], \\
&= \underbrace{\sum_a \pi(a|s) \sum_r p(r|s, a)r}_{\text{mean of immediate rewards}} + \gamma \underbrace{\sum_a \pi(a|s) \sum_{s'} p(s'|s, a) v_\pi(s')}_{\text{mean of future rewards}}, \\
&= \sum_a \pi(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a) v_\pi(s') \right], \quad \forall s \in \mathcal{S}.
\end{align*}
$$
***这就是大名鼎鼎的Bellman Equation，它构建起了不同state value之间的联系***
- 贝尔曼公式是依赖于policy的。$\pi(a|s)$ 是一个已知概率，求解bellman 方程就叫做policy evaluation，因为$v_\pi$ 的高低代表着policy的好坏。
- $p(r|s,a)$ 和 $p(s'|s,a)$ 就是 dynamic model，类似于深度学习中的权重和偏置，是让机器去学习的部分。

通过ex来巩固理解：![[Pasted image 20250119112127.png]]
使用bellman equation来求解图中所有的state value：
以 $v_\pi(s_1)$ 为例，$\pi(a=a_3|s_1)=1$ 且 $\pi(a\neq a_3|s_1)=0$
$r=0$, $p(r=0|s_1,a_3) = p(s'=s_3|s_1,a_3) = 1$
可得：
$$v_\pi(s_1)=0+\gamma v_\pi(s_3)$$*这就是它的bellman equation表达式*
使用相同方法也可以列出$s_2, s_3, s_4$ 的表达式，令 $\gamma = 0.9$，有：
$$
\begin{align*}
v_\pi(s_4) &= \frac{1}{1 - 0.9} = 10, \\
v_\pi(s_3) &= \frac{1}{1 - 0.9} = 10, \\
v_\pi(s_2) &= \frac{1}{1 - 0.9} = 10, \\
v_\pi(s_1) &= \frac{0.9}{1 - 0.9} = 9.
\end{align*}
$$这些state value越大，代表往这个state越有价值。
接下来为了求解bellman equation，将其转换为matrix-vector form
首先简化bellman equation：
$$
v_\pi(s) = \underbrace{r_\pi(s)}_{\text{E of imm rewards}} + \gamma \underbrace{\sum_{s'} p_\pi(s'|s) v_\pi(s')}_{\text{E of future rewards}}
$$这其实是在推导bellman equation出现过的，没有完全展开的形态
因为要把所有状态放在一起，所以要标号：$$v_\pi(s_i) = r_\pi(s_i) + \gamma \sum_{s_j} p_\pi(s_j|s_i) v_\pi(s_j)
$$其中，$s_i(i=1,...,n)$
$v_\pi = [v_\pi(s_1), \dots, v_\pi(s_n)]^T \in \mathbb{R}^n$
$r_\pi = [r_\pi(s_1), \dots, r_\pi(s_n)]^T \in \mathbb{R}^n$
$P_\pi \in \mathbb{R}^{n \times n}, \text{ where } [P_\pi]_{ij} = p_\pi(s_j|s_i)$

这样就可以写出matrxi-vector form：
$$v_\pi=r_\pi+\gamma P_\pi v_\pi$$
拿4个state来举例子：
$$
\underbrace{
\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}
}_{\text{v}_\pi}
=
\underbrace{
\begin{bmatrix}
r_\pi(s_1) \\
r_\pi(s_2) \\
r_\pi(s_3) \\
r_\pi(s_4)
\end{bmatrix}
}_{\text{r}_\pi}
+ \gamma
\underbrace{
\begin{bmatrix}
p_\pi(s_1|s_1) & p_\pi(s_2|s_1) & p_\pi(s_3|s_1) & p_\pi(s_4|s_1) \\
p_\pi(s_1|s_2) & p_\pi(s_2|s_2) & p_\pi(s_3|s_2) & p_\pi(s_4|s_2) \\
p_\pi(s_1|s_3) & p_\pi(s_2|s_3) & p_\pi(s_3|s_3) & p_\pi(s_4|s_3) \\
p_\pi(s_1|s_4) & p_\pi(s_2|s_4) & p_\pi(s_3|s_4) & p_\pi(s_4|s_4)
\end{bmatrix}
}_{\text{P}_\pi}
\underbrace{
\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}
}_{\text{v}_\pi}
$$
具体来讲：![[Pasted image 20250119121043.png]]
再来一个更为复杂的例子：![[Pasted image 20250119121315.png]]
两种解决matrix-vector form的bellman equation的方法，
1. 第一种，矩阵运算：$$
\begin{aligned}
\mathbf{v}&=\mathbf{r}+\gamma\mathbf{Pv}\\
(\mathbf{I}-\gamma\mathbf{P})\mathbf{v}&=\mathbf{r}\\
\mathbf{v}&=(\mathbf{I}-\gamma\mathbf{P})^{-1}\mathbf{r}
\end{aligned}
$$但是当矩阵非常大的时候，矩阵求逆能算死人，实际当中没有人用
2. 第二种，迭代算法：$$v_{k+1}=r_\pi+\gamma P_\pi v_k$$
	- 计算方法，将一个随机值 $v_0$ 带入公式，得到 $v_1$ ，再次带入，得到 $v_2$, ......最终得到 $v_k$ 会收敛到 $v_\pi$ 
	- 证明方法不再给出。

# 下面是Action Value
定义：从一个state开始采取一个action后得到的平均回报
**Mathematically**:$$q_\pi(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$看公式不难发现，和state value差别不大，只不过condition里多了一个 $A_t$
**Depending Arguments**：
1. state-action pair $(s,a)$
2. 策略 $\pi$

state value和action value之间存在紧密的联系，从state value的初阶展开式可见一斑：
$$
\underbrace{\mathbb{E}[G_t | S_t = s]}_{v_\pi(s)} = \sum_a \underbrace{\mathbb{E}[G_t | S_t = s, A_t = a]}_{q_\pi(s,a)} \pi(a|s)
$$
$$
v_\pi(s) = \sum_a \pi(a|s) q_\pi(s, a)
$$
由Bellman Equation，不难得出 $q_{\pi}(s,a)$的表达式 $$
\begin{aligned}
v_\pi(s) 
&= \sum_a \pi(a|s) \underbrace{ \left[ \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v_\pi(s') \right] }_{q_\pi(s,a)}
\end{aligned}
$$
即：$$
q_\pi(s, a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s, a) v_\pi\color{lime}(s')
$$这个式子说明了只要知道每一个state的state value，就能知道所有的action value
同时，如果知道了一个state的所有的action value，将它们求平均就能得到这个state的state value

# 总结
实际上只讲了4件事：
1. State value：$$v_\pi(s)=\mathbb{E}[G_t|S_t=s]$$这是State-value Function的Bellman Expectation Equation，后面还会见到这种形式（经过数学变化）：$$V^\pi(s) = \mathbb{E}_\pi \left[ r(s, a) + \gamma V^\pi(s') \right]$$
2. Action value：$$q_\pi(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]$$这是Action-value Function的Bellman Expectation Equation，后面会见到这种形式：$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ r(s, a) + \gamma Q^\pi(s', a') \right]$$
3. Bellman Equation：$$\begin{aligned}v_\pi(s) &=\sum_a\pi(a|s)\underbrace{\left[\sum_rp(r|s,a)r+\gamma\sum_{s'} p(s'|s,a)v_\pi(s') \right] }_{q_\pi(s,a)}\\&= \sum_a \pi(a|s)q_\pi(s,a)\end{aligned}$$
4. How to slove BE：
	1. Matrix-Vector form of Bellman Equation (Closed-form solution)：$$v_\pi=r_\pi+\gamma P_\pi v_\pi$$
	2. Iterative solution：$$v_{k+1}=r_\pi+\gamma P_\pi v_k$$

# The END