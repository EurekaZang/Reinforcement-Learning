MC Exploring starts是MC basic的一种扩展，或者说优化方案。前面提到MC basic的效率极低，所以提出了MC ES。它通过两方面显著提高了数据的利用率，且确保了机器算力利用率最大化。
首先我们要给出一个概念，叫做**visit**。
- 很好理解，**visit**发生于一个episode中：从一个 (state, action) pair出发的episode (ex. $\{(s_1,a)，(s_2,a)，(s_1,a)，(s_5,a)，(s_7,a)，(s_{11},a)\}$) 中我们可以计算这个episode 的 discounted return（g），计算公式如下：$$g=r_1+\gamma r_2+\gamma^2 r_3+...$$而这就是对这个episode中所有的 state-action pair做了一次visit。agent处于一个state然后选择了一个action被称为agent对这个pair的visit。
MC basic中使用的策略叫做 **"Initial-visit method"** 也就是生硬的计算每一个e的g然后估计q。
# 第一个改进点：更高效的利用数据
我们不难发现从一个s-a pair出发的e visit了很多其他的s-a pair，这说明一个e的g可以由从其他的s-a pair开始的e的g计算得来。
$$
\begin{aligned}
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} ...& &\text{[original episode]} \\
s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} ...& &\text{[episode starting from $(s_2, a_4)$]} \\
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} ...& &\text{[episode starting from $(s_1, a_2)$]} \\
s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} ...& &\text{[episode starting from $(s_2, a_3)$]} \\
s_5 \xrightarrow{a_1} ...& &\text{[episode starting from $(s_5, a_1)$]}
\end{aligned}
$$

Can estimate $q_\pi(s_1, a_2)$, $q_\pi(s_2, a_4)$, $q_\pi(s_2, a_3)$, $q_\pi(s_5, a_1)$,...
上面的策略叫做 **"Data-efficient Methods"** ，细分为两种：
1. First-visit method
	- 比如上面这个例子， $(s_1,a_2)$ 这个对一个e中visit了两次，我们只用第一次的visit来计算这个对的discounted return。第二次出现的时候不使用它来估计。
2. Every-visit method
	- 同一个对的每一次都进行一次估计。
这就实现了如何让数据被利用的更充分的方法。

# 第二个改进点：更高效的更新策略
state-action pair的收集部分也可以做优化，**在MC basic中我们要收集所有的episode，获得所有相应的g然后求average才能计算该state的所有action value (q)。** 但是收集所有的episode需要时间，而这个时间可以被优化掉。***MC exploring starts中只收集了第一条episode就使用它的g计算该state的所有action value (q)。*** 当然，这样计算精度不是很高，==但是事实证明这是更加有效的。==

# GPI Generalized policy iteration
GPI是一种思想或者框架，是所有具有 “PE”->“PI”->“PE”->... 结构策略的统称
最后再来看一下MC exploring starts的伪代码实现，有利于理顺逻辑：
**Initialization:** Initial guess $\pi_0$.
**Aim:** Search for an optimal policy.

For each episode, do
  *Episode generation:* Randomly select a starting state-action pair $(s_0, a_0)$ and ensure that all pairs can be possibly selected. Following the current policy, generate an episode of length $T$: $s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T$.
  *Policy evaluation and policy improvement:*
  Initialization: $g \leftarrow 0$
  For each step of the episode, $t = T - 1, T - 2, ..., 0$, do
    $g \leftarrow \gamma g + r_{t+1}$
    Use the **first-visit strategy**:
    If $(s_t, a_t)$ does not appear in $(s_0, a_0, s_1, a_1, ..., s_{t-1}, a_{t-1})$, then
      $\text{Returns}(s_t, a_t) \leftarrow \text{Returns}(s_t, a_t) + g$
      $q(s_t, a_t) = \text{average}(\text{Returns}(s_t, a_t))$
      $\pi(a|s_t) = 1$ if $a = \arg \max_a q(s_t, a)$ otherwise $\pi(a|s_t) = 0$

其中的 if 语句对应的是first-visit策略，也就是如果这个state-action pair是第一次出现，我才对其进行后续操作（计算g，计算return的均值来计算action value，挑选最大的action value对应的策略）
其中 $(s_0, a_0, s_1, a_1, ..., s_{t-1}, a_{t-1})$ 是past的时间步中visit过的s-a pair。

最后再来解释一下Exploring 和 starts是啥意思：
1. Exploring：指的是我从每一个 (s,a) 出发我都要得到一个episode，只有这样我才能保证每一个 (s,a) 都能计算相应所有的q值
2. Starts：从 (s,a) 开始生成一段episode成为starts，如果只是在从其他的 (s*,a*) 开始的episode中经过了该 (s,a) 就不叫starts，就叫做visit（**也就是** MC $\varepsilon$-greedy **采用的策略**）

不过在实践过程中 Exploring 这个条件是比较苛刻的，比如一个网格世界若想让机器人从每一个 (s,a) 出发就得把机器人搬过去，设置好程序，很麻烦。怎么把exploring starts这个条件去掉呢？这就引出了MC $\varepsilon$-greedy，且听下回分解。