我们先看一下mean是怎么求解的（然后介绍一种新的求 $\text{mean}$ 的算法）：$$w_{k+1} = \frac{1}{k} \sum_{i=1}^{k} x_i, \quad k = 1, 2, ...$$递推一步，有：
$$w_k = \frac{1}{k-1} \sum_{i=1}^{k-1} x_i, \quad k = 2, 3, ...$$Then, $w_{k+1}$ can be expressed in terms of $w_k$ as$$\begin{aligned} w_{k+1} &= \frac{1}{k} \sum_{i=1}^{k} x_i = \frac{1}{k} \left( \sum_{i=1}^{k-1} x_i + x_k \right) \\ &= \frac{1}{k} \left( (k-1)w_k + x_k \right) = w_k - \frac{1}{k}(w_k - x_k). \end{aligned}$$Therefore, we obtain the following iterative algorithm:$$\color{lime}w_{k+1} = w_k - \frac{1}{k}(w_k - x_k).$$再将这个式子推广得到，其中 $\alpha_k$ 不再是 $\frac{1}{k}$ ，而可以是一个常数。事实证明即使是常数（满足三个条件）也能向着$\text{mean}$收敛：$$\color{olive}w_{k+1} = w_k - \alpha_k(w_k - x_k).$$
讲完这个，我们回过来定义一下什么是Stochastic Approximation (SA) 算法。
SA算法是一大类算法，是**随机迭代算法**的统称，上面给出的求解$\text{mean}$的算法就是一种SA算法。每一个 $x_k$ 都是一个随机变量，而且算式是迭代式的。
SA算法通常用于求解一个方程的解或者优化问题。
1. Stochastic随机：涉及到对变量的随机采样
2. Iterative迭代：迭代式算法
***SA的优势***：不需要知道自己求解的方程是什么，***可以视为黑箱***，即不需要知道方程本身，方程的导数和梯度，各种性质等。

讲完了SA的概念，我们来讲一个SA领域内非常具有开创性的工作：
# Robbins-Monro algorithm
我们知道，求解一个类似：$$g(w)=0$$是一个非常普遍的问题。而Robbins-Monro算法就是为了求解这类方程而生的。
正常情况肯定有解法，如果我知道 $g(w)$ 的表达式。
但是Robbins-Monro解决的是一种非常有意思的情况，那就是 **我不知道方程表达式** 的情况。可能会问，这种情况真的存在吗？不知道方程却要求解？
想想FC神经网络，就是一个 $y=g(w)$ ，我们只能输入一个 $w$ 得到一个输出 $y$ ，也就是一个黑箱情况，我们没法知道神经网络的表达式（不局限于FCNN）。这就是一个经典的例子。
像这种 ***不知道待解方程表达式的情况*** 就是Robbins-Monro算法的用武之地。下面我们正式介绍Robbins-Monro：$$w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k), \quad k = 1, 2, 3, ...$$where
*   $w_k$ is the $k$th estimate of the root
*   $\tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k$ is the $k$th noisy observation (是对 $g$ 带有噪音的观测)
*   $a_k$ is a positive coefficient.
现在我们来讨论一下**为什么这个迭代算法最终能收敛到方程的解**：
先给出一个直观的证明：
- $g(w) = \tanh(w - 1)$
- The true root of $g(w) = 0$ is $w^* = 1$.
- Parameters: $w_1 = 3$, $a_k = 1/k$, $\eta_k \equiv 0$ (no noise for the sake of simplicity)
现在使用Robbins-Monro算法来逼近解。
下图使用Robbins-Monro算法，初始猜测 $x_0=5$，步长 $a=2$，最大迭代次数 $\text{max iter} =100$ ![[tanh(x-1).png]]
直觉上的证明 $w_{k+1}$ is closer to $w^*$ than $w_k$.
- When $w_k > w^*$, we have $g(w_k) > 0$. Then,
  $w_{k+1} = w_k - a_k g(w_k) < w_k$ and hence $w_{k+1}$ is closer to $w^*$ than $w_k$.
- When $w_k < w^*$, we have $g(w_k) < 0$. Then,
  $w_{k+1} = w_k - a_k g(w_k) > w_k$ and $w_{k+1}$ is closer to $w^*$ than $w_k$.
严谨的数学证明：
In the Robbins-Monro algorithm, if
1) $0 < c_1 \leq \nabla_w g(w) \leq c_2$ for all $w$;
2) $\sum_{k=1}^{\infty} a_k = \infty$ and $\sum_{k=1}^{\infty} a_k^2 < \infty$;
3) $\mathbb{E}[\eta_k | \mathcal{H}_k] = 0$ and $\mathbb{E}[\eta_k^2 | \mathcal{H}_k] < \infty$;

where $\mathcal{H}_k = \{w_k, w_{k-1}, ...\}$, then $w_k$ converges with probability 1 (w.p.1) to the root $w^*$ satisfying $g(w^*) = 0$.

第一个条件 $0 < c_1 \leq \nabla_w g(w) \leq c_2$ for all $w$ 什么意思呢？
1. 首先要求 $\nabla_w g(w)$ 是**有界**的，不能在某一个 $w$ 趋向于无穷
2. 其次要求 $\nabla_w g(w)$ 是大于零的，也就是函数必须是**递增**的
第二个条件，$\sum_{k=1}^{\infty} a_k = \infty$ and $\sum_{k=1}^{\infty} a_k^2 < \infty$：
1. 首先要求 $a_k$ 收敛的不能太快，如果在第一、第二个 $a_k$ 就收敛为0了，那肯定相加不等于无穷。
2. 其次要求 $a_k$ 最终收敛到0，当 $k \rightarrow \infty$ 时。
	- 为什么？如果 $a_k$ 是个非零值，那么无穷个 $a_k$ 相加必定在 $k \rightarrow \infty$ 时趋近于无穷。所以当 $k \rightarrow \infty$ 时$a_k$ 收敛到0。
第三个条件，$\mathbb{E}[\eta_k | \mathcal{H}_k] = 0$ and $\mathbb{E}[\eta_k^2 | \mathcal{H}_k] < \infty$：
1. $\eta$ 的$\text{mean}$一定是0
2. $\eta$ 的variance一定是有界的。
一个特殊但常见的例子是：{$\eta_k$} 是一个独立同分布 (**iid**) 的随机序列，满足 $\mathbb{E}[\eta_k] = 0$ 且 $\mathbb{E}[\eta_k^2] < \infty$。其中观测误差 $\eta_k$ 不需要是高斯分布。

下面我们详细关注一下第二个条件，因为这个条件在后面的*时序差分方法*中也需要，论文中也频繁的出现。
首先讨论第二个不等式：$$\sum_{k=1}^{\infty} a_k^2 < \infty$$将前述式子移项得到：
$$w_{k+1} - w_k = -a_k\tilde{g}(w_k, \eta_k)$$
$a_k \rightarrow 0 \Rightarrow a_k\tilde{g}(w_k, \eta_k) \rightarrow 0 \Rightarrow \color{violet}w_{k+1} - w_k \rightarrow 0$。
我们需要 $w_{k+1} - w_k \rightarrow 0$ 这个事实，所以需要 $a_k \rightarrow 0$。
再看第一个等式：$$\sum_{k=1}^{\infty} a_k = \infty$$总结一下，有 $w_2 = w_1 - a_1\tilde{g}(w_1, \eta_1)$，$w_3 = w_2 - a_2\tilde{g}(w_2, \eta_2)$，...，以此类推得到 $w_{k+1} = w_k - a_k\tilde{g}(w_k, \eta_k)$，这可以推导出：$$w_\infty - w_1 = \sum_{k=1}^{\infty} a_k\tilde{g}(w_k, \eta_k)$$如果 $w_k$ 收敛，则 $w_\infty = w^*$。如果 $\sum_{k=1}^{\infty} a_k < \infty$，那么 $\sum_{k=1}^{\infty} a_k\tilde{g}(w_k, \eta_k)$ 可能是有界的。那么，如果初始猜测 $w_1$ 选择得离 $w^*$ 非常远，上述等式将不再成立。所以我们必须要求 $\sum_{k=1}^{\infty} a_k = \infty$，**这保证了我们能够随便选初始猜测值。** 也就是初始猜测 $w_1$ 选择得离 $w^*$ 多远都可以。
那么怎么选择这个 $\alpha$ 值呢？有两个经典选择：
One typical sequence is $a_k = \frac{1}{k}$.
* 证明：
$$\lim_{n\rightarrow\infty} \left( \sum_{k=1}^{n} \frac{1}{k} - \ln n \right) = \kappa$$where $\kappa \approx 0.577$ is called the Euler-Mascheroni constant (also called Euler's constant).
* 另外一个是 $1/k^2$，证明：
$$\sum_{k=1}^{\infty} \frac{1}{k^2} = \frac{\pi^2}{6} < \infty$$The limit $\sum_{k=1}^{\infty} 1/k^2$ also has a specific name in the number theory: Basel problem.
但是在实际的应用场景中，我们一般不使用 $\frac{1}{k}$ 或者 $\frac{1}{k^2}$ 作为 $\alpha$，我们一般选择一个很小的常数作为 $\alpha$，因为我们希望后面的数据也能得到利用。
下面我们Recall一下一开始讲的求 $\text{mean}$ 的算法，讨论一下如果 $\alpha$ 不是 $1/k$ 的情况，还能不能收敛：$$w_{k+1} = w_k + \alpha_k(w_k - x_k).$$思路就是证明此时它是一个Robbins-Monro算法。
如果想要用Robbins-Monro算法求 $\text{mean}$，我们首先想到的就是构建等式：$$g(w)=0$$且在迭代计算中，我们希望 $\{w_1,w_2,...,w_n\}$ 最终能趋近于随机变量 $x$ 的期望 $\mathbb{E}[X]$。也就是 $w^*=w_\infty=\mathbb{E}[X]$。此时等式已经出现了，就是 $w=\mathbb{E}[X]$，我们令 $$g(w)=w-\mathbb{E}[X]$$expectation实际上是不知道的，但是我们可以对 $x$ 进行采样。
所以，我们获得的带噪音测量 (**Observation**) 是（为什么它是带噪音观测？因为x很明显不是 $\mathbb{E}[X]$，只是用x来近似（观测它））： $$\tilde{g}(w,x)=w-x$$其中 $\tilde{g}(w,x)$ 也可以写成 $g(w)$ 加上一个噪声：$$\begin{aligned}
\tilde{g}(w,\eta) &= w - x = w - x + \mathbb{E}[X] - \mathbb{E}[X] \\
&= (w - \mathbb{E}[X]) + (\mathbb{E}[X] - x) \doteq g(w) + \eta,
\end{aligned}$$最后，我们将 $\tilde{g}$ 的值带入Robbins-Monro公式：$$w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, \eta_k) = w_k - \alpha_k (w_k - x_k),$$***综上，我们证明了这个mean estimation的算法是一个Robbins-Monro算法。***
***只要 $\alpha_k$ 满足和等于无穷，平方和小于无穷，迭代算法就能收敛。***

Robbins-Monro算法和使用这个算法求解 $\text{mean estimation}$ 的算法已经讲完了，*至于该算法收敛的**严格数学证明**，可以看看一下论证思路*：
[[Robbins-Monro收敛性证明]]
