这是一个在AC方法基础上改进的模型。
AC方法的Actor公式可以进行一个变换，变成A2C的公式。
$$
\begin{aligned}
∇_{θ}J(θ) &= E_{S∼η,A∼π}\left[∇_{θ}\ln π(A|S,θ_{t})q_{π}(S,A)\right] \\
&= E_{S∼η,A∼π}\left[∇_{θ}\ln π(A|S,θ_{t})(q_{π}(S,A) - b(S))\right]
\end{aligned}
$$
为什么这个变换成立呢？我们能证明：$$E_{S∼η,A∼π}\left[∇_{θ}\ln π(A|S,θ_{t})b(S)\right] = 0$$如下
$$
\begin{aligned}
E_{S∼η,A∼π}\left[∇_{θ}\ln π(A|S,θ_{t})b(S)\right] 
&= \sum_{S}η(S) \sum_{A}π(A|S,θ_t) ∇_{θ}\ln π(A|S,θ_t) b(S) \\
&= \sum_{S}η(S)b(S) \sum_{A}π(A|S,θ_t) \frac{∇_{θ}π(A|S,θ_t)}{π(A|S,θ_t)} \\
&= \sum_{S}η(S)b(S) ∇_{θ} \sum_{A}π(A|S,θ_t) \\
&= \sum_{S}η(S)b(S) ∇_{θ}(1) \\
&= 0
\end{aligned}
$$
那么为什么要引入这个baseline $b(S)$ 呢？其实是让采样能够更加快速的逼近平均值，也就是让sample的方差最小，这样每一次采的样本都是接近期望的。
来看两个例子：
图1，图2
下面给出最优的 $b(S)$ 的公式，使用这个 $b(S)$ 能让数据的方差达到最小
$$  
b^*(s) = \frac{  
\mathbb{E}_{A \sim \pi} \left[ \lVert \nabla_{\theta} \ln \pi(A|s, \theta_t) \rVert^2 \cdot q(s, A) \right]  
}{  
\mathbb{E}_{A \sim \pi} \left[ \lVert \nabla_{\theta} \ln \pi(A|s, \theta_t) \rVert^2 \right]  
}  
$$
但是在实践中我们不使用这个，因为它太复杂了。
在实践中，一般会把权重去掉，也就是将 $\color{violet}\lVert \nabla_{\theta} \ln \pi(A|s, \theta_t) \rVert^2$ 去掉。剩下：$$b(s)=\mathbb{E}_{A\sim\pi}\left[q(s,A)\right]=v_\pi(s)$$也就是状态值函数。
带回原来的式子，也就有了：$$E_{S∼η,A∼π}\left[∇_{θ}\ln π(A|S,θ_{t})(q_{π}(S,A) - b(S))\right]$$后面的那一坨就是优势函数 $\text{Advantage Function}$ ，用 $A(s,a)$ 表示。
$$E_{S∼η,A∼π}\left[∇_{θ}\ln π(A|S,θ_{t})A(S,A)\right]$$
为什么叫优势函数？其实很好理解，$v_\pi(s)$ 是 $q_\pi(s,a)$ 的期望（均值），
1. 如果一个 $q$ 比均值大，那么说明这个动作是好的，那么在梯度策略中后面的一项就是正的，也就是在让参数 $\theta$ 向最大化该动作的概率的方向优化。也就是梯度上升。
2. 如果一个 $q$ 比均值小，那么这个动作不太好，后面一项就是负数，也就是做一个梯度下降，最小化这个动作的概率。
$$
\begin{aligned}
\theta_{t+1} 
&= \theta_t + \alpha \nabla_{\theta} \ln \pi(a_t|S_t, \theta_t) \delta_t(S_t, a_t) \\
&= \theta_t + \alpha \frac{\nabla_{\theta} \pi(a_t|S_t, \theta_t)}{\pi(a_t|S_t, \theta_t)} \delta_t(S_t, a_t) \\
&= \theta_t + \alpha \underbrace{\left( \frac{\delta_t(S_t, a_t)}{\pi(a_t|S_t, \theta_t)} \right)}_{\text{step size }(\beta_t)} \nabla_{\theta} \pi(a_t|S_t, \theta_t) \\
\end{aligned}
$$这样我们就得到了新的actor。
我们仍然可以使用AC中的critic对 $q$ 值进行更新，但是在这里可以引入TD target，使用TD target替换掉 $q$ 的估计 ( $q_t$ )。 
也就是：$$\theta_t + \alpha \underbrace{\left( \frac{r_{t+1}+\gamma v_\pi(s_{t+1})-v_\pi(s_t)}{\pi(a_t|S_t, \theta_t)} \right)}_{\text{step size }(\beta_t)} \nabla_{\theta} \pi(a_t|S_t, \theta_t)$$这样做明显的好处就是不再需要维护另外一个网络去储存 $q$ 的估计，只需要一个网络去估计 $v_\pi(s)$ 就好了。
下面给出伪代码：

**Aim:** Search for an optimal policy by maximizing $J(\theta)$.
At time step $t$ in each episode, do:  
1. Generate $a_t$ following $\pi(a|s_t, \theta_t)$  
   Observe $r_{t+1},\ s_{t+1}$

TD error (advantage function):
$$  
\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)  
$$
Critic (value update):  
$$  
w_{t+1} = w_t + \alpha_w \delta_t \nabla_w v(s_t, w_t)  
$$
Actor (policy update):  
$$  
\theta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla_\theta \ln \pi(a_t|s_t, \theta_t)  
$$
因为使用了TD error，所以A2C也被称为TD Actor-Critic。

