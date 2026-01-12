
`这一篇证明分为三个点，写的有点乱，但是环环相扣。建议先全部读一遍，再单独看一遍Dvoretzky's Theorem。`
[[#Supermartingale Convergence Theorem]]
[[#Lyapunov函数]]
[[#Dvoretzky's Theorem]]


# Supermartingale Convergence Theorem
#### 1. 超鞅（Supermartingale）的定义
在概率论中，**超鞅**是一类随机过程，其未来值的条件期望不超过当前值。具体定义如下：
设$\{X_k\}$是一个随机过程，$\{ \mathcal{F}_k \}$是其适应的σ-代数（即历史信息集合）。若满足：
$$
E[X_{k+1} | \mathcal{F}_k] \leq X_k \quad \text{（几乎必然）},
$$则称$\{X_k\}$为**超鞅**。
- **直观理解**：超鞅的“能量”（$X_k$）随时间推移趋于减少或保持稳定，不会无界增长。
#### 2. 超鞅收敛定理（Supermartingale Convergence Theorem）
**定理内容**：
若非负超鞅$\{X_k\}$满足 $X_k \geq 0$ 对任意$k$成立，则存在一个随机变量$X_\infty \geq 0$，使得：
$$
X_k \to X_\infty \quad \text{几乎必然},
$$
且 $E[X_\infty] \leq E[X_0]$（期望值不增）。
- **核心结论**：非负超鞅几乎必然收敛到一个有限极限。
#### 3. 在Dvoretzky定理中的应用步骤
在Dvoretzky定理的证明中，构造Lyapunov函数 $V_k = \Delta_k^2$，并推导出条件期望不等式：
$$
\color{lime}E[V_{k+1} | H_k] \leq (1 - 2\alpha_k)V_k + C\beta_k^2. \tag{1}
$$
接下来将证明$V_k$满足超鞅的条件，并应用超鞅收敛定理。
##### 步骤1：验证$V_k$为非负超鞅
1. **非负性**：$V_k = \Delta_k^2 \geq 0$ 显然成立。
2. **超鞅性质**：需证明 $E[V_{k+1} | H_k] \leq V_k - 2\alpha_k V_k + C\beta_k^2$。
   - 由式(1)，直接得到：$$
     E[V_{k+1} | H_k] \leq V_k - 2\alpha_k V_k + C\beta_k^2.
     $$
   - 若进一步构造辅助过程 $W_k = V_k + \sum_{i=k}^\infty C\beta_i^2$（假设$\sum \beta_i^2 < \infty$），则：$$
     E[W_{k+1} | H_k] \leq W_k - 2\alpha_k V_k.
     $$
     由于 $2\alpha_k V_k \geq 0$，这表明$W_k$是一个非负超鞅。
##### 步骤2：应用超鞅收敛定理
根据超鞅收敛定理，非负超鞅$W_k$几乎必然收敛到某个有限随机变量$W_\infty$，即：
$$
W_k \to W_\infty \quad \text{（几乎必然）}.
$$
由于$V_k = W_k - \sum_{i=k}^\infty C\beta_i^2$，且$\sum_{i=k}^\infty C\beta_i^2 \to 0$（因$\sum \beta_i^2 < \infty$），可得：
$$
V_k \to W_\infty \quad \text{（几乎必然）}.
$$
##### 步骤3：证明极限$W_\infty = 0$
从式(1)的期望不等式出发，递推可得：
$$
E[V_{k+1}] \leq E[V_k] - 2E[\alpha_k V_k] + C\beta_k^2.
$$
对两边求和：
$$
\sum_{k=1}^\infty E[\alpha_k V_k] \leq \frac{1}{2}\left( E[V_1] + C\sum_{k=1}^\infty \beta_k^2 \right) < \infty.
$$
由于$\sum \alpha_k = \infty$，若极限$W_\infty > 0$，则存在某个时刻后$V_k \geq \epsilon > 0$，导致$\sum \alpha_k V_k \geq \epsilon \sum \alpha_k = \infty$，与$\sum E[\alpha_k V_k] < \infty$矛盾。
因此，**几乎必然有$W_\infty = 0$**，即：
$$
V_k \to 0 \quad \text{（几乎必然）} \implies \Delta_k \to 0 \quad \text{（几乎必然）}.
$$
#### 4. 关键条件的作用
- **$\sum \alpha_k = \infty$**：确保长期衰减的累积效应足够强，驱动$V_k$最终趋于0。
- **$\sum \alpha_k^2 < \infty$ 和 $\sum \beta_k^2 < \infty$**：限制步长和噪声的波动，避免扰动破坏收敛性。
- **噪声的鞅性质**（$E[\eta_k | H_k] = 0$ 和方差有界）：保证随机项的期望影响为零，且波动可控。
#### 5. 直观总结
通过构造Lyapunov函数$V_k = \Delta_k^2$，将其转化为超鞅，并利用超鞅收敛定理：
1. **能量递减**：条件期望不等式表明“能量”$V_k$随时间递减（尽管存在噪声）。
2. **噪声压制**：$\sum \beta_k^2 < \infty$确保噪声的总能量有限，最终被确定性衰减项压制。
3. **极限唯一性**：级数条件$\sum \alpha_k = \infty$迫使极限必须为0，排除其他可能性。
这一过程将复杂的随机收敛问题转化为对能量函数的分析，体现了Lyapunov方法和鞅理论的强大结合。


# Lyapunov函数
#### 1. Lyapunov函数的定义
Lyapunov函数是用于分析系统稳定性的数学工具，其核心思想是构造一个“能量函数” $V_k$，满足以下性质：
1. **非负性**：$V_k \geq 0$（能量非负）。
2. **递减性**：$E[V_{k+1} | \text{历史信息}] \leq V_k$（能量随时间递减）。
3. **收敛性**：若$V_k \to 0$，则系统状态趋于目标（如误差$\Delta_k \to 0$）。
在随机过程中，满足上述条件的$V_k$称为**超鞅（Supermartingale）**，其期望值随时间递减。
### Dvoretzky定理中的Lyapunov函数 $V_k = \Delta_k^2$
#### 1. 构造目的
在Dvoretzky定理的证明中，选择$V_k = \Delta_k^2$作为Lyapunov函数，其作用是将随机过程$\Delta_k$的收敛性问题转化为分析$V_k$的收敛性。
- **直观解释**：平方函数$\Delta_k^2$直接衡量误差的大小（非负性），且平方操作便于数学处理。
#### 2. 条件期望分析
从递推公式$\Delta_{k+1} = (1 - \alpha_k)\Delta_k + \beta_k \eta_k$出发，计算$V_{k+1} = \Delta_{k+1}^2$的条件期望：
$$
E[V_{k+1} | H_k] = E\left[ \left( (1 - \alpha_k)\Delta_k + \beta_k \eta_k \right)^2 \Big| H_k \right].
$$
展开平方项并利用定理条件：
$$
E[V_{k+1} | H_k] = (1 - \alpha_k)^2 \Delta_k^2 + 2(1 - \alpha_k)\beta_k \Delta_k \underbrace{E[\eta_k | H_k]}_{=0} + \beta_k^2 \underbrace{E[\eta_k^2 | H_k]}_{\leq C}.
$$
由于噪声项$\eta_k$的条件均值为零（条件(b)），中间项消失，最终得到：
$$
E[V_{k+1} | H_k] \leq (1 - 2\alpha_k + \alpha_k^2)V_k + C\beta_k^2.
$$
当$\alpha_k$较小时（例如$\alpha_k^2 \ll \alpha_k$），可近似为：
$$
\color{lime}E[V_{k+1} | H_k] \leq (1 - 2\alpha_k)V_k + C\beta_k^2.
$$
#### 3. 超鞅收敛定理的应用
通过构造不等式：
$$
E[V_{k+1} | H_k] \leq V_k - 2\alpha_k V_k + C\beta_k^2,
$$
可以证明$V_k$是一个**超鞅**（因其条件期望递减）。结合定理条件：
- $\sum \alpha_k = \infty$：确保长期累积衰减足够强，驱动$V_k \to 0$。
- $\sum \alpha_k^2 < \infty$和$\sum \beta_k^2 < \infty$：保证扰动项的总和有限。
根据**超鞅收敛定理**（Supermartingale Convergence Theorem），非负超鞅$V_k$几乎必然收敛到某个有限值，且在此极限下：
$$
\lim_{k \to \infty} V_k = 0 \quad \text{（几乎必然）},
$$
从而推出$\Delta_k \to 0$。
### 总结：Lyapunov函数的作用
1. **转化问题**：将复杂的随机过程收敛性问题转化为分析能量函数$V_k$的递减性。
2. **数学简化**：通过平方操作和条件期望展开，将噪声的影响分解为可控制的部分。
3. **收敛保证**：结合级数条件（$\sum \alpha_k = \infty$等）和超鞅定理，严格证明$\Delta_k \to 0$。
在Dvoretzky定理中，$V_k = \Delta_k^2$的构造是收敛性证明的核心工具，体现了Lyapunov方法在随机分析中的强大威力。


# Dvoretzky's Theorem
Dvoretzky收敛定理是随机逼近领域的重要结果，用于分析RM算法及强化学习算法的收敛性。以下是对该定理的逐步解释：
### Dvoretzky定理的核心思想
定理研究形如$$
\Delta_{k+1} = (1 - \alpha_k)\Delta_k + \beta_k \eta_k
$$的随机过程，目标是证明当满足一定条件时，$\Delta_k$ **几乎必然（almost surely）收敛到0**。其核心是通过控制确定性衰减项 $(1 - \alpha_k)\Delta_k$ 和随机噪声项 $\beta_k \eta_k$ 的影响，确保整体过程收敛。
$\Delta_k$​：*状态误差变量*
- **含义**：表示第kk步的误差或偏差，通常是当前估计值与目标值（如最优解）的差异。
    - **在RM算法中**：可能对应参数估计误差，例如$\Delta_k=θ_k−θ^∗$，其中$θ^∗$是待收敛的最优参数。
- **作用**：作为迭代过程的核心变量，其收敛性（趋于0）是定理的目标。
$\beta_k$：*噪声系数*
- **含义**：非负的系数（$\beta_k≥0$），调节随机噪声项 $η_k$ 的强度。
- **作用**：
    - **控制噪声影响**：通过乘积项 $\beta_kη_k$ ​将噪声引入迭代过程。
    - **收敛条件**：$\sum^{k=1}_\infty β_k^2<\infty$，确保噪声的累积能量有限，避免长期扰动。
### 定理条件解析
定理要求以下两个条件：
#### **条件 (a)：级数收敛性**
1.  $\sum_{k=1}^\infty \alpha_k = \infty$
    - **作用**：确保累积衰减足够强，驱动$\Delta_k$趋向0。例如，若$\alpha_k = 1/k$，则级数发散，但每个$\alpha_k$逐渐减小。
2.  $\sum_{k=1}^\infty \alpha_k^2 < \infty$ 和 $\sum_{k=1}^\infty \beta_k^2 < \infty$
    - **作用**：控制步长和噪声系数的波动，避免随机扰动过大。例如，$\alpha_k = 1/k$满足$\sum \alpha_k^2 = \pi^2/6 < \infty$。
    - *注*：“uniformly almost surely”指这些级数的收敛性在几乎每个样本路径上成立，*即使$\alpha_k, \beta_k$是依赖于历史的随机变量。*
#### **条件 (b)：噪声的鞅性质**
1.  $$E[\eta_k | H_k] = 0$$
    - **作用**：噪声$\eta_k$在给定历史$H_k$下均值为0，即无偏，不引入系统性偏差。
2.  $$E[\eta_k^2 | H_k] \leq C$$
    - **作用**：噪声的方差被常数$C$控制，防止异常扰动。
$H_k$​：*历史信息集合*
- **含义**：包含截至第 $k$ 步的所有历史信息，即：
    $H_k=\{Δ_k,Δ_{k−1},…,η_{k−1},…,α_{k−1},β_{k−1},… \}.$
- **作用**：
    - 条件期望 $E[η_k∣H_k]$ 的依赖对象，确保噪声 $η_k$ 仅依赖于历史信息，而非未来。
    - 允许 $α_k$​ 和 $β_k$ ​是随机的（如自适应步长），但需满足定理条件。
### 定理的直观理解
1.  **确定性衰减项 $(1 - \alpha_k)\Delta_k$**
    - 当$\sum \alpha_k = \infty$时，累积衰减效应会将$\Delta_k$逐步压缩。例如，若$\alpha_k = 1/k$，则乘积$\prod_{k=1}^n (1 - \alpha_k)$趋于0（类似调和级数发散时的性质）。
2.  **随机噪声项 $\beta_k \eta_k$**
    - 由于$\sum \beta_k^2 < \infty$，噪声的累积能量有限；结合$\eta_k$的条件方差有界，噪声的影响被逐步抑制。
### 证明思路（简化版）
1.  **构造Lyapunov函数**
    定义$V_k = \Delta_k^2$，分析其条件期望：$$
    E[V_{k+1} | H_k] \leq (1 - 2\alpha_k + \alpha_k^2)V_k + C\beta_k^2.
    $$当$\alpha_k$较小时，近似为：$$
    E[V_{k+1} | H_k] \leq (1 - 2\alpha_k)V_k + C\beta_k^2.
    $$
2.  **递推不等式与收敛性**
    - 由于$\sum \alpha_k = \infty$， $V_k$的期望被指数级衰减。
    - $\sum \beta_k^2 < \infty$确保噪声项的累积影响有限。
    - 应用超鞅收敛定理（Supermartingale Convergence Theorem），可得$V_k \to 0$几乎必然。
### 与RM算法的联系
在RM算法中，更新规则为：$$
\theta_{k+1} = \theta_k + \alpha_k (g(\theta_k) + \eta_k),
$$对应$\Delta_k = \theta_k - \theta^*$，定理中的$\alpha_k$为步长，$\beta_k = \alpha_k$。Dvoretzky定理的条件转化为RM算法的经典收敛条件：
- $\sum \alpha_k = \infty$（充分探索）
- $\sum \alpha_k^2 < \infty$（抑制噪声）。
### 总结
Dvoretzky定理通过以下机制保证收敛：
1.  **衰减项**：$\sum \alpha_k = \infty$确保长期衰减效应。
2.  **噪声控制**：$\sum \beta_k^2 < \infty$和$\eta_k$的鞅性质抑制随机扰动。
3.  **数学工具**：Lyapunov函数与超鞅收敛定理的结合。

此定理为分析随机迭代算法（如RM算法、Q学习）提供了通用框架，是强化学习收敛性证明的基础工具。



 



