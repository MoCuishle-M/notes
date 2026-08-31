# CCE（Cut Cross Entropy）原理、正确性证明与工程代价

## 1. 先给结论

CCE 不改变语言模型交叉熵的定义。它改变的是计算顺序：普通交叉熵先生成全部 token 对全部词表项的 logits，再执行 softmax 和交叉熵；CCE 沿词表维逐块生成 logits，一边计算 online log-sum-exp，一边提取正确类别的 logit，用完当前块后立即复用缓冲区。

在精确实数运算下，CCE 的前向 loss 和反向梯度都与普通交叉熵完全相同。证明依赖三个事实：

1. online log-sum-exp 合并后仍是完整词表上的 log-sum-exp；
2. 正确类别只属于一个词表块，因此扫描全部词表块一定能且只会提取一次正确类别 logit；
3. 矩阵乘法对词表分块可加，逐块计算的梯度可以无损地拼接或求和。

CCE 的主要收益是把与 logits 相关的峰值激活显存从随完整词表大小增长，改为随词表 tile 大小增长。它不会减少 LM-Head 权重、LM-Head 权重梯度或 Transformer 最后一层输出本身的显存，也不会减少理论矩阵乘法量。为了不保存完整 logits，反向需要重新计算 logits tile，因此要付出重计算、更多 kernel launch、事件同步和实现复杂度。

本文讲解的是当前 MindSpeed-MM 中的 NPU/Triton CCE 实现。它借鉴 Apple CCE 的核心思想，但没有照搬 Apple 版本的梯度过滤等近似优化；本文的正确性证明对应当前代码实际执行的“完整词表逐 tile 精确扫描”。

## 2. GitHub Markdown 中的公式写法

GitHub 官方文档说明，Markdown 文件支持 LaTeX 数学表达式，使用 MathJax 渲染；行内公式可以使用单个美元符号，块级公式可以使用一对双美元符号或 `math` 代码块。本文采用更保守的写法：

- 所有正式公式均独占一段，并使用一对双美元符号包围；
- 正文中的短符号使用反引号，例如 `N`、`V`、`Q`，不把复杂公式放入表格；
- 公式前后保留空行，避免公式和列表、表格或普通段落粘连；
- 使用 MathJax 支持的基础 LaTeX 命令，例如 `\mathbf`、`\mathbb`、`\sum`、`\exp`、`\log`、`\partial` 和 `\mathsf`。

对应的 GitHub 官方说明见文末参考资料。

## 3. 一组贯穿全文的真实 shape

本文采用 MindSpeed-MM 仓库 `tests/ut/loss/test_chunkloss.py` 中声明的工程量级参数。该 legacy 测试会手工使用 `input[:, :-1]`，所以它实际送入 legacy ChunkLoss 的序列长度是 8191；当前 FSDP CCE 接口则允许 `[B,S,D]` 输入，并通过在 labels 右侧补 `ignore_index` 保持 shift 前后的长度均为 `S`。为了完整展示当前 CCE 接口，本文使用测试中声明的 `S=8192`，即 `[2,8192,4096]`。这组数字是实际仓库参数的组合，但不表示该 legacy 单元测试已经覆盖 CCE。

| 符号 | 含义 | 数值 |
|---|---|---:|
| `B` | micro batch size | 2 |
| `S` | 序列长度 | 8192 |
| `D` | 特征维度 | 4096 |
| `V` | 词表大小 | 151674 |
| `N` | 展平后的 token 数，`N=B×S` | 16384 |
| `Q` | CCE 词表 tile 大小 | 4096 |
| `K` | 词表 tile 数，`K=ceil(V/Q)` | 38 |

这里把 Transformer 最后一层输出统一记为：

$$
\mathbf{X}\in\mathbb{R}^{N\times D}.
$$

LM-Head 不带 bias，其权重统一记为：

$$
\mathbf{W}\in\mathbb{R}^{V\times D}.
$$

普通路径会执行：

$$
\mathbf{Z}=\mathbf{X}\mathbf{W}^{\mathsf T}
\in\mathbb{R}^{N\times V}.
$$

代入真实 shape：

$$
\mathbf{X}:[16384,4096],
\qquad
\mathbf{W}:[151674,4096],
\qquad
\mathbf{Z}:[16384,151674].
$$

完整 logits 的元素数是：

$$
N V
=16384\times151674
=2\,485\,026\,816.
$$

因此仅一个 BF16 完整 logits 张量就需要：

$$
2\,485\,026\,816\times2\ \mathrm{bytes}
\approx4.63\ \mathrm{GiB}.
$$

若完整 logits 转成 FP32，则仅 logits 就约为：

$$
2\,485\,026\,816\times4\ \mathrm{bytes}
\approx9.26\ \mathrm{GiB}.
$$

这还没有计入 softmax、交叉熵中间量、反向状态以及内存分配器保留空间。CCE 要解决的正是这个与 `N×V` 成正比的显存尖峰。

## 4. 普通交叉熵的朴素计算

令第 `n` 个 token 的标签为 `y_n`。有效标签属于从 `0` 到 `V-1` 的词表下标；被忽略的位置使用 `ignore_index`。定义有效位置指示量：

$$
a_n=
\begin{cases}
1, & y_n\ne-100,\\
0, & y_n=-100.
\end{cases}
$$

LM-Head 对第 `n` 个 token 和第 `j` 个词表项产生的 logit 为：

$$
z_{n,j}
=\sum_{d=1}^{D}x_{n,d}w_{j,d}.
$$

该 token 在第 `j` 个词表项上的 softmax 概率为：

$$
p_{n,j}
=\frac{\exp(z_{n,j})}
{\sum_{r=0}^{V-1}\exp(z_{n,r})}.
$$

单 token 交叉熵为：

$$
\ell_n
=-\log p_{n,y_n}
=\log\sum_{r=0}^{V-1}\exp(z_{n,r})-z_{n,y_n}.
$$

定义归一化系数 `A`。对于当前代码的 `default` loss，`A` 是有效 token 数；对于 `per_token_loss`，`A` 是训练 step 约定的平均有效 token 数。最终 loss 为：

$$
\mathcal{L}
=\frac{1}{A}\sum_{n=0}^{N-1}a_n\ell_n.
$$

朴素实现很直接：

```text
Z = X @ W.T                    # [N,V]
P = softmax(Z, vocab_dim)      # [N,V]
loss = nll(P, labels) / A
```

问题不在公式，而在执行计划。`Z` 含有 `N×V` 个元素，且很多框架还要保留或生成同量级的概率和反向中间量。

## 5. CCE 的朴素原理

### 5.1 沿词表维切块

把完整词表下标集合划分为 `K` 个互不相交的连续集合：

$$
\mathcal{V}
=\{0,1,\ldots,V-1\}
=\bigcup_{k=0}^{K-1}\mathcal{V}_k,
\qquad
\mathcal{V}_i\cap\mathcal{V}_j=\varnothing\quad(i\ne j).
$$

第 `k` 个词表块的大小记为 `q_k`，满足：

$$
q_k=|\mathcal{V}_k|\le Q,
\qquad
\sum_{k=0}^{K-1}q_k=V.
$$

对本文真实 shape，前 37 个块的大小都是 4096，最后一个块大小为：

$$
151674-37\times4096=122.
$$

相应地，把 LM-Head 权重按行切成：

$$
\mathbf{W}^{(k)}
\in\mathbb{R}^{q_k\times D}.
$$

每次只计算一个局部 logits tile：

$$
\mathbf{Z}^{(k)}
=\mathbf{X}\left(\mathbf{W}^{(k)}\right)^{\mathsf T}
\in\mathbb{R}^{N\times q_k}.
$$

普通交叉熵一次生成 `[16384,151674]`；CCE 前 37 次生成 `[16384,4096]`，最后一次生成 `[16384,122]`。所有词表项仍然都被计算，没有漏掉任何类别。

### 5.2 流式计算时真正需要保留什么

由单 token 交叉熵公式可知，前向最终只需要两个量：

1. 完整词表 logits 的 log-sum-exp；
2. 正确类别的 logit。

对第 `n` 个 token，把这两个量分别记为：

$$
\lambda_n
=\log\sum_{j=0}^{V-1}\exp(z_{n,j}),
\qquad
c_n=z_{n,y_n}.
$$

于是：

$$
\ell_n=\lambda_n-c_n.
$$

CCE 不保存完整的 `z_(n,j)`。它在词表块到来时更新 `λ_n` 所需的在线状态，并在正确类别所在的块到来时记录 `c_n`。

### 5.3 最朴素的 CCE 伪代码

```text
for each token n:
    running_max[n] = -infinity
    running_sum[n] = 0
    correct_logit[n] = 0

for each vocab tile k:
    tile = X @ W_tile[k].T
    update running_max and running_sum with tile
    if label[n] belongs to tile k:
        correct_logit[n] = tile[n, local_label]
    release or reuse tile

lse = running_max + log(running_sum)
loss_sum = sum(valid[n] * (lse[n] - correct_logit[n]))
loss = loss_sum / A
```

这就是 CCE 的朴素核心：把“先存下全部 logits，再归约”改成“生成一块，立即归约一块”。

## 6. 前向：online log-sum-exp 的推导

### 6.1 为什么不能直接累计指数和

数学上可以累计：

$$
\sum_{j=0}^{V-1}\exp(z_{n,j}).
$$

但当 logit 较大时，直接计算指数容易上溢。稳定做法是减去最大值。对任意有限集合 `𝒜`，定义：

$$
m_{n,\mathcal{A}}
=\max_{j\in\mathcal{A}}z_{n,j},
$$

$$
s_{n,\mathcal{A}}
=\sum_{j\in\mathcal{A}}
\exp\left(z_{n,j}-m_{n,\mathcal{A}}\right).
$$

因为每个指数的输入都不大于 0，所以不容易上溢。原始指数和可恢复为：

$$
\sum_{j\in\mathcal{A}}\exp(z_{n,j})
=\exp(m_{n,\mathcal{A}})s_{n,\mathcal{A}}.
$$

因此 log-sum-exp 为：

$$
\log\sum_{j\in\mathcal{A}}\exp(z_{n,j})
=m_{n,\mathcal{A}}+\log s_{n,\mathcal{A}}.
$$

### 6.2 两个词表块如何合并

设 `𝒜` 是已经扫描的词表项，`ℬ` 是新到来的词表块，且两者不相交。已有状态由块 `𝒜` 的局部最大值和移位指数和组成，新块状态同理。为简化书写，将合并后的最大值简记为：

$$
m_n^{\prime}
=m_{n,\mathcal{A}\cup\mathcal{B}}
=\max\left(m_{n,\mathcal{A}},m_{n,\mathcal{B}}\right).
$$

相应地，合并后的移位指数和为：

$$
s_n^{\prime}
=s_{n,\mathcal{A}}
\exp\left(m_{n,\mathcal{A}}-m_n^{\prime}\right)
+s_{n,\mathcal{B}}
\exp\left(m_{n,\mathcal{B}}-m_n^{\prime}\right).
$$

这里的撇号只表示“合并后得到的新状态”，不是求导符号。这个公式的关键是：两个块保存的虽然都是移位指数和，但它们分别以各自的块内最大值为基准，不能直接相加。合并前必须先把两个指数和换算到共同的合并最大值基准。

根据上一节的恢复公式，两个块对应的原始指数和分别为：

$$
\sum_{j\in\mathcal{A}}\exp(z_{n,j})
=\exp(m_{n,\mathcal{A}})s_{n,\mathcal{A}},
$$

$$
\sum_{j\in\mathcal{B}}\exp(z_{n,j})
=\exp(m_{n,\mathcal{B}})s_{n,\mathcal{B}}.
$$

合并后，原始指数和的数学值不能改变，因此必须满足：

$$
\exp(m_n^{\prime})s_n^{\prime}
=\exp(m_{n,\mathcal{A}})s_{n,\mathcal{A}}
+\exp(m_{n,\mathcal{B}})s_{n,\mathcal{B}}.
$$

等式两边除以合并最大值对应的指数，就得到上面的合并公式。第一个换基准因子的作用，是把块 `𝒜` 从自己的局部最大值基准换算到合并最大值基准；块 `ℬ` 的因子作用相同。

之所以选择两个块最大值中的较大者作为合并后的新最大值，是因为此时：

$$
m_{n,\mathcal{A}}-m_n^{\prime}\leq 0,
\qquad
m_{n,\mathcal{B}}-m_n^{\prime}\leq 0.
$$

两个换基准因子都不大于 1，不会因为计算很大的指数而上溢，而且至少有一个因子等于 1。数学上也可以使用其他共同基准，但使用合并后的最大值数值最稳定。

例如，假设同一个 token 在两个词表块上的 logits 分别为 `𝒜=[1,2]` 和 `ℬ=[4,3]`。两个块的状态为：

$$
m_{n,\mathcal{A}}=2,
\qquad
s_{n,\mathcal{A}}
=\exp(-1)+1
\approx 1.367879,
$$

$$
m_{n,\mathcal{B}}=4,
\qquad
s_{n,\mathcal{B}}
=1+\exp(-1)
\approx 1.367879.
$$

合并后的最大值是 4。块 `𝒜` 原来以 2 为基准，现在要换成以 4 为基准，所以它的贡献需要乘以从基准 2 到基准 4 的换算因子；块 `ℬ` 的基准已经是 4，换基准因子为 1：

$$
s_n^{\prime}
=1.367879\exp(2-4)
+1.367879\exp(4-4)
\approx 1.553002.
$$

最终得到：

$$
m_n^{\prime}+\log s_n^{\prime}
=4+\log(1.553002)
\approx 4.440190.
$$

这与一次性计算完整 logits 的结果相同：

$$
\log\left(\exp(1)+\exp(2)+\exp(4)+\exp(3)\right)
\approx 4.440190.
$$

如果错误地直接把两个块的移位指数和相加，会得到 `2.735758`，进而得到约 `5.006409`，与正确结果不同。错误的原因就是把两个基准不同的移位指数和当成了同一尺度上的量。

这就是当前 CCE kernel 使用的 online log-sum-exp 合并公式。

### 6.3 合并公式正确性证明

由定义：

$$
s_{n,\mathcal{A}}
=\sum_{j\in\mathcal{A}}
\exp\left(z_{n,j}-m_{n,\mathcal{A}}\right).
$$

乘以换基准因子后：

$$
s_{n,\mathcal{A}}
\exp\left(m_{n,\mathcal{A}}-m_n^{\prime}\right)
=\sum_{j\in\mathcal{A}}
\exp\left(z_{n,j}-m_n^{\prime}\right).
$$

同理：

$$
s_{n,\mathcal{B}}
\exp\left(m_{n,\mathcal{B}}-m_n^{\prime}\right)
=\sum_{j\in\mathcal{B}}
\exp\left(z_{n,j}-m_n^{\prime}\right).
$$

两式相加，因为两个集合不相交：

$$
s_n^{\prime}
=\sum_{j\in\mathcal{A}\cup\mathcal{B}}
\exp\left(z_{n,j}-m_n^{\prime}\right).
$$

这恰好是集合 `𝒜∪ℬ` 在合并后的新最大值基准下的稳定指数和。因此合并前后的数学值完全一致。

以空集为初始状态，再对 `K` 个词表块重复应用该结论。根据数学归纳法，扫描完第 `K` 个块后：

$$
m_n
=\max\left(z_{n,0},z_{n,1},\ldots,z_{n,V-1}\right),
$$

$$
s_n
=\sum_{j=0}^{V-1}\exp(z_{n,j}-m_n).
$$

最终得到：

$$
m_n+\log s_n
=\log\sum_{j=0}^{V-1}\exp(z_{n,j})
=\lambda_n.
$$

所以 CCE 的 online log-sum-exp 与一次性在完整 logits 上计算的 log-sum-exp 相同。

### 6.4 正确类别 logit 为什么不会取错

词表块组成互不相交的完备划分。对于任意有效标签 `y_n`，存在唯一的 `k_n` 使得：

$$
y_n\in\mathcal{V}_{k_n}.
$$

当扫描到第 `k_n` 个词表块时，从局部矩阵 `Z^(k_n)` 读取对应位置即可得到：

$$
c_n=z_{n,y_n}.
$$

其他词表块不包含 `y_n`，因此不会重复贡献。被忽略位置最后由 `a_n=0` 置零，不参与 loss。

### 6.5 前向 loss 等价性定理

CCE 扫描完成后得到的逐 token loss 是：

$$
\ell_n^{\mathrm{CCE}}
=m_n+\log s_n-c_n.
$$

代入前两节已经证明的结果：

$$
\ell_n^{\mathrm{CCE}}
=\log\sum_{j=0}^{V-1}\exp(z_{n,j})-z_{n,y_n}
=\ell_n.
$$

于是：

$$
\mathcal{L}_{\mathrm{CCE}}
=\frac{1}{A}\sum_{n=0}^{N-1}a_n\ell_n^{\mathrm{CCE}}
=\frac{1}{A}\sum_{n=0}^{N-1}a_n\ell_n
=\mathcal{L}.
$$

因此，在精确实数运算下，CCE 前向结果与普通交叉熵严格相等。

## 7. 反向：梯度公式与分块正确性证明

### 7.1 loss 对 logits 的梯度

先考虑有效 token。根据：

$$
\ell_n
=\log\sum_{r=0}^{V-1}\exp(z_{n,r})-z_{n,y_n},
$$

对任意 `z_{n,j}` 求偏导：

$$
\frac{\partial}{\partial z_{n,j}}
\log\sum_{r=0}^{V-1}\exp(z_{n,r})
=\frac{\exp(z_{n,j})}
{\sum_{r=0}^{V-1}\exp(z_{n,r})}
=p_{n,j}.
$$

正确类别项的偏导为：

$$
\frac{\partial z_{n,y_n}}{\partial z_{n,j}}
=\mathbb{1}[j=y_n].
$$

加入有效位置掩码和归一化后：

$$
g_{n,j}
=\frac{\partial\mathcal{L}}{\partial z_{n,j}}
=\frac{a_n}{A}
\left(p_{n,j}-\mathbb{1}[j=y_n]\right).
$$

令完整 logits 梯度矩阵为：

$$
\mathbf{G}
\in\mathbb{R}^{N\times V}.
$$

普通反向可以写成：

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}
=\mathbf{G}\mathbf{W},
$$

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}}
=\mathbf{G}^{\mathsf T}\mathbf{X}.
$$

### 7.2 CCE 如何在不保存完整概率时恢复梯度

前向已经保存每个 token 的 `λ_n`。反向重新计算第 `k` 个局部 logits tile 后，可以直接恢复这个 tile 内的概率：

$$
p_{n,j}
=\exp(z_{n,j}-\lambda_n),
\qquad
j\in\mathcal{V}_k.
$$

于是该 tile 的局部 logits 梯度为：

$$
g_{n,j}^{(k)}
=\frac{a_n}{A}
\left(
\exp(z_{n,j}-\lambda_n)
-\mathbb{1}[j=y_n]
\right),
\qquad
j\in\mathcal{V}_k.
$$

当前 MindSpeed-MM 实现先在自定义 backward 中生成未除以 `A` 的 `softmax-onehot`，外层的 `loss/A` 通过 autograd 的 `grad_output` 把 `1/A` 乘回 `X` 的梯度和 LM-Head 权重梯度。两种执行顺序在代数上等价。

### 7.3 Transformer 最后一层输出梯度的分块证明

按词表块把完整梯度矩阵横向分块：

$$
\mathbf{G}
=\left[
\mathbf{G}^{(0)}\ \mathbf{G}^{(1)}\ \cdots\ \mathbf{G}^{(K-1)}
\right].
$$

按相同边界把 LM-Head 权重纵向堆叠：

$$
\mathbf{W}
=\begin{bmatrix}
\mathbf{W}^{(0)}\\
\mathbf{W}^{(1)}\\
\vdots\\
\mathbf{W}^{(K-1)}
\end{bmatrix}.
$$

利用分块矩阵乘法：

$$
\mathbf{G}\mathbf{W}
=\sum_{k=0}^{K-1}
\mathbf{G}^{(k)}\mathbf{W}^{(k)}.
$$

因此 CCE 可初始化一个 `[N,D]` 的累加器，并逐 tile 累加：

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}
\leftarrow
\frac{\partial\mathcal{L}}{\partial\mathbf{X}}
+\mathbf{G}^{(k)}\mathbf{W}^{(k)}.
$$

扫描完全部词表块后，这个累加器恰好等于普通反向的 `G W`。

### 7.4 LM-Head 权重梯度的分块证明

完整 LM-Head 权重梯度按词表维分块为：

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}}
=\begin{bmatrix}
\left(\mathbf{G}^{(0)}\right)^{\mathsf T}\mathbf{X}\\
\left(\mathbf{G}^{(1)}\right)^{\mathsf T}\mathbf{X}\\
\vdots\\
\left(\mathbf{G}^{(K-1)}\right)^{\mathsf T}\mathbf{X}
\end{bmatrix}.
$$

第 `k` 个词表块只对应 LM-Head 权重的第 `k` 段行，因此可以直接写入：

$$
\frac{\partial\mathcal{L}}
{\partial\mathbf{W}^{(k)}}
=\left(\mathbf{G}^{(k)}\right)^{\mathsf T}\mathbf{X}.
$$

不同 tile 写入不重叠的 LM-Head 权重梯度区间，不会遗漏或重复累加。把所有区间按原顺序拼接后，恰好得到普通反向的 `G^T X`。

### 7.5 反向正确性结论

每个 `g_(n,j)` 都通过前向保存的完整词表 `λ_n` 精确恢复；所有词表项组成互不相交的完备划分；Transformer 最后一层输出梯度按 tile 求和，LM-Head 权重梯度按 tile 拼接。因此：

$$
\frac{\partial\mathcal{L}_{\mathrm{CCE}}}
{\partial\mathbf{X}}
=\frac{\partial\mathcal{L}}
{\partial\mathbf{X}},
$$

$$
\frac{\partial\mathcal{L}_{\mathrm{CCE}}}
{\partial\mathbf{W}}
=\frac{\partial\mathcal{L}}
{\partial\mathbf{W}}.
$$

至此，CCE 的前向值和两个可训练输入的反向梯度都完成了数学等价性证明。

## 8. 用真实 shape 走一遍当前实现

### 8.1 输入与展平

输入阶段：

```text
Transformer 最后一层输出 X: [B,S,D] = [2,8192,4096], BF16
LM-Head 权重 W:              [V,D]   = [151674,4096], BF16
shift_labels:                [B,S]   = [2,8192], INT64
```

CCE 公共入口展平 token 维：

```text
X:            [16384,4096]
shift_labels: [16384]
W:            [151674,4096]
```

当前 FSDP loss 路径先在 labels 右侧补 `ignore_index=-100`，再取右移后的 labels。因此 shape 仍为 `[2,8192]`，最后一个位置被忽略；`default` loss 的 `A` 是其余有效位置总数。

### 8.2 前向 38 个词表块

默认 `Q=4096`：

```text
tile 0:   W[0:4096]           -> logits [16384,4096]
tile 1:   W[4096:8192]        -> logits [16384,4096]
...
tile 36:  W[147456:151552]    -> logits [16384,4096]
tile 37:  W[151552:151674]    -> logits [16384,122]
```

每个 tile 完成三件事：

1. 通过矩阵乘得到当前词表范围的 logits；
2. 更新长度为 `N=16384` 的 `m` 和 `s`；
3. 标签若落入当前词表范围，则更新长度为 `N=16384` 的 `correct`。

扫描结束后生成：

```text
lse = m + log(s): [16384], FP32
loss_sum:         scalar, FP32
```

前向只把 `X`、`W`、`shift_labels` 和 `lse` 保存给 backward，不保存 `[16384,151674]` 的完整 logits。

### 8.3 反向 38 个词表块

对每个词表块，反向执行：

```text
重算局部 logits:       X @ W_tile.T
原地变为局部梯度:     exp(logits-lse) - onehot(label)
计算 LM-Head 梯度块:  G_tile.T @ X
累加 X 的梯度:        grad_X += G_tile @ W_tile
```

最后得到：

```text
grad_X: [16384,4096]
grad_W: [151674,4096]
```

再由外层 `loss/A` 传入的 `grad_output` 完成归一化缩放。

### 8.4 三缓冲区与双 stream

当前实现不是只有一个 tile buffer，而是轮转使用 3 个 buffer：

```text
slot 0 <- tile 0, tile 3, tile 6, ...
slot 1 <- tile 1, tile 4, tile 7, ...
slot 2 <- tile 2, tile 5, tile 8, ...
```

一个 stream 负责 LM-Head 矩阵乘，另一个 stream 负责 online-softmax 和 `softmax-onehot` Triton kernel。event 保证：

- tile 矩阵乘完成后，向量 kernel 才能读取；
- 向量 kernel 或梯度矩阵乘完成后，对应 slot 才能被后续 tile 复用；
- backward 可预取下一个 tile 的重算矩阵乘，以重叠部分计算。

三缓冲区增加常数倍显存，但让矩阵乘与向量运算可以流水化，降低串行逐 tile 的性能损失。

## 9. 显存收益：到底省了多少

### 9.1 仅比较 logits 工作区

普通 BF16 完整 logits 为：

$$
M_{\mathrm{full}}
=N V\times2\ \mathrm{bytes}
\approx4.63\ \mathrm{GiB}.
$$

一个满大小 BF16 CCE tile 为：

$$
M_{\mathrm{tile}}
=N Q\times2\ \mathrm{bytes}
=16384\times4096\times2\ \mathrm{bytes}
=128\ \mathrm{MiB}.
$$

当前实现使用 3 个轮转 slot，满载时合计：

$$
M_{\mathrm{slots}}
=3NQ\times2\ \mathrm{bytes}
=384\ \mathrm{MiB}.
$$

因此，仅对比 BF16 logits 工作区，完整 logits 与一个 tile 的大小比约为：

$$
\frac{V}{Q}
=\frac{151674}{4096}
\approx37.03.
$$

对比当前实现的 3 个 slot，大小比约为：

$$
\frac{V}{3Q}
=\frac{151674}{3\times4096}
\approx12.34.
$$

这两个比值只描述 logits 工作区，不等于端到端训练峰值显存降幅。

### 9.2 复杂度表达

忽略固定输入、权重和输出梯度，普通路径需要物化与 `N×V` 同量级的 logits 激活：

$$
M_{\mathrm{CE}}=\Theta(NV).
$$

使用固定数量 `R` 个轮转 slot 的 CCE，其 tile 工作区为：

$$
M_{\mathrm{CCE\ workspace}}
=\Theta(RNQ+N).
$$

当前实现 `R=3`。其中 `N` 项来自 `m`、`s`、`correct` 和 `lse` 等逐 token 状态。因为 `Q` 是可调的固定 tile 大小，而 `V` 可以很大，所以 CCE 消除了工作区对完整 `N×V` logits 的依赖。

若额外配置外层 token 分段大小 `C`，每次只处理最多 `C` 个展平 token，则 tile 工作区进一步变为：

$$
M_{\mathrm{CCE\ workspace}}
=\Theta\left(R\min(N,C)Q+N\right).
$$

但外层 token 分段会增加自定义 autograd 调用次数和调度开销。

### 9.3 CCE 不会省掉的显存

在本文真实 shape 下：

| 张量 | shape | BF16 大小 | CCE 是否消除 |
|---|---|---:|---|
| Transformer 最后一层输出 `X` | `[16384,4096]` | 128 MiB | 否 |
| LM-Head 权重 `W` | `[151674,4096]` | 约 1.16 GiB | 否 |
| LM-Head 权重梯度 `grad_W` | `[151674,4096]` | 约 1.16 GiB | 否 |
| `grad_X` FP32 累加器 | `[16384,4096]` | 256 MiB | CCE backward 新增，结束时转回输入 dtype |
| 完整 logits `Z` | `[16384,151674]` | 约 4.63 GiB | 是 |
| 3 个 CCE tile slot | `3×[16384,4096]` | 384 MiB | CCE 新增工作区 |

优化器状态、参数副本、FSDP 通信缓冲区、其他层激活以及分配器碎片也不在上述估算内。实际收益必须以目标设备上的峰值显存测量为准。

## 10. 计算量：为什么省显存不等于省 FLOPs

### 10.1 普通 LM-Head 加交叉熵

只按主要矩阵乘计，普通实现包含：

1. forward：`X @ W.T`；
2. backward：`G @ W`，得到 `grad_X`；
3. backward：`G.T @ X`，得到 `grad_W`。

三次矩阵乘的主量级都是 `N×V×D`：

$$
F_{\mathrm{baseline}}
\approx3NVD.
$$

### 10.2 当前 CCE 实现

CCE 包含：

1. forward 逐 tile 计算 logits；
2. backward 逐 tile重算 logits；
3. backward 逐 tile 计算 `grad_X`；
4. backward 逐 tile 计算 `grad_W`。

所有 tile 的词表宽度之和仍是 `V`，因此：

$$
F_{\mathrm{CCE}}
\approx4NVD.
$$

只看 LM-Head 和交叉熵这部分的主要矩阵乘，CCE 比保存 logits 的基线多出一次 logits 重算，理论增量约为三分之一：

$$
\frac{F_{\mathrm{CCE}}-F_{\mathrm{baseline}}}
{F_{\mathrm{baseline}}}
\approx\frac{1}{3}.
$$

这不是整模型训练计算量增加三分之一。Transformer 主体通常占据大量计算，CCE 只改变 LM-Head 与交叉熵部分。实际耗时还取决于内存带宽、矩阵乘效率、kernel 融合、双 stream 重叠和设备特性；显存压力降低后，也可能允许更大 batch 或减少其他内存策略的开销。

## 11. CCE 的收益

### 11.1 避免完整 logits 常驻设备内存

这是最核心的收益。对长序列和大词表，`N×V` 很容易成为训练中的最大单个激活。CCE 只让 `N×Q` 大小的局部 logits 短暂存在。

### 11.2 不改变训练目标

当前实现没有跳过词表项，也没有近似 log-sum-exp。精确实数意义下，前向 loss、Transformer 最后一层输出梯度和 LM-Head 权重梯度都与普通交叉熵相同。

### 11.3 数值稳定

online log-sum-exp 始终以当前最大 logit 为指数基准；`m`、`s`、`correct`、`lse` 和 `grad_X` 累加器使用 FP32。相比直接累计指数和，这能显著降低溢出风险。

### 11.4 tile 大小可调

减小 `Q` 会降低单个 slot 显存，增大 `Q` 会减少 tile 数和 kernel/event 数。它提供了显存与调度效率之间的显式旋钮。

### 11.5 可叠加外层 token 分段

当 `N×Q` 仍然过大时，可以再按展平 token 维设置外层分段。这样同时限制 token 维和词表维的局部 logits 大小。

## 12. CCE 的代价

### 12.1 backward 必须重算 logits

前向不保存完整 logits，反向为了恢复 softmax 概率必须再次执行 `X @ W_tile.T`。这是用计算换显存的本质代价。

### 12.2 kernel launch 与 event 数量增加

本文真实 shape 有 38 个词表 tile。前向和反向都要遍历这些 tile，并在两个 stream 之间建立依赖。tile 越小，循环、kernel launch、event 和 host 调度开销越明显。

### 12.3 工作区并非零

当前实现为了流水化使用 3 个 tile slot。本文 shape 下它们共占 384 MiB，还要加逐 token FP32 状态、`grad_X` FP32 累加器和正常的权重梯度。CCE 是把超大的工作区变小，不是让交叉熵完全不占显存。

### 12.4 浮点结果不保证逐 bit 相同

数学公式等价，但 CCE 改变了归约顺序，并混合使用 BF16 tile 与 FP32 累加。浮点加法不满足严格结合律，因此与普通交叉熵之间可能有舍入级差异。正确的验证方式是比较合理容差内的 loss 和梯度，而不是要求逐 bit 相同。

### 12.5 实现和硬件约束更强

当前 MindSpeed-MM CCE kernel 依赖 Triton/Ascend 运行环境、设备 stream 和 event 语义。相比框架内置交叉熵，它有更多并发顺序、尾块和 dtype 边界需要验证。

### 12.6 当前功能边界

基于当前源码：

- LM-Head bias 不受支持；
- CCE 只支持 `default` 和 `per_token_loss`，不支持 `per_sample_loss`；
- CCE 与 dynamic ChunkLoss 互斥；
- `vocab_tile_size` 默认是 4096；
- CCE 的 `chunk_size` 表示展平 token 维的可选外层分段，不是 legacy ChunkLoss 中每个样本的序列块长；
- 当前仓库基线未找到 CCE kernel 的专用单元测试。

## 13. 数学正确与工程正确不是一回事

前文证明了算法在满足以下条件时的数学正确性：

1. 词表块互不重叠且完整覆盖从 `0` 到 `V-1`；
2. 每个局部 logits 都由同一组 `X` 和对应的 `W` 行计算；
3. online log-sum-exp 状态严格按依赖顺序合并；
4. 正确类别在所属 tile 中被提取一次；
5. ignore mask 和归一化系数 `A` 与基线一致；
6. backward 使用前向对应的完整词表 `lse`；
7. 所有局部 `grad_X` 被累加，所有 `grad_W` 块被写回正确区间。

但异步多 stream 实现还必须保证 buffer 生命周期正确。例如 slot 在向量 kernel 或梯度矩阵乘读取完成前被覆盖，会破坏上述第 2、3 或 6 条前提。当前代码使用 event 约束复用顺序，正是为了让工程执行满足数学证明的假设。

建议在目标 NPU 上至少验证：

1. CCE 与普通交叉熵的前向 loss；
2. CCE 与普通交叉熵的 `grad_X` 和 `grad_W`；
3. `ignore_index`、全 ignore、不同有效 token 数；
4. `V` 不能被 `Q` 整除的尾 tile，本文 shape 的尾 tile 是 122；
5. `N` 不能被 Triton 行块整除的情况；
6. 设置与不设置外层 token 分段的等价性；
7. 多轮训练中的 stream/event 稳定性；
8. 峰值显存、吞吐和 loss 曲线。

## 14. 如何选择 tile 大小

词表 tile 大小 `Q` 的选择是典型的显存与调度折中：

- `Q` 变小：单 slot 显存按比例下降，但 tile 数增加，矩阵乘更窄，kernel/event 开销更高；
- `Q` 变大：tile 数减少，矩阵乘通常更高效，但 3 个 slot 的显存按比例上升；
- 外层 token 分段 `C` 变小：`N×Q` 工作区进一步缩小，但自定义 autograd 调用和循环次数增加。

当前实现建议从 `Q=4096` 开始。本文 shape 下：

| `Q` | tile 数 | 单个 BF16 slot | 3 个 slot |
|---:|---:|---:|---:|
| 2048 | 75 | 64 MiB | 192 MiB |
| 4096 | 38 | 128 MiB | 384 MiB |
| 8192 | 19 | 256 MiB | 768 MiB |

表中只计算满大小 slot；最后一个词表 tile 可能更小。最终参数应由目标设备上的峰值显存和端到端吞吐共同决定。

## 15. 与 Apple 原始 CCE 的关系

Apple CCE 的核心命题同样是：不把所有 token 对所有词表项的 logits 写入全局显存，而是即时计算正确类别 logit 和完整词表 log-sum-exp。Apple 实现还提供梯度过滤、Kahan 累加、vocabulary parallel 等选项。

当前 MindSpeed-MM 版本采用了词表 tile、online log-sum-exp、backward 重算和 Triton kernel 等核心思路，并针对 NPU 使用双 stream 与 3 个轮转 buffer。当前代码没有实现 Apple 版本的梯度过滤，所以不能把 Apple 论文中依赖梯度稀疏性的性能结论直接当作本实现的实测结果。本文所有显存数字都来自当前 shape 的张量容量计算，不冒充端到端 benchmark。

## 16. 总结

CCE 的本质可以浓缩为一句话：

> 交叉熵真正需要的是完整词表的 log-sum-exp 和正确类别 logit，而不是需要长期保存完整 logits 矩阵。

它通过词表分块和 online log-sum-exp，在前向得到与普通交叉熵相同的 loss；通过保存 `lse`、反向重算局部 logits，并按分块矩阵乘法求 `grad_X` 和 `grad_W`，得到与普通交叉熵相同的梯度。

对本文真实 shape `[B,S,D]=[2,8192,4096]`、`V=151674`，普通 BF16 完整 logits 约 4.63 GiB。使用 `Q=4096` 时，一个 tile 为 128 MiB，当前 3-slot 实现的 tile 工作区为 384 MiB。收益是大幅压低 logits 工作区；代价是反向重算一次 LM-Head 投影、更多逐 tile 调度以及更复杂的异步正确性要求。

## 17. 参考资料

- [GitHub Docs：Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
- [Apple：ml-cross-entropy](https://github.com/apple/ml-cross-entropy)
- [Cut Your Losses in Large-Vocabulary Language Models，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/aa963ac256590bb7ad5fc26c68229a3a-Abstract-Conference.html)
- MindSpeed-MM 当前实现：`mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_cce_fused.py`
- MindSpeed-MM 当前 kernel：`mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_cce_kernels.py`
- MindSpeed-MM loss 接入：`mindspeed_mm/fsdp/loss/loss_func.py`
- 本文 shape 来源：`tests/ut/loss/test_chunkloss.py`
