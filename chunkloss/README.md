# ChunkLoss：背景、原理与使用边界

> 本文基于 MindSpeed-MM `master` 分支提交 `192b2a13`（2026-08-27）的代码整理。代码仍在演进，尤其是 CCE 实现比仓库中的入口文档更新；实际使用时应以当前代码和配置校验逻辑为准。

## 1. ChunkLoss 解决什么问题

自回归语言模型最后通常通过线性层把隐藏状态投影到词表：

| 阶段 | 张量或运算 | 形状/结果 |
|---|---|---|
| 输入 | hidden states | `[B,S,H]` |
| 投影 | `lm_head`，权重 `W` | `W` 的形状为 `[V,H]` |
| 输出 | logits | `[B,S,V]` |
| 损失 | cross entropy | 标量 loss |

其中：

- `B`：micro batch size；
- `S`：序列长度；
- `H`：hidden size；
- `V`：词表大小。

多模态大模型常同时具有长序列和大词表。完整 logits 的元素数是 `B × S × V`，其显存随序列长度和词表大小线性增长，通常远大于 `[B,S,H]` 的隐藏状态。交叉熵还可能产生 FP32 logits、softmax 中间量和反向所需状态，因此 loss 阶段会形成明显的显存尖峰；动态 shape 下，反复申请大块张量也容易加剧碎片化。

以仓库单元测试中的一组形状为例：`B=2`、`S=8192`、`V=151674`。仅 BF16 完整 logits 就约为：

$$
2 \times 8192 \times 151674 \times 2\ \mathrm{bytes}
\approx 4.63\ \mathrm{GiB}
$$

式中最后的 `2 bytes` 表示一个 BF16 元素占 2 字节，`GiB` 是二进制容量单位，$1\ \mathrm{GiB}=2^{30}$ bytes。因此元素总数为
$2\times8192\times151674=2,485,846,016$，乘以 2 bytes 后再除以 $2^{30}$，得到约 $4.63\ \mathrm{GiB}$。

这还没有计入 FP32 转换和交叉熵内部临时张量。

ChunkLoss 的核心目标不是改变损失函数，而是避免在同一时刻物化整个 `[B,S,V]` logits。

## 2. 为什么可以分块

语言模型交叉熵在 token 维度上可加：

$$
L = \frac{1}{\alpha}\sum_t \mathrm{CE}\!\left(h_t W^{\mathsf T}, y_t\right)
$$

这里各符号的含义是：

- $L$：完成归一化后的总 loss；$t$：token 位置索引；
- $h_t\in\mathbb{R}^{H}$：第 $t$ 个 token 的隐藏向量；$W\in\mathbb{R}^{V\times H}$：lm_head 权重；$\mathbb{R}^{H}$ 表示由 $H$ 个实数组成的向量空间；
- $W^{\mathsf T}$：$W$ 的转置，$h_tW^{\mathsf T}\in\mathbb{R}^{V}$ 是该 token 对词表中 $V$ 个类别的 logits；
- $y_t$：正确类别在词表中的整数下标；$\mathrm{CE}$：交叉熵；$\alpha$：由 loss 类型决定的归一化系数，例如有效 token 总数；
- $\sum_t$：把所有有效 token 的损失相加，ignore token 不参与求和。

交叉熵衡量预测概率与正确类别之间的差异。对单个 token，若 logits 为
$z=[2,1,0]$、正确类别 $y=0$，则先用 softmax 得到
$p_j=e^{z_j}/\sum_k e^{z_k}\approx[0.665,0.245,0.090]$，再计算
$\mathrm{CE}(z,y)=-\log p_y\approx0.408$。正确类别的预测概率越接近 1，交叉熵越接近 0。

把序列划分为若干互不重叠的块 `C_i` 后：

$$
L = \frac{1}{\alpha}\sum_i\sum_{t\in C_i}
\mathrm{CE}\!\left(h_t W^{\mathsf T}, y_t\right)
$$

式中 $i$ 是块编号，$C_i$ 是第 $i$ 个块包含的 token 位置集合，$t\in C_i$ 表示位置 $t$ 属于该块。各 $C_i$ 互不重叠且并集覆盖全部待计算位置。例如 4 个 token 可分为
$C_0=\{0,1\}$、$C_1=\{2,3\}$；若逐 token loss 为 $[0.2,0.4,0.3,0.1]$、$\alpha=4$，则
$L=((0.2+0.4)+(0.3+0.1))/4=0.25$，与不分块直接求平均相同。

只要所有块使用与原始 loss 一致的：

- label shift；
- `ignore_index` 掩码；
- 归约方式；
- 归一化系数 `α`；

逐块 loss 的和就与一次性计算完整 loss 等价。梯度同样满足可加性：

$$
\frac{\partial L}{\partial H}
= \mathrm{concat}_{i}\!\left(\frac{\partial L_i}{\partial H_i}\right),
\qquad
\frac{\partial L}{\partial W}
= \sum_i\frac{\partial L_i}{\partial W}
$$

$L_i$ 表示第 $i$ 块尚未与其他块合并的损失；$H_i$ 是该块的 hidden states，$H$ 是完整 hidden states。$\partial L/\partial H$ 和 $\partial L/\partial W$ 分别表示 loss 对 hidden states、权重的梯度，即输入或参数发生微小变化时 loss 的变化率。$\mathrm{concat}_i$ 表示按原序列位置拼接：各 token 只属于一个块，所以 hidden 梯度放回各自位置；$\sum_i$ 表示逐元素相加：所有块共用同一个 $W$，所以其参数梯度需要累加。例如两块给同一权重元素的梯度分别为 $0.3$ 和 $-0.1$，完整梯度就是 $0.3+(-0.1)=0.2$。

因此可以让每个块的 logits 在计算完 loss 和梯度后立即释放，只保留最终标量和必要梯度。

## 3. 传统 ChunkLoss 的工作方式

假设固定 `chunk_size` 为 `C`，传统实现沿序列维 `dim=1` 切分：

| 分块 | hidden states | logits | loss |
|---|---|---|---|
| chunk `i` | `[B,C,H]` | `[B,C,V]` | `L_i` |
| 全部 chunk | 按序列维拼接 | 按序列维拼接 | `L = sum(L_i)` |

任意时刻只需持有一个 `[B,C,V]` logits。仅考虑 logits，理想峰值约从 `B × S × V` 降为 `B × C × V`，降幅近似为 `S/C`。

MindSpeed-MM 的实现不是简单地在普通 forward 后等待统一 backward，而是在自定义 `torch.autograd.Function` 的 forward 内，对每个块调用 `torch.func.grad_and_value`，同时得到：

- 当前块的 loss；
- 当前块对 hidden states 的梯度；
- 当前块对 `lm_head.weight` 的梯度。

随后它把 hidden 梯度写入预分配的 `[B,S,H]` 张量，把各块 weight 梯度累加到 `[V,H]` 张量。自定义 backward 不再重新执行 lm_head 和交叉熵，只把 forward 中保存的梯度乘以上游 `grad_output` 后返回。

这意味着传统实现的显存收益主要来自“不保留完整 logits 及其计算图”，并不意味着 loss 阶段没有额外内存：它仍需保留 hidden 梯度、weight 梯度和单块 logits。

## 4. 静态分块与动态分块

### 4.1 静态分块

静态模式直接使用配置中的 `chunk_size`：

```yaml
features:
  loss_cfg:
    loss_type: default
  enable_chunk_loss: true
  enable_dynamic_chunk_loss: false
  chunkloss_plan:
    apply_module: lm_head
    impl_type: legacy
    chunk_size: 1024
```

传统实现中，`chunk_size` 表示每个样本在序列维上的 token 数；若记 `C=chunk_size`，单块 token 总数近似为 `B × C`。

### 4.2 动态分块

动态模式配置单次计算允许的总 token 上限 `T=total_chunk_size`。运行时根据当前 batch size 计算：

$$
C_{\max}=\left\lfloor\frac{T}{B}\right\rfloor,
\qquad
C=2^{\left\lfloor\log_2 C_{\max}\right\rfloor}
\quad (C\le C_{\max})
$$

式中 $T$ 是一次分块允许的总 token 上限，$B$ 是当前 batch size，$C_{\max}$ 是理论上每个样本最多可放入的 token 数，$C$ 是最终采用的 `chunk_size`。$\lfloor x\rfloor$ 是向下取整，例如 $\lfloor3.7\rfloor=3$；$\log_2x$ 表示“2 的多少次方等于 $x$”。第二步因此是在不超过 $C_{\max}$ 的前提下取最大的 2 的整数次幂。例如 $T=4096$、$B=3$ 时，$C_{\max}=\lfloor4096/3\rfloor=1365$；因为 $2^{10}=1024\le1365<2048=2^{11}$，所以 $C=1024$。

例如 `T=4096`：

| Batch size | 理论上限 `floor(T/B)` | 实际 `chunk_size` |
|---:|---:|---:|
| 1 | 4096 | 4096 |
| 2 | 2048 | 2048 |
| 3 | 1365 | 1024 |
| 8 | 512 | 512 |

当参数无效，或 `B >= T` 时，代码退化为 `chunk_size=1`。动态模式适合 batch size 或输入 shape 经常变化的场景，可让不同 batch 的单块工作量更稳定。

```yaml
features:
  loss_cfg:
    loss_type: default
  enable_chunk_loss: false
  enable_dynamic_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    impl_type: legacy
    total_chunk_size: 4096
```

静态开关与动态开关应二选一。

## 5. 三种 loss 语义如何保持不变

原生 FSDP2 的预置 loss 支持以下归约方式：

| `loss_type` | 归一化含义 | 传统 ChunkLoss |
|---|---|---|
| `default` | 当前 batch 所有有效 token 的平均 loss | 支持 |
| `per_sample_loss` | 每个样本先按自己的有效 token 数平均，再对样本平均 | 支持，也处理 data packing 的 `cu_seqlens` |
| `per_token_loss` | 按一个训练 step 的平均有效 token 数归一化 | 支持；要求 `PrefetchGradAccDataLoader` 提供统计量 |

代码先完成 next-token label shift：在 labels 右侧补一个 `-100`，再取 `labels[..., 1:]`。这样 hidden position `t` 对应原始 label `t+1`，最后一个位置被忽略。之后根据 loss 类型计算全局或逐样本归一化系数 `α`，再把 labels 和必要的逐 token `α` 与 hidden states 按相同边界切块。

关键点是 `α` 按完整 batch/step 的语义生成，而不是每个 chunk 各自重新统计。因此 chunk 边界不会改变 loss 权重。

## 6. CCE：同时沿词表维流式计算

当前代码还提供 `impl_type: cce`。它源自 Cut Cross Entropy 思路，针对传统 ChunkLoss 仍需生成 `[块内 token 数,V]` logits 的问题，再沿词表维切成大小为 `vocab_tile_size` 的 tile。

### 6.1 前向

对每个词表区间 `[v0:v1]`：

1. 计算局部 logits：`tile = H W[v0:v1]^T`；
2. 用 online log-sum-exp 合并当前 tile 的最大值与指数和；
3. 如果正确类别落在当前 tile，提取对应 logit；
4. 复用轮转 tile buffer，不生成完整 `[N,V]` logits。

所有 tile 扫描完成后：

$$
\mathrm{CE}_{t}
=\mathrm{logsumexp}\!\left(\mathrm{logits}_{t}\right)
-\mathrm{logit}_{t,\,y_t}
$$

$\mathrm{logits}_t\in\mathbb{R}^{V}$ 是 token $t$ 的全部词表 logits，$\mathrm{logit}_{t,y_t}$ 是其中正确类别的那一个。`logsumexp` 定义为
$\mathrm{logsumexp}(z)=\log\sum_j e^{z_j}$。它把“先指数、求和、再取对数”写成一个可数值稳定实现的运算；实践中会先减去最大值，避免 $e^{z_j}$ 溢出。仍以 $z=[2,1,0]$、$y=0$ 为例，
$\mathrm{logsumexp}(z)\approx2.408$，所以 $\mathrm{CE}=2.408-2=0.408$，与前面的 softmax 计算一致。CCE 只改变这个量的流式计算方式，不改变交叉熵定义。

忽略位置贡献 0，最后对 token 求和。

### 6.2 反向

CCE 前向只保存 hidden states、head weight、labels 和每个 token 的 `logsumexp`。反向再次逐 tile 重算 logits，然后计算：

$$
G=\mathrm{softmax}\!\left(\mathrm{logits}\right)-\mathrm{onehot}\!\left(\mathrm{labels}\right),
\qquad
dH=GW,
\qquad
dW=G^{\mathsf T}H
$$

这里公式中的大写 $H$ 表示 hidden states 矩阵，不是前文单独表示 hidden size 的 $H$。为消除形状书写中的歧义，暂用 $D$ 表示 hidden size，则参与计算的 token 展平后有：$H\in\mathbb{R}^{N\times D}$，$W\in\mathbb{R}^{V\times D}$，$G\in\mathbb{R}^{N\times V}$。$N$ 是 token 数，$G$ 是 loss 对 logits 的梯度；`onehot(labels)` 把每个整数标签变成只有正确类别为 1、其余为 0 的向量。$dH$、$dW$ 分别是对 hidden states 和 lm_head 权重的梯度，$G^{\mathsf T}$ 是 $G$ 的转置。

例如 logits 为 $[2,1,0]$、label 为 0 时，softmax 约为 $[0.665,0.245,0.090]$，one-hot 向量为 $[1,0,0]$，所以
$G\approx[-0.335,0.245,0.090]$。这表示提高正确类别 logit 会降低 loss，而提高错误类别 logit 会增大 loss。随后按矩阵乘法计算 $GW$，即可把这三个类别方向上的影响传回 hidden states；计算 $G^{\mathsf T}H$，则把同一权重在所有 token 上的贡献累加起来。

`G` 也只在 tile 范围内存在。实现用两个设备 stream 分别承载矩阵乘和向量 kernel，并以 3 个轮转 buffer 和 event 建立依赖，重叠相邻 tile 的计算。

### 6.3 配置与限制

```yaml
features:
  loss_cfg:
    loss_type: default
  enable_chunk_loss: true
  enable_dynamic_chunk_loss: false
  chunkloss_plan:
    apply_module: lm_head
    impl_type: cce
    vocab_tile_size: 4096
    chunk_size: null
```

- CCE 当前只支持 `default` 和 `per_token_loss`，不支持 `per_sample_loss`。
- CCE 与动态 ChunkLoss 互斥。
- `vocab_tile_size` 默认 4096；增大它可减少 tile 和 event 数，但会增大轮转 buffer。
- CCE 下的 `chunk_size` 是可选的外层 token 分段大小。实现先把 `[B,S,H]` 展平为 `[N,H]`，因此这里按展平 token 数切段，不再是传统实现中“每个样本的序列长度”。
- CCE kernel 依赖 Triton/Ascend 环境，但采用延迟导入；未启用 CCE 时不会因缺少 Triton 而影响普通训练代码导入。

## 7. 如何选择参数

建议从传统静态 ChunkLoss 和 `chunk_size=1024` 开始，这是当前入口文档和样例配置的默认方向。

- 仍然 OOM：逐步减小 `chunk_size`，如 `1024 -> 512 -> 256`。
- batch size 经常变化：考虑动态模式，用 `total_chunk_size` 约束 `B × chunk_size`。
- 词表特别大，传统分块后单块 logits 仍是主要峰值：在满足 NPU/Triton 和 loss 类型限制时评估 CCE。
- 吞吐下降明显：适当增大 chunk；块越小，循环、kernel launch 和重复调度开销越多。
- CCE 调优：先用 `vocab_tile_size=4096`；显存允许且 host/event 开销突出时再评估 8192。

参数没有脱离模型形状和训练并行配置的通用最优值。应同时观察峰值显存、step time、loss 曲线和梯度稳定性。

## 8. 明确的使用边界

- 当前仓库文档将 ChunkLoss 定位为 FSDP2 特性；原生 FSDP2 是推荐路径，Megatron-FSDP2 是过渡路径。
- 原生 FSDP2 使用 ChunkLoss 时，`features.loss_cfg.loss_type` 不能保持 `raw`；训练引擎在 `raw` 时不会注入预置 loss 函数。
- 被替换的目标模块必须唯一匹配，并且必须是 `torch.nn.Linear`。
- 传统实现和 CCE 都不支持带 bias 的 lm_head；传入 bias 会抛出 `NotImplementedError`。
- ChunkLoss 优化的是输出投影与交叉熵阶段，不会减少 transformer 主干本身的激活、参数或优化器状态。
- 数学上等价不代表浮点结果逐 bit 相同。分块改变了归约顺序，合理预期是容差范围内一致。

## 9. 阅读下一篇

具体配置如何进入训练引擎、`lm_head` 如何被替换、自定义 autograd 如何保存梯度，以及各模型如何调用 loss，请继续阅读 [MindSpeed-MM ChunkLoss 源码实现](./implementation.md)。
