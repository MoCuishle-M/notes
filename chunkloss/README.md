# ChunkLoss 原理：以 Qwen3.5 35B 为例

## 1. 本文讨论什么

自回归语言模型训练到最后，需要把每个 token 的隐藏向量投影到整个词表，再用下一个 token 作为标签计算交叉熵。词表很大、序列很长时，输出分数张量 `logits` 会成为显存峰值。

ChunkLoss 的核心思想是：沿序列维把 token 切成若干块，每次只为一个块执行 logits、loss 和局部梯度计算，不让全部 `K` 个块的词表 logits 一起进入普通 autograd 图。所有块的 loss 与梯度按原来的归约规则合并，因此数学上的优化目标不变。

本文既解释算法，也沿实际代码说明配置如何把 `lm_head` 改造成 ChunkLoss 入口，以及前向和反向分别做了什么。文中的实例参数来自 Qwen3.5 35B 训练脚本、模型配置，以及 2026-09-01 在 8 张 NPU 上完成的一次 20 iteration smoke run。

## 2. 先认识参与 loss 的张量

### 2.1 符号和形状

| 符号 | 含义 | 形状 |
|---|---|---|
| `B` | 当前设备一次参与 loss 计算的样本数 | 标量 |
| `S` | 当前 batch 对齐后的序列长度 | 标量 |
| `D` | 每个 token 的隐藏维度 | 标量 |
| `V` | 词表大小，也就是分类类别数 | 标量 |
| `H` | 主干网络输出的 hidden states | `[B,S,D]` |
| `W` | 输出投影权重 | `[V,D]` |
| `Y` | 与 hidden states 对齐后的标签 | `[B,S]` |
| `Z` | 所有 token 对所有词表项的 logits | `[B,S,V]` |

为了避免把 hidden states 张量 `H` 和 hidden size 混在一起，本文用 `D` 表示 hidden size。

### 2.2 Qwen3.5 35B 实例参数

本次实例确认到的关键值如下：

| 参数 | 数值 | 含义 |
|---|---:|---|
| NPU 数量 | 8 | 一次训练使用 8 个 rank |
| `micro_batch_size` | 4 | 每个 rank 的本地 batch size，即 `B=4` |
| global batch size | 32 | `4 × 8`；它不是单个 rank 上 loss 张量的 `B` |
| `cutoff_len` | 16384 | 预处理允许的最大序列长度，不代表每个 batch 都达到该长度 |
| `D` | 2048 | 文本 hidden size |
| `V` | 248320 | 词表大小 |
| `C` | 512 | 每个样本在一个 loss 块中的最大 token 数 |
| hidden/weight dtype | BF16 | 实际运行探针观测到的输入和权重类型 |

这次 smoke run 的模型权重与 `lm_head` 形状来自 `Qwen3.5-35B-A3B`，但服务器工作区为了缩短验证时间，临时把文本 decoder 从完整层数裁成了 12 层。这个改动不改变本文关心的 `D=2048`、`V=248320`、`lm_head.weight=[248320,2048]` 和 ChunkLoss 切分方式；因此本文把日志用作 loss 路径和 shape 的实测证据，不把它当作完整 35B 模型的吞吐或显存基准。

输出投影权重的形状为 `[248320,2048]`，共有：

$$
248320 \times 2048 = 508559360
$$

个参数。仅按 BF16 计算，这个权重约占：

$$
\frac{508559360 \times 2}{2^{30}}
=0.9473\ \mathrm{GiB}
$$

ChunkLoss 不会缩小 `W`，它优化的是随 `B` 和 `S` 增长的 logits 及其临时状态。

## 3. 实际代码怎样进入 ChunkLoss

### 3.1 启动脚本和有效配置

服务器上的启动脚本是：

```text
examples/qwen3_5/finetune_qwen3_5_35B.sh
```

脚本设置 `NPUS_PER_NODE=8`，通过 `torchrun` 启动 `mindspeed_mm/fsdp/train/trainer.py`，读取 `examples/qwen3_5/qwen3_5_35B_config.yaml`。与本文直接相关的配置是：

本文对照的是服务器仓库基线 `874c57eb7cc9857e488d2f75fc6f7d3a33f22f36` 及其 2026-09-01 实验工作区。启动脚本、YAML 和 12 层 smoke 配置是该工作区的实验改动，所以这里记录的是实际运行值，不把它们误写成仓库默认值。

```yaml
features:
  loss_cfg:
    loss_type: default
  enable_chunk_loss: true
  chunkloss_plan:
    apply_module: lm_head
    chunk_size: 512

training:
  micro_batch_size: 4
```

`ChunkLossPlanConfig.impl_type` 的默认值是 `legacy`。配置没有覆盖它，所以本次实际执行的是本文介绍的 PyTorch legacy 路径，不是按词表 tile 流式计算的 CCE 路径。

同一份配置还启用了 `chunk_mbs=2`，但它只包装 `model.language_model.layers.{*}`：每个 decoder layer 依次处理两个大小为 2 的 micro-batch，再沿 batch 维拼接输出。`lm_head` 位于这些 layer 之后，所以它看到的仍是完整本地 batch `B=4`。这与探针里的 `hidden=(4,S,2048)` 一致。

### 3.2 初始化时：定位并改写 `lm_head.forward`

初始化阶段的调用关系是：

```text
FeaturesApplier.pre_fully_shard_apply(model)
└── FeaturesApplier.apply_chunkloss(model)
    ├── model.enable_chunk_loss = True
    ├── model.chunk_size = 512
    ├── get_chunkloss_module(model, plan)
    └── apply_chunkloss_module(lm_head)
```

相关代码位于：

- `mindspeed_mm/fsdp/features/apply_features.py`
- `mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_lm_head.py`

`get_chunkloss_module()` 按 `apply_module: lm_head` 匹配模块，要求只能命中一个模块，而且该模块必须是 `torch.nn.Linear`。随后 `apply_chunkloss_module()` 用 `types.MethodType` 替换它的 `forward`。替换前，`lm_head(hidden_states)` 返回 logits；替换后，它接收 `hidden_states` 和 loss closure，直接返回标量 loss：

```python
def chunkloss_forward(self, hidden_states, loss_func, labels=None):
    w = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
    b = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
    return loss_func(hidden_states, w, b, labels)
```

这里把 FSDP 使用的 `DTensor` 参数转成本 rank 的 local tensor。实测 `weight=[248320,2048]`、`bias=None`；legacy ChunkLoss 不支持非空 bias。

### 3.3 每个 batch：先构造 loss closure，再让模型调用它

每个训练 step 中，`TrainEngine.train_step()` 先取得 `batch_data`，然后执行：

```text
TrainEngine.set_loss_func(batch_data)
└── build_loss_func(loss_type="default", chunk_size=512, **batch_data)
    └── 返回捕获当前 batch labels 的 loss_func closure
```

`set_loss_func()` 把这个 closure 写入 `self.model.loss_function`。Qwen3.5 模型 forward 看到 `enable_chunk_loss=True` 后，不再先生成完整 logits，而是执行：

```python
logits = None
loss = self.lm_head(hidden_states[:, slice_indices, :], self.loss_function)
```

于是实际调用链为：

```text
Qwen3_5MoeForConditionalGeneration.forward
└── patched lm_head.forward(hidden_states, loss_func)
    └── loss_func(hidden_states, local_weight, None)
        ├── get_loss_func_params(...)
        └── chunk_loss(...)
            └── ChunkLoss.apply(...)
```

关键点是 `logits=None`：完整 `[B,S,V]` 从模型主路径上消失，ChunkLoss 内部改为逐块物化 `[B×C_i,V]`。

## 4. 普通 loss 为什么容易形成显存峰值

### 4.1 从 hidden states 到 logits

把前两个维度展平，令 token 总数 `N=B×S`：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 主干输出 | `H` | `[B,S,D]` |
| 展平 | `X` | `[N,D]` |
| 输出投影 | `Z=XWᵀ` | `[N,V]` |
| 恢复 batch 视角 | `Z` | `[B,S,V]` |

矩阵乘法为：

$$
Z=XW^{\mathsf T}
$$

其中 `X` 的形状是 `[N,D]`，`Wᵀ` 的形状是 `[D,V]`，所以结果 `Z` 的形状是 `[N,V]`。

对 Qwen3.5 35B，`V/D=248320/2048=121.25`。在元素类型相同的前提下，logits 的元素数是 hidden states 的 121.25 倍。

### 4.2 达到序列上限时有多大

假设一个 rank 上真的出现 `B=4`、`S=16384` 的 batch：

| 张量 | 形状 | 元素数 |
|---|---|---:|
| hidden states | `[4,16384,2048]` | 134217728 |
| labels | `[4,16384]` | 65536 |
| 完整 logits | `[4,16384,248320]` | 16273899520 |

完整 logits 的理论容量为：

$$
M_{\mathrm{logits}}=B\times S\times V\times b
$$

式中 `b` 表示每个元素的字节数。若 logits 是 BF16，`b=2`：

$$
\frac{4\times16384\times248320\times2}{2^{30}}
=30.3125\ \mathrm{GiB}
$$

若交叉熵前将 logits 转为 FP32，`b=4`：

$$
\frac{4\times16384\times248320\times4}{2^{30}}
=60.625\ \mathrm{GiB}
$$

这里只计算一个 logits 张量，没有计入 softmax、交叉熵、反向传播所需的其他状态。相比之下，同一组形状的 BF16 hidden states 只有：

$$
\frac{4\times16384\times2048\times2}{2^{30}}
=0.25\ \mathrm{GiB}
$$

所以 loss 阶段的问题不是标量 loss 本身，而是为了得到它而产生的巨大 `[B,S,V]` 中间张量。

## 5. 交叉熵为什么允许沿 token 分块

### 5.1 单个 token 的 loss

对第 `t` 个有效 token，它的 logits 是长度为 `V` 的向量 `z_t`，正确类别下标是 `y_t`。交叉熵可以写成：

$$
\ell_t
=-z_{t,y_t}
+\log\left(\sum_{j=0}^{V-1}\exp(z_{t,j})\right)
$$

例如 `z_t=[2,1,0]`，正确类别为 0。softmax 后正确类别概率约为 0.665，因此：

$$
\ell_t=-\log(0.665)\approx0.408
$$

一个 token 的交叉熵只依赖该 token 自己的 logits 和标签，不依赖同一 batch 中其他 token 的 logits。

### 5.2 一个 batch 的 loss 是逐 token 求和再归一化

令 `I` 为所有有效标签位置的集合，`M` 为有效 token 数。默认平均 loss 为：

$$
L=\frac{1}{M}\sum_{t\in\mathcal I}\ell_t
$$

如果把 `I` 拆成互不重叠的 `K` 个块，每个有效位置只属于一个块，那么：

$$
L
=\frac{1}{M}
\sum_{i=0}^{K-1}
\sum_{t\in\mathcal I_i}\ell_t
$$

这只是改变求和顺序，没有改变任何 token 的 logits、标签、权重或总归一化因子 `M`。

假设四个有效 token 的 loss 分别为 `[0.2,0.4,0.3,0.1]`，分成前两个和后两个两块：

$$
L
=\frac{(0.2+0.4)+(0.3+0.1)}{4}
=0.25
$$

一次性相加同样得到 `(0.2+0.4+0.3+0.1)/4=0.25`。

这里有一个关键条件：`M` 必须按整个 batch 统计，不能在每个块内各自求平均后再直接相加。否则较短块会被错误地赋予与较长块相同的权重。

## 6. ChunkLoss 怎样切分张量

### 6.1 沿序列维切，而不是沿 hidden 维或词表维切

设每块的最大序列长度为 `C`，块数为：

$$
K=\left\lceil\frac{S}{C}\right\rceil
$$

第 `i` 块实际长度记为 `C_i`。除最后一块外通常有 `C_i=C`；最后一块取剩余 token，所以可能小于 `C`。

每块的形状变化如下：

| 阶段 | 形状 |
|---|---|
| hidden 块 `H_i` | `[B,C_i,D]` |
| 展平 hidden `X_i` | `[B×C_i,D]` |
| 标签块 `Y_i` | `[B,C_i]`，展平后为 `[B×C_i]` |
| 当前块 logits `Z_i` | `[B×C_i,V]` |
| 当前块逐 token loss | `[B×C_i]` |
| 当前块归约结果 | 标量 |

词表仍然完整保留，因为每个 token 的交叉熵必须比较全部 `V` 个类别。ChunkLoss 缩小的是同一时刻参与投影的 token 数。

### 6.2 Qwen3.5 35B 的满块

本例 `B=4`、`C=512`、`D=2048`、`V=248320`。一个满块经历：

```text
hidden block     [4,512,2048]
    ↓ 展平 B 和 C
hidden matrix    [2048,2048]
    × weightᵀ    [2048,248320]
    ↓
logits block     [2048,248320]
labels block     [4,512] -> [2048]
    ↓ 逐 token 交叉熵并按整个 batch 的有效 token 数归一化
chunk loss       scalar
```

满块 logits 有 `508559360` 个元素，容量约为：

| dtype | 单个满块 logits |
|---|---:|
| BF16 | 0.9473 GiB |
| FP32 | 1.8945 GiB |

这也是为什么 `C` 的选择直接影响 loss 阶段峰值：`C` 减半，单块 logits 的元素数也近似减半。

### 6.3 达到 16384 上限时

因为 `16384/512=32`，序列可以整齐地分为 32 块：

```text
完整 hidden states: [4,16384,2048]

chunk 0:  [4,512,2048] -> logits [2048,248320]
chunk 1:  [4,512,2048] -> logits [2048,248320]
...
chunk 31: [4,512,2048] -> logits [2048,248320]
```

只看一个逻辑块的 logits 载荷，FP32 理论容量从完整张量的 60.625 GiB 降到单个满块的约 1.8945 GiB，比例为 32 倍。这里说的是单块张量元素数的理论缩减，不等于训练总峰值也必然缩减 32 倍，因为参数、hidden states、梯度、相邻循环对象及其他网络状态仍然存在。

### 6.4 最后一块不足 512 时

若 `S=576`：

$$
K=\left\lceil\frac{576}{512}\right\rceil=2
$$

形状为：

| 块 | hidden | 展平 hidden | logits | labels |
|---|---|---|---|---|
| 第 0 块 | `[4,512,2048]` | `[2048,2048]` | `[2048,248320]` | `[4,512]` |
| 第 1 块 | `[4,64,2048]` | `[256,2048]` | `[256,248320]` | `[4,64]` |

第二块只计算实际存在的 64 个 token，不需要补成 512。它的 FP32 logits 约为 0.2368 GiB。

## 7. 源码怎样完成逐块 loss 和梯度

核心实现位于：

```text
mindspeed_mm/fsdp/loss/loss_func.py
mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss.py
```

### 7.1 标签先整体移位和计数，再按同一边界切块

`get_loss_func_params()` 先处理完整 `[B,S]` labels：

```python
labels = F.pad(labels, (0, 1), value=-100)
shift_labels = labels[..., 1:].contiguous()
loss_mask = shift_labels > -1
alpha = loss_mask.sum()
chunk_labels = torch.split(shift_labels, chunk_size, dim=1)
```

`F.pad(..., (0,1))` 在末尾补一个 `-100`，再取 `[...,1:]`，所以 shape 仍是 `[B,S]`，但语义变成“位置 `t` 预测原标签位置 `t+1`”；最后一个位置被忽略。`alpha` 是整个本地 batch 的有效 token 数 `M`，它在切块前计算，并作为同一个标量传给每个块。

`loss_type=default` 时，每块执行“本块有效 token loss 求和，再除以全 batch 的 `M`”。最后把所有块相加，正好得到：

$$
L
=\sum_{i=0}^{K-1}
\frac{\sum_{t\in\mathcal I_i}\ell_t}{M}
=\frac{\sum_{t\in\mathcal I}\ell_t}{M}
$$

这就是分块后平均 loss 不变的代码依据。

### 7.2 `ChunkLoss.forward()` 预分配最终梯度

`ChunkLoss` 是自定义 `torch.autograd.Function`。它的 forward 先分配：

```python
grad_inputs = torch.empty_like(hidden_states)  # [B,S,D]
grad_weight = torch.zeros_like(head_weight)    # [V,D]

grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)
```

因此实现从一开始就明确了两种合并方式：hidden 梯度写回 `[B,S,D]` 的对应切片，公共 `lm_head.weight` 的梯度累加到一个 `[V,D]` 张量。

### 7.3 每个循环只计算一个块，不累计 `K` 份计算图

forward 的核心循环可缩写为：

```python
for hidden_chunk, grad_input_chunk, loss_kwargs in zip(...):
    (dx, dw), (chunk_loss_value, _) = torch.func.grad_and_value(
        calculate_lm_loss,
        argnums=(0, 1),
        has_aux=True,
    )(hidden_chunk, head_weight, None, **loss_kwargs)

    accumulated_loss.add_(chunk_loss_value)
    grad_input_chunk.copy_(dx)
    grad_weight.add_(dw)
```

`calculate_lm_loss()` 内部才真正执行投影和交叉熵：

```python
hidden_chunk = hidden_chunk.reshape(-1, hidden_chunk.size(-1))
logits = F.linear(hidden_chunk, head_weight).float()
loss = fixed_cross_entropy(logits, shift_labels.reshape(-1), alpha=alpha, reduction="sum")
```

对本例满块，逐行 shape 是：

```text
hidden_chunk                    [4,512,2048]
reshape                         [2048,2048]
head_weight                     [248320,2048]
F.linear 后、转 FP32 的 logits   [2048,248320]
shift_labels.reshape(-1)        [2048]
chunk_loss_value                scalar
dx                              [4,512,2048]
dw                              [248320,2048]
```

`torch.func.grad_and_value()` 在自定义 Function 的 forward 阶段同时得到当前块的 loss、hidden 梯度和 weight 梯度。代码不把 `K` 个块的计算图存入列表；每轮得到的局部梯度立刻复制或累加到最终缓冲区。

需要精确区分“未使用”和“立刻释放”：`(chunk_loss_value, _)` 中的 `_` 在 Python 里仍是普通变量，可能一直引用该轮作为辅助返回值的 logits，直到下一次赋值。因此源码能保证不会累计 `K` 份 logits 图，但不能仅凭这段代码断言设备峰值严格等于一份单块 logits；相邻循环求值、算子 workspace 和缓存分配器都可能造成额外重叠。第 6 节的 GiB 数值应理解为张量载荷估算，不是 allocator 峰值承诺。

### 7.4 外层 backward 只缩放并返回已算好的梯度

所有块处理完后，forward 只保存 `grad_inputs` 和累计的 `grad_weight`：

```python
ctx.save_for_backward(grad_inputs, grad_weight)
```

外层训练调用 `loss.backward()` 时，`ChunkLoss.backward()` 不再重新生成 logits。它读取这两个张量；如果上游标量梯度不是 1，就同时乘以该标量，然后返回：

```text
d(hidden_states)  = grad_inputs   [B,S,D]
d(head_weight)    = grad_weight   [V,D]
d(head_bias)      = None
```

所以 legacy ChunkLoss 的实际策略不是“把 K 个 logits 计算图都留到普通 backward”，而是“在自定义 forward 内逐块完成局部求导，外层 backward 只交回合并后的梯度”。

## 8. 真实首步 shape：为什么不能把上限当成实际长度

`cutoff_len=16384` 只是样本截断上限。实际数据经过动态 padding 后，每个 rank 的首个 batch 有不同的 `S`。运行探针观测结果如下：

| rank | hidden states | 分块长度 |
|---:|---|---|
| 0 | `[4,504,2048]` | 504 |
| 1 | `[4,536,2048]` | 512 + 24 |
| 2 | `[4,496,2048]` | 496 |
| 3 | `[4,576,2048]` | 512 + 64 |
| 4 | `[4,592,2048]` | 512 + 80 |
| 5 | `[4,576,2048]` | 512 + 64 |
| 6 | `[4,552,2048]` | 512 + 40 |
| 7 | `[4,592,2048]` | 512 + 80 |

所有 rank 上都确认：

- hidden states 为 BF16；
- 输出权重为 BF16，形状 `[248320,2048]`；
- bias 不存在；
- `B=4`；
- 标签块与 hidden 块在前两个维度严格对齐。

以 rank 0 为例，`S=504` 小于 `C=512`，所以只有一个块。此时 ChunkLoss 不会从 token 分块中获得峰值缩减，完整 FP32 logits 理论容量约为 1.8649 GiB。

以 rank 5 为例，`S=576`，完整 FP32 logits 理论容量约为 2.1313 GiB，最大块为 1.8945 GiB，缩减比例只有 `576/512=1.125` 倍。

因此本实例中：

- 对当前首步的短序列，ChunkLoss 的收益有限；
- 随 `S` 增大，块数增加，收益逐渐明显；
- 到 `S=16384` 时，单看 logits 才达到理论上的 32 倍缩减。

这说明评估 ChunkLoss 不能只看 `cutoff_len`，还必须看训练数据实际产生的序列长度分布。

## 9. 标签为什么要移位，忽略位置怎样处理

### 9.1 next-token prediction 的对齐关系

自回归训练中，位置 `t` 的 hidden state 用来预测位置 `t+1` 的 token。假设原始 token 是：

```text
input:  [A, B, C, D]
target: [B, C, D, ignore]
```

因此 hidden states 和移位后标签仍然都是 `[B,S,...]` 的长度；最后一个位置没有下一个 token，标签设为忽略值。

多模态监督微调还会把用户提示、视觉占位符、padding 等不参与监督的位置标为忽略值。这些位置仍可参与主干网络上下文计算，但不贡献语言模型交叉熵。

### 9.2 分块时 hidden 和 labels 必须使用相同边界

如果 hidden states 的第 0 块覆盖序列位置 0 到 511，那么标签第 0 块也必须覆盖完全相同的位置。否则某个位置的预测会与别的位置标签比较，loss 将失去语义。

忽略位置的逐 token loss 和 logits 梯度都按 0 处理。有效 token 总数 `M` 在切块前按整个 batch 统计，所以一个块中即使大部分位置被忽略，也不会改变其他块的权重。

## 10. 为什么分块后的梯度也与完整计算相同

### 10.1 对 hidden states 的梯度

不同 token 的 hidden states 位于不同的行。第 `i` 块只产生自己位置的 hidden 梯度 `dX_i`，最后按原序列顺序拼回：

$$
\frac{\partial L}{\partial X}
=\mathrm{concat}
\left(
\frac{\partial L}{\partial X_0},
\frac{\partial L}{\partial X_1},
\ldots,
\frac{\partial L}{\partial X_{K-1}}
\right)
$$

因为每个 token 只属于一个块，不会出现同一 hidden 行需要跨块相加的问题。

### 10.2 对输出权重的梯度

所有块共用同一个 `W`。每块产生一份形状 `[V,D]` 的权重梯度，完整梯度是逐元素相加：

$$
\frac{\partial L}{\partial W}
=\sum_{i=0}^{K-1}
\frac{\partial L_i}{\partial W}
$$

例如某个权重元素从两个块分别得到 `0.3` 和 `-0.1`，累加结果为 `0.2`，与把两个块一次性放进同一矩阵计算得到的结果一致。

### 10.3 从 softmax 梯度看形状

对一个块，令 `G_i` 是 loss 对 logits 的梯度，形状为 `[B×C_i,V]`。忽略归一化系数时，每个有效 token 行满足：

$$
G_i=\mathrm{softmax}(Z_i)-\mathrm{onehot}(Y_i)
$$

于是：

$$
\frac{\partial L_i}{\partial X_i}=G_iW
$$

$$
\frac{\partial L_i}{\partial W}=G_i^{\mathsf T}X_i
$$

形状检查如下：

| 计算 | 输入形状 | 输出形状 |
|---|---|---|
| `G_i W` | `[B×C_i,V] × [V,D]` | `[B×C_i,D]` |
| `G_iᵀ X_i` | `[V,B×C_i] × [B×C_i,D]` | `[V,D]` |

这正好分别对应 hidden 梯度和输出权重梯度。

## 11. 为什么必须及时完成每块的梯度计算

如果只是把 forward 写成循环，却把所有块的计算图一直保留到最后统一 backward，那么虽然某一时刻没有完整 `[B,S,V]`，多个块的反向状态仍可能同时驻留，显存收益会被削弱。

legacy ChunkLoss 的实现顺序是：

1. 取出一个 hidden/label 块；
2. 生成该块 logits；
3. 计算该块 loss；
4. 立即得到该块的 hidden 梯度和权重梯度；
5. 保存 hidden 梯度到对应序列位置，累加权重梯度；
6. 不把本块计算图加入跨块列表，再处理下一块。

跨越整个 `ChunkLoss.forward()` 必须保留的是完整形状的 hidden 梯度、累计后的权重梯度和标量 loss，而不是所有 token 的词表 logits。辅助返回值、算子 workspace 与分配器缓存仍会影响实际峰值，所以最终应以设备实测为准。

## 12. 显存收益怎样估算

只考虑 logits，普通计算的理论容量是：

$$
M_{\mathrm{full}}=B\times S\times V\times b
$$

ChunkLoss 的单个逻辑块 logits 容量近似是：

$$
M_{\mathrm{chunk}}
=B\times\min(C,S)\times V\times b
$$

当 `S` 大于 `C` 时，logits 部分的理论缩减比例约为：

$$
R
=\frac{M_{\mathrm{full}}}{M_{\mathrm{chunk}}}
=\frac{S}{C}
$$

这个估算有五个边界：

1. 它只计算 logits，不是完整训练显存；
2. hidden 梯度 `[B,S,D]` 和权重梯度 `[V,D]` 仍需要存在；
3. 交叉熵和矩阵乘可能有额外 workspace；
4. `S` 不超过 `C` 时只有一个块，理论比例为 1。
5. legacy 源码的辅助返回值和缓存分配行为可能让相邻块的存储短暂重叠，所以它不是严格的 allocator 峰值公式。

因此应使用真实序列长度分布估算，而不是只代入最大长度。

## 13. `chunk_size` 的时间—显存权衡

`C` 越小：

- 单块 logits 越小，峰值显存越低；
- 块数 `K` 越多；
- 循环调度、矩阵乘启动、梯度累加次数越多；
- 小矩阵更可能降低设备利用率，训练时间可能增加。

`C` 越大：

- 块数更少，计算更接近一次大矩阵乘；
- 单块 logits 更大，显存收益下降；
- 当 `C` 不小于实际 `S` 时，等价于没有进行 token 分块。

所以 `chunk_size` 不是越小越好。合理选择方式是：先根据设备可承受的 loss 阶段峰值确定上限，再在不超显存的候选值中选择吞吐更好的值。

对本例，`C=512` 的含义是每个 rank、每个样本每次最多处理 512 个序列位置；一个满块实际同时处理 `B×C=2048` 个 token。

## 14. ChunkLoss 改变了什么，没有改变什么

ChunkLoss 改变的是：

- logits 的物化粒度；
- loss 和局部梯度的计算顺序；
- 显存峰值与矩阵调度次数之间的权衡。

ChunkLoss 不改变的是：

- 输出投影权重 `W`；
- 每个 token 与全部词表类别的比较；
- next-token 标签对齐关系；
- 忽略位置；
- 整个 batch 的 loss 归一化语义；
- hidden 梯度和权重梯度的数学表达式；
- transformer 主干的参数、激活和优化器状态。

数学上，只要标签边界、忽略掩码和全局归一化保持一致，ChunkLoss 与完整 logits 计算具有相同的 loss 和梯度。实际 BF16/FP32 计算会因矩阵形状、求和顺序和 BF16 权重梯度累加产生舍入差异，因此不能承诺逐 bit 一致。

一句话概括：legacy ChunkLoss 把“完整 logits 后统一 backward”改成“按序列分块，在自定义 forward 内逐块求 loss 和局部梯度，再由自定义 backward 交回合并结果”，以更多块级计算和梯度累加换取更低的 loss 阶段显存压力。
