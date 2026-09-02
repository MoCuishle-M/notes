# MindSpeed-MM ChunkMBS：从原理、源码到真实 shape

ChunkMBS 是 MindSpeed-MM 原生 FSDP2 训练中的 **layer-level batch 切分**：训练器仍把一个完整 micro batch 交给模型，但被选中的 Transformer 层会在一次 module 调用内部，将 batch 切成更小的块顺序计算，最后沿 batch 维拼回完整输出。

它的目标不是减少样本数，也不是改变 loss，而是在保持较大训练 MBS 的同时：

- 让一次 FSDP 参数 unshard 服务更多样本；
- 用较小的单块计算控制层内瞬时激活；
- 配合 recompute 和 activation offload，避免完整大 batch 的层内激活同时驻留在设备上。

本文基于 2026-09-02 的远端 MindSpeed-MM `master`：

- 代码提交：`874c57eb7cc9857e488d2f75fc6f7d3a33f22f36`；
- 运行脚本：`examples/qwen3_5/finetune_qwen3_5_35B.sh`；
- 模型：Qwen3.5-35B-A3B；
- 8 个 NPU rank，EP=8；
- 每个 rank 的 `micro_batch_size=4`，`gradient_accumulation_steps=1`；
- `chunk_mbs=2`，ChunkMBS 包装 `model.language_model.layers.{*}`；
- 对第一层加只读运行探针并完成 1 个训练 step，进程退出码为 0。

远端工作树存在模型实验相关的本地修改，但本文引用的 ChunkMBS 核心文件相对上述提交没有 diff。文中的 shape 来自这次实际运行，而不是根据配置猜测。

## 1. 先区分三个容易混淆的量

| 名称 | 本例数值 | 含义 |
|---|---:|---|
| training MBS | 4 | 单个数据并行 rank 一次模型 forward 接收的样本数 |
| `chunk_mbs` | 2 | 一个目标层每次内部 forward 最多处理的样本数 |
| GBS | 32 | 8 个 rank 一次参数更新共处理的样本数，即 `4 × 8 × 1` |

`chunk_mbs` 表示 **每个 chunk 的 batch size**，不是 chunk 数。设完整 batch size 为 `B`，每块大小为 `C`，块数为 `K`：

$$
K=\left\lceil\frac{B}{C}\right\rceil
$$

代码没有调用浮点除法和 `ceil`，而是用正整数等价式：

```python
num_micros = (full_batch_size + chunk_mbs - 1) // chunk_mbs
```

本例中：

```text
B = 4
C = 2
K = (4 + 2 - 1) // 2 = 2

chunk 0: [0:2)
chunk 1: [2:4)
```

若 `B=5`、`C=2`，则边界是 `[0:2)`、`[2:4)`、`[4:5)`，最后一块允许不足 2 个样本。

## 2. 为什么需要 ChunkMBS

FSDP2 平时只保存参数分片。目标层开始计算前，需要将该层参数恢复为可计算状态；计算后再按策略 reshard：

```text
参数分片
  │
  ├─ FSDP pre-forward：unshard / all-gather
  ▼
完整参数 ── layer forward ── FSDP post-forward：reshard
```

有两种朴素选择：

1. 小 MBS 加梯度累积：层内激活小，但同一层会被多次独立调用，参数 unshard 的次数也可能增加；
2. 直接增大 MBS：一次 unshard 覆盖更多样本，但层内激活随 batch 增大，容易 OOM。

ChunkMBS 把“训练器看到的大 MBS”和“层内部一次计算的小 MBS”分开：

```text
一次 FSDP layer 调用，输入 B=4
  │
  ├─ chunk 0，B=2 ── 原始 layer forward ── output 0
  └─ chunk 1，B=2 ── 原始 layer forward ── output 1
                                              │
                                      cat(batch_dim=0)
                                              ▼
                                         输出 B=4
```

两个 chunk 是顺序执行的，不是并行执行。收益来自参数通信摊薄和激活控制，不来自 chunk 间并发。

## 3. Qwen3.5 实测 shape：到底切了什么

### 3.1 切分前

rank 0 第一层收到的真实参数如下：

| 参数 | shape / 类型 | 是否配置切分 |
|---|---|---|
| `hidden_states` | `[4, 504, 2048]`，BF16 | 是，位置参数 0 |
| `position_embeddings[0]` | `[4, 504, 64]`，BF16 | 是，随 tuple 递归切分 |
| `position_embeddings[1]` | `[4, 504, 64]`，BF16 | 是，随 tuple 递归切分 |
| `attention_mask` | `[4, 504]`，Int64 | 是 |
| `position_ids` | `[4, 504]`，Int64 | 是 |
| `rope_deltas` | `[4, 1]`，Int64 | 是 |
| `cache_position` | `[504]`，Int64 | 否，它没有 batch 维 |
| `seq_mask` | `[4, 504]`，Int64 | 否，当前 YAML 未列出 |
| `past_key_values` | `None` | 否 |
| `use_cache` | `bool` | 否 |

这里 `504` 是这一个 rank、这一个 step 数据整理后的实际序列长度；配置中的 `cutoff_len=16384` 只是上限，不代表每个 step 的序列长度固定为 16384。

### 3.2 两次内部 forward

`batch_dim=0`、`chunk_mbs=2`，所以第一层实际执行两次：

```text
chunk 0
  hidden_states:       [2, 504, 2048] BF16
  position_embeddings: ([2, 504, 64], [2, 504, 64]) BF16
  attention_mask:      [2, 504] Int64
  position_ids:        [2, 504] Int64
  rope_deltas:         [2, 1] Int64
  output:              [2, 504, 2048] BF16

chunk 1
  hidden_states:       [2, 504, 2048] BF16
  position_embeddings: ([2, 504, 64], [2, 504, 64]) BF16
  attention_mask:      [2, 504] Int64
  position_ids:        [2, 504] Int64
  rope_deltas:         [2, 1] Int64
  output:              [2, 504, 2048] BF16
```

`cache_position=[504]` 与 `seq_mask=[4,504]` 在两个 chunk 中保持原值，因为它们没有出现在 `chunk_kwarg_names` 中。最后：

```python
torch.cat([output_0, output_1], dim=0)
```

得到 `[4,504,2048]` BF16，下一层再次按相同方式切成两个 `[2,504,2048]`。所有被包装的 language layer 执行完后，最终 norm 和 `lm_head` 看到的仍是完整 `B=4`，所以 ChunkMBS 不会把模型最终 batch 永久改成 2。

### 3.3 用元素数理解显存尺度

仅计算第一层输入 `hidden_states`：

```text
完整输入元素数 = 4 × 504 × 2048 = 4,128,768
完整 BF16 字节数 = 4,128,768 × 2 = 8,257,536 bytes ≈ 7.875 MiB

单块元素数 = 2 × 504 × 2048 = 2,064,384
单块 BF16 字节数 = 2,064,384 × 2 = 4,128,768 bytes ≈ 3.938 MiB
```

这只说明单次 layer 内部计算的 batch 尺度从 4 降为 2，**不是整层峰值显存的精确预测**。真实峰值还包含 attention、MoE、通信 buffer、参数、梯度、输出列表以及 allocator 行为。

另一个重要边界是：两个 `[2,504,2048]` 输出在 `outputs` 列表中都会保留到 `torch.cat`，拼接还会生成完整 `[4,504,2048]` 输出。因此 ChunkMBS 不减少最终输出大小；它主要控制每次原始 layer forward 内部产生的临时激活。

## 4. 源码调用链

配置到运行时的完整路径是：

```text
qwen3_5_35B_config.yaml
  │
  ▼
FeatureArguments.enable_chunk_mbs
ChunkMbsPlanConfig
  │
  ▼
Trainer.get_model()
  │
  ├─ FeaturesApplier.pre_fully_shard_apply(model)
  │    ├─ apply_recompute_models(model)
  │    ├─ apply_activation_offload_modules(model)
  │    ├─ apply_chunk_mbs(model)
  │    │    ├─ get_chunkmbs_modules(...)
  │    │    └─ apply_chunkmbs_module(...)
  │    │           └─ module.forward = chunk_mbs_forward(...)(old_forward)
  │    └─ apply_chunkloss(model)
  │
  ├─ model_parallel_applier(model)
  │    └─ fully_shard(target modules)
  │
  └─ training forward
       └─ decoder_layer(...)
            └─ ChunkMBS wrapper：slice -> forward × K -> cat
```

对应源码位置：

| 职责 | 文件与位置 |
|---|---|
| 配置结构 | `mindspeed_mm/fsdp/params/feature_args.py:198-222,255-259` |
| 特性安装顺序 | `mindspeed_mm/fsdp/features/apply_features.py:114-140` |
| 安装发生在 fully-shard 前 | `mindspeed_mm/fsdp/train/trainer.py:208-222` |
| 模块匹配、切片、循环与拼接 | `mindspeed_mm/fsdp/features/communication/chunk_mbs.py:14-192` |
| Qwen 层调用参数 | `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py:1976-2009` |
| Qwen decoder layer forward | `mindspeed_mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py:1332-1383` |

## 5. 核心实现逐步拆解

### 5.1 匹配目标 module

`get_chunkmbs_modules()` 对 plan 中的每个模式遍历 `model.named_modules()`，用 `module_name_match()` 判断是否命中：

```python
for plan_name in plan:
    for name, module in modules.named_modules():
        if module_name_match(plan_name, name):
            matched_modules.append((name, module))
```

一个目标都没有时会抛出 `RuntimeError`。当前函数不去重；若两个模式命中同一个 module，该 module 可能被重复包装。

### 5.2 monkey-patch forward

`apply_chunkmbs_module()` 保存 module 当前的 bound forward，并用 decorator 返回的新函数替换它：

```python
module.forward = chunk_mbs_forward(
    chunk_mbs=cfg.chunk_mbs,
    batch_dim=cfg.batch_dim,
    chunk_arg_indexs=cfg.chunk_arg_indexs,
    chunk_kwarg_names=cfg.chunk_kwarg_names,
)(module.forward)
```

这是实例级替换，不会修改 module 类，也没有自定义 NPU 算子。

### 5.3 推断完整 batch size

wrapper 优先使用第一个位置参数配置项：

```python
if chunk_arg_indexs:
    full_batch_size = args[chunk_arg_indexs[0]].shape[batch_dim]
elif chunk_kwarg_names:
    full_batch_size = kwargs[chunk_kwarg_names[0]].shape[batch_dim]
else:
    raise ValueError("No tensor input found to infer batch size.")
```

它只从一个输入推断 `B`，不会自动校验所有被切 Tensor 的 batch 长度是否相同。若 `B <= chunk_mbs`，直接调用旧 forward，不执行 slice 和 cat。

### 5.4 递归切输入

`_slice_batch_recursive()` 的行为是：

```text
Tensor     -> 沿 batch_dim 取 [start:end]
tuple/list -> 保持容器类型并递归处理每个元素
dict       -> 保持 key 并递归处理每个 value
其他对象   -> 原样传入
```

Tensor 分支的实际代码等价于：

```python
slices = [slice(None)] * data.ndim
slices[batch_dim] = slice(start, end)
micro = data[tuple(slices)]
```

这通常是 view，并保留标准 autograd 关系。Qwen 的 `position_embeddings` 是包含两个 Tensor 的 tuple，因此一个 kwarg 配置就会递归切两个 Tensor。

配置粒度是整个 arg/kwarg，不支持只切嵌套容器中的某一个路径。若一个被配置的 dict 同时含 batch Tensor 和非 batch Tensor，内部所有 Tensor 都会沿同一个 `batch_dim` 被切。

### 5.5 顺序调用并拼接输出

每一块重新构造 `micro_args` 和 `micro_kwargs`，再调用捕获的旧 forward：

```python
for i in range(num_micros):
    start = i * chunk_mbs
    end = min(start + chunk_mbs, full_batch_size)
    out = forward_func(*micro_args, **micro_kwargs)
    outputs.append(out)
```

输出支持：

- 单个 Tensor：直接沿 `batch_dim` 拼接；
- 一层 tuple/list：每个位置分别 `torch.cat`；
- 其他类型：抛出 `TypeError`。

tuple/list 拼接并不是递归实现，内部含 `None`、标量、dict、嵌套容器或 Hugging Face `ModelOutput` 都不属于当前直接支持范围。

## 6. 为什么前向和反向通常等价

设被包装层为 `F`，共享参数为 `W`，完整输入沿 batch 维切为 `X_0` 到 `X_{K-1}`，并用 `C` 表示沿 batch 维拼接。ChunkMBS 的前向是：

$$
Y=C\left(F(X_0;W),F(X_1;W),\ldots,F(X_{K-1};W)\right)
$$

若 `F` 的每个样本不读取其他样本的数据，则它与一次完整 batch 计算相同：

$$
F\left(C(X_0,X_1,\ldots,X_{K-1});W\right)=Y
$$

`slice`、原始 forward 和 `torch.cat` 都是 PyTorch 标准可微操作，所以不需要自定义 `autograd.Function`。

反向时，`cat` 的上游梯度先按原边界拆回每个 chunk。所有 chunk 使用同一参数 `W`，因此参数梯度由 autograd 相加：

$$
\frac{\partial L}{\partial W}
=
\sum_{i=0}^{K-1}
\frac{\partial L}{\partial F(X_i;W)}
\frac{\partial F(X_i;W)}{\partial W}
$$

等价性有明确前提：

- 层内没有跨样本归约、batch statistics 或样本间 attention；
- 所有随 batch 变化且会被读取的输入都使用相同边界切分；
- 输出能按 batch 维无损拼接；
- 随机算子允许容差内而不是逐 bit 对齐，因为分块会改变调用次数和随机数消费顺序；
- 浮点归约顺序变化也可能产生小数值差异。

## 7. 为什么一次 layer 调用只触发一组 FSDP forward 边界

`Trainer.get_model()` 先执行 `pre_fully_shard_apply(model)`，再执行 `model_parallel_applier(model)`。也就是说，ChunkMBS 先成为 module 的 forward，随后 FSDP2 才管理这个 module。

运行时关系是：

```text
FSDP module pre-forward
  └─ ChunkMBS wrapper
       ├─ chunk 0 -> 原始 layer forward
       ├─ chunk 1 -> 原始 layer forward
       └─ cat
FSDP module post-forward
```

因此 `K` 次原始 layer forward 位于一次外层 module 调用内部。相对于让训练器用 `K` 次独立小 MBS forward 做梯度累积，ChunkMBS 可以让一次 forward 参数 unshard 覆盖这 `K` 块计算。

需要避免把这个描述扩大成“整个训练永远只通信一次”。不同层、反向阶段、prefetch 策略、reshard 配置和不同 optimizer step 仍有各自的 FSDP 参数生命周期。

## 8. 与 recompute、activation offload 的 wrapper 顺序

`pre_fully_shard_apply()` 的实际安装顺序是：

```python
self.apply_recompute_models(model)
self.apply_activation_offload_modules(model)
self.apply_chunk_mbs(model)
self.apply_chunkloss(model)
```

每个功能都捕获当时的旧 forward，再装一层新 wrapper，所以对 language layer 的调用关系是：

```text
ChunkMBS wrapper
  └─ activation-offload wrapper
       └─ recompute/checkpoint wrapper
            └─ original decoder layer forward
```

这意味着 ChunkMBS 先切输入，每个 `[2,504,2048]` 再分别进入 offload 和 checkpoint 边界。

ChunkMBS 单独使用时，autograd 仍可能为所有 chunk 保留反向所需的计算图，不能简单认为“第一个 chunk forward 结束后，它的全部激活立即释放”。当前 Qwen 配置同时使用：

- recompute：forward 少保存层内中间结果，backward 按 chunk 重算；
- activation offload：把符合条件的保存 Tensor 移到 Host，需要时再取回；
- ChunkMBS：使每次原始 layer forward 的输入和临时计算按 `C=2` 而不是 `B=4` 展开。

三者共同决定峰值显存与性能，不能只根据 `B/C` 推断最终节省比例。

## 9. Qwen3.5 配置与每个字段的真实含义

本次运行使用：

```yaml
features:
  recompute: true
  recompute_plan:
    apply_modules:
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}

  enable_activation_offload: true
  activation_offload_plan:
    apply_modules:
      - model.visual.blocks.{*}
      - model.language_model.layers.{*}

  enable_chunk_mbs: true
  chunkmbs_plan:
    apply_modules:
      - model.language_model.layers.{*}
    chunk_mbs: 2
    batch_dim: 0
    chunk_arg_indexs: [0]
    chunk_kwarg_names:
      - position_embeddings
      - position_ids
      - rope_deltas
      - attention_mask

training:
  micro_batch_size: 4
  gradient_accumulation_steps: 1
```

| 字段 | 实现语义 |
|---|---|
| `enable_chunk_mbs` | 是否安装 ChunkMBS wrapper |
| `apply_modules` | 交给 `module_name_match()` 的目标 module 模式列表 |
| `chunk_mbs` | 每块最多包含的样本数 |
| `batch_dim` | 对所有被切 Tensor 使用的 batch 维 |
| `chunk_arg_indexs` | 需要切分的位置参数索引 |
| `chunk_kwarg_names` | 需要切分的关键字参数名 |

### `seq_mask` 的实测边界

Qwen text model 在调用 decoder layer 前执行 `kwargs["seq_mask"] = attention_mask`。本次探针确认：`attention_mask` 被切成 `[2,504]`，但 `seq_mask` 因未配置而仍是 `[4,504]`。

当前运行能够完成，是因为 `features.skip_moe_pad_tokens` 没有开启，EP dispatcher 不读取 `seq_mask` 做 pad-token 过滤。若开启该功能，`ep_dispatcher.py:96-104` 会展平 `seq_mask` 并用它索引当前 chunk 的 hidden states；此时应把 `seq_mask` 加入 `chunk_kwarg_names`，否则 batch 对不齐。

更一般地说，不能只照抄上面的四个 kwargs。迁移到新模型或开启新特性时，应根据目标 forward 的真实调用参数重新检查所有 batch-shaped Tensor。

## 10. 当前实现的边界与故障模式

### 10.1 配置缺少主动校验

当前代码没有独立校验以下条件：

- `chunk_mbs` 必须为正整数；
- `batch_dim` 对每个被切 Tensor 都合法；
- `chunk_arg_indexs` 中的索引存在；
- `chunk_kwarg_names` 中用于推断 batch 的 key 存在；
- 所有被切输入的 batch 长度一致；
- 多个 plan 没有重复命中同一个 module。

配置错误通常要到第一次 forward 才以除零、索引、KeyError、shape mismatch 或更隐蔽的语义错误暴露。

### 10.2 不应切的输入

本次 `cache_position=[504]` 描述序列位置，没有 batch 维，因此保持不变是正确的。标量、布尔开关、cache 对象和真正按全局共享的数据也应原样传入。

不要因为某个对象里“包含 Tensor”就一律配置切分。递归函数会对整个容器中的所有 Tensor 使用同一个 `batch_dim`。

### 10.3 输出限制

Qwen decoder layer 返回单个 `[B,S,H]` Tensor，适配当前实现。若新目标层返回：

- dict 或 `ModelOutput`；
- tuple 中含 `None`；
- batch 输出与标量 loss 混合；
- 不同元素的 batch 维不同；

则不能直接使用当前拼接逻辑。

### 10.4 目标层必须满足 batch 独立性

BatchNorm、跨样本负样本构造、跨样本 attention、整 batch 统计或全 batch 路由策略可能破坏分块等价性。应先做 forward、输入梯度、参数梯度的对齐测试，再做性能测试。

### 10.5 代码库测试现状

当前源码树没有 `_slice_batch_recursive` 或 `chunk_mbs_forward` 的独立 UT。仓库有启用 ChunkMBS 的 Qwen3.5 ST 配置，但它不能替代对整除/尾块、嵌套输入、复杂输出、梯度等价性和 FSDP unshard 次数的专项测试。

## 11. 本次实测结果

本次只把训练步数临时改为 1，并通过临时启动器包装第一层的旧 forward；没有改动 MindSpeed-MM 工作树中的 ChunkMBS 源码。结果：

```text
world size:          8
target layer input: [4, 504, 2048] BF16
chunk count:         2
chunk 0 input:       [2, 504, 2048] BF16
chunk 0 output:      [2, 504, 2048] BF16
chunk 1 input:       [2, 504, 2048] BF16
chunk 1 output:      [2, 504, 2048] BF16
merged output:       [4, 504, 2048] BF16
training:            iteration 1/1
global batch size:   32
loss:                11.41470
grad norm:           58.894
process exit code:   0
```

这个结果证明了当前 Qwen 配置在真实 NPU 训练中完成了：

```text
[4,504,2048]
  -> [2,504,2048] + [2,504,2048]
  -> 两次原始 layer forward
  -> cat
  -> [4,504,2048]
  -> loss
  -> backward
```

它不单独证明 ChunkMBS 带来多少吞吐或显存收益。要回答性能问题，还需要在相同模型层数、数据、随机种子和训练步数下，对比关闭 ChunkMBS、直接小 MBS 加梯度累积、以及不同 `chunk_mbs` 的峰值显存、Host 内存、step time 和通信 trace。

## 12. 新模型接入检查表

1. 确认目标 module 的一次 forward 在 batch 维上样本独立。
2. 记录实际 forward 的所有 args、kwargs、输出类型和 shape。
3. 找出每个随 batch 变化且在目标层中会被读取的 Tensor。
4. 确认这些 Tensor 的 batch 维一致，再填写 `chunk_arg_indexs` 和 `chunk_kwarg_names`。
5. 确认非 batch Tensor 不会被递归误切。
6. 确认输出能由当前 Tensor 或一层 tuple/list 规则拼接。
7. 确认 ChunkMBS 目标集合被 recompute 和 activation offload 覆盖。
8. 用整除和非整除 batch 分别比较 forward、loss、输入梯度与参数梯度。
9. 统计实际 FSDP unshard 次数，不只看最终 loss。
10. 最后再比较显存与吞吐，逐步寻找合适的 `chunk_mbs`。
