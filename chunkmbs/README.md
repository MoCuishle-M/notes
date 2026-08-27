# ChunkMBS：背景、原理与使用指南

> 本文基于 MindSpeed-MM `master` 分支提交 `192b2a139d0c`（2026-08-27）的代码整理。这里的 ChunkMBS 特指原生 FSDP2 后端的 layer-level batch 切分，不是训练调度器中的梯度累积 micro-batch，也不是用于降低 logits 显存的 ChunkLoss。

## 1. 要解决的问题

FSDP2 将模型参数分片保存在多个数据并行 rank 上。计算某个 Transformer Block 时，目标参数需要经历：

```text
分片参数
  │ copy-in / all-gather（unshard）
  ▼
完整参数 ── Block forward/backward ── reshard
```

当模型很大而单次计算量较小时，会出现两个问题：

1. 参数 all-gather、copy-in/copy-out 的通信与同步开销占比较高；
2. 通信与计算争用总线带宽，计算不一定能完全掩盖通信。

常见的摊薄方式是增大训练 `micro_batch_size`，让一次参数 unshard 服务更多样本。但完整大 batch 会同时扩大每层反向所需的激活峰值，往往先遇到 OOM。另一种方式是保持小 MBS、增加梯度累积步数；它能控制激活峰值，却会让同一层在多个梯度累积轮次中反复 unshard。

ChunkMBS 试图同时保留两者的优点：训练调度器向模型提交一个较大的 batch，而目标 Block 内部再把它切成小块串行执行。这样 FSDP2 在 Block 外层只做一次 unshard，小块计算又把瞬时激活控制在较小规模。

## 2. 核心思想

设当前训练 micro batch 的 batch size 为 `B`，`chunk_mbs=C`。ChunkMBS 计算：

```text
K = ceil(B / C)

input [B, ...]
  ├─ [0:C, ...]       ── Block ── output_0
  ├─ [C:2C, ...]      ── Block ── output_1
  └─ [...]            ── Block ── output_K-1
                                      │
                         cat(batch_dim)
                                      ▼
                               output [B, ...]
```

最后一块允许小于 `C`。如果 `B <= C`，代码直接调用原始 forward，不做切分与拼接。

### 2.1 为什么数学上可行

对普通逐样本 Transformer 层，batch 中各样本的前向互不依赖：

```text
F(concat(x_0, x_1, ...)) = concat(F(x_0), F(x_1), ...)
```

`torch.cat` 本身可求导，反向时上游梯度会按同样边界分发到各 chunk；每个 chunk 对共享参数产生的梯度由 autograd 累加。因此实现不需要自定义 `autograd.Function`。

这个等价关系有前提：层内不能存在依赖整个 batch 的计算，例如跨样本归约、batch statistics，或输入之间的交叉注意力。随机算子还可能因调用次数和随机数消费顺序变化而不能做到逐 bit 一致。

### 2.2 为什么能减少 unshard 次数

关键不只是“切 batch”，而是 wrapper 的位置。MindSpeed-MM 在 FSDP fully-shard 之前按以下顺序改写目标层：

```text
原始 forward
  ↑ recompute wrapper
  ↑ activation-offload wrapper
  ↑ ChunkMBS wrapper（最外层）
  ↑ FSDP2 module hooks / fully_shard
```

运行时，调用一次 FSDP module，FSDP 的外层 hook 先 unshard 参数；随后 ChunkMBS wrapper 在同一次 module 调用内部循环执行 `K` 个小块；所有小块完成并拼接后，FSDP 才执行 forward 后处理和 reshard。因此目标层的一次大 batch 调用对应一次参数 unshard，而不是每个 chunk 一次。

### 2.3 为什么仍需要重计算和激活卸载

仅做串行 forward 并不会自动释放所有 chunk 的反向激活：外层 loss 尚未 backward 时，autograd 通常仍需保留各 chunk 的计算图。

仓库方案要求目标层同时开启：

- **recompute**：forward 只保存 checkpoint 边界输入，backward 时按 chunk 重算层内计算；
- **async activation offload**：通过 `saved_tensors_hooks` 将符合条件的入口激活异步 D2H 搬到 Host，backward 取用时 H2D 搬回并预取；
- **ChunkMBS**：把进入 checkpoint/offload wrapper 的输入先切小，使重算时的瞬时层内激活以 `C` 而不是 `B` 为尺度。

三者共同作用，才形成“一次参数聚合、多个小块计算、设备侧瞬时激活接近单块规模”的设计。

## 3. 与大 MBS、梯度累积的区别

以下比较固定单个数据并行 rank 每次参数更新处理 `B` 个样本，忽略 pipeline 等其他并行因素：

| 方案 | 调度器 MBS | 梯度累积 | Block 每次更新的调用/Unshard 次数 | 层内激活尺度 |
|---|---:|---:|---:|---:|
| 小 MBS + 梯度累积 | `C` | `B/C` | 约 `B/C` 次 | `C` |
| 直接大 MBS | `B` | 1 | 1 次 | `B` |
| 大 MBS + ChunkMBS | `B` | 1 | 1 次 | 目标接近 `C` |

ChunkMBS 不会改变训练调度器对 MBS、GBS 和梯度累积的定义，也不会把一个 optimizer step 拆成多个 step。它只改写选定 module 的一次 forward。

## 4. 配置方法

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

这与仓库 Qwen3.5 35B 样例一致：调度器每次给模型 4 个样本，每个目标 language layer 内按 2 个样本一块执行两次原始 forward。

### 4.1 参数语义

| 参数 | 代码中的真实含义 |
|---|---|
| `enable_chunk_mbs` | 开启目标层 forward 包装 |
| `apply_modules` | 用 `module_name_match` 匹配的 module 路径列表；一个都匹配不到会报错 |
| `chunk_mbs` | **切分后每一块的 batch size**，不是块数 |
| `batch_dim` | 所有被切 Tensor 的 batch 所在维度；`[B,S,H]` 通常为 0，`[S,B,H]` 通常为 1 |
| `chunk_arg_indexs` | 需要切分的位置参数索引；默认 `[0]` |
| `chunk_kwarg_names` | 需要切分的关键字参数名；默认空列表 |

如果希望把 `B=8` 恢复成传统 `MBS=2` 的单块计算尺度，应设置 `chunk_mbs: 2`，此时块数是 `ceil(8/2)=4`。上游入口文档效果章节写的 `chunk_mbs=GBS/MBS` 容易把“块大小”和“块数”混淆；应以实现中的 `start=i*chunk_mbs`、`end=start+chunk_mbs` 为准。

### 4.2 哪些输入必须切

凡是 batch 维长度随主输入一起变化、且每个 chunk forward 会读取的输入，都必须配置为切分对象。例如 decoder layer 调用：

```python
decoder_layer(
    hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=layer_mask,
    position_ids=text_position_ids,
    past_key_values=past_key_values,
    use_cache=use_cache,
)
```

通常 `hidden_states` 是位置参数 0；`position_embeddings`、`attention_mask`、`position_ids` 是关键字参数。标量、开关、cache 对象等与 batch 无关的参数应原样传递，不应列入切分配置。

切分函数会递归处理 Tensor、tuple、list 和 dict，所以一个配置项可以是嵌套 Tensor 容器。但它会对该容器中的**每一个 Tensor**使用同一个 `batch_dim` 和 `[start:end]`，配置前必须确认它们的 layout 一致。

## 5. 收益与代价

上游文档及特性提交说明记录：Qwen3.5 35B 在其测试配置中端到端收益约 4%～5%。这不是跨模型保证，实际收益取决于参数通信占比、互联带宽、chunk 大小、Host 传输、重计算开销和算子效率。

主要收益：

- 降低目标 Block 在一个 optimizer update 中的重复参数 unshard 次数；
- 让单次 unshard 覆盖更多计算，提高通信摊销；
- 在重计算与激活卸载配合下，把设备激活峰值控制在小 chunk 尺度；
- 最后一块自动支持非整除 batch。

主要代价：

- 原始 layer forward 被调用更多次，增加 Python、dispatcher 和 kernel launch 开销；
- chunk 太小会降低矩阵乘规模和算子效率；
- D2H/H2D、重计算仍有成本，Host 内存也会增加；
- 分块改变执行与浮点归约顺序，通常只应要求容差内精度对齐；
- 当前实现缺少专门 UT，特性提交记录的测试是 Qwen3.5 35B 自验精度对齐。

## 6. 使用边界与排错

### 6.1 当前代码限制

- 只在 `enable_chunk_mbs=true` 且存在 `chunkmbs_plan` 时启用；`apply_modules` 为空或无匹配会失败。
- 推断 `B` 时优先读取第一个 `chunk_arg_indexs` 对应参数；没有位置参数配置时才读取第一个关键字参数。
- 代码没有显式校验 `chunk_mbs > 0`、索引存在、关键字存在、所有被切输入 batch 长度一致；错误配置可能产生除零、索引、KeyError、shape 或语义错误。
- 输出只支持单个 Tensor，或由 Tensor 构成的一层 tuple/list；dict、模型输出对象、嵌套输出以及 tuple 中的 `None` 当前不支持。
- tuple/list 输出的每个元素都沿同一个 `batch_dim` 拼接，非 batch 型辅助输出不适用。
- 同一个 module 被多个 plan 重复匹配时，当前匹配函数不去重，可能被重复包装。

### 6.2 常见现象

| 现象 | 优先检查 |
|---|---|
| shape mismatch | 是否漏配 batch 相关 kwargs；所有输入的 `batch_dim` 是否一致 |
| 输出拼接报错 | forward 是否返回 dict/ModelOutput/包含 `None` 的 tuple；各输出是否都能沿同一维拼接 |
| 无显存收益 | recompute 与 activation offload 是否覆盖 ChunkMBS 的全部目标层；wrapper 是否实际打印 apply 日志 |
| 吞吐下降 | chunk 是否过小；原本通信占比是否不高；offload/重计算成本是否超过通信节省 |
| 训练结果偏差 | 层中是否有跨 batch 运算或随机算子；切分输入是否完整、边界是否对齐 |

## 7. 调参建议

1. 先用不开 ChunkMBS 的小 MBS 配置记录 loss、峰值显存和 step time。
2. 将调度器 `micro_batch_size` 增大，并把 `chunk_mbs` 设为原先已验证的小 MBS。
3. 保证 ChunkMBS 目标集合是 recompute 和 activation offload 目标集合的子集。
4. 验证若干 step 的 loss 和梯度在合理容差内对齐，再做长时间吞吐测试。
5. 显存有余量时逐步增大 `chunk_mbs`；如果 OOM 则减小。不要只看平均 step time，也要观察 Host 内存、通信时间和最后非整除 chunk 的波动。

具体代码入口、wrapper 嵌套关系和逐行行为见 [MindSpeed-MM ChunkMBS 核心代码实现](./implementation.md)。
