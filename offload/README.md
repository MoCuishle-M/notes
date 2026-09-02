# MindSpeed-MM 异步激活值卸载：从直觉到源码，再到重计算协同

异步激活值卸载（Async Activation Offload）可以先理解成一句话：

> 前向时，把反向以后才会用到的部分激活值从 NPU 搬到 CPU 内存；反向快用到它们之前，再提前搬回 NPU。数据搬运走单独的 stream，尽量与模型计算重叠。

这里的 Host 侧只负责**暂存**激活值，不会在 CPU 上“更新激活值”，也不会在 CPU 上替模型完成正常的前向或反向计算。

MindSpeed-MM 当前有两套采用相同核心思想、但入口不同的实现：

- `mindspeed_mm/utils/async_offload.py`：通用模型代码手动进入 context，正是原说明文档示例对应的实现；
- `mindspeed_mm/fsdp/features/memory/async_offload.py`：原生 FSDP2 通过 YAML 自动包装 module 的实现，也是本文 Qwen3.5 训练脚本实际走的路径。

本文先用通俗方式解释两者共有的机制，再重点对照当前 Qwen3.5 真正使用的 FSDP2 源码说明：

- 前向如何异步执行 D2H（Device to Host）；
- NPU 上的原 storage 何时释放；
- 反向如何执行 H2D（Host to Device）和预取；
- `saved_tensors_hooks` 为什么能介入 autograd；
- 激活值卸载和 checkpoint 重计算怎样组合；
- Qwen3.5 中 `skip_flash_attn_recompute`、`skip_gdn_recompute` 又做了什么；
- 当前 Qwen3.5 训练配置实际形成了怎样的包装顺序。

建议阅读顺序：只想先建立直觉，读第一部分；想看搬运源码，继续读第二部分；想弄清“卸载 + 重计算 + 跳过昂贵算子重算”，重点读第三部分。

## 0. 本文对应的代码与运行环境

本文核对的是远端目录：

```text
/home/z30071651/code/qwen3.5/2026-09-01/MindSpeed-MM
```

源码基线与环境：

| 项目 | 当前值 |
|---|---|
| MindSpeed-MM commit | `874c57eb7cc9857e488d2f75fc6f7d3a33f22f36` |
| 容器 | `ci-master-cann9.1.0-torch_npu2.10.0.post4-py3.12-fla` |
| conda 环境 | `ci_master` |
| PyTorch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post4` |
| 训练脚本 | `examples/qwen3_5/finetune_qwen3_5_35B.sh` |
| 配置 | `examples/qwen3_5/qwen3_5_35B_config.yaml` |

远端工作树有 Qwen3.5 实验相关修改，但本文引用的下列通用核心文件相对该 commit 没有未提交修改：

- `docs/zh/features/async_activation_offload.md`；
- `mindspeed_mm/utils/async_offload.py`；
- `mindspeed_mm/fsdp/features/memory/async_offload.py`；
- `mindspeed_mm/fsdp/features/memory/recompute.py`；
- `mindspeed_mm/fsdp/features/apply_features.py`。

当前实验日志 `logs/train_20260901_065739.log` 记录了 `1/20` 到 `20/20` 的完整迭代，并打印了 3 个视觉 block 和 12 个语言 layer 的 recompute/offload 安装信息。这能证明当前组合路径实际跑通过，但**不能单独证明卸载带来的显存收益或性能收益**；性能结论仍需有关闭特性后的同条件对照组。

---

## 第一部分：先通俗理解

### 1. 什么是激活值

训练一个层时，可以简化成：

```text
输入 x ── layer forward ──> 输出 y ── 后续层 ──> loss
```

反向传播不只需要最终的 `loss`。为了计算梯度，很多算子的 backward 还需要前向时的输入或中间结果，例如：

- Linear backward 可能需要输入；
- Attention backward 需要 `q`、`k`、`v` 和部分 attention 中间量；
- 激活函数 backward 可能需要前向输入或输出。

这些为 backward 保留下来的 Tensor，通常统称为**激活值**或 **saved-for-backward tensors**。

训练层数多、序列长、micro batch 大时，前面层的激活要一直等到反向走回来才能释放，因此会形成很高的显存峰值：

```text
前向：L0 -> L1 -> L2 -> ... -> LN
       x0    x1    x2          xN    都在等待未来的 backward

反向：LN -> ... -> L2 -> L1 -> L0
```

### 2. 激活值卸载做了什么

假设第 `i` 层的输入是 `x_i`，shape 为 `[B, S, D]`：

- `B`：batch size；
- `S`：序列长度；
- `D`：hidden dimension，通常也叫 `hidden_size`。

普通训练会让 `x_i` 留在 NPU，直到第 `i` 层反向使用它。激活值卸载改成：

```text
前向刚保存 x_i
    │
    ├─ NPU --D2H--> CPU pinned memory
    │
    └─ D2H 完成后，把 x_i 的 NPU storage 缩为 0

反向快走到第 i 层
    │
    ├─ 恢复 x_i 的 NPU storage
    ├─ CPU --H2D--> NPU
    └─ event 确认 H2D 完成后，才让 backward 使用 x_i
```

对于 BF16 的连续 `x_i`，仅 Tensor 数据量可以粗略写成：

```text
bytes = B × S × D × 2
```

例如 `x_i=[4,504,2048]`：

```text
4 × 504 × 2048 × 2 = 8,257,536 bytes ≈ 7.875 MiB
```

这只是一个层输入的理论数据量，不等于整个模型准确的显存节省量。实际显存还包括其它激活、参数、梯度、通信 buffer、allocator 碎片，以及被选择跳过重计算的算子中间量。

### 3. “异步”到底是什么意思

如果先停下计算、等 D2H 完成，再继续前向，搬运时间会完整暴露在 step time 中：

```text
同步方式：D2H 等待 | layer 计算 | D2H 等待 | 下一层计算
```

两套实现都会让 D2H/H2D 走专用交换流，使其尽量与默认计算流重叠。通用 API 由调用方传入 stream；当前 FSDP2 实现内部维护一条 `swap_stream`：

```text
计算流：       layer 0 forward  | layer 1 forward  | layer 2 forward
交换流： x0 D2H ---------------| x1 D2H ----------|
```

“异步”不表示永远不等待，而是把等待推迟到**真正存在数据依赖**的位置：

- D2H 前：交换流等待前向流已经产生 `x_i`；
- 释放 NPU storage 前：计算流等待 D2H 已完成；
- backward 使用前：计算流等待 H2D 已完成。

因此，event 不是多余开销，而是异步执行仍然保持数据正确性的关键。

### 4. 反向时发生什么

以 3 个连续卸载层 `L0`、`L1`、`L2` 为例，当前实现不卸载最后一个 block 的入口 `x2`，避免刚结束前向就马上搬回：

```text
前向

L0：保存 x0 -> 异步 x0 D2H -> 计算 L0
L1：确认 x0 D2H 完成并释放 x0 的 NPU storage
    保存 x1 -> 异步 x1 D2H -> 计算 L1
L2：确认 x1 D2H 完成并释放 x1 的 NPU storage
    x2 保留在 NPU -> 计算 L2
```

反向从最后一层开始：

```text
反向

L2 backward：x2 本来就在 NPU，直接使用

L1 backward：
    恢复 x1 的 NPU storage
    异步 x1 H2D
    等待 x1 H2D 完成
    使用 x1 做 L1 的重计算/反向
    同时预取下一步要用的 x0

L0 backward：
    x0 通常已在 L1 计算期间提前 H2D
    等待对应 event，随后做 L0 的重计算/反向
```

所以反向仍然只有训练代码里的那一次 `loss.backward()`。Autograd 需要某个已卸载 Tensor 时，自动触发恢复逻辑；用户不需要为每个层手写一次 backward。

### 5. 卸载、重计算分别节省什么

两者都在用其它资源换 NPU 显存，但代价不同：

| 技术 | 前向后怎样处理激活 | 反向代价 | 主要消耗 |
|---|---|---|---|
| 保留激活 | 一直留在 NPU | 直接使用 | NPU 显存 |
| 激活值卸载 | 保存到 Host | 搬回 NPU | Host 内存、D2H/H2D 带宽 |
| 重计算 | 丢弃层内中间激活 | 重跑部分前向 | NPU 算力 |

直观的选择原则是：

- **Tensor 大、重新计算便宜**：更适合丢弃后重计算；
- **Tensor 相对小、重新计算很贵**：更适合保存并卸载；
- **层入口 hidden states**：可以卸载，作为整层重计算的起点；
- **Flash Attention/GDN 的关键中间结果**：可以卸载，使重计算阶段跳过昂贵内核。

这也是 MindSpeed-MM 把两项技术组合起来的核心思想：不是在“全部卸载”和“全部重算”之间二选一，而是让不同 Tensor 走适合自己的路径。

---

## 第二部分：核心代码实现

### 6. 两套实现，以及当前脚本走哪一套

原说明文档中的 context 示例对应通用实现：

```text
mindspeed_mm/utils/async_offload.py
  │
  └─ 模型代码显式使用 with async_save_on_cpu(...)
       └─ 在 context 内调用目标算子或 decoder layer
```

例如 `mindspeed_mm/models/transformers/qwen3vl/modeling_qwen3_vl.py` 会在 decoder layer 循环中显式传入 `h2d_stream`、`d2h_stream`、`block_idx` 和 `depth`。

本文指定的启动脚本设置了：

```bash
export NON_MEGATRON=true
```

并在 YAML 的 `features` 下打开 offload，因此本次 Qwen3.5 走的是原生 FSDP2 实现，而不是上面的模型内手动 context 路径。

当前 Qwen3.5 配置为：

```yaml
model:
  attn_implementation: flash_attention_2
  gdn_implementation: ascendc
  skip_gdn_recompute: true
  skip_flash_attn_recompute: true

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
```

配置没有填写 `activation_offload_plan.impl`，所以使用默认的 `legacy` 实现。仓库还提供显式设置 `impl: stash` 的另一条共享 swap-cache 路径，但当前脚本没有启用它，本文不展开其内部行为。配置结构位于：

```text
mindspeed_mm/fsdp/params/feature_args.py:183-195,249-253
```

当前 Qwen3.5 的完整调用链如下：

```text
qwen3_5_35B_config.yaml
  │
  ▼
FeatureArguments
  │
  ▼
FeaturesApplier.pre_fully_shard_apply(model)
  │
  ├─ recompute_modules(...)
  │    └─ module.forward = checkpoint_wrapper(old_forward)
  │
  ├─ get_offload_modules(...)
  └─ async_offload_modules(...)
       └─ module.forward = offload_wrapper(current_forward)
```

相关源码：

| 职责 | 文件与行号 |
|---|---|
| 原说明文档示例对应的通用 context | `mindspeed_mm/utils/async_offload.py:6-272` |
| 通用 context 的模型内调用示例 | `mindspeed_mm/models/transformers/qwen3vl/modeling_qwen3_vl.py:533-550` |
| 配置结构与默认实现 | `mindspeed_mm/fsdp/params/feature_args.py:86-108,183-195,249-253` |
| 特性安装顺序 | `mindspeed_mm/fsdp/features/apply_features.py:52-78,126-140` |
| checkpoint 包装 | `mindspeed_mm/fsdp/features/memory/recompute.py:17-33,105-125` |
| 异步卸载核心 | `mindspeed_mm/fsdp/features/memory/async_offload.py:17-370` |
| 训练阶段切换 | `mindspeed_mm/fsdp/train/train_engine.py:176-194` |
| Flash Attention 跳过重算 | `mindspeed_mm/fsdp/ops/flash_attn/skip_recompute_flash_attn.py:13-180` |
| Qwen3.5 AscendC GDN 跳过重算 | `mindspeed_mm/fsdp/ops/gdn/flash_gated_delta_rule.py:410-520` |

### 7. 第一个核心：`saved_tensors_hooks`

PyTorch autograd 在前向中需要为 backward 保存 Tensor 时，允许用 `saved_tensors_hooks(pack, unpack)` 改写“怎样保存”和“怎样取回”。两套实现都以它为入口；下面展示当前 FSDP2 版本的签名：

```python
class async_save_on_cpu(saved_tensors_hooks):
    def __init__(self, block_idx, depth, custom_check_fn=None, prefetch=True):
        ...
        super().__init__(_pack_to_cpu, _unpack_from_cpu)
```

可以把这两个 hook 理解成：

```text
autograd 想保存 Tensor
        │
        ▼
      pack(Tensor)
        │
        └─ autograd 实际保存 pack 的返回对象

backward 想取回 Tensor
        │
        ▼
      unpack(保存对象)
        │
        └─ 必须返回 backward 能使用的 Tensor
```

MindSpeed-MM 的 `pack` 不再让 autograd 直接保存原 Tensor，而是返回一个 `SwapTensor` 对象。`unpack` 收到它后，把 CPU 数据恢复到 NPU，再返回原 Tensor。

### 8. 哪些 Tensor 会被卸载

第一层筛选是 `base_check_fn()`：

```python
def base_check_fn(tensor) -> bool:
    if isinstance(tensor._base, torch.nn.parameter.Parameter) \
            or isinstance(tensor, torch.nn.parameter.Parameter):
        return False
    if tensor.storage().size() <= 0:
        return False
    return True
```

它会排除：

- Parameter 本身；
- Parameter 的 view；
- storage 为空的 Tensor。

第二层筛选来自 module wrapper：

```python
hidden_states = args[hidden_states_idx]

context = async_save_on_cpu(
    block_idx=layer_idx,
    depth=depth,
    custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
    prefetch=prefetch,
)

with context:
    return forward_func(*args, **kwargs)
```

默认 `hidden_states_idx=0`。`custom_check_fn` 用 `data_ptr()` 相等来筛选，因此通用 layer wrapper 的目标不是“层内所有 saved tensor”，而是和该层入口 `hidden_states` 指向同一数据地址的 Tensor。

这个区别很重要：

> 当前 legacy activation offload 的通用 wrapper，主要卸载的是层级重计算边界的输入 hidden states，不是无差别搬走该层全部激活。

### 9. `SwapTensor`：一份数据的状态机

`SwapTensor` 保存了四类信息：

```python
self.tensor = tensor
self.storage_size = tensor.storage().size()
self.tensor_cpu = torch.empty(
    tensor.shape,
    dtype=tensor.dtype,
    pin_memory=True,
    device="cpu",
)
self.stat = "device"
```

其中 `pin_memory=True` 很关键。Pinned memory 让设备 DMA 能更高效地执行异步 D2H/H2D；普通 pageable CPU 内存通常无法得到相同的异步效果。

它的状态可以简化成：

```text
device
  │ launch_d2h()
  ▼
host（D2H 已提交，event 记录在交换流）
  │ wait_d2h_finished()
  ├─ 等 event
  └─ NPU storage.resize_(0)
  │
  │ launch_h2d()
  ├─ NPU storage 恢复到原大小
  └─ CPU 数据异步 copy 回 NPU
  ▼
device（H2D event 记录完成点）
```

#### 9.1 前向 D2H

核心代码是：

```python
forward_event = create_event()
forward_event.record()

with switch_to_specified_stream(stream):
    stream.wait_event(forward_event)
    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
    self.d2h_event.record()
    self.stat = "host"
```

顺序含义：

1. 在当前计算流记录 `forward_event`；
2. 交换流等待该 event，保证 Tensor 已经可读；
3. 在交换流提交 non-blocking D2H；
4. 用 `d2h_event` 标记复制完成点。

此时 `stat="host"` 表示已切换到 Host 管理阶段，不表示 CPU copy 在 Python 赋值那一刻已经同步完成；真正释放 NPU storage 前仍要等待 `d2h_event`。

#### 9.2 D2H 完成后释放 NPU storage

```python
get_current_stream().wait_event(self.d2h_event)
self.tensor.storage().resize_(0)
```

当前 block 的 D2H 在本 block 计算期间异步进行。进入下一个 block、第一次保存 Tensor 时，`OffloadManager().del_npu_tensor()` 会让当前流等待上一 block 的 D2H，然后把对应 NPU storage 缩成 0。

因此不是 `launch_d2h()` 一执行就立刻释放显存，而是：

```text
先提交 copy -> 用本层计算掩盖 copy -> 下一层入口确认 copy 完成 -> 再释放
```

#### 9.3 反向 H2D

```python
backward_event = create_event()
backward_event.record()

with switch_to_specified_stream(h2d_stream):
    h2d_stream.wait_event(backward_event)
    self.tensor.storage().resize_(self.storage_size)
    self.tensor.storage().copy_(
        self.tensor_cpu.storage(), non_blocking=True
    )
    self.h2d_event.record()
```

完整含义是：

1. 恢复原 NPU Tensor 的 storage 大小；
2. 在交换流提交 H2D；
3. 记录 `h2d_event`；
4. 默认计算流在消费 Tensor 前等待该 event。

对于 slice Tensor，源码走 `tensor.copy_(tensor_cpu)`，而不是直接复制整个底层 storage。

### 10. `_pack_to_cpu()`：前向保存路径

下面继续以当前 Qwen3.5 使用的 FSDP2 版本为准。删去异常分支后，核心逻辑如下：

```python
key, after_block = OffloadManager().get_cnt(block_idx)
d2h_stream = OffloadManager().swap_stream

if after_block:
    OffloadManager().del_npu_tensor(f"{block_idx - 1}_")

if block_idx == depth - 1:
    return tensor

swap_tensor = SwapTensor(tensor, key)
swap_tensor.launch_d2h(d2h_stream)
OffloadManager().put(key, swap_tensor)
return swap_tensor
```

每个保存对象用 `block_idx_tensor_idx` 做 key，例如：

```text
3_0：第 3 个 offload block 保存的第 0 个 Tensor
3_1：第 3 个 offload block 保存的第 1 个 Tensor
```

最后一个 offload block 直接返回原 Tensor，不执行 D2H。这是因为前向结束后马上就会从最后一层开始反向，卸载后立刻加载通常得不偿失。

### 11. `_unpack_from_cpu()`：反向恢复和预取路径

核心代码如下：

```python
if isinstance(swap_tensor, torch.Tensor):
    return swap_tensor

swap_tensor.launch_h2d(OffloadManager().swap_stream)
get_current_stream().wait_event(swap_tensor.h2d_event)

block_idx, tensor_idx = swap_tensor.key.split("_")
OffloadManager().clear(swap_tensor.key)
TrainingContext().set_layer_index(int(block_idx))

if prefetch:
    OffloadManager().prefetch_get(
        int(block_idx),
        int(tensor_idx),
        OffloadManager().swap_stream,
        OffloadManager().swap_stream,
    )

return swap_tensor.tensor
```

这里做了四件事：

1. 当前 Tensor 若根本没有卸载，直接返回；
2. 对已卸载 Tensor 发起 H2D，并在消费前等待完成；
3. 从 Manager 删除已经恢复的条目；
4. 预取反向下一步将访问的前一个 block。

当前 `prefetch_get()` 假设 block 索引连续，并以 `block_idx - 1` 构造前一层的 key。因此 `activation_offload_plan.apply_modules` 应按照真实前向顺序排列，匹配出的 block 也应保持连续执行。它不是一个可随意打乱的集合。

### 12. 两套 API 的关键差异

原说明文档的使用示例写成：

```python
with async_save_on_cpu(
    h2d_stream=h2d_stream,
    d2h_stream=d2h_stream,
    block_idx=block_idx,
    depth=depth,
    custom_check_fn=your_check_fn,
):
    output = layer(input)
```

这个示例是有效的，它对应：

```text
mindspeed_mm/utils/async_offload.py:217-272
```

通用构造函数确实接收：

```python
async_save_on_cpu(
    h2d_stream,
    d2h_stream,
    block_idx,
    depth,
    custom_check_fn=None,
    prefetch=True,
)
```

而当前 Qwen3.5 FSDP2 路径使用的是另一个同名 class：

```python
async_save_on_cpu(
    block_idx,
    depth,
    custom_check_fn=None,
    prefetch=True,
)
```

在 FSDP2 版本中，`h2d_stream` 和 `d2h_stream` 不由调用者传入。`OffloadManager` 内部创建一条：

```python
self.swap_stream = create_stream()
```

随后 D2H 和 H2D 都使用这条 `swap_stream`。两套实现的差异可归纳为：

| 对比项 | 通用 `utils` 实现 | 当前 Qwen3.5 的 FSDP2 实现 |
|---|---|---|
| 接入方式 | 模型代码手写 `with async_save_on_cpu(...)` | YAML 选 module，框架自动包装 `forward` |
| stream 来源 | 调用方传入 H2D/D2H stream | Manager 内部创建一条 `swap_stream` |
| 设备接口 | 直接调用 `torch_npu.npu` | 经 `fsdp.utils.device` 适配函数 |
| 预取映射 | 按当前/前一 block Tensor 数量做区间映射 | 当前实现按相邻 block 批量预取 key |
| 与训练阶段联动 | context 本身不设置 FSDP `TrainingContext` | unpack 更新 layer index，供 skip-recompute 使用 |
| 适用当前脚本 | 否 | 是 |

当前 Qwen3.5 FSDP2 路径的准确表述是：

> D2H/H2D 共用一条专用交换流；它们在这条流上彼此有序，但可以和默认计算流异步重叠。

因此不能只看 class 名就混用参数：从 `mindspeed_mm.utils.async_offload` 导入时应传入两个 stream；从 FSDP2 的 `mindspeed_mm.fsdp.features.memory.async_offload` 导入时则没有这两个参数。使用当前仓库的 `features.enable_activation_offload` 配置时，不需要自己构造 context。

---

## 第三部分：异步激活值卸载怎样与重计算结合

### 13. 先看 wrapper 安装顺序

`FeaturesApplier.pre_fully_shard_apply()` 明确按下面顺序安装：

```python
self.apply_recompute_models(model=model)
self.apply_activation_offload_modules(model=model)
self.apply_chunk_mbs(model=model)
```

因为每一步都用新 wrapper 包住 module 当前的 `forward`，所以调用时的嵌套顺序正好反过来：

```text
若同时开启 ChunkMBS：

ChunkMBS wrapper
  └─ activation offload wrapper
       └─ checkpoint wrapper
            └─ 原始 layer.forward
```

如果不启用 ChunkMBS，则去掉最外层即可。

这段顺序不是普通的代码风格选择，而是决定了特性语义：

- offload context 包住 checkpoint 调用；
- checkpoint 为重计算保存的边界输入会经过外层 `saved_tensors_hooks`；
- checkpoint 内部原始层前向产生的大量中间 Tensor 不需要长期保留；
- ChunkMBS 最外层先切 batch，每个 chunk 分别进入 offload + checkpoint。

### 14. 为什么 offload 能拿到 checkpoint 的输入

当前 `RecomputePlanConfig.use_reentrant` 默认是 `False`。非 reentrant checkpoint 会通过 autograd 的保存通道保存它的入口 Tensor，以便 backward 时重跑前向。

在容器的 PyTorch 2.10.0 上做最小验证，外层 `saved_tensors_hooks` 包住一次非 reentrant checkpoint 时，观察到 pack/unpack 对象为：

```text
pack:   shape [0] 的 checkpoint 内部占位 Tensor
pack:   用户输入 Tensor
unpack: shape [0] 的 checkpoint 内部占位 Tensor
unpack: 用户输入 Tensor
```

MindSpeed-MM 的 `base_check_fn` 会排除空 storage，占位 Tensor不会被卸载；`custom_check_fn` 又只接受与 `hidden_states` 地址相同的 Tensor。最后被卸载的就是 checkpoint 重计算真正需要的层入口。

### 15. 两项技术组合后的完整前向

对第 `i` 个 layer，可以按以下步骤理解：

```text
1. 进入 activation offload context
2. 调用 checkpoint(layer_forward, hidden_states, ...)
3. checkpoint 请求保存重计算入口 hidden_states
4. 外层 pack hook 创建 pinned CPU buffer
5. 在 swap_stream 发起 hidden_states D2H
6. 原始 layer.forward 在计算流执行
7. checkpoint 不长期保存普通层内中间激活
8. 进入下一 offload block 时，等待上一 block D2H 完成
9. 把上一 block 已卸载 Tensor 的 NPU storage 缩成 0
```

最终效果不是“只卸载”或“只重算”，而是：

```text
层入口 hidden_states：保存，但放到 Host
普通层内中间激活：不保存，反向重新计算
特定昂贵算子中间量：按 skip-recompute 策略保存并卸载
```

### 16. 两项技术组合后的完整反向

训练器只做一次：

```python
TrainingContext().set_training_stage(TrainingStage.BACKWARD)
loss.backward()
```

当 autograd 反向走到第 `i` 层 checkpoint 时：

```text
1. checkpoint 请求取回入口 hidden_states
2. 外层 unpack hook 恢复 NPU storage
3. 在 swap_stream 发起 hidden_states H2D
4. 计算流等待 h2d_event
5. checkpoint 用恢复后的 hidden_states 重跑 layer.forward
6. 重跑过程中重新产生 backward 需要的普通中间激活
7. autograd 执行该层真正的 backward，计算输入梯度和参数梯度
8. 同时在 swap_stream 预取反向下一层需要的 Host Tensor
```

可以画成两条时间线：

```text
计算流： L2 recompute+bwd | L1 recompute+bwd | L0 recompute+bwd
交换流：       x1 H2D ----| x0 H2D ---------|
                         ↑                  ↑
                   真正使用前等待       真正使用前等待
```

预取成功时，`x0 H2D` 可以被 `L1` 的重计算和反向掩盖。若计算太短、Tensor 太大、Host 到 Device 带宽不足，H2D 无法及时结束，计算流仍会在 event 处等待。

### 17. `skip_flash_attn_recompute`：重跑 layer，但不重跑昂贵 Attention 内核

只使用普通 checkpoint 时，反向重计算会再次执行完整 Attention forward。当前 Qwen3.5 配置额外开启：

```yaml
skip_flash_attn_recompute: true
```

模型将它传给 Flash Attention 接口；NPU fused attention 路径会改用 `SkipRecomputeFlashAttention` 自定义 autograd Function。

#### 17.1 第一次前向

训练阶段为 `FORWARD` 时，代码真正调用：

```python
attn_output, softmax_max, softmax_sum, *_ = \
    torch_npu.npu_fusion_attention(q, k, v, ...)
```

然后把三个关键结果交给同一个 `OffloadManager`：

```python
swap_tensors = [attn_output, softmax_max, softmax_sum]

for tensor in swap_tensors:
    swap_tensor = SwapTensor(tensor, key)
    swap_tensor.launch_d2h(OffloadManager().swap_stream)
    OffloadManager().put(key, swap_tensor)
```

最后一个 block 的这些 Tensor 仍然留在 NPU，理由与层入口相同：马上就会反向使用。

#### 17.2 checkpoint 重计算阶段

反向前，训练器已把全局阶段设为 `BACKWARD`。checkpoint 重跑 layer.forward，再次进入该自定义 Function 的 `forward()` 时，它不会第二次调用 `npu_fusion_attention`，而是：

```text
Host -> NPU 恢复 attn_output、softmax_max、softmax_sum
```

随后自定义 backward 用恢复的中间结果调用：

```python
torch_npu.npu_fusion_attention_grad(
    q,
    k,
    v,
    grad_output,
    softmax_max=softmax_max,
    softmax_sum=softmax_sum,
    attention_in=attn_output,
    ...,
)
```

所以这里的“跳过重计算”准确含义是：

- 整个 Transformer layer 仍位于 checkpoint 中；
- q/k/v 投影等普通计算仍可能重跑；
- 昂贵的 fused Attention forward 不重跑；
- 它需要的三个结果在第一次前向保存到 Host，反向再恢复；
- Attention backward 仍正常执行并产生 q/k/v 梯度。

### 18. `skip_gdn_recompute`：同样的混合策略

当前配置还开启：

```yaml
gdn_implementation: ascendc
skip_gdn_recompute: true
```

在当前 Qwen3.5 MoE 实现中，`gdn_implementation: ascendc` 会把 `self.chunk_gated_delta_rule` 指向 `flash_gated_delta_rule`。这个自定义 autograd Function 使用相同思想。

第一次前向实际计算后，保存并卸载：

```python
swap_tensors = [g, o, A]
if output_final_state:
    swap_tensors.append(final_state)
```

checkpoint 重计算进入 `BACKWARD` 阶段时，从 Host 恢复这些结果，不再调用 `flash_chunk_gated_delta_rule_fwd()`。随后真正的 Function backward 使用恢复的 `g`、`A` 等数据调用 `flash_chunk_gated_delta_rule_bwd()`。

Qwen3.5 是混合 attention 架构：`full_attention` layer 使用 Flash Attention 路径，`linear_attention` layer 使用 GDN 路径，并不是同一个 layer 同时执行这两种核心算子。

因此 Qwen3.5 当前策略可以概括成：

| 数据或计算 | 当前处理方式 |
|---|---|
| 每层 checkpoint 入口 `hidden_states` | 异步卸载到 Host，重计算前恢复 |
| 普通层内临时激活 | checkpoint 不保留，反向重新生成 |
| Flash Attention 昂贵 forward 的关键结果 | 异步卸载，重计算阶段直接恢复 |
| GDN 昂贵 forward 的关键结果 | 异步卸载，重计算阶段直接恢复 |
| 真正 backward | 仍在 NPU 正常执行 |

### 19. 与 ChunkMBS 一起开启时

当前配置还设置：

```yaml
training:
  micro_batch_size: 4

features:
  enable_chunk_mbs: true
  chunkmbs_plan:
    apply_modules:
      - model.language_model.layers.{*}
    chunk_mbs: 2
    batch_dim: 0
```

因为 ChunkMBS 是最外层 wrapper，一个 language layer 的实际调用变成：

```text
完整输入 [B=4,S,D]
  │
  ├─ chunk 0 [2,S,D]
  │    └─ offload context -> checkpoint -> 原 layer.forward
  │
  └─ chunk 1 [2,S,D]
       └─ offload context -> checkpoint -> 原 layer.forward
```

所以语言层的 checkpoint 边界输入是两个 `[2,S,D]` chunk，各自会得到独立的 `block_idx_tensor_idx` key。到下一 language layer 的第一个 chunk 时，上一层各 chunk 已完成的 D2H Tensor 会统一等待并释放 NPU storage。

反向则由 `torch.cat` 的梯度图把完整输出梯度拆回各 chunk；每个 chunk 对应的 checkpoint 在需要时触发自己的 H2D、重计算和 backward。参数梯度仍然累加到同一份 layer 参数上，优化器最后只更新一次。

也就是说，当前组合不是：

```text
完整 B=4 layer 激活 -> 一次 offload
```

而是：

```text
两个 B=2 的 layer 内部前向
  -> 每个 chunk 各自建立 checkpoint/offload 记录
  -> 输出拼回 B=4
```

### 20. 当前 Qwen3.5 实验中的实际覆盖范围

当前运行配置将下面两组 module 同时交给 recompute 和 activation offload：

```text
model.visual.blocks.{*}
model.language_model.layers.{*}
```

现有日志显示本次运行实际匹配：

- 3 个视觉 block；
- 12 个语言 layer；
- 合计 15 个连续 offload block，索引为 0 到 14。

当前模型配置的 12 个语言 layer 中，0-based 的第 3、7、11 层是 `full_attention`，其余 9 层是 `linear_attention`。因此语言模型部分分别由 `skip_flash_attn_recompute` 和 `skip_gdn_recompute` 覆盖对应类型，而不是让每层同时保存两套算子中间量。

日志同时记录：

```text
iteration 1/20  ... loss 11.41471 ...
...
iteration 20/20 ... loss 5.103907 ...
```

第 2 次迭代后日志报告：

```text
allocated: 16013.60 MB
max allocated: 27796.48 MB
reserved: 33748.00 MB
```

这些数字只描述“当前组合配置”的一次运行。没有同数据、同 shape、同层数、同 allocator 状态下的关闭 offload 基线，因此不能据此宣称节省了多少显存，也不能把首步编译/预热时间当作稳定性能。

---

## 第四部分：边界、风险与验证方法

### 21. 常见误解

#### 21.1 Host 会更新激活值吗

不会。Host 侧保存的是前向产生的数据副本。真正的层计算、重计算和梯度计算仍在 NPU 上进行。

#### 21.2 所有激活都会卸载吗

不会。通用 `utils` context 由调用方提供 `custom_check_fn`；当前 FSDP2 自动 wrapper 则只选择和入口 `hidden_states` 数据地址相同的 saved Tensor。两套实现都会跳过 Parameter、Parameter view 和空 storage。Flash Attention/GDN 的额外中间量由 FSDP2 skip-recompute 实现显式管理。

#### 21.3 异步以后就没有传输开销吗

不会。异步只能尝试用计算掩盖传输。如果计算时间不足，H2D event 仍会让计算流等待；Host 内存分配、总线带宽和 pinned memory 也都有成本。

#### 21.4 反向会变成很多次 `backward()` 吗

不会。训练代码仍调用一次 `loss.backward()`。H2D、checkpoint 重计算和各层 backward 都由 autograd 图按需触发。

#### 21.5 最后一层为什么不卸载

因为它是最先开始反向的层。前向结束后立刻把数据搬回通常无法获得驻留时间收益，反而增加一次 D2H/H2D 往返。

### 22. 当前 legacy 实现需要注意的约束

1. **执行顺序约束**：预取按相邻 block 索引工作，plan 应与真实前向顺序一致。
2. **Host 内存约束**：每个被卸载 Tensor 都会创建 pinned CPU buffer；显存压力会部分转移为 Host 内存压力。
3. **带宽约束**：长序列、大 batch 可能让传输时间超过可用于掩盖的计算时间。
4. **先区分导入路径**：通用 `utils` API 由调用方传 stream；当前 FSDP2 API 内部让 D2H/H2D 共用 `OffloadManager().swap_stream`。
5. **全局 Manager**：`OffloadManager` 是 Singleton，依赖 block 编号和调用顺序维护全局状态。
6. **最后 block 特例**：最后 block 的通用入口和 skip-recompute 中间量保持在 NPU。
7. **PyTorch 兼容性**：当前运行日志中 `tensor.storage()` 出现 TypedStorage deprecation warning；功能跑通，但未来 PyTorch 版本需要迁移到合适的 untyped storage 接口。
8. **收益不能只看 allocated**：应同时比较 `max allocated`、`max reserved`、稳定 step time、Host 内存和 H2D/D2H 时间线。

### 23. 怎样做可信的 A/B 验证

至少对同一批数据、相同 shape、相同随机种子、相同层数执行以下组合：

| 组别 | recompute | activation offload | skip Attention/GDN recompute |
|---|---:|---:|---:|
| A：基线 | 关 | 关 | 关 |
| B：只重计算 | 开 | 关 | 关 |
| C：重计算 + 入口卸载 | 开 | 开 | 关 |
| D：完整组合 | 开 | 开 | 开 |

建议记录：

- 预热后至少若干稳定 step 的平均/中位 step time；
- `max_memory_allocated` 与 `max_memory_reserved`；
- Host RSS 和 pinned memory 使用量；
- profiler 中 D2H/H2D 与计算 kernel 的重叠；
- 各组 loss、grad norm 是否在允许误差内一致；
- 是否存在 H2D event 长时间阻塞计算流。

只有 C 对 B，才能回答“入口激活卸载额外节省多少显存、增加多少传输”；只有 D 对 C，才能回答“保存昂贵算子结果、跳过其重计算是否值得”。

### 24. 最后总结

MindSpeed-MM 当前 legacy 异步激活值卸载的核心不是“把模型搬到 CPU”，而是把 autograd 在 checkpoint 边界保存的特定 Tensor 换一种保存方式：

```text
前向 pack：
NPU Tensor -> pinned CPU buffer -> D2H event -> 安全释放 NPU storage

反向 unpack：
恢复 NPU storage -> H2D -> 等待 h2d event -> 返回原 Tensor
```

与重计算结合后，分工是：

```text
入口 hidden_states 放 Host，作为重计算起点
普通层内激活不保存，反向重新生成
Flash Attention/GDN 昂贵中间结果放 Host，重计算时直接恢复
```

因此它本质上是在三种资源之间做调度：

```text
NPU 显存 <-> NPU 算力 <-> Host 内存与传输带宽
```

是否收益取决于 Tensor 大小、算子计算量、层数、序列长度、micro batch、Host 带宽和预取能否被计算掩盖，不能只根据“开启了异步”就假定性能一定更快。
