# Copy-In 与 Copy-Out 操作详解

> 基于 PyTorch v2.12.0 源码分析
> 核心源码：
>
> - `_fsdp_collectives.py`：`torch.ops.fsdp.all_gather_copy_in` / `split_with_sizes_copy` / `chunk_cat`
> - `_fsdp_param_group.py`：`unshard` / `wait_for_unshard` / `foreach_reduce`

## 0. 什么是 Copy-In 和 Copy-Out

在 FSDP2 的通信中，参数从一个布局转换为另一个布局时需要经过 Copy-In 和 Copy-Out 操作。这些操作本质上是**内存重排（memory rearrangement）**，确保数据以 NCCL 集体通信期望的格式存在。

```text
Copy-In:  将 "参数原生布局" → "通信缓冲区布局"
Copy-Out: 将 "通信缓冲区布局" → "参数原生布局"
```

## 1. 示例代码

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

mesh = init_device_mesh("cuda", (4,))  # 4 GPU

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)   # weight (512, 1024)
        self.fc2 = nn.Linear(512, 256)    # weight (256, 512)
        self.fc3 = nn.Linear(256, 128)    # weight (128, 256)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel().cuda()

# 对每层分别 FSDP
for layer in [model.fc1, model.fc2, model.fc3]:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

x = torch.randn(32, 1024).cuda()
output = model(x)      # ← Copy-In/Copy-Out 发生在 all-gather 过程中
loss = output.sum()
loss.backward()         # ← Copy-In/Copy-Out 发生在 reduce-scatter 过程中
```

## 2. All-Gather 的 Copy-In

### 2.1 为什么需要 Copy-In？

FSDP2 中每个参数是独立分片的，但 all-gather 需要**一个平坦的 1D 输入缓冲区**。Copy-In 将多个参数的分片数据复制到一个连续的缓冲区中。

```text
分片参数（独立的）:
  fc1.weight_shard: shape (128, 1024), 1D numel=131072
  fc2.weight_shard: shape (64, 512),   1D numel=32768
  fc3.weight_shard: shape (32, 256),   1D numel=8192

Copy-In 后（all-gather 输入缓冲区）:
  all_gather_input: shape (131072+32768+8192=172032,), 1D
  ┌─────────────────┬────────────┬─────────┐
  │   fc1_shard      │ fc2_shard  │fc3_shard│
  │   numel=131072   │ numel=32768│numel=8192│
  └─────────────────┴────────────┴─────────┘
```

### 2.2 源码实现

```python
# _fsdp_collectives.py: foreach_all_gather()
def foreach_all_gather(all_gather_inputs, all_gather_numels, fsdp_params, ...):
    # all_gather_inputs: 各参数的 1D 分片数据列表
    # all_gather_numels: 各参数的 numel 列表

    with device_handle.stream(copy_in_stream):
        # 1. 获取 all-gather 输入数据
        all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)

        # 2. 计算输入总大小
        self_all_gather_input_numel = sum(t.numel() for t in all_gather_inputs)

        # 3. 分配 all-gather 输出缓冲区
        #    大小为 input_numel × world_size（4 GPU → ×4）
        all_gather_output = all_gather_comm.allocate(
            self_all_gather_input_numel * group.size()
        )

        # 4. 执行 Copy-In: torch.ops.fsdp.all_gather_copy_in
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_1d, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,      # 输入：分片数据列表
            all_gather_output,      # 输出：all-gather 缓冲区
            inp_split_sizes,        # 各参数的 numel
            self_all_gather_input_numel,
            group.rank(),
        )
```

### 2.3 `all_gather_copy_in` 的 CUDA 实现

```python
# _fsdp_collectives.py: torch.ops.fsdp.all_gather_copy_in 的实现
@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
def all_gather_copy_in_cuda(inputs, output, inp_split_sizes, all_gather_input_numel, rank):
    # 1. 计算当前 rank 在输出缓冲区中的偏移
    all_gather_input = output.narrow(
        0,
        all_gather_input_numel * rank,  # rank 0: offset 0, rank 1: offset 172032, ...
        all_gather_input_numel,
    )

    # 2. 按参数拆分目标区间
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)

    # 3. 批量复制（高效的 foreach 操作）
    torch._foreach_copy_(foreach_copy_dsts, inputs)

    return all_gather_input, output
```

**图解**：

```text
输出缓冲区 all_gather_output (size = 4 × 172032):
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Rank 0 分区  │  Rank 1 分区  │  Rank 2 分区  │  Rank 3 分区  │
│  (172032,)   │  (172032,)   │  (172032,)   │  (172032,)   │
└──────────────┴──────────────┴──────────────┴──────────────┘
       ▲
       │ copy_in: rank 0 的数据写入这里
       │ rank 1 写入第2个分区，...
```

### 2.4 混合精度下的 Copy-In 快速路径

```python
# _fsdp_collectives.py: _get_param_all_gather_inputs
# 当使用混合精度且不在 offload 时，使用快速路径：
# 1. 第一遍：收集可以 foreach copy 的输入
# 2. 第二遍：分配平坦缓冲区，使用 torch._foreach_copy_ 批量复制
#    这比逐个 copy_ 有更低的 CPU overhead
```

## 3. All-Gather 的 Copy-Out

### 3.1 为什么需要 Copy-Out？

All-gather 输出是一个**平坦的 1D 缓冲区**，包含所有 rank 的完整数据。Copy-Out 将缓冲区重新分布到各参数的输出存储中。

```text
All-gather 输出 (1D 缓冲区):
  [rank0_fc1 | rank0_fc2 | rank0_fc3 | rank1_fc1 | rank1_fc2 | rank1_fc3 | ...]

Copy-Out 后（各参数获得完整数据）:
  fc1_weight: shape (512, 1024)  ← 从各 rank 的分区中收集
  fc2_weight: shape (256, 512)
  fc3_weight: shape (128, 256)
```

### 3.2 源码实现

```python
# _fsdp_collectives.py: foreach_all_gather_copy_out()
def foreach_all_gather_copy_out(all_gather_state, fsdp_params, group, ...):
    all_gather_output, all_gather_event, all_gather_work = all_gather_state

    device_handle = _get_device_handle(group.device_type)

    # 1. 同步：等待 all-gather 完成
    if all_gather_event is not None:
        device_handle.current_stream().wait_event(all_gather_event)
    if all_gather_work is not None:
        all_gather_work.wait()

    # 2. 计算各参数在输出中的 split 大小
    all_gather_output_splits = [
        fsdp_param.all_gather_output_numel
        for fsdp_param in fsdp_params
    ]

    # 3. reshaped output: 从 1D 变为包含 world_size 的 2D
    #    all_gather_output shape: (world_size * total_numel,)
    #    reshaped: (total_numel, world_size)

    # 4. Copy-Out: torch.ops.fsdp.split_with_sizes_copy
    per_param_all_gather_outputs = [
        fsdp_param.all_gather_output for fsdp_param in fsdp_params
    ]
    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output,           # all-gather 输出（1D）
        all_gather_output_splits,    # 各参数的 numel 大小
        per_param_all_gather_outputs, # 各参数的输出缓冲区
        # split_with_sizes_copy 内部将每个 rank 的分区复制到对应参数的输出
    )

    # 5. 初始化非分片参数视图
    for fsdp_param in fsdp_params:
        fsdp_param.init_unsharded_param()
```

### 3.3 `split_with_sizes_copy` 详解

```python
# torch.ops.fsdp.split_with_sizes_copy 的作用
# 将 all_gather_output (world_size * total_numel,) 重组到各参数：

# 输入: all_gather_output (1D tensor, size = world_size * total_numel)
# 对于参数 fc1 (numel = 131072):
#   fc1_output[0:131072]  ← 来自所有 rank 的 fc1 数据（已 all-gathered）
#   fc1_output 现在包含完整的 fc1 参数数据
#
# 实现：
#   for i, param_output in enumerate(per_param_all_gather_outputs):
#       # 从 all_gather_output 中提取第 i 个参数在各 rank 的数据
#       chunks = []
#       for r in range(world_size):
#           offset = r * total_numel + sum(splits[:i])
#           chunks.append(all_gather_output[offset : offset + splits[i]])
#       param_output.copy_(torch.cat(chunks))
```

### 3.4 非 dim=0 分片的 Copy-Out 特殊处理

```python
# _fsdp_collectives.py
if fsdp_placement.dim != 0:
    # 非 dim=0 分片需要额外的维度转换
    # 示例: fc1.weight (512, 1024) 沿 dim=1 分片
    # all-gather 沿 dim=0 输出 → 需要用 chunk + cat 转换为 dim=1 的分片
    
    temp = all_gather_output_narrowed
    chunks = torch.chunk(temp, world_size, dim=0)
    per_param_output = torch.cat(chunks, dim=fsdp_placement.dim)
```

## 4. Reduce-Scatter 的 Copy-In

### 4.1 Chunk-Cat 操作

在反向计算中，Copy-In 使用 `chunk_cat` 操作将非分片梯度重组：

```python
# _fsdp_collectives.py: foreach_reduce 中的 copy-in

# 输入: unsharded_grads
#   fc1_weight.grad: shape (512, 1024)   ← 完整梯度
#   fc2_weight.grad: shape (256, 512)
#   fc3_weight.grad: shape (128, 256)

# Copy-In: chunk_cat
reduce_scatter_input = torch.empty(total_grad_numel)
torch.ops.fsdp.chunk_cat(
    unsharded_grads,
    dim=0,
    num_chunks=world_size,  # 4
    out=reduce_scatter_input,
)
```

### 4.2 Chunk-Cat 的内存布局

```text
输入: [fc1_grad (512,1024), fc2_grad (256,512), fc3_grad (128,256)]

chunk_cat 操作（world_size=4）:
  1. 将每个梯度沿 dim=0 分成 4 份
  2. 按 world_size 分组交错排列

输出布局:
┌──────────────────────────────────────────────────────────────────────────┐
│ chunk 0              │ chunk 1              │ chunk 2              │ chunk 3│
│ fc1[0:128] fc2[0:64] │ fc1[128:256] fc2[64:128] │ ...              │ ...    │
│ fc3[0:32]            │ fc3[32:64]           │                   │        │
└──────────────────────────────────────────────────────────────────────────┘
     ↑ rank 0 负责              ↑ rank 1 负责

# chunk_cat 的 CUDA 实现:
#   for each grad in unsharded_grads:
#       for each chunk in world_size:
#           out[chunk_offset + param_offset + grad_offset] = grad[chunk]
```

**为什么需要这种布局？** Reduce-scatter 要求每个 rank 对自己负责的分区进行 reduce。Chunk-cat 后的布局使每个 rank 的数据在缓冲区中连续，NCCL 可以直接操作。

## 5. Reduce-Scatter 的 Copy-Out

### 5.1 写入分片梯度

```python
# _fsdp_collectives.py: foreach_reduce 中的 copy-out

# reduce-scatter 输出: reduce_output
#   每个 rank 得到自己负责的梯度分片（1D）

# Copy-Out: 将 reduce_output 写入各参数的分片梯度
for i, fsdp_param in enumerate(fsdp_params):
    # as_strided: 从 1D reduce_output 中创建对应参数的视图
    sharded_grad = torch.as_strided(
        reduce_output,
        size=fsdp_param.sharded_size,
        stride=...,
        storage_offset=offset,
    )

    # 写入分片梯度
    fsdp_param.sharded_param.grad = sharded_grad
```

### 5.2 CPU Offload 的 Copy-Out

```python
# 如果启用了 CPU offload，梯度需要复制到 CPU
if offload_to_cpu:
    with device_handle.stream(post_reduce_stream):
        cpu_grad = sharded_grad.to("cpu", non_blocking=True)
        fsdp_param.cpu_sharded_grad = cpu_grad
```

## 6. 完整通信流程图

### 6.1 All-Gather 流程

```text
分片参数 (每个 param 独立)           All-Gather 通信                   非分片参数
┌──────────────────┐              ┌─────────────────┐           ┌──────────────────┐
│ fc1_shard:       │              │                 │           │ fc1_weight:      │
│  (128, 1024)     │──flatten──→  │  copy_in_stream │           │  (512, 1024)     │
│                  │              │                 │           │                  │
│ fc2_shard:       │              │  all_gather_    │──copy──→  │ fc2_weight:      │
│  (64, 512)       │──flatten──→  │  copy_in        │   out     │  (256, 512)      │
│                  │              │     ↓           │           │                  │
│ fc3_shard:       │              │  all_gather_    │           │ fc3_weight:      │
│  (32, 256)       │──flatten──→  │  stream         │           │  (128, 256)      │
└──────────────────┘              │     ↓           │           └──────────────────┘
                                  │  all_gather     │
                                  │  (NCCL kernel)  │
                                  └─────────────────┘
```

### 6.2 Reduce-Scatter 流程

```text
非分片梯度                          Reduce-Scatter 通信                分片梯度
┌──────────────────┐              ┌─────────────────┐           ┌──────────────────┐
│ fc1_grad:        │              │  chunk_cat      │           │ fc1_shard_grad:  │
│  (512, 1024)     │──chunk──→    │  (copy-in)      │           │  (128, 1024)     │
│                  │              │     ↓           │           │                  │
│ fc2_grad:        │              │  reduce_scatter │──copy──→  │ fc2_shard_grad:  │
│  (256, 512)      │──chunk──→    │  stream         │   out     │  (64, 512)       │
│                  │              │     ↓           │           │                  │
│ fc3_grad:        │              │  reduce_scatter │           │ fc3_shard_grad:  │
│  (128, 256)      │──chunk──→    │  (NCCL kernel)  │           │  (32, 256)       │
└──────────────────┘              └─────────────────┘           └──────────────────┘
```

## 7. Copy-In/Copy-Out 的优化技巧

### 7.1 foreach_copy 批量操作

```python
# 不使用 foreach: O(n) 次 Python-CUDA 调用
for dst, src in zip(dsts, srcs):
    dst.copy_(src)

# 使用 foreach: 1 次 Python-CUDA 调用
torch._foreach_copy_(dsts, srcs)
```

### 7.2 内存复用

```python
# alloc_storage / free_storage 模式
# 参数 storage 在分片/非分片之间 resize，但 tensor 对象不变
# 避免了创建新 tensor 的 overhead
fsdp_param.alloc_storage(unsharded_size)   # resize storage
# ... 使用完整参数 ...
fsdp_param.free_storage()                   # resize back
```

### 7.3 Copy-In 与 All-Gather 的重叠

```python
# 源码注释说明:
# "we want to overlap the next copy-in with the current all-gather"
#
# 使用独立的 copy_in_stream 使当前 all-gather 在 all_gather_stream 上执行时，
# 下一层的 copy-in 在 copy_in_stream 上同步执行
```

---

> **源码参考**：
>
> - `all_gather_copy_in`: `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py` (torch.ops.fsdp library)
> - `split_with_sizes_copy`: 同上文件
> - `chunk_cat`: 同上文件
> - `foreach_all_gather`: 同上文件，描述完整 all-gather 流程
> - `foreach_reduce`: 同上文件，描述完整 reduce-scatter 流程
> - 快速路径 `foreach_copy`: `_get_param_all_gather_inputs` in `_fsdp_collectives.py`
