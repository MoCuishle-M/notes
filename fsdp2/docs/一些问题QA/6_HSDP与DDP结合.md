# HSDP：FSDP2 与 DDP 结合 —— All-Reduce 如何工作

> 基于 PyTorch v2.12.0 源码分析
> 核心源码：
>
> - `_fsdp_common.py`：`HSDPMeshInfo` / `FSDPMeshInfo` / `DDPMeshInfo` 类定义
> - `_fsdp_init.py`：`_get_mesh_info` / `_get_post_forward_mesh_info`
> - `_fsdp_collectives.py`：`foreach_reduce` 中的 HSDP all-reduce 路径
> - `_fsdp_param.py`：HSDP 的 `_init_sharding_spec`

## 0. HSDP 是什么

HSDP（Hybrid Sharded Data Parallel）将 FSDP 的**参数分片**与 DDP 的**梯度复制**结合：

```text
                    DeviceMesh (2×4)
         ┌──────────┬──────────┬──────────┬──────────┐
  Rank0  │  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │  ← Shard Group 0
  Rank1  │  GPU 4   │  GPU 5   │  GPU 6   │  GPU 7   │  ← Shard Group 1
         └──────────┴──────────┴──────────┴──────────┘
              ↑          ↑          ↑          ↑
         Replicate Group 0, 1, 2, 3 (跨 shard group 复制)

- Shard dim (dim=1): 4 GPU 间分片参数（FSDP 行为）
- Replicate dim (dim=0): 2 个 shard group 间复制参数（DDP 行为）
```

**HSDP 的优势**：

- FSDP 减少了单个 GPU 的内存占用（分片参数）
- DDP 减少了跨节点通信（all-reduce 只在 shard group 内？错—all-reduce 跨 replicate group）

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

# ====== HSDP: 2D mesh ======
# dim 0: dp_replicate (DDP 复制维度) — 2 个复制组
# dim 1: dp_shard (FSDP 分片维度) — 4 个分片
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 4),  # 总共 8 GPU
    mesh_dim_names=("dp_replicate", "dp_shard"),
)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x

model = Model().cuda()

# 逐层 FSDP（在 2D mesh 上自动变为 HSDP）
for block in [model.block0, model.block1]:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)

x = torch.randn(32, 4096).cuda()
output = model(x)
loss = output.sum()
loss.backward()  # ← HSDP 的 all-reduce 发生在这里
```

## 2. HSDP 的 Mesh 初始化

### 2.1 源码中的 Mesh 解析

```python
# _fsdp_init.py: _get_mesh_info()
def _get_mesh_info(mesh, dp_mesh_dims=None):
    if mesh.ndim == 1:
        return FSDPMeshInfo(mesh, shard_mesh_dim=0)  # 纯 FSDP
    else:
        # 2D mesh → HSDP
        return HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
```

### 2.2 HSDPMeshInfo 的多重继承

```python
# _fsdp_common.py
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    """同时具有 shard 和 replicate 维度"""
    def __post_init__(self):
        super().__post_init__()
        # MRO: HSDPMeshInfo → FSDPMeshInfo → DDPMeshInfo → DataParallelMeshInfo
        # 先后调用 FSDPMeshInfo.__post_init__（设置 shard_mesh_dim）
        # 和 DDPMeshInfo.__post_init__（设置 replicate_mesh_dim）
```

**HSDPMeshInfo 的关键属性**：

```python
# 继承自 FSDPMeshInfo:
self.shard_mesh_dim = 1       # 分片维度
self.shard_mesh_size = 4      # 4 GPU 间分片
self.shard_process_group      # shard dim 的进程组
self.shard_mesh_rank          # 在 shard group 中的 rank

# 继承自 DDPMeshInfo:
self.replicate_mesh_dim = 0   # 复制维度
self.replicate_mesh_size = 2  # 2 个复制组
self.replicate_process_group  # replicate dim 的进程组
self.replicate_mesh_rank      # 在 replicate group 中的 rank
```

### 2.3 使用命名维度配置

```python
# 方式 1: 隐式 2D mesh（dim 0=replicate, dim 1=shard）
mesh = init_device_mesh("cuda", (2, 4))
# → HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)

# 方式 2: 命名维度（推荐，更清晰）
mesh = init_device_mesh(
    "cuda", (2, 4),
    mesh_dim_names=("dp_replicate", "dp_shard")
)
# → HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)

# 方式 3: dp_mesh_dims（显式指定哪个维度做什么）
from torch.distributed.fsdp import fully_shard

mesh = init_device_mesh(
    "cuda", (2, 4),
    mesh_dim_names=("dp_replicate", "dp_shard")
)
fully_shard(
    model,
    mesh=mesh,
    # 不指定 dp_mesh_dims 时，2D mesh 自动使用 HSDP
)
```

## 3. HSDP 的前向计算

HSDP 的前向计算与纯 FSDP 基本相同，区别在于参数的分片方式：

### 3.1 DTensor 的 ShardingSpec

```python
# _fsdp_param.py: _init_sharding_spec_plain()

# 纯 FSDP (1D mesh):
# placements = [Shard(0)]
# fc1.weight = DTensor(local=(1024, 4096), placements=(Shard(0),))

# HSDP (2D mesh):
# placements = [Replicate(), Shard(0)]
# fc1.weight = DTensor(local=(1024, 4096), placements=(Replicate(), Shard(0)))
#                                                                  ↑            ↑
#                                                          dim 0: replicas  dim 1: shard
```

### 3.2 All-Gather 在 HSDP 中

```python
# 在 HSDP 中，all-gather 只在 shard 维度上进行
# 因为 replicate 维度上的参数已经是完整的

# _fsdp_param_group.py: unshard()
def unshard(self):
    # self.mesh_info.shard_process_group 包含 shard dim 上的 GPU
    # 对于 (2, 4) mesh，shard_process_group 有 4 个成员
    # replicate dim 上的 GPU 不参与 all-gather
    all_gather_group = self.mesh_info.shard_process_group
```

**HSDP 的 All-Gather 范围**：

```text
(2, 4) mesh:
  Rank 0 (SG0, RG0): all-gather with {0, 1, 2, 3}      ← 只在 shard group 内
  Rank 1 (SG0, RG1): all-gather with {0, 1, 2, 3}
  Rank 4 (SG1, RG0): all-gather with {4, 5, 6, 7}      ← 只在 shard group 内
  Rank 5 (SG1, RG1): all-gather with {4, 5, 6, 7}
```

## 4. HSDP 的反向计算与 All-Reduce

这是 HSDP 与纯 FSDP 最大的不同。反向计算分两步：

### 4.1 步骤 1：Reduce-Scatter（Shard Dim 内）

```python
# _fsdp_collectives.py: foreach_reduce()

# 阶段 1: Reduce-Scatter（FSDP 部分）
# 在 shard_process_group（4 GPU）内执行
with device_handle.stream(reduce_scatter_stream):
    reduce_scatter_work = reduce_scatter_comm(
        reduce_output,          # 输出：每个 rank 的分片梯度
        reduce_scatter_input,   # 输入：chunk_cat 后的完整梯度
        group=shard_process_group,  # FSDP shard group
    )
    # 等价于: dist.reduce_scatter_tensor(
    #     reduce_output, reduce_scatter_input,
    #     group=shard_process_group
    # )
```

**此时的数据状态**：

```text
Reduce-scatter 后（以 fc1.weight 为例，shape 4096×4096）：
  Shard Group 0:
    Rank 0: gradient[0:1024, :]       ← 4 GPU 间分片
    Rank 1: gradient[1024:2048, :]
    Rank 2: gradient[2048:3072, :]
    Rank 3: gradient[3072:4096, :]
  
  Shard Group 1:
    Rank 4: gradient[0:1024, :]       ← 相同分片，但值可能不同（因为不同 rank 的输入数据不同）
    Rank 5: gradient[1024:2048, :]
    Rank 6: gradient[2048:3072, :]
    Rank 7: gradient[3072:4096, :]

注意：Rank 0 和 Rank 4 持有相同的分片区间 [0:1024, :]
     但由于输入数据不同，它们的梯度值不同
     需要 all-reduce 跨 replicate group 同步！
```

### 4.2 步骤 2：All-Reduce（Replicate Dim 内）

```python
# _fsdp_collectives.py: foreach_reduce()
# HSDP all-reduce 路径

if all_reduce_group is not None:  # replicate_process_group
    with device_handle.stream(all_reduce_stream):
        # 等待 reduce-scatter 完成
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        
        # 跨 replicate group 执行 all-reduce
        # Rank 0 ←→ Rank 4 交换梯度（相同分片区间）
        # Rank 1 ←→ Rank 5
        # ...
        dist.all_reduce(
            reduce_output,         # 输入/输出相同 buffer
            group=all_reduce_group,  # replicate_process_group
            op=all_reduce_op,       # ReduceOp.AVG 或 SUM
        )
        
        all_reduce_event = all_reduce_stream.record_event()
```

### 4.3 All-Reduce 的 Stream 同步

```python
# Stream 之间的同步链：
# 1. reduce_scatter_stream.wait_stream(current_stream)
#    → 确保 chunk_cat (copy-in) 完成
# 2. all_reduce_stream.wait_stream(reduce_scatter_stream)
#    → 确保 reduce-scatter 完成
# 3. post_reduce_stream.wait_stream(all_reduce_stream)
#    → 确保 all-reduce 完成，然后写入分片梯度
```

### 4.4 All-Reduce 后的数据状态

```text
All-Reduce 后（以 fc1.weight 为例）：
  Rank 0 和 Rank 4: gradient[0:1024, :]  ← 现在相同（已 all-reduce）
  Rank 1 和 Rank 5: gradient[1024:2048, :] ← 相同
  Rank 2 和 Rank 6: gradient[2048:3072, :] ← 相同
  Rank 3 和 Rank 7: gradient[3072:4096, :] ← 相同

每个 rank 的优化器只更新自己持有的分片参数
```

## 5. 梯度累积场景下的 HSDP

```python
# 梯度累积时，可以先累积 reduce-scatter 的结果
# 在最后一个 micro-batch 再执行 all-reduce

# 非最后一个 micro-batch:
model.set_requires_all_reduce(False)  # 跳过 all-reduce
# foreach_reduce:
#   if not all_reduce_grads:
#       partial_reduce_output += reduce_output  # 累加到 partial
#       return  # 不执行 all-reduce

# 最后一个 micro-batch:
model.set_requires_all_reduce(True)   # 执行 all-reduce
# foreach_reduce:
#   reduce_output += partial_reduce_output  # 合并累积的梯度
#   dist.all_reduce(reduce_output, ...)     # 执行 all-reduce
```

```python
# _fsdp_param_group.py
# set_requires_all_reduce 控制 HSDP 的 all-reduce 行为
def set_requires_all_reduce(self, requires_all_reduce, recurse=True):
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.all_reduce_grads = requires_all_reduce
```

## 6. All-Reduce Hook（用户自定义）

```python
# 用户可以注册 all-reduce hook 来检查或修改 all-reduce 的输出
def my_all_reduce_hook(reduce_output: torch.Tensor) -> None:
    # 检查梯度范数
    grad_norm = reduce_output.norm()
    print(f"Gradient norm after all-reduce: {grad_norm}")

model.set_all_reduce_hook(my_all_reduce_hook)
```

```python
# _fsdp_collectives.py: foreach_reduce
# All-reduce hook 在 all_reduce_stream 上执行
if self._all_reduce_hook is not None:
    with device_handle.stream(all_reduce_stream):
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        self._all_reduce_hook(reduce_output)  # 用户 hook
```

## 7. HSDP vs FSDP vs DDP 对比

| 特性 | DDP | FSDP | HSDP |
| ------ | ----- | ------ | ------ |
| 参数存储 | 完整（每 GPU） | 分片（跨所有 GPU） | 分片（跨 shard group） |
| 前向通信 | 无 | All-gather | All-gather（shard group 内） |
| 反向通信 | All-reduce | Reduce-scatter | Reduce-scatter + All-reduce |
| All-reduce 范围 | 所有 GPU | 无 | Replicate group 内 |
| 跨节点通信 | 所有 GPU | 只 shard group 内 | shard group 内 all-gather + replicate group 内 all-reduce |
| 内存效率 | 低 | 高 | 中 |
| 通信效率 | 跨节点带宽受限 | 跨节点通信少 | 灵活权衡 |

### HSDP 的典型部署拓扑

```text
节点 1 (8 GPU):             节点 2 (8 GPU):
┌─────────────────────┐    ┌─────────────────────┐
│ GPU0 GPU1 GPU2 GPU3 │    │ GPU4 GPU5 GPU6 GPU7 │
│  ↑     ↑     ↑     ↑ │    │  ↑     ↑     ↑     ↑ │
│  └─ Shard Group 0 ─┘ │    │  └─ Shard Group 1 ─┘ │
│  Replicate Group      │    │  Replicate Group      │
└─────────────────────┘    └─────────────────────┘
     │         跨节点 all-reduce          │

mesh_shape = (2, 4)  # 2 replicate groups, 4 shard GPUs each
shard dim = 1 (节点内高速 NVLink)
replicate dim = 0 (跨节点 InfiniBand/RoCE)
```

## 8. Post-Forward Reshard 与 HSDP

HSDP 还支持**部分 reshard** 以进一步节省内存：

```python
# reshard_after_forward 可以是整数
# 例如在 (2, 4) mesh 上，reshard_after_forward=2:
# 前向后参数被 reshard 到 2 个 GPU 的 mesh 上
# 这比完全 reshard (4 GPU) 使用更多内存，但下次前向的 all-gather 更快

fully_shard(model, mesh=mesh, reshard_after_forward=2)
```

```python
# _fsdp_init.py: _get_post_forward_mesh_info
if isinstance(reshard_after_forward, int):
    post_forward_mesh = DeviceMesh(
        mesh.device_type,
        mesh.mesh.view(-1, reshard_after_forward)
    )
    # 创建新的 HSDP mesh 用于 post-forward reshard
    post_forward_mesh_info = HSDPMeshInfo(
        post_forward_mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0,
    )
```

## 9. Gradient Division 在 HSDP 中的特殊处理

```python
# _fsdp_collectives.py: _get_gradient_divide_factors()

# HSDP 需要分配 division factor 到两个阶段:
# 阶段 1 (reduce-scatter): 除以 predivide_factor
# 阶段 2 (all-reduce):     乘以 NCCL AVG (或显式除以 postdivide_factor)

# 总 factor = world_size = shard_mesh_size * replicate_mesh_size
# 例如 (2, 4) mesh: factor = 8

# 使用 ReduceOp.AVG 时:
#   reduce-scatter 使用 AVG (除以 4)
#   all-reduce 使用 AVG (除以 2)
#   等效于除以 8
```

---

> **源码参考**：
>
> - `HSDPMeshInfo`: `torch/distributed/fsdp/_fully_shard/_fsdp_common.py`
> - `_get_mesh_info`: `torch/distributed/fsdp/_fully_shard/_fsdp_init.py`
> - `_init_sharding_spec_plain` (HSDP placements): `torch/distributed/fsdp/_fully_shard/_fsdp_param.py`
> - `foreach_reduce` (HSDP all-reduce path): `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`
> - `_get_post_forward_mesh_info`: `torch/distributed/fsdp/_fully_shard/_fsdp_init.py`
> - `set_requires_all_reduce`: `torch/distributed/fsdp/_fully_shard/_fully_shard.py` (FSDPModule)
