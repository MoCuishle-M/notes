# DTensor 切分详解：为什么是 dim=0，激活值/梯度/优化器状态

> 基于 PyTorch v2.12.0 源码分析
> 核心源码：
>
> - `_fsdp_param.py`：`_init_sharded_param` / `_init_sharding_spec` / `to_sharded` / `to_unsharded`
> - `_fsdp_common.py`：`ShardPlacementFnResult` / `resolve_shard_placement`
> - `_fsdp_init.py`：`_init_param_group` / `_get_mesh_info`

## 0. 示例代码

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# 4 GPU FSDP
mesh = init_device_mesh("cuda", (4,))

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)   # weight shape: (512, 1024)
        self.fc2 = nn.Linear(512, 256)    # weight shape: (256, 512)
        self.fc3 = nn.Linear(256, 128)    # weight shape: (128, 256)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel().cuda()

# 默认 dim=0 分片
fully_shard(model, mesh=mesh)

# 如果在 4 GPU 上运行，每个参数的分片情况如下：
# fc1.weight: 原始 (512, 1024) → 分片后每 rank 持有 (128, 1024)
# fc2.weight: 原始 (256, 512)  → 分片后每 rank 持有 (64, 512)
# fc3.weight: 原始 (128, 256)  → 分片后每 rank 持有 (32, 256)
```

## 1. 为什么默认分片 dim=0

### 1.1 源码中的默认值

```python
# _fsdp_param.py: FSDPParam._init_sharded_param()
def _init_sharded_param(self):
    fsdp_placement = self.fsdp_placement
    if fsdp_placement is None:
        fsdp_placement = Shard(0)  # ← 硬编码默认值
```

### 1.2 技术原因

#### 原因 1：通信效率 — All-Gather 沿 dim=0 最自然

All-gather 沿 dim=0 拼接分片是最直接的操作。NCCL 的 `all_gather_into_tensor` 天然支持沿 dim=0 的拼接，不需要额外的转置或重排。

```text
4 GPU 分片 fc1.weight (512, 1024) 沿 dim=0：

Rank 0: weight[0:128, :]     shape (128, 1024)
Rank 1: weight[128:256, :]   shape (128, 1024)
Rank 2: weight[256:384, :]   shape (128, 1024)
Rank 3: weight[384:512, :]   shape (128, 1024)

All-gather 后 → (512, 1024)  # 直接沿 dim=0 拼接
```

#### 原因 2：线性层数学特性

对于 `y = xW^T`（即 `F.linear(x, weight)`），weight 沿 dim=0（输出维度）分片后：

- 前向：`x @ weight_chunk^T` 产生部分输出，需要 all-gather 才能得到完整输出
- 但 FSDP2 在**参数层面**做 all-gather，而非激活层面
- 参数 all-gather 后，计算等价于单卡，输出自然完整

#### 原因 3：Reduce-Scatter 沿 dim=0 最自然**

Reduce-scatter 沿 dim=0 进行 reduce 后 scatter 是最标准的模式。梯度沿 dim=0 分片后，每个 rank 只保留自己负责的那部分梯度。

```text
梯度 ∂L/∂W shape (512, 1024)：

reduce-scatter 沿 dim=0 后:
Rank 0 得到: grad[0:128, :]       ← 只更新 weight[0:128, :]
Rank 1 得到: grad[128:256, :]
...
```

#### 原因 4：内存均匀分布

dim=0 通常是参数的最大维度（尤其是在线性层中），沿 dim=0 分片可以最大程度均匀分配内存。

### 1.3 自定义分片维度

```python
# 方式 1: 使用 shard_placement_fn
from torch.distributed.tensor import Shard

def my_shard_fn(param: nn.Parameter):
    # 对 2D 参数沿 dim=1 分片
    if param.dim() >= 2:
        return Shard(1)
    return Shard(0)

fully_shard(model, mesh=mesh, shard_placement_fn=my_shard_fn)

# 方式 2: 使用 ShardPlacementResult（指定自定义 mesh）
from torch.distributed.fsdp._fully_shard._fsdp_common import ShardPlacementResult

def my_shard_fn(param: nn.Parameter):
    return ShardPlacementResult(Shard(0), custom_mesh_info)
```

### 1.4 非 dim=0 分片的额外开销

```python
# _fsdp_param.py 中对非 dim=0 分片的处理
def _init_sharded_param(self):
    if fsdp_placement.dim != 0:
        # 要求均匀分片
        assert param.size(fsdp_placement.dim) % self.world_size == 0, \
            "Require even sharding for non-zero dim sharding"
```

```python
# _fsdp_collectives.py: foreach_all_gather_copy_out
# 非 dim=0 分片需要额外的内存重排
if fsdp_placement.dim != 0:
    # all-gather 沿 dim=0 输出，但需要沿 shard_dim 切分
    # 先保存到临时缓冲区，再 chunk + cat 转换维度
    temp = all_gather_output_narrowed
    chunks = torch.chunk(temp, world_size, dim=0)
    output = torch.cat(chunks, dim=fsdp_placement.dim)
```

## 2. 激活值需要切分吗？

**答案：在纯 FSDP 中，激活值不需要切分。**

### 原因

FSDP2 的策略是 **参数分片 + 激活完整**：

1. **前向计算**：参数通过 all-gather 恢复为完整参数，计算输出完整的激活值
2. **反向计算**：使用完整激活值 + 完整参数计算梯度
3. **梯度同步**：通过 reduce-scatter 将梯度分片

```python
# 计算流程示意
# 前向: x (完整) × W(all-gather后完整) → output (完整)
# 反向: grad_output (完整) × W(all-gather后完整) → grad_input (完整), grad_W (完整)
#       grad_W reduce-scatter → 分片梯度
```

### 激活值不分片的原因

- 激活值的形状取决于 batch size 和序列长度，形状不规则
- 在 Transformer 中，激活值需要被后续层使用，分片会增加通信
- FSDP 的核心思路是"参数分片，让通信发生在参数上"而非激活上

### 例外：与序列并行 (SP) 或张量并行 (TP) 结合时

```python
# 如果使用 Context Parallel（序列并行的变体）
# 激活值会被切分
# 但这是 SP/TP 的职责，不是 FSDP 的
```

## 3. 梯度需要切分吗？

**答案：是的，梯度必须切分。**

### 3.1 梯度的分片方式

梯度在反向计算时是**完整的**，但通过 reduce-scatter 后变为**分片的**：

```python
# 反向计算流程：
# 1. Autograd 计算完整梯度
#    例如 fc1.weight.grad shape (512, 1024)  ← 完整梯度

# 2. Reduce-scatter: 各 rank 对自己持有的分片做 reduce，然后 scatter
#    Rank 0 得到 grad[0:128, :]
#    Rank 1 得到 grad[128:256, :]
#    ...

# 3. 写入分片梯度到 DTensor
#    fc1.weight.grad = DTensor(local_grad, placements=(Shard(0),))
```

### 3.2 源码中的梯度处理

```python
# _fsdp_param.py: FSDPParam
@property
def unsharded_grad_data(self):
    """获取非分片梯度的 inner tensor"""
    grad = self._unsharded_param.grad
    if isinstance(grad, DTensor):
        # 对于 SPMD mesh，只有非 DP 维度需要重分布
        # DP 维度保持 Partial placement，由 FSDP 的 reduce-scatter 处理
        grad = grad.redistribute(
            placements=[Partial() if i in dp_dims else p
                       for i, p in enumerate(grad.placements)]
        )
    return grad._local_tensor if isinstance(grad, DTensor) else grad
```

### 3.3 梯度分片与参数分片的关系

梯度分片维度与参数分片维度一致：

```python
# 参数分片 = Shard(0)  →  梯度分片 = Shard(0)
# 参数分片 = Shard(1)  →  梯度分片 = Shard(1)
# 这是因为 optimizer.step() 需要 param.grad 和 param 有相同的 shape
```

## 4. 优化器状态需要切分吗？

**答案：是的，优化器状态自动切分。**

### 4.1 原理

FSDP2 的优化器状态切分是**隐式**的——不需要像 ZeRO-1/2 那样显式处理：

```python
# 参数以分片 DTensor 形式存在
# fc1.weight = DTensor(local_shape=(128, 1024), placements=(Shard(0),))

# 优化器创建
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 优化器状态自动基于分片参数创建
# AdamW 的 exp_avg 和 exp_avg_sq 形状与参数相同
# exp_avg shape = (128, 1024)  ← 自动分片！
# exp_avg_sq shape = (128, 1024)
```

**关键**：优化器创建时参数已经是分片 DTensor，`param.data` 是分片的 local tensor，优化器状态 (`exp_avg`, `exp_avg_sq`) 自动与分片参数一样大。

### 4.2 优化器步骤与参数状态同步

```python
# _fsdp_param.py: FSDPParam 的 alloc_storage / free_storage

# 训练循环中：
# 1. 前向：向分片参数的 storage 中写入完整数据
#    fsdp_param.alloc_storage(unsharded_size)
#    ↓ 参数 storage 扩展为完整大小
#    _unsharded_param.data  → shape (512, 1024)

# 2. 前向后：释放完整 storage
#    fsdp_param.free_storage()
#    ↓ 参数 storage 恢复为分片大小
#    _sharded_param._local_tensor  → shape (128, 1024)

# 3. Optimizer.step() 时参数是分片的
#    优化器状态也是分片的
#    参数更新作用于分片参数
```

### 4.3 为什么 storage 重分配不会破坏优化器状态

```python
# _fsdp_param.py 中的设计
# 关键: alloc_storage 通过 resize_ 修改 storage，不改变 tensor 对象
# 优化器状态通过 tensor 对象（而非 storage）关联参数
# 因此 storage  resize 不影响优化器状态
```

## 5. DTensor 分片的三种 ShardingSpec

### 5.1 纯 FSDP (1D Mesh)

```python
# mesh = (4,)  # 1D mesh
# sharding_spec = DTensorSpec(mesh, (Shard(0),))

# fc1.weight 在 4 GPU 上的分片：
# Rank 0: weight[0:128, :]
# Rank 1: weight[128:256, :]
# Rank 2: weight[256:384, :]
# Rank 3: weight[384:512, :]
```

### 5.2 HSDP (2D Mesh)

```python
# mesh = (2, 4)  # 2D mesh: dim0=replicate(2), dim1=shard(4)
# sharding_spec = DTensorSpec(mesh, (Replicate(), Shard(0)))

# fc1.weight 在 8 GPU 上的分片：
# 复制组 0 (rank 0-3):  每个持有 weight[0:128, :], weight[128:256, :], ...
# 复制组 1 (rank 4-7):  每个持有 weight[0:128, :], weight[128:256, :], ...
# 两个复制组持有完全相同的分片数据
```

### 5.3 DDP (纯复制)

```python
# mesh = (4,)  with DDPMeshInfo
# sharding_spec = DTensorSpec(mesh, (Replicate(),))

# fc1.weight 在 4 GPU 上：每个持有完整 weight
```

### 5.4 源码中的 ShardingSpec 初始化

```python
# _fsdp_param.py: _init_sharding_spec()

def _init_sharding_spec_plain(self, mesh_info):
    """非 DTensor 参数的 sharding spec 初始化"""
    if isinstance(mesh_info, HSDPMeshInfo):
        placements = [Replicate(), Shard(self.fsdp_placement.dim)]
    elif isinstance(mesh_info, FSDPMeshInfo):
        placements = [Shard(self.fsdp_placement.dim)]
    else:  # DDPMeshInfo
        placements = [Replicate()]
    
    return DTensorSpec(mesh=mesh_info.mesh, placements=placements)

def _init_sharding_spec_spmd(self, mesh_info):
    """SPMD mesh 上的 DTensor 参数"""
    # 将 DP 维度的 Replicate 转为 _StridedShard
    # 保持 TP/PP 维度的 placements 不变
    
def _init_sharding_spec_tp(self, mesh_info):
    """TP + FSDP 组合"""
    # 拼接 DP mesh 和 TP mesh
    # placements = TP_placements + FSDP_placements
```

## 6. 分片状态的完整生命周期

```python
# FSDPParam 的三种分片状态
class ShardedState(Enum):
    SHARDED = 1              # 分片参数已注册到模块
    SHARDED_POST_FORWARD = 2 # 前向后分片（HSDP 用更小的 mesh）
    UNSHARDED = 3            # 完整参数已注册到模块

# 状态转换：
# SHARDED ──unshard()──→ UNSHARDED ──reshard()──→ SHARDED
# SHARDED ──unshard()──→ UNSHARDED ──reshard(HSDP)──→ SHARDED_POST_FORWARD
# SHARDED_POST_FORWARD ──unshard()──→ UNSHARDED
```

### 6.1 状态转换源码

```python
# _fsdp_param.py
def to_sharded(self):
    """切换到分片状态"""
    if self.sharded_state == ShardedState.SHARDED:
        return
    self._setattr_on_modules(self._sharded_param)  # 注册分片 DTensor
    self.sharded_state = ShardedState.SHARDED

def to_unsharded(self):
    """切换到非分片状态"""
    if self.sharded_state == ShardedState.UNSHARDED:
        return
    self._setattr_on_modules(self._unsharded_param)  # 注册完整参数
    self.sharded_state = ShardedState.UNSHARDED

def to_sharded_post_forward(self):
    """切换到前向后分片状态（HSDP 专用）"""
    self._setattr_on_modules(self._sharded_param_post_forward)
    self.sharded_state = ShardedState.SHARDED_POST_FORWARD
```

## 7. 完整分片示例对比

```python
# ====== 纯 FSDP (4 GPU, 1D mesh) ======
mesh = init_device_mesh("cuda", (4,))
fully_shard(model, mesh=mesh)

# fc1.weight (512, 1024):
#   Rank 0: DTensor(local=(128, 1024), placements=(Shard(0),))
#   Rank 1: DTensor(local=(128, 1024), placements=(Shard(0),))
#   Rank 2: DTensor(local=(128, 1024), placements=(Shard(0),))
#   Rank 3: DTensor(local=(128, 1024), placements=(Shard(0),))

# fc1.weight.grad (512, 1024):
#   Rank 0: DTensor(local=(128, 1024), placements=(Shard(0),))
#   ... (同参数分片)

# optimizer exp_avg:
#   Rank 0: Tensor(128, 1024)  # 自动匹配分片参数形状

# ====== HSDP (8 GPU, 2D mesh, replicate=2, shard=4) ======
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))
fully_shard(model, mesh=mesh)

# fc1.weight (512, 1024):
#   Rank 0-3 (复制组 1): DTensor(local=(128, 1024), placements=(Replicate(), Shard(0)))
#   Rank 4-7 (复制组 2): DTensor(local=(128, 1024), placements=(Replicate(), Shard(0)))
#   两个复制组的数据相同
```

---

> **源码参考**：
>
> - 默认 Shard(0): `_fsdp_param.py: _init_sharded_param()`
> - ShardingSpec 初始化: `_fsdp_param.py: _init_sharding_spec()`
> - 参数状态切换: `_fsdp_param.py: to_sharded() / to_unsharded() / to_sharded_post_forward()`
> - Shard placement 解析: `_fsdp_common.py: resolve_shard_placement()`
> - 非 dim=0 分片处理: `_fsdp_collectives.py: foreach_all_gather_copy_out()`
