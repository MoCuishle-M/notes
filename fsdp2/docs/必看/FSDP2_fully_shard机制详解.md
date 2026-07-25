# FSDP2 `fully_shard` 机制详解：从普通 Tensor 到 DTensor 的全流程剖析

> 基于 PyTorch 源码（v2.12.1）分析
> 源码根目录：`torch/distributed/fsdp/_fully_shard/`
>
> 本文以源码为准，辅以行为示例与 DTensor 切分原理，自顶向下解析 `fully_shard` 的工作机制，重点阐述**模型权重向 DTensor 转换**的全过程，以及前向 Unshard / 反向 Reduce-Scatter 的内存生命周期。
>
> 关键源码文件：
>
> - 入口与 `FSDPModule`：`_fully_shard.py`
> - 初始化（mesh/param group/MRO 替换）：`_fsdp_init.py`
> - mesh 信息与公共工具：`_fsdp_common.py`
> - 单参数分片（DTensor 转换核心）：`_fsdp_param.py`
> - 参数组生命周期（unshard/reshard/reduce-scatter）：`_fsdp_param_group.py`
> - 通信原语（all-gather/reduce-scatter）：`_fsdp_collectives.py`
> - 状态与 hook 注册：`_fsdp_state.py`

---

## 目录

- [第一部分：通俗原理解析（概念层）](#第一部分通俗原理解析概念层)
  - [1. 一个简单模型与故事主线](#1-一个简单模型与故事主线)
  - [2. 初始化与切分：普通 Tensor 如何变成 DTensor](#2-初始化与切分普通-tensor-如何变成-dtensor)
  - [3. 前向传播（Unshard）：权重的临时聚合](#3-前向传播unshard权重的临时聚合)
  - [4. 反向传播（Reduce Scatter）：梯度的分片聚合](#4-反向传播reduce-scatter梯度的分片聚合)
  - [5. 内存生命周期总览](#5-内存生命周期总览)
- [第二部分：源码深度剖析（实现层）](#第二部分源码深度剖析实现层)
  - [6. `fully_shard` 入口：从 API 到 MRO 替换](#6-fully_shard-入口从-api-到-mro-替换)
  - [7. DTensor 转换核心：`FSDPParam._init_sharded_param`](#7-dtensor-转换核心fsdpparam_init_sharded_param)
  - [8. Unshard 源码：all-gather 的 copy-in / 通信 / copy-out](#8-unshard-源码all-gather-的-copy-in--通信--copy-out)
  - [9. Reshard 源码：storage 缩放释放内存](#9-reshard-源码storage-缩放释放内存)
  - [10. Reduce-Scatter 源码：梯度归约与 DTensor 写回](#10-reduce-scatter-源码梯度归约与-dtensor-写回)
  - [11. Hook 编排：前向 / 反向如何被触发](#11-hook-编排前向--反向如何被触发)
- [附录：关键数据结构与速查表](#附录关键数据结构与速查表)

---

# 第一部分：通俗原理解析（概念层）

## 1. 一个简单模型与故事主线

我们用一个非常小的模型作为贯穿全文的例子，运行在 **4 张 GPU** 上：

```python
import torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
mesh = init_device_mesh("cuda", (4,))   # 1D mesh，纯 FSDP

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)   # weight: (512, 1024)
        self.fc2 = nn.Linear(512, 256)    # weight: (256, 512)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP().cuda()
fully_shard(model, mesh=mesh)            # 只对顶层包装
```

故事主线只有三幕，围绕 `fc1.weight`（shape `(512, 1024)`，float32，约 2MB）展开：

| 阶段 | `fc1.weight` 在每张 GPU 上的形态 | 单卡显存占用 |
|------|----------------------------------|--------------|
| ① 初始化切分 | DTensor，`local=(128,1024)`，`placements=(Shard(0),)` | 0.5 MB（1/4） |
| ② 前向 Unshard | 完整 Tensor `(512,1024)`（临时聚合） | 2.0 MB（全量） |
| ③ 前向结束 Reshard | DTensor，`local=(128,1024)`（释放聚合） | 0.5 MB |
| ④ 反向 Unshard | 完整 Tensor `(512,1024)`（再次聚合） | 2.0 MB |
| ⑤ 反向 Reduce-Scatter | 完整梯度先归约再分片 → `fc1.weight.grad` 是 DTensor `local=(128,1024)` | 0.5 MB |

核心思想一句话：**参数平时分片存放（省显存），计算时临时拼回全量（保正确），算完立刻释放（控峰值），梯度同样先算全量再分片归约。** DTensor 就是“分片存放”这一状态的载体——它是一个带分布信息（mesh + placements）的本地分片张量。

---

## 2. 初始化与切分：普通 Tensor 如何变成 DTensor

### 2.1 调用 `fully_shard` 之后发生了什么（概念版）

调用 `fully_shard(model, mesh=mesh)` 时，FSDP2 并不立即把所有参数一次性搬走，而是做这几件事：

1. **校验模块与 mesh**：拒绝 `nn.ModuleList` 这类无 `forward` 的容器；把 1D mesh 解析为 `FSDPMeshInfo`（2D 解析为 `HSDPMeshInfo`）。
2. **收集参数**：遍历 `model` 的子模块，收集所有尚未被内层 `fully_shard` 接管的参数（“bottom-up”原则——应先对子模块 `fully_shard`，最后才对根模块 `fully_shard`）。
3. **就地改类（MRO 替换）**：把 `model.__class__` 从 `SimpleMLP` 改成一个新类 `FSDPSimpleMLP(FSDPModule, SimpleMLP)`。这样 `FSDPModule` 的方法会优先于原始 `forward` 被调用，FSDP2 从而能拦截 forward、插入 all-gather/reshard 钩子，而**不改变模型结构、不替换对象**。
4. **逐参数切分**：对每个参数沿 `dim=0`（默认）切成 4 份，每张卡保留一份，并包装成 **DTensor** 注册回模块。

### 2.2 普通 Tensor → DTensor 的切分策略

切分策略由 **placement**（放置方式）描述，FSDP2 默认是 `Shard(0)`。对 `fc1.weight (512, 1024)`、4 GPU、`Shard(0)`：

```text
原始完整参数 (512, 1024)：
  Rank 0: weight[  0:128, :]   shape (128, 1024)
  Rank 1: weight[128:256, :]
  Rank 2: weight[256:384, :]
  Rank 3: weight[384:512, :]
```

切分后，模块上注册的不再是普通 `Tensor`，而是一个 **DTensor**：

```python
# fc1.weight 现在是：
DTensor(
    local_tensor: Tensor(128, 1024),   # 本卡只持有 1/4
    placements: (Shard(0),),           # 沿 dim=0 切分在 1D mesh 上
    device_mesh: DeviceMesh(shape=(4,)),
)
```

**为什么默认 `Shard(0)`？** 四个原因（详见辅助文档 B）：

- **通信效率**：NCCL 的 `all_gather_into_tensor` 天然沿 dim=0 拼接，无需转置。
- **线性层数学特性**：参数 all-gather 后计算等价于单卡，输出自然完整。
- **Reduce-Scatter 友好**：沿 dim=0 reduce 后 scatter 最标准，梯度分片维度与参数一致。
- **内存均匀**：dim=0 通常是参数最大维度，分片最均匀。

**自定义分片维度**：通过 `shard_placement_fn` 返回 `Shard(1)` 或 `ShardPlacementResult(Shard(1), custom_mesh_info)`，可让不同参数用不同 mesh（如 MoE 的 expert 参数）。

### 2.3 HSDP（2D mesh）的切分

若 mesh 是 2D `(2, 4)`（2 个复制组 × 4 个分片组），placements 变为 `(Replicate(), Shard(0))`：复制组间数据完全相同，分片组内沿 dim=0 切分。梯度反向时多一步跨复制组的 `all_reduce`。

```text
mesh = (2, 4)，fc1.weight (512, 1024)：
  复制组 0 (rank 0-3)：各自持 weight[0:128,:], [128:256,:], ...
  复制组 1 (rank 4-7)：与复制组 0 完全相同的数据
```

---

## 3. 前向传播（Unshard）：权重的临时聚合

### 3.1 Unshard 的概念

前向计算需要**完整参数**才能得到正确输出。FSDP2 在模块 `forward` 之前插入一个 **pre-forward hook**，它做两件事：

1. **All-Gather**：每张卡把自己的分片发出去，同时收集其余 3 张卡的分片，沿 dim=0 拼成完整 `(512, 1024)`。
2. **注册回模块**：把完整参数临时写回 `fc1.weight`（实际是写回预分配的 `_unsharded_param` 对象），供 `nn.Linear` 的 `F.linear(x, weight)` 使用。

```text
前向前：  fc1.weight = DTensor(local=(128,1024), Shard(0))     ← 0.5 MB
Unshard： all-gather(dim=0)
前向时：  fc1.weight = Tensor(512, 1024)                        ← 2.0 MB（临时）
```

### 3.2 内存占用的“冲高与回落”

对“顶层包装”的例子（所有参数在一个 group 里），前向时一次性 all-gather **所有**参数，内存峰值高。最佳实践是**逐层包装**：每个 Transformer block 单独 `fully_shard`，这样 all-gather 逐层进行，且下一层的 all-gather 可与当前层的计算重叠（前向预取），峰值大幅降低。

### 3.3 Reshard after forward

前向算完后，**post-forward hook** 会立即释放完整参数、重新注册回分片 DTensor（除非 `reshard_after_forward=False`，根模块默认为 `False`，因为反向一开始就要用）。这就是“用完即弃”——内存从 2.0 MB 回落回 0.5 MB。

```text
前向后：  fc1.weight = DTensor(local=(128,1024), Shard(0))     ← 0.5 MB（释放聚合）
```

> **关键实现技巧**：FSDP2 不销毁 `_unsharded_param` 对象，而是用 **storage resize** 把它的底层存储缩到 0（`free_storage`），下次再 all-gather 时再扩回来（`alloc_storage`）。这样 autograd 保存的引用/视图不会失效，别名关系得以保留。

---

## 4. 反向传播（Reduce Scatter）：梯度的分片聚合

### 4.1 反向的两个动作

反向传播对每个 FSDP 模块也有两个钩子：

1. **pre-backward（Unshard）**：反向开始前，再次 all-gather 把完整参数拼回来——因为算梯度 `grad_W = grad_output^T @ x` 需要完整权重。
2. **post-backward（Reduce-Scatter）**：autograd 算出的梯度是**完整**的 `(512, 1024)`，但每张卡都算出了同样的梯度副本（因为输入 `x` 在各卡上是复制的）。需要把 4 张卡的完整梯度 **求和（reduce）后再分片（scatter）**，让每卡只保留自己负责的那 1/4 梯度。

### 4.2 梯度为什么必须切分

- **内存**：若保留完整梯度，每卡都要存全量，违背 FSDP 省显存的初衷。
- **与参数对齐**：optimizer.step() 要求 `param.grad` 与 `param` 形状一致。参数是分片 DTensor，梯度也必须是同形状的分片 DTensor。

```text
反向算梯度： grad_W (512, 1024)  ← 完整，4 张卡各有一份相同副本
Reduce-Scatter(dim=0)：
  Rank 0 得到: grad_W[0:128,:]   ← 只更新 weight[0:128,:]
  Rank 1 得到: grad_W[128:256,:]
  ...
最终：  fc1.weight.grad = DTensor(local=(128,1024), Shard(0))   ← 与参数同构
```

### 4.3 优化器状态自动分片

参数以分片 DTensor 存在，`param.data`（即 local tensor）是 `(128, 1024)`。AdamW 的 `exp_avg`、`exp_avg_sq` 创建时自动与参数同形，因此优化器状态**隐式分片**——无需像 ZeRO-1/2 那样显式处理。

---

## 5. 内存生命周期总览

下图展示单次训练迭代中，`fc1.weight` 相关显存随时间的演变（4 GPU、顶层包装、`reshard_after_forward=True`）：

```text
显存
 2.0MB ┤        ┌──┐                ┌──┐
       │        │  │                │  │         ← 完整参数/完整梯度
 0.5MB ┤──┐  ┌──┘  └──┐  ┌──┐  ┌──┘  └──┐  ┌──  ← 分片参数/分片梯度
       │  └──┘       └──┘  └──┘       └──┘
       └─────────────────────────────────────────► 时间
        init  fwd   fwd  bwd   bwd   bwd   optim
              unshard end  unshard rs    end   step
              ↑           ↑      ↑
            all-gather  all-gather reduce-scatter
```

| 时刻 | 参数形态 | 梯度形态 |
|------|----------|----------|
| 初始化后 | 分片 DTensor | 无 |
| 前向 Unshard 后 | 完整 Tensor | 无 |
| 前向 Reshard 后 | 分片 DTensor | 无 |
| 反向 Unshard 后 | 完整 Tensor | 无 |
| 反向 Reduce-Scatter 后 | 分片 DTensor | 分片 DTensor |
| optimizer.step() | 分片 DTensor（已更新） | 分片 DTensor（被清零） |

> 注意：根模块默认 `reshard_after_forward=False`，所以根模块参数在前向后**不释放**，省去反向开始时的一次 all-gather。

---

# 第二部分：源码深度剖析（实现层）

下面结合 PyTorch v2.12.1 源码，定位每个关键函数并给出代码佐证。

## 6. `fully_shard` 入口：从 API 到 MRO 替换

### 6.1 入口函数 `fully_shard`

源码：`_fully_shard.py:97-272`。`@contract(state_cls=FSDPState)` 装饰器会在模块上挂一个 `FSDPState`（可通过 `fully_shard.state(module)` 访问）。

```python
# _fully_shard.py
@contract(state_cls=FSDPState)
def fully_shard(
    module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn=None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params=None,
    dp_mesh_dims=None,
):
    _validate_module(module, "fully_shard")
    mesh = mesh or _init_default_mesh()
    _validate_mesh(mesh, dp_mesh_dims)
    mesh_info = _get_mesh_info(mesh, dp_mesh_dims)          # ① 解析 mesh
    device = _get_device_from_mesh(mesh)
    ...
    arg_module, modules, managed_modules, params, buffers = _get_modules_and_states(
        module, device, ignored_params                      # ② 收集参数、搬到 device
    )
    state = fully_shard.state(modules[0])
    state.init(modules, device, mp_policy, auto_reshard_after_forward)
    _init_param_group(state, params, modules, mesh_info, ...)  # ③ 建参数组
    ...
    _apply_to_module(modules, cls_to_fsdp_cls, FSDPModule, "FSDP", _unimplemented_deepcopy)  # ④ MRO 替换
    return arg_module
```

四步含义：

- **① 解析 mesh**：`_get_mesh_info`（`_fsdp_init.py:87`）把 1D mesh 变成 `FSDPMeshInfo(shard_mesh_dim=0)`，2D mesh 变成 `HSDPMeshInfo(shard_mesh_dim=1, replicate_mesh_dim=0)`。`FSDPMeshInfo` 会缓存 `shard_mesh_size`、`shard_process_group`、`shard_mesh_rank`（`_fsdp_common.py:52`）。
- **② 收集参数**：`_get_modules_and_states`（`_fsdp_init.py:536`）用 DFS 收集 managed modules，再 `_get_managed_states` 去重收集 params/buffers，`_move_states_to_device` 搬到 device。
- **③ 建参数组**：见 6.3。
- **④ MRO 替换**：见 6.2。

### 6.2 MRO 替换：`_apply_to_module`（核心机制）

源码：`_fsdp_init.py:404-430`。FSDP2 不创建 wrapper 模块，而是**直接改 `__class__`**：

```python
# _fsdp_init.py
def _apply_to_module(modules, cls_to_wrapper_cls, wrapper_module_cls,
                     wrapper_cls_prefix, unimplemented_deepcopy):
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_wrapper_cls.get(cls)
        if not new_cls:
            dct = {"__deepcopy__": unimplemented_deepcopy}
            new_cls = type(
                f"{wrapper_cls_prefix}{cls.__name__}",   # 类名 "FSDPSimpleMLP"
                (wrapper_module_cls, cls),              # 继承 (FSDPModule, SimpleMLP)
                dct,
            )
            cls_to_wrapper_cls[cls] = new_cls
        module.__class__ = new_cls                       # 直接改类！
```

改类后 MRO（方法解析顺序）变为：

```text
FSDPSimpleMLP → FSDPModule → SimpleMLP → nn.Module → ...
```

由于 `FSDPModule` 在 `SimpleMLP` **之前**，`FSDPModule` 的方法优先被调用。但 FSDP2 并没有重写 `forward`——它通过 **forward pre/post hook** 来拦截（见 11.1）。`FSDPModule` 只提供 `unshard()`、`reshard()`、`set_...` 等管理 API（`_fully_shard.py:294`）。

`FSDPModule.__new__`（`_fully_shard.py:300`）被重写，用于在容器索引等场景下退回原始类构造：

```python
class FSDPModule:
    _orig_cls_mro_index: int = 2   # MRO 中原始类的位置
    def __new__(cls, *args, **kwargs):
        orig_cls = cls.__mro__[cls._orig_cls_mro_index]   # SimpleMLP
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        if _enable_fsdp_module_new_init:
            self.__init__(*args, **kwargs)
        return self
```

### 6.3 参数组初始化：`_init_param_group`

源码：`_fsdp_init.py:433-533`。无 `shard_placement_fn` 时，所有参数放进**一个** `FSDPParamGroup`；有 `shard_placement_fn` 时，按 process group 分组（支持 MoE 等 per-param mesh 场景）。

```python
# _fsdp_init.py（无 shard_placement_fn 的快路径）
if shard_placement_fn is None:
    state._fsdp_param_groups.append(
        FSDPParamGroup(params, modules, mesh_info,
                       post_forward_mesh_info, device,
                       shard_placement_fn, mp_policy, offload_policy)
    )
    return
```

`FSDPParamGroup.__init__`（`_fsdp_param_group.py:137`）为每个参数创建一个 `FSDPParam`——**这就是 DTensor 转换发生的地方**：

```python
# _fsdp_param_group.py
self.fsdp_params = [
    FSDPParam(param, module_info, mesh_info, post_forward_mesh_info,
              device, shard_placement_fn, mp_policy, offload_policy)
    for param, module_info in zip(params, param_module_infos)
]
```

---

## 7. DTensor 转换核心：`FSDPParam._init_sharded_param`

这是“普通 Tensor → DTensor”的**核心函数**。源码：`_fsdp_param.py:223-328`。

### 7.1 全貌

```python
# _fsdp_param.py
@torch.no_grad()
def _init_sharded_param(self, param, device, shard_placement_fn, mesh_info):
    # 1. 解析分片 placement（默认 Shard(0)）
    if callable(shard_placement_fn):
        shard_result = resolve_shard_placement(shard_placement_fn(param), mesh_info)
        self.mesh_info = shard_result.mesh_info
        fsdp_placement = shard_result.placement
    else:
        self.mesh_info = mesh_info
        fsdp_placement = None
    ...
    if fsdp_placement is None:
        fsdp_placement = Shard(0)              # ← 硬编码默认值
    ...
    self.fsdp_placement = fsdp_placement
    shard_dim = fsdp_placement.dim             # 默认 0

    # 2. 构建 _sharding_spec（DTensorSpec），返回待切分的 param_data
    param_data = self._init_sharding_spec(param, fsdp_placement, shard_dim)

    # 3. 沿 shard_dim 切成 world_size 份，取本 rank 的一份
    shard_world_size = self.mesh_info.shard_mesh_size
    chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
    sharded_param = chunks[self.mesh_info.shard_mesh_rank]

    # 4. 预填充（pre-pad），避免 all-gather 前再 padding
    padded_sharded_size = chunks[0].size()
    padded_sharded_param = param_data.new_zeros(padded_sharded_size)
    if sharded_param.numel() > 0:
        padded_sharded_param.narrow(dim=shard_dim, start=0,
                                    length=sharded_param.size(shard_dim)).copy_(sharded_param)
    # 5. 展平为 1D，作为 all-gather 的输入缓冲
    self._sharded_param_data = padded_sharded_param.view(-1)

    # 6. 包装为 DTensor 并注册回模块
    sharded_param = padded_sharded_param.narrow(dim=shard_dim, start=0, length=length)
    self.sharded_param = nn.Parameter(
        self.to_sharded_dtensor(sharded_param),          # ← DTensor 诞生
        requires_grad=param.requires_grad,
    )
    self._setattr_on_modules(self.sharded_param)         # 注册回 fc1.weight
    self.sharded_state = ShardedState.SHARDED
```

### 7.2 三个关键细节

**(a) `_sharding_spec` 的构建 —— DTensor 的“身份证”**

`_init_sharding_spec`（`_fsdp_param.py:330-502`）根据 mesh 类型生成 `DTensorSpec`，决定了 DTensor 的 placements：

```python
# _fsdp_param.py: _init_sharding_spec_plain（普通 Tensor 路径）
def _init_sharding_spec_plain(self, param, fsdp_placement):
    self._spmd_mesh = self.mesh_info.mesh
    if isinstance(self.mesh_info, HSDPMeshInfo):
        self._spmd_placements = (Replicate(), fsdp_placement)   # (Replicate, Shard(0))
    elif isinstance(self.mesh_info, FSDPMeshInfo):
        self._spmd_placements = (fsdp_placement,)               # (Shard(0),)
    elif isinstance(self.mesh_info, DDPMeshInfo):
        self._spmd_placements = (Replicate(),)                  # (Replicate,)
    self._sharding_spec = DTensorSpec(
        self._spmd_mesh, self._spmd_placements,
        tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
    )
    return param
```

这个 `_sharding_spec` 记录了三件事：**全局 mesh**、**每个 mesh 维度的 placement**（`Shard(0)`/`Replicate()`）、**全局张量元信息**（shape/stride/dtype）。DTensor 本质上 = local_tensor + DTensorSpec。

**(b) 预填充（pre-pad）—— 为了通信效率**

当参数不能被 world_size 整除时（如 510 行切 4 份），最后一份会少。FSDP2 在切分时就**预先把每份 pad 到等长**（`chunks[0].size()` 总是 padded 的），这样 all-gather 出来直接就是等长拼接，无需在通信前临时 padding：

```python
padded_sharded_size = chunks[0].size()                # 0 号总是 padded 的
padded_sharded_param = param_data.new_zeros(padded_sharded_size)
padded_sharded_param.narrow(...).copy_(sharded_param) # 真实数据拷到前部，尾部补 0
```

**(c) `to_sharded_dtensor` —— 真正的 DTensor 构造**

```python
# _fsdp_param.py:696
def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
    return _from_local_no_grad(tensor, self._sharding_spec)
```

`_from_local_no_grad`（`_fsdp_common.py:132`）就是 `DTensor(local_tensor, spec, requires_grad=...)` 的轻量非可分版本——把 local 分片张量 + `_sharding_spec` 组装成 DTensor。

### 7.3 转换前后对照

```python
# 转换前（fully_shard 调用前）
fc1.weight: Parameter(shape=(512,1024), dtype=float32)   # 每卡都有完整副本

# 转换后（_init_sharded_param 完成后）
fc1.weight: Parameter(DTensor(
    _local_tensor: shape=(128,1024),                      # 本卡 1/4
    _spec: DTensorSpec(
        mesh=DeviceMesh((4,)),
        placements=(Shard(0),),
        tensor_meta=TensorMeta((512,1024), ..., float32), # 全局 shape
    ),
))
# 同时，FSDPParam 内部还保留：
#   _sharded_param_data: 1D Tensor(128*1024,)   ← all-gather 的输入缓冲
#   _orig_size: (512, 1024)                      ← 全局形状，用于 unshard 还原
```

---

## 8. Unshard 源码：all-gather 的 copy-in / 通信 / copy-out

Unshard 分两阶段：`unshard()` 发起通信，`wait_for_unshard()` 拷出结果并注册完整参数。源码：`_fsdp_param_group.py:337-454`。

### 8.1 `unshard()`：发起 all-gather

```python
# _fsdp_param_group.py:337
def unshard(self, async_op: bool = False):
    ...
    with record_function(self._with_fqn("FSDP::all_gather")):
        self._all_gather_result = foreach_all_gather(
            self.fsdp_params,
            self._all_gather_process_group,
            async_op,
            *self.comm_ctx.get_all_gather_streams(async_op, self._training_state),
            self.device,
            self._all_gather_comm,
        )
```

### 8.2 `foreach_all_gather`：copy-in + NCCL all-gather

源码：`_fsdp_collectives.py:324-378`。三步走：

```python
# _fsdp_collectives.py
@torch.no_grad()
def foreach_all_gather(fsdp_params, group, async_op,
                       all_gather_copy_in_stream, all_gather_stream,
                       device, all_gather_comm):
    world_size, rank = group.size(), group.rank()
    with device_handle.stream(all_gather_copy_in_stream):
        # 1. 取每个参数的 all-gather 输入（即 _sharded_param_data，1D）
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
        all_gather_input_numel = sum(t.numel() for t in all_gather_inputs)
        # 2. 分配 all-gather 输出缓冲（本 rank 的输入大小 × world_size）
        all_gather_output = all_gather_comm.allocate(
            (all_gather_input_numel * world_size,), dtype=dtype, device=device)
        # 3. copy-in：把本 rank 的输入拷到输出缓冲的对应 rank 切片
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs, all_gather_output, inp_split_sizes,
            all_gather_input_numel, rank)
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        # 4. 真正的 NCCL all-gather（in-place 风格）
        all_gather_work = all_gather_comm(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group, async_op=async_op)
        ...
```

底层通信类 `DefaultAllGather`（`_fsdp_collectives.py:111`）直接调用 `dist.all_gather_into_tensor`：

```python
class DefaultAllGather(DefaultAllocMixin, AllGather):
    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        return dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=async_op)
```

> **设计要点**：一个 `FSDPParamGroup` 的所有参数被**拼成一个** 1D 缓冲做**一次** all-gather，而不是每参数一次——这是 FSDP2 通信效率的关键（“grouping is first class”）。

### 8.3 `foreach_all_gather_copy_out` + `init_unsharded_param`：拷出并还原 ND

源码：`_fsdp_collectives.py:430-521` 与 `_fsdp_param.py:572-620`。

```python
# _fsdp_param.py:572  init_unsharded_param
def init_unsharded_param(self):
    ...
    # 默认路径：all-gather 输出就是完整参数数据
    unsharded_tensor = self.all_gather_outputs[0]
    # 用 as_strided 把 1D 缓冲还原成 ND 形状（_orig_size）
    unsharded_param = torch.as_strided(
        unsharded_tensor, self._orig_size,
        self._contiguous_orig_stride, storage_offset=0)
    # 若是 DTensor 参数（SPMD/TP），再包一层 DTensor
    if self._unsharded_dtensor_spec is not None:
        unsharded_param = _from_local_no_grad(unsharded_param, self._unsharded_dtensor_spec)
    self._unsharded_param = nn.Parameter(
        unsharded_param, requires_grad=self.sharded_param.requires_grad)
```

> 注意 `as_strided` 是**视图**操作——`_unsharded_param` 与 all-gather 输出共享 storage。这正是后续能用 storage resize 释放内存的前提。

### 8.4 `to_unsharded`：注册完整参数回模块

源码：`_fsdp_param.py:674-685`。

```python
# _fsdp_param.py:674
def to_unsharded(self) -> None:
    set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
    self._setattr_on_modules(self._unsharded_param)   # fc1.weight = 完整参数
    ...
    self.sharded_state = ShardedState.UNSHARDED
```

`_setattr_on_modules`（`_fsdp_param.py:687`）通过 `unsafe_setattr_param` 直接改 `module._parameters[name]`，绕过 `nn.Module.__setattr__` 的开销。此后 `fc1.weight` 就是完整参数，`nn.Linear` 的 forward 可以正常计算。

非 dim=0 分片时，`foreach_all_gather_copy_out` 还会做额外的 chunk+cat 重排（`_fsdp_collectives.py:495-520`），把沿 dim=0 拼接的结果转成沿目标 dim 拼接。

---

## 9. Reshard 源码：storage 缩放释放内存

前向结束后，`post_forward` → `reshard`（`_fsdp_param_group.py:463`）→ `_to_sharded`（`_fsdp_param_group.py:713`）→ 每个 param 调 `to_sharded`（`_fsdp_param.py:630`）：

```python
# _fsdp_param.py:630
def to_sharded(self) -> None:
    self._setattr_on_modules(self.sharded_param)   # 重新注册分片 DTensor
    self.free_unsharded_param()                    # 释放完整参数内存
    self.sharded_state = ShardedState.SHARDED

def free_unsharded_param(self) -> None:            # _fsdp_param.py:753
    for tensor in itertools.chain(
        self.all_gather_outputs, self._unsharded_inner_tensors
    ):
        free_storage(tensor)
```

`free_storage` / `alloc_storage`（`_fsdp_param.py:988-996`）是内存管理的精髓——**不删对象，只缩 storage**：

```python
# _fsdp_param.py:988
def alloc_storage(tensor: torch.Tensor) -> None:
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)

def free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)        # ← storage 缩到 0，显存立即归还
```

为什么这样安全？源码注释（`_fsdp_param.py:71 [Note: FSDP and autograd]`）解释：autograd 可能在反向时引用 `_unsharded_param` 或它的视图。用 storage resize（而非销毁对象）释放内存，能**保留别名关系**——下次 all-gather 时 `alloc_storage` 把 storage 扩回来，原对象和所有视图自动指向新数据。优化器状态通过 tensor 对象关联，storage resize 不影响它。

---

## 10. Reduce-Scatter 源码：梯度归约与 DTensor 写回

反向 post-backward 钩子 `post_backward`（`_fsdp_param_group.py:517`）收集完整梯度，调用 `foreach_reduce`（`_fsdp_collectives.py:523`）做 reduce-scatter 并把分片梯度写回 `sharded_param.grad`。

### 10.1 `post_backward`：收集梯度、reshard、发起 reduce

```python
# _fsdp_param_group.py:517
def post_backward(self, *unused):
    ...
    # 1. 收集每个参数的完整梯度（unsharded_grad_data）
    fsdp_params_with_grad, unsharded_grads = [], []
    for fsdp_param in self.fsdp_params:
        ...
        if fsdp_param.unsharded_param.grad is not None:
            fsdp_params_with_grad.append(fsdp_param)
            unsharded_grads.append(fsdp_param.unsharded_grad_data)   # 完整 (512,1024)
            fsdp_param.unsharded_param.grad = None                   # 取走梯度引用
    if self.reshard_after_backward:
        self.reshard()                                              # 2. 释放完整参数
    ...
    # 3. 发起 reduce-scatter（HSDP 还含 all-reduce）
    (reduce_scatter_input, reduce_scatter_event, self._post_reduce_event,
     all_reduce_input, all_reduce_event, self._partial_reduce_output
    ) = foreach_reduce(fsdp_params_with_grad, unsharded_grads, ...)
```

### 10.2 `foreach_reduce`：copy-in + NCCL reduce-scatter + 写回

源码：`_fsdp_collectives.py:523-747`。核心步骤：

```python
# _fsdp_collectives.py:523  foreach_reduce
# 非 dim-0 分片：先把梯度沿 shard_dim 的分块重排成沿 dim-0（便于 reduce-scatter）
for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
    if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
        continue
    chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
    unsharded_grads[i] = torch.cat(chunks, dim=0)

# 计算各梯度 padded 后的大小，分配 reduce-scatter 输入缓冲
padded_unsharded_sizes = tuple(_get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads)
reduce_scatter_input = reduce_scatter_comm.allocate((reduce_scatter_input_numel,), ...)

# copy-in：把所有梯度按 world_size 份 chunk-cat 进输入缓冲（每 rank 一行）
foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
#  ↓ 即 torch.ops.fsdp.chunk_cat(unsharded_grads, dim=0, num_chunks=world_size, out=...)
unsharded_grads.clear()                         # 梯度引用清空，释放内存

with device_handle.stream(reduce_scatter_stream):
    reduce_output = reduce_scatter_comm.allocate((reduce_scatter_output_numel,), ...)
    _div_if_needed(reduce_scatter_input, predivide_factor)
    # 真正的 NCCL reduce-scatter
    reduce_scatter_comm(output_tensor=reduce_output,
                        input_tensor=reduce_scatter_input,
                        group=reduce_scatter_group, op=reduce_scatter_op)
    # HSDP：再跨复制组 all-reduce
    if all_reduce_group is not None:
        dist.all_reduce(reduce_output, group=all_reduce_group, op=all_reduce_op)
    _div_if_needed(reduce_output, postdivide_factor)
    reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
```

底层 `DefaultReduceScatter`（`_fsdp_collectives.py:175`）调用 `dist.reduce_scatter_tensor`。

### 10.3 梯度写回：DTensor 形态

reduce-scatter 输出 `reduce_output` 是本 rank 拿到的**分片梯度**（1D）。源码把它 view 成各参数的 `sharded_size` 并包装成 DTensor 写回 `sharded_param.grad`：

```python
# _fsdp_collectives.py:688  view out and accumulate sharded gradients
flat_grad_offset = 0
for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params):
    new_sharded_grad = torch.as_strided(
        reduce_output,
        size=fsdp_param.sharded_size,                 # (128, 1024)
        stride=fsdp_param.contiguous_sharded_stride,
        storage_offset=flat_grad_offset,
    )
    if to_accumulate_grad:
        fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
    else:
        new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
        fsdp_param.sharded_param.grad = new_sharded_dtensor_grad   # ← 梯度变 DTensor
    flat_grad_offset += padded_sharded_numel
```

注意 `to_sharded_dtensor` 用的仍是参数的 `_sharding_spec`（`_fsdp_param.py:696`），所以 **`sharded_param.grad` 与 `sharded_param` 共享同一 placements**——梯度分片维度 = 参数分片维度，optimizer.step() 直接可用。

### 10.4 梯度内层张量提取（DTensor 参数）

对于本身就是 DTensor 的参数（SPMD mesh / TP），autograd 算出的梯度是 DTensor。`unsharded_grad_data` 通过 `_get_grad_inner_tensor`（`_fsdp_param.py:856`）做 redistribute 后取 local tensor：

```python
# _fsdp_param.py:856  _get_grad_inner_tensor
def _get_grad_inner_tensor(self, grad):
    if self.is_dtensor:
        ...
        if self.mesh_info.is_spmd_mesh:
            # DP 维度保持 Partial（交给 FSDP 的 reduce-scatter 处理），其余维度 redistribute
            target_placements = tuple(
                grad.placements[i] if i in self._dp_dim_indices else placements[i]
                for i in range(len(placements)))
            if target_placements != grad.placements:
                grad = grad.redistribute(placements=target_placements)
        else:
            if placements != grad.placements:
                grad = grad.redistribute(placements=placements)
        grad = grad._local_tensor
    return grad
```

---

## 11. Hook 编排：前向 / 反向如何被触发

FSDP2 的 unshard/reshard/reduce-scatter 都靠 `nn.Module` 的 hook 触发。注册在 `FSDPState.init`（`_fsdp_state.py:101`）。

### 11.1 前向 hook

```python
# _fsdp_state.py:115
self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
    self._pre_forward, prepend=True, with_kwargs=True)
self._post_forward_hook_handle = modules[0].register_forward_hook(
    self._post_forward, prepend=False)
```

- `_pre_forward`（`_fsdp_state.py:271`）→ `fsdp_param_group.pre_forward`（`_fsdp_param_group.py:475`）：设 `FORWARD` 状态 → `unshard` + `wait_for_unshard` → 注册 post-backward hook。
- `_post_forward`（`_fsdp_state.py:306`）→ `fsdp_param_group.post_forward`（`_fsdp_param_group.py:486`）：`reshard`（若需要）→ 记录 post-forward 顺序（供反向预取）。

### 11.2 反向 hook

反向的 unshard 和 reduce-scatter 分别用两种机制：

**(a) pre-backward（Unshard）—— 用 tensor hook**

`_post_forward` 调 `_register_pre_backward_hook`（`_fsdp_state.py:391`），对输出张量注册 `t.register_hook(self._pre_backward)`：

```python
# _fsdp_state.py:333
def _pre_backward(self, grad):
    self._training_state = TrainingState.PRE_BACKWARD
    self._register_root_post_backward_final_callback()
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.pre_backward(default_prefetch)   # 内部调 unshard
    return grad
```

**(b) post-backward（Reduce-Scatter）—— 用 autograd.Function**

`pre_forward` 中调 `_register_post_backward_hook`（`_fsdp_param_group.py:753`），把 `RegisterPostBackwardFunction.apply(self, *inp_tensors)` 插入计算图：

```python
# _fsdp_param_group.py:912
class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group, *inputs):
        ctx.param_group = param_group
        return inputs
    @staticmethod
    def backward(ctx, *grads):
        ctx.param_group.post_backward()      # ← reduce-scatter 在此触发
        ...
```

autograd 在该节点反向时自动调用 `backward`，从而触发 `post_backward`。对于 forward 输入不需要梯度的边缘情况，`_root_post_backward_final_callback`（`_fsdp_state.py:348`，通过 `Variable._execution_engine.queue_callback` 注册到反向末尾）兜底调用未触发的 `post_backward`。

### 11.3 三态机：`ShardedState` 与 `TrainingState`

参数状态（`_fsdp_param.py:111`）：

```python
class ShardedState(Enum):
    SHARDED = auto()                # 分片参数已注册（唯一占用内存）
    SHARDED_POST_FORWARD = auto()   # 前向后 reshard 到更小 world size（HSDP）
    UNSHARDED = auto()              # 完整参数已注册（分片+完整都占内存）
```

转换：`SHARDED ──unshard──→ UNSHARDED ──reshard──→ SHARDED`。`to_sharded`/`to_unsharded`/`to_sharded_post_forward`（`_fsdp_param.py:630/674/635`）通过 `_setattr_on_modules` 切换注册的参数对象。

训练状态（`_fsdp_common.py:80`）：`IDLE / FORWARD / PRE_BACKWARD / POST_BACKWARD`，控制通信流选择与预取逻辑（`_fsdp_param_group.py:102 get_all_gather_streams`）。

---

# 附录：关键数据结构与速查表

## A.1 FSDP 张量术语（`_fsdp_param.py:44 [Note: FSDP tensors]`）

| 术语 | 含义 |
|------|------|
| Original parameter | 传入 `FSDPParam` 的原始参数 |
| Sharded parameter | 原参数沿 dim-0（或自定义 dim）切分后的 **DTensor** |
| All-gather inputs | 从 sharded param 派生、送入 all-gather 的张量（即 `_sharded_param_data`，1D）|
| All-gather output | all-gather 结果（1D，拼接所有 rank 的输入）|
| Unsharded parameter | 从 all-gather 输出派生、用于前向/反向计算的完整参数；autograd leaf |

## A.2 关键源码定位速查

| 机制 | 函数 | 文件:行 |
|------|------|---------|
| `fully_shard` 入口 | `fully_shard` | `_fully_shard.py:97` |
| MRO 替换 | `_apply_to_module` | `_fsdp_init.py:404` |
| mesh 解析 | `_get_mesh_info` | `_fsdp_init.py:87` |
| 参数组创建 | `_init_param_group` | `_fsdp_init.py:433` |
| **DTensor 转换** | `FSDPParam._init_sharded_param` | `_fsdp_param.py:223` |
| sharding spec 构建 | `_init_sharding_spec` / `_init_sharding_spec_plain` | `_fsdp_param.py:330` / `:484` |
| DTensor 构造 | `to_sharded_dtensor` / `_from_local_no_grad` | `_fsdp_param.py:696` / `_fsdp_common.py:132` |
| 默认 Shard(0) | `_init_sharded_param` 中 `fsdp_placement = Shard(0)` | `_fsdp_param.py:251` |
| Unshard（发起） | `FSDPParamGroup.unshard` | `_fsdp_param_group.py:337` |
| Unshard（拷出） | `wait_for_unshard` / `foreach_all_gather_copy_out` | `_fsdp_param_group.py:381` / `_fsdp_collectives.py:430` |
| all-gather 通信 | `foreach_all_gather` / `DefaultAllGather` | `_fsdp_collectives.py:324` / `:111` |
| 完整参数还原 | `init_unsharded_param` / `to_unsharded` | `_fsdp_param.py:572` / `:674` |
| Reshard（释放） | `to_sharded` / `free_storage` | `_fsdp_param.py:630` / `:994` |
| 内存管理 | `alloc_storage` / `free_storage` | `_fsdp_param.py:988` |
| Reduce-Scatter | `post_backward` / `foreach_reduce` | `_fsdp_param_group.py:517` / `_fsdp_collectives.py:523` |
| reduce-scatter 通信 | `DefaultReduceScatter` / `dist.reduce_scatter_tensor` | `_fsdp_collectives.py:175` |
| 梯度写回 DTensor | `to_sharded_dtensor(new_sharded_grad)` | `_fsdp_collectives.py:724` |
| 梯度内层提取 | `_get_grad_inner_tensor` | `_fsdp_param.py:856` |
| 前向 hook 注册 | `FSDPState.init` | `_fsdp_state.py:115` |
| pre-forward | `_pre_forward` / `FSDPParamGroup.pre_forward` | `_fsdp_state.py:271` / `_fsdp_param_group.py:475` |
| post-forward | `_post_forward` / `post_forward` | `_fsdp_state.py:306` / `_fsdp_param_group.py:486` |
| pre-backward（unshard） | `_pre_backward` / `pre_backward` | `_fsdp_state.py:333` / `_fsdp_param_group.py:504` |
| post-backward（reduce-scatter） | `RegisterPostBackwardFunction.backward` → `post_backward` | `_fsdp_param_group.py:921` / `:517` |
| 反向兜底回调 | `_root_post_backward_final_callback` | `_fsdp_state.py:348` |
| 非法 deepcopy 拦截 | `_unimplemented_deepcopy` | `_fully_shard.py:275` |

## A.3 一次迭代调用链总览

```text
forward(x)
 └─ _pre_forward hook                         [_fsdp_state.py:271]
     └─ _root_pre_forward → _lazy_init        (首次：识别 root、lazy_init 各 group)
     └─ FSDPParamGroup.pre_forward            [_fsdp_param_group.py:475]
         └─ unshard → foreach_all_gather      (all_gather_copy_in → dist.all_gather_into_tensor)
         └─ wait_for_unshard
             └─ foreach_all_gather_copy_out   (split_with_sizes_copy)
             └─ init_unsharded_param          (as_strided 还原 ND，可选 _from_local_no_grad)
             └─ to_unsharded                  (_setattr_on_modules 注册完整参数)
         └─ _register_post_backward_hook      (RegisterPostBackwardFunction.apply)
 └─ 原始 forward (nn.Linear 计算)             ← 用完整参数
 └─ _post_forward hook                        [_fsdp_state.py:306]
     └─ FSDPParamGroup.post_forward           [_fsdp_param_group.py:486]
         └─ reshard → to_sharded              (_setattr_on_modules 注册分片 + free_storage)
     └─ _register_pre_backward_hook           (t.register_hook(_pre_backward))

loss.backward()
 └─ _pre_backward (每个输出张量梯度到来时)     [_fsdp_state.py:333]
     └─ FSDPParamGroup.pre_backward           [_fsdp_param_group.py:504]
         └─ unshard + wait_for_unshard        (再次 all-gather 完整参数)
 └─ autograd 计算完整梯度
 └─ RegisterPostBackwardFunction.backward     [_fsdp_param_group.py:921]
     └─ FSDPParamGroup.post_backward          [_fsdp_param_group.py:517]
         └─ 收集 unsharded_grads → reshard
         └─ foreach_reduce                    (chunk_cat copy-in → dist.reduce_scatter_tensor
                                               → [HSDP: dist.all_reduce] → 写回 sharded_param.grad 为 DTensor)
 └─ _root_post_backward_final_callback        [_fsdp_state.py:348] (兜底 + finalize_backward)

optimizer.step()
 └─ 对分片 DTensor 参数的 local tensor 更新（优化器状态自动分片）
```

---

> **总结**：FSDP2 的 `fully_shard` 以 DTensor 为分片载体，通过 MRO 替换就地包装模块、用 forward/backward hook 编排通信。初始化时 `_init_sharded_param` 把普通 Tensor 沿 dim-0 切分并包装成 `Shard(0)` 的 DTensor；前向用 `all_gather_into_tensor` 临时聚合、`as_strided` 还原，靠 `storage resize` 实现零拷贝释放；反向用 `reduce_scatter_tensor` 把完整梯度归约分片，写回与参数同构的 DTensor 梯度。整套机制把“省显存”的代价压缩到通信上，而参数/梯度/优化器状态的分片全部由 DTensor 这一层统一表达。
