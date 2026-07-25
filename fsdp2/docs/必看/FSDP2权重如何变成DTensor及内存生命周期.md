# FSDP2：权重如何变成 DTensor 及内存生命周期

> 源码位置：`torch/distributed/fsdp/_fully_shard/_fsdp_param.py`
> 关键辅助：`torch/distributed/fsdp/_fully_shard/_fsdp_common.py`、`torch/distributed/tensor/_dtensor_spec.py`

---

## 一、总览

源代码注释（`_fsdp_param.py:44-79`）定义了 5 个关键张量概念，对应 5 个时间段：

| 概念 | 含义 |
| --- | --- |
| **Original parameter** | 传给 `FSDPParam` 的原始参数（应用 FSDP 时模块上的那个） |
| **Sharded parameter** | 在主 mesh 上按 dim-0（或指定 dim）分片后的 DTensor |
| **All-gather inputs** | 传给 all-gather 的张量，由 sharded parameter 派生 |
| **All-gather output** | all-gather 得到的张量 |
| **Unsharded parameter** | 前向/反向计算用的参数，由 all-gather output 派生，是 autograd leaf |

框架关系：

```text
all-gather-inputs  = pre-all-gather-transform(sharded-parameter)
unsharded-parameter = post-all-gather-transform(all-gather-outputs)
```

对默认 `torch.Tensor` 路径：

- all-gather 输入与 sharded parameter **共享同一底层数据**；
- all-gather 输出与 unsharded parameter **共享同一底层数据**。

本文以最常用的 **普通张量路径**（plain tensor，即 `param` 不是 `DTensor`，使用 1D FSDP mesh）为主线说明，并补充 SPMD/TP 路径的差异。

---

## 二、DTensor 是怎么"变"出来的——核心机制

一个普通权重变成 DTensor 的本质是 **`_from_local_no_grad`**（`_fsdp_common.py:132`）：

```python
def _from_local_no_grad(local_tensor, sharding_spec):
    return DTensor(
        local_tensor,
        sharding_spec,
        requires_grad=local_tensor.requires_grad,
    )
```

它 **不复制、不分配数据**，只是把已有的 local tensor 包一层 DTensor 外壳，并附加一个 `DTensorSpec`（记录 mesh、placements、全局 tensor_meta）。DTensor 的 `_local_tensor` 直接指向原 local tensor 的存储。

具体到 FSDP：

1. 原始 `param`（完整）→ chunk 取本 rank 一块 → pad → 得到 `_sharded_param_data`。
2. `_sharded_param_data` 的 narrow view → 套上 `_sharding_spec`（`Shard(0)` on FSDP mesh）→ 变成 **sharded DTensor Parameter**。
3. 这个 DTensor 的 `tensor_meta` 记录的是 **全局原始形状**，而 `_local_tensor` 只是本 rank 的一块——这就是"分片"在数据结构上的体现。

对于 SPMD/TP 路径，参数本身已是 DTensor，FSDP 做的是 **修改 spec 的 placements**（把 DP 维的 `Replicate` 换成 `Shard`/`_StridedShard`），local tensor 数据不变。

---

## 三、内存生命周期详解（按时间段）

### T0：原始参数（Original Parameter）

**入口**：`FSDPParam.__init__`（`_fsdp_param.py:188`）被调用前。

用户模块上挂着一个完整的 `nn.Parameter`，形状如 `[out_features, in_features]`，在某个 GPU 上，连续内存。这是唯一的张量，占据 `numel * itemsize` 字节。

```python
module.weight = nn.Parameter  # 完整权重，shape=[O, I]
```

---

### T1：初始化分片参数（`_init_sharded_param`）

**入口**：`__init__` → `_init_sharded_param`（`_fsdp_param.py:223`）

这是 **权重变成 DTensor 的核心步骤**，具体子步骤如下：

#### 步骤 1a：确定 placement（`_fsdp_param.py:250-258`）

```python
if fsdp_placement is None:
    fsdp_placement = Shard(0)   # 默认按 dim-0 分片
self.fsdp_placement = fsdp_placement
shard_dim = fsdp_placement.dim  # 通常为 0
```

#### 步骤 1b：构建 sharding spec（`_init_sharding_spec`，`_fsdp_param.py:330`）

根据 `param` 是否为 `DTensor`，走三条不同路径：

**路径 A — plain tensor**（`_init_sharding_spec_plain`，`_fsdp_param.py:484`）：

```python
self._spmd_mesh = self.mesh_info.mesh          # DeviceMesh (1D)
self._spmd_placements = (fsdp_placement,)       # (Shard(0),)
self._sharding_spec = DTensorSpec(
    self._spmd_mesh,
    self._spmd_placements,
    tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),  # 全局形状
)
return param  # 返回原始 param 作为待分片数据
```

**路径 B — SPMD DTensor**（`_init_sharding_spec_spmd`，`_fsdp_param.py:353`）：
参数本身已是 DTensor（在完整 SPMD mesh 上，DP 维 Replicate）。保留原 `_unsharded_dtensor_spec`，把 DP 维的 `Replicate` 改成 `Shard/_StridedShard`，返回 `param._local_tensor`。

**路径 C — TP DTensor**（`_init_sharding_spec_tp`，`_fsdp_param.py:428`）：
参数是 TP 分片的 DTensor，DP mesh 与 TP mesh 拼接，构造 `_StridedShard` 以表达多 mesh 维共切同一 tensor 维。

> **关键点**：`_sharding_spec` 是一个 `DTensorSpec`，它 **不持有数据**，只记录 mesh、placements、全局 tensor_meta。它是后续把 local tensor 包装成 DTensor 的"配方"。

#### 步骤 1c：chunk 切分（`_fsdp_param.py:294-298`）

```python
chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
sharded_param = chunks[shard_rank]   # 取本 rank 对应的那一块
```

`_chunk_with_empty`（`_fsdp_common.py:113`）用 `torch.chunk` 切分，不够的补空张量（处理不均分）：

```python
def _chunk_with_empty(tensor, num_chunks, dim):
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks
```

#### 步骤 1d：pre-padding（`_fsdp_param.py:300-312`）

```python
padded_sharded_size = chunks[0].size()          # 0 号块总是 padded（最大尺寸）
padded_sharded_param = param_data.new_zeros(padded_sharded_size)
padded_sharded_param.narrow(...).copy_(sharded_param)  # 把本 rank 真实数据拷进前段
self._sharded_param_data = padded_sharded_param.view(-1)  # 1D 视图，作为 all-gather 输入
```

> **为什么要 pad？** 这样 all-gather 时每个 rank 输入等长，免去了 gather 前后的 pad/unpad 操作。padding 部分在 all-gather 后被丢弃。

#### 步骤 1e：包装成 DTensor（`_fsdp_param.py:314-324`）

```python
sharded_param = padded_sharded_param.narrow(dim=shard_dim, start=0, length=length)
self.sharded_param = nn.Parameter(
    self.to_sharded_dtensor(sharded_param),   # ← 关键：local tensor -> DTensor
    requires_grad=param.requires_grad,
)
```

其中 `to_sharded_dtensor`（`_fsdp_param.py:696`）调用 `_from_local_no_grad`（`_fsdp_common.py:132`）：

```python
return DTensor(local_tensor, sharding_spec, requires_grad=local_tensor.requires_grad)
```

**直接用 local tensor 作为 DTensor 的 `_local_tensor`，不复制数据**。

#### 步骤 1f：替换模块上的参数（`_fsdp_param.py:327-328`）

```python
self._setattr_on_modules(self.sharded_param)   # module.weight = sharded DTensor Parameter
self.sharded_state = ShardedState.SHARDED
```

#### T1 结束后内存中存在

| 对象 | 类型 | 内容 | 说明 |
| --- | --- | --- | --- |
| `param`（原参数） | `nn.Parameter` | 完整权重 | 引用计数归零后会被 GC 释放 |
| `self._sharded_param_data` | `torch.Tensor` (1D) | padded 分片数据 | all-gather 的实际输入，**持有真实显存** |
| `self.sharded_param` | `nn.Parameter(DTensor)` | DTensor，`_local_tensor` 是 `_sharded_param_data` 的 narrow view | 注册到 module 上 |
| `self._sharding_spec` | `DTensorSpec` | mesh + placements + 全局 tensor_meta | DTensor 的元数据"配方" |
| `module.weight` | 同 `self.sharded_param` | | 指向同一个 DTensor |

> 原 `param` 在 `fully_shard()` 调用返回后自然释放（注释 `_fsdp_param.py:325-326`）。

---

### T2：SHARDED 状态（前向触发前）

**状态**：`sharded_state = ShardedState.SHARDED`

此时 module 上挂的是 **sharded DTensor Parameter**，只占 `padded_sharded_size` 的显存。如果开了 CPU offload，`_sharded_param_data` 在 CPU（可 pin memory）。

**内存中**：

- `self._sharded_param_data`：1D 显存（或 CPU），padded 分片。
- `self.sharded_param`：DTensor，`_local_tensor` 是上面的 narrow view。
- `_unsharded_param`：**不存在**（尚未创建）。
- `all_gather_outputs`：`[]`（空 list）。

---

### T3：unshard（前向前，all-gather）

前向触发时 FSDP 调用 `to_unsharded`，但数据准备分几步：

#### 步骤 3a：构造 all-gather 输入（`all_gather_inputs` 属性，`_fsdp_param.py:759`）

```python
sharded_param_data = self._sharded_param_data  # 1D
# 若 CPU offload: .to(device, non_blocking=True)
# 若 mixed precision: _to_dtype_if_needed(..., self.param_dtype)
return [sharded_param_data]
```

plain 路径下输入与 sharded param 共享存储（注释 `_fsdp_param.py:63-65`）。

#### 步骤 3b：分配 all-gather 输出（`init_all_gather_outputs`，`_fsdp_param.py:557`）

```python
self.all_gather_outputs = [
    torch.empty([numel * world_size], dtype=dtype, device=device)
]
```

此时 **分配了 full size 的显存**（全局大小），但未填数据。

#### 步骤 3c：执行 all-gather（在 FSDPParamGroup 中，本文件外）

NCCL all-gather 把各 rank 的 `_sharded_param_data` 拼进 `all_gather_outputs[0]`。

#### 步骤 3d：构造 unsharded param（`init_unsharded_param`，`_fsdp_param.py:572`）

首次调用（无 `_unsharded_param` 属性）走 `_fsdp_param.py:600-620`：

```python
unsharded_tensor = self.all_gather_outputs[0]  # 1D 全量
unsharded_param = torch.as_strided(
    unsharded_tensor, self._orig_size, self._contiguous_orig_stride, storage_offset=0
)  # reshape 成原始 ND 形状，共享存储
if self._unsharded_dtensor_spec is not None:   # 仅 DTensor 参数(SPMD/TP)走这里
    unsharded_param = _from_local_no_grad(unsharded_param, self._unsharded_dtensor_spec)
self._unsharded_param = nn.Parameter(unsharded_param, requires_grad=...)
```

> 注意：plain tensor 路径下 `_unsharded_dtensor_spec is None`，所以 **unsharded param 是普通 Tensor Parameter，不是 DTensor**。只有 SPMD/TP 路径才会把 unsharded 也包成 DTensor（用原始 param 的 spec）。

#### 步骤 3e：注册到 module（`to_unsharded`，`_fsdp_param.py:674`）

```python
self._setattr_on_modules(self._unsharded_param)
self.sharded_state = ShardedState.UNSHARDED
```

#### T3 结束后内存中（plain 路径）

| 对象 | 内容 | 显存 |
| --- | --- | --- |
| `self._sharded_param_data` | padded 分片 (1D) | 分片大小 |
| `self.sharded_param` | sharded DTensor Parameter | （view，不额外占） |
| `self.all_gather_outputs[0]` | 全量 1D | **全局大小** |
| `self._unsharded_param` | as_strided view → all_gather_outputs | （view，不额外占） |
| `module.weight` | = `_unsharded_param`（普通 Tensor Parameter） | |

> sharded 与 unsharded **同时占显存**（见 `ShardedState` 注释 `_fsdp_param.py:120-121`）。

---

### T4：UNSHARDED 状态（前向/反向计算中）

此时 module 上是完整的 unsharded Parameter，供 autograd 前向/反向计算。梯度计算在 unsharded param 上。

反向后，梯度需要 reduce-scatter 回 sharded 形态（在 `unsharded_grad_data` / `_get_grad_inner_tensor`，`_fsdp_param.py:856` 处理 DTensor 梯度的 redistribute）。

---

### T5：reshard（前向后 / 反向后回到 SHARDED）

调用 `to_sharded`（`_fsdp_param.py:630`）：

```python
def to_sharded(self):
    self._setattr_on_modules(self.sharded_param)  # module.weight 换回 sharded DTensor
    self.free_unsharded_param()                    # 释放 all_gather_outputs 显存
    self.sharded_state = ShardedState.SHARDED
```

`free_unsharded_param`（`_fsdp_param.py:753`）用 **storage resize 到 0** 释放 `all_gather_outputs` 的显存：

```python
def free_unsharded_param(self):
    for tensor in itertools.chain(
        self.all_gather_outputs, self._unsharded_inner_tensors
    ):
        free_storage(tensor)
```

```python
def free_storage(tensor):
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)
```

> 注释 `_fsdp_param.py:71-78` 解释：用 storage resize 而非 del，是为了保留 autograd 可能持有的 aliasing。unsharded param 对象构造一次后复用，后续只 in-place 写入。

#### T5 结束后内存中

回到 T2 状态——只剩 sharded param。

---

### 可选 T4'：SHARDED_POST_FORWARD（post-forward reshard）

若配置了 `post_forward_mesh_info`（如 HSDP 把 forward 后的 unsharded 重分到更小 world size），前向后调用 `to_sharded_post_forward`（`_fsdp_param.py:635`）：

- 从 `all_gather_outputs[0]` narrow 出本 rank 的一块并 `clone()`（`_fsdp_param.py:655-659`）。
- 包成 post-forward DTensor（placement `(Replicate(), Shard(0))`）。
- 释放 unsharded param。
- 此时 `_sharded_post_forward_param_data` 和 `sharded_param` **同时占显存**（见注释 `_fsdp_param.py:116-119`）。

---

## 四、ShardedState 状态机

`ShardedState` 枚举（`_fsdp_param.py:111-126`）定义了三种状态：

```text
          to_unsharded
   SHARDED ────────────────► UNSHARDED
      ▲                        │
      │ to_sharded             │ to_sharded_post_forward
      │                        ▼
      └──────────── SHARDED_POST_FORWARD
```

| 状态 | 含义 | 占显存 |
| --- | --- | --- |
| `SHARDED` | sharded param 注册到 module，唯一贡献者 | 仅 sharded param |
| `SHARDED_POST_FORWARD` | unsharded 重分到更小 world size，不注册到 module | sharded param + post-forward param |
| `UNSHARDED` | unsharded param 注册到 module | sharded param + unsharded param |

---

## 五、内存关键时间线速查表

| 时间 | module.weight 类型 | 占显存的对象 | 状态 |
| --- | --- | --- | --- |
| T0 调用前 | 完整 `nn.Parameter` | `param` | - |
| T1 `fully_shard` 中 | sharded `DTensor` Parameter | `_sharded_param_data`(padded 分片) + 原 param(待释放) | SHARDED |
| T2 训练空闲 | sharded `DTensor` Parameter | `_sharded_param_data` | SHARDED |
| T3 all-gather 后 | unsharded `Parameter`(plain) 或 `DTensor`(SPMD) | `_sharded_param_data` + `all_gather_outputs`(全局) | UNSHARDED |
| T4 前向/反向 | 同 T3 | 同 T3 | UNSHARDED |
| T5 reshard | sharded `DTensor` Parameter | `_sharded_param_data`（all-gather 输出 storage resize 0） | SHARDED |

---

## 六、关键设计要点

### 1. 为什么用 storage resize 而非 del 来释放显存？

FSDP 动态释放/分配 unsharded param，但 autograd 可能在反向时需要引用它或它的 view（保存在计算图里）。storage resize 到 0 释放显存但 **保留 aliasing 关系**，下次重新分配时 view 仍有效。因此 unsharded param 对象只构造一次，之后 in-place 写入（注释 `_fsdp_param.py:71-78`）。

### 2. 为什么 pre-pad？

all-gather 要求每个 rank 输入等长。若不均分（如 10 维分 4 份），先 pad 到最大块大小，gather 后丢弃多余部分。这避免了每次 all-gather 前后的 pad/unpad 拷贝。

### 3. plain 路径下 unsharded param 不是 DTensor

`_unsharded_dtensor_spec` 只在参数原本就是 DTensor（SPMD/TP）时设置。plain tensor 路径下 unsharded param 是 `torch.as_strided` 出来的普通 Tensor Parameter，前向计算不经过 DTensor 机制。

### 4. `_sharded_param_data` 与 `sharded_param._local_tensor` 的关系

`_sharded_param_data` 是 padded 的 1D 视图（长度 = padded_size.numel()）。`sharded_param._local_tensor` 是它的 narrow view（长度 = 真实分片长度，去掉尾部 padding）。两者共享 storage，all-gather 时用 `_sharded_param_data`（含 padding，等长）。

### 5. DTensorSpec 的角色

`DTensorSpec`（`_dtensor_spec.py:77`）是 dataclass，包含：

- `mesh`：DeviceMesh
- `placements`：tuple[Placement, ...]
- `tensor_meta`：TensorMeta（全局 shape/stride/dtype）
- `shard_order`：分片执行顺序（多 mesh 维共切同一 tensor 维时用）

它是 DTensor 的"分片配方"，**不持有数据**，只描述 local tensor 如何对应全局张量。

---

## 七、三条初始化路径对比

| 路径 | 触发条件 | `_sharding_spec` 的 placements | 返回的待分片数据 |
| --- | --- | --- | --- |
| **plain** | param 非 DTensor | `(Shard(0),)` 或 `(Replicate(), Shard(0))` (HSDP) | `param` 本身 |
| **SPMD** | param 是 DTensor 且 mesh 是 SPMD mesh | DP 维由 `Replicate` 改为 `Shard`/`_StridedShard` | `param._local_tensor` |
| **TP** | param 是 DTensor，DP mesh 独立于 TP mesh | `(Replicate(), ...tp_placements)` 或含 `_StridedShard` | `param._local_tensor` |

三条路径最终都通过 `_from_local_no_grad` 把 local tensor 包成 sharded DTensor，区别仅在于 spec 的 placements 构造方式。
