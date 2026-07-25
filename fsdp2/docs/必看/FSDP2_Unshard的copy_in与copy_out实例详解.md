# FSDP2：Unshard 的 copy-in / copy-out 实例详解

> 配套文档：`FSDP2_fully_shard机制详解.md`、`FSDP2权重如何变成DTensor及内存生命周期.md`
>
> 源码：
> - `_fsdp_param.py`：单参数分片、`init_unsharded_param`、`to_unsharded`、`free_storage`
> - `_fsdp_collectives.py`：`foreach_all_gather`（copy-in + NCCL）、`foreach_all_gather_copy_out`
> - `_fsdp_param_group.py`：`unshard` / `wait_for_unshard` 编排
>
> 本文用一个**具体例子**完整追踪一遍，标出每个时刻内存里实际存在的变量（storage 级别）。

---

## 一、例子设定

4 张 GPU，`fc1.weight = Parameter((512, 1024), float32)` ≈ 2MB，`Shard(0)`，4 张卡的参数放在**同一个** `FSDPParamGroup` 里。以下取 **rank 0 视角**，且**无混合精度**（`param_dtype=None`）。

为便于追踪，给每个 storage 命名：

| storage 名 | 含义 | 大小 |
|-----------|------|------|
| **S1** | 本 rank 的分片（padded） | 0.5MB |
| **S2** | all-gather 合并输出缓冲（整个 group 共用一个） | 2.0MB（临时） |
| **S3** | 每个 param 自己的 `all_gather_outputs[0]`（全量） | 2.0MB |

---

## 二、阶段 ①：普通 Tensor → DTensor（`_init_sharded_param`，`_fsdp_param.py:223`）

调用前每卡都有完整 `(512,1024)`。函数逐步执行：

| 步骤 | 源码行 | 产生/赋值的变量 | 指向的 storage |
|------|--------|-----------------|----------------|
| 构造 spec | `:497` | `_sharding_spec = DTensorSpec(mesh=(4,), placements=(Shard(0),), tensor_meta=(512,1024,f32))` | — (元数据) |
| 记录全局形状 | `:276` | `_orig_size = (512,1024)`，`_contiguous_orig_stride` | — |
| 切分 | `:294` | `chunks = _chunk_with_empty(param, 4, dim=0)` → 4 份 `(128,1024)` | 临时 |
| 取本 rank | `:295` | `sharded_param = chunks[0]` → `(128,1024)` | 临时 |
| **预分配 padded 缓冲** | `:303` | `padded_sharded_param = param.new_zeros((128,1024))` | **S1 (0.5MB)** |
| 拷入真实数据 | `:305` | `padded_sharded_param.narrow(0,0,128).copy_(sharded_param)` | S1 |
| **all-gather 输入缓冲** | `:312` | `_sharded_param_data = padded_sharded_param.view(-1)` → `(131072,)` 1D | **S1**（别名！）|
| 造 DTensor | `:321` | `sharded_param = nn.Parameter(to_sharded_dtensor(padded_sharded_param.narrow(...)))` | S1 |
| 注册回模块 | `:327` | `_setattr_on_modules` → `fc1.weight = DTensor(local=(128,1024))` | S1 |

### 关键别名关系

`_sharded_param_data`（1D, 131072）和 `sharded_param._local_tensor`（2D, 128×1024）是**同一个 storage S1 的两个视图**：

- `_sharded_param_data`（1D）→ 供 all-gather 取 1D 输入
- `sharded_param._local_tensor`（2D）→ 供模块计算身份

`_init_sharded_param` 返回后，原始全量 `param`（2MB）引用计数归零被释放。

### 初始化后 rank 0 内存

```
fc1.weight (DTensor) ─┐
                      ├─→ sharded_param._local_tensor (128,1024) ─┐
fsdp_param._sharded_param_data (131072,) 1D ──────────────────────┴─→ S1 (0.5MB)
fsdp_param._sharding_spec (元数据, 不占显存)
fsdp_param._orig_size = (512,1024)
# 原始 2MB 全量 tensor 已释放
```

**显存：0.5MB**

---

## 三、阶段 ②：Unshard —— copy-in / all-gather / copy-out

`pre_forward`（`_fsdp_param_group.py:475`）→ `unshard()` + `wait_for_unshard()`。

### (a) copy-in：把本 rank 分片填进 all-gather 输出缓冲的对应槽位

源码 `foreach_all_gather`（`_fsdp_collectives.py:324`）：

```python
# :337 取每个 param 的 1D 输入（= _sharded_param_data，视图 S1）
param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
# :351 分配 all-gather 输出缓冲：本rank输入 × world_size
all_gather_output = all_gather_comm.allocate((131072*4,), f32, cuda)   # → S2 (2.0MB)
# :354 copy-in
all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
    all_gather_inputs, all_gather_output, inp_split_sizes, 131072, rank)
```

`all_gather_copy_in` 实现（`:263`）：

```python
all_gather_input = all_gather_output.narrow(0, 131072*rank, 131072)  # S2 里本rank的槽位
foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)            # S1 → S2[本rank槽]
```

#### copy-in 作用

`dist.all_gather_into_tensor` 是 **in-place 风格**——输出缓冲里**本 rank 自己那段**必须预先填好本 rank 的数据，NCCL 才能把各 rank 的段拼成完整缓冲。所以 copy-in = "把自己的分片放到输出缓冲的自己那格"。

copy-in 后 S2 状态（rank 0 视角，只有 `[0:131072]` 有数据，其余待通信填充）：

```
S2: [rank0的128行 | 空 | 空 | 空 ]
```

### (b) NCCL all-gather：真正通信

```python
# :364
all_gather_work = all_gather_comm(output=all_gather_output, input=all_gather_input, ...)
# → dist.all_gather_into_tensor(S2, S2的rank槽, group)
```

通信后 S2 = `[rank0分片 | rank1分片 | rank2分片 | rank3分片]` = 完整 (512,1024) 的 1D 展开。

### (c) copy-out：从合并缓冲拆到每个 param 的 all_gather_output

`wait_for_unshard`（`_fsdp_param_group.py:381`）→ `foreach_all_gather_copy_out`（`_fsdp_collectives.py:430`）：

```python
# :459 每个 param 创建自己的 all_gather_output（首次）
fsdp_param.init_all_gather_outputs(...)   # → all_gather_outputs = [empty((524288,), f32)] = S3 (2.0MB)
fsdp_param.alloc_all_gather_outputs()     # storage 已分配，no-op
# :487 split_with_sizes_copy 把合并缓冲 S2 按 param 拆分拷到各 param 的 S3
torch.ops.fsdp.split_with_sizes_copy(
    all_gather_output.view(4,-1), all_gather_input_split_sizes, dim=1, out=[S3.view(4,-1)]
)
```

#### copy-out 作用

一个 `FSDPParamGroup` 的所有参数**共用一次 all-gather**（合并成一个 1D 缓冲 S2 提高通信效率，即"grouping is first class"）。copy-out 把合并结果**按参数边界拆开**，写回每个 `FSDPParam.all_gather_outputs[0]`（S3），这样每个参数能独立地 `as_strided` 还原成自己的 ND 形状。

> 单参数 group 时 S2→S3 看似多余，但代码路径统一；多参数时这一步是必须的拆分。

copy-out 后 `self._all_gather_result = None`（`:454`）→ 合并缓冲 S2 被释放。

### (d) init_unsharded_param + to_unsharded：还原完整参数并注册回模块

`init_unsharded_param`（`_fsdp_param.py:572`）：

```python
unsharded_tensor = self.all_gather_outputs[0]          # S3 (1D, 524288)
unsharded_param = torch.as_strided(unsharded_tensor, (512,1024), stride, offset=0)  # S3 的 ND 视图
self._unsharded_param = nn.Parameter(unsharded_param, requires_grad=...)            # 新 Parameter 对象，数据 = S3 视图
```

`to_unsharded`（`:674`）→ `_setattr_on_modules(self._unsharded_param)` → **`fc1.weight` 变成完整 (512,1024)**。

> `as_strided` 是**视图操作**——`_unsharded_param` 与 S3 共享 storage。这正是后续能用 storage resize 释放内存的前提。

### Unshard 完成后 rank 0 内存

```
fc1.weight = _unsharded_param (512,1024) ──→ S3 (2.0MB)   ← 完整参数，供 F.linear 计算
fsdp_param.all_gather_outputs[0] (524288,) ─→ S3 (别名)
fsdp_param.sharded_param (DTensor local 128,1024) ──→ S1 (0.5MB)   ← 分片仍在！
fsdp_param._sharded_param_data (131072,) ──→ S1 (别名)
# S2 (合并缓冲) 已释放
```

**显存：2.5MB**（S1 分片 + S3 全量）。这正对应 `ShardedState.UNSHARDED` 注释（`_fsdp_param.py:120`）："Both it and the sharded parameter contribute to parameter memory"。

---

## 四、阶段 ③：Reshard —— storage 缩 0 释放

`post_forward`（`_fsdp_param_group.py:486`）→ `reshard` → `to_sharded`（`_fsdp_param.py:630`）：

```python
def to_sharded(self):
    self._setattr_on_modules(self.sharded_param)   # fc1.weight = DTensor(分片) 又指回 S1
    self.free_unsharded_param()                     # ← 关键
    self.sharded_state = ShardedState.SHARDED

def free_unsharded_param(self):                     # :753
    for tensor in itertools.chain(self.all_gather_outputs, self._unsharded_inner_tensors):
        free_storage(tensor)                        # storage.resize_(0)
```

`free_storage`（`:994`）**不删对象，只把 storage 缩到 0**：

```python
def free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)        # ← storage 缩到 0，显存立即归还
```

### Reshard 后 rank 0 内存

```
fc1.weight = sharded_param (DTensor local 128,1024) ──→ S1 (0.5MB)
fsdp_param._sharded_param_data (131072,) ──→ S1 (别名)
fsdp_param.all_gather_outputs[0] (524288,) ──→ S3 但 storage.size()==0  ← 对象还在，显存已归还
fsdp_param._unsharded_param (512,1024) ──→ S3 但 storage==0  ← 对象还在，0 显存
```

**显存：0.5MB**

### 为什么不销毁对象只缩 storage？

源码注释 `[Note: FSDP and autograd]`（`_fsdp_param.py:71`）解释：autograd 反向时可能持有 `_unsharded_param` 或其视图的引用。用 storage resize 释放内存能**保留别名关系**——下次 unshard 时 `alloc_storage` 把 S3 扩回来（`_fsdp_param.py:988`），原对象和所有视图自动指向新 all-gather 的数据，autograd 保存的引用不会失效。

```python
def alloc_storage(tensor: torch.Tensor) -> None:     # :988
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)
```

---

## 五、内存变量时间线总表（rank 0，单参数 group）

| 时刻 | `fc1.weight` 指向 | 存活的 storage | 显存 |
|------|-------------------|----------------|------|
| `fully_shard` 前 | 全量 (512,1024) | 原始全量 | 2.0MB |
| 初始化切分后 | DTensor(local 128,1024)→S1 | S1(0.5) | 0.5MB |
| **copy-in** 时 | DTensor→S1 | S1 + **S2**(2.0，正在填本rank槽) | 2.5MB |
| **all-gather** 后 | DTensor→S1 | S1 + S2(满) | 2.5MB |
| **copy-out** 后 | DTensor→S1 | S1 + **S3**(2.0) ；S2 释放 | 2.5MB |
| `to_unsharded` 后（前向计算中） | **全量→S3** | S1 + S3 | 2.5MB |
| **reshard** 后 | DTensor→S1 | S1 ；S3 storage=0 | 0.5MB |
| 反向 unshard | 全量→S3（`alloc_storage` 扩回） | S1 + S3 | 2.5MB |
| 反向 reduce-scatter 后 | DTensor→S1 | S1 + 分片梯度 | 0.5MB+0.5MB |

---

## 六、多参数 group 时的 copy-out 详解

当 `FSDPParamGroup` 含 `fc1.weight` 和 `fc2.weight` 两个参数时，copy-out 才真正体现"拆分"作用：

```
copy-in 阶段：
  S2 = [fc1_rank0 | fc2_rank0 | 空 | 空 | 空 | 空 | 空 | 空]
       (按 inp_split_sizes 把本rank的 fc1分片、fc2分片 依次填进 S2 的本rank槽)

all-gather 后：
  S2 = [fc1_r0|fc2_r0 | fc1_r1|fc2_r1 | fc1_r2|fc2_r2 | fc1_r3|fc2_r3]
       (每rank槽里含两个param的分片，concat在一起)

copy-out（split_with_sizes_copy, dim=1）：
  把 S2 按 param 边界拆成两份：
    fc1 的 all_gather_output = S3 = [fc1_r0|fc1_r1|fc1_r2|fc1_r3]  (2.0MB)
    fc2 的 all_gather_output = S4 = [fc2_r0|fc2_r1|fc2_r2|fc2_r3]  (1.0MB)
```

`split_with_sizes_out` 列表的顺序与 `inp_split_sizes` 对应（`:454-474`），即每个 param 的 `all_gather_outputs` 被依次填入。

### 非 dim=0 分片的额外 copy-out

若某参数 `fsdp_placement = Shard(1)`，all-gather 沿 dim=0 拼接的结果不是该参数需要的拼接维度。此时 `foreach_all_gather_copy_out`（`:467-473`）先拷到临时缓冲，再 chunk-cat 重排成沿 dim=1 拼接（`:495-520`）：

```python
if fsdp_param.fsdp_placement.dim != 0:
    param_all_gather_outputs = [torch.empty_like(t) for t in param_all_gather_outputs]
    shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
# ...
for fsdp_param, param_all_gather_outputs in shard_i_copy_infos:
    chunks = torch.chunk(param_all_gather_output.view(pre_param_size), world_size, dim=0)
    torch.cat(chunks, dim=shard_dim, out=cat_out)   # 沿目标 dim 重新拼接
```

---

## 七、三个核心问题速答

### 1. copy-in 发生在什么时候？有什么用？

**时机**：all-gather **之前**，在 `all_gather_copy_in_stream` 中（`_fsdp_collectives.py:336`）。
**操作**：把本 rank 的 `_sharded_param_data`（S1 的 1D 视图）拷进合并输出缓冲 S2 的本 rank 槽位。
**作用**：满足 `all_gather_into_tensor` 的 in-place 语义——输出里本 rank 那段必须先有数据，NCCL 才能正常 gather。

### 2. copy-out 发生在什么时候？有什么用？

**时机**：all-gather **之后**，在 `wait_for_unshard` 中（`_fsdp_param_group.py:427`）。
**操作**：用 `split_with_sizes_copy` 把合并缓冲 S2 按参数边界拆分拷到每个 param 的 `all_gather_outputs[0]`（S3），再 `as_strided` 还原 ND。
**作用**：一次 all-gather 服务整个 group 的多参数（通信批量化），copy-out 负责把合并结果拆回各参数独立缓冲，供后续 `as_strided` 还原成 `(512,1024)`。

### 3. unshard 创建了什么变量？

| 变量 | 类型 | 生命周期 |
|------|------|----------|
| `all_gather_output` (S2) | 合并缓冲 | copy-in 到 copy-out，之后释放 |
| `all_gather_outputs[0]` (S3) | 每 param 一个全量缓冲 | unshard 创建，reshard 时 storage 缩 0 但对象保留 |
| `_unsharded_param` | Parameter，`as_strided` 视图 S3 | 首次 unshard 创建，对象永久保留，storage 动态伸缩 |
| `fc1.weight` | 通过 `_setattr_on_modules` 指向 `_unsharded_param` | unshard 后到 reshard 前 |

---

## 八、一句话总结

**DTensor = local 分片张量(S1) + DTensorSpec(placements/mesh/全局 meta)**；unshard 通过 copy-in→all-gather→copy-out 把 S1 聚合成 S3，用 `as_strided` 让 `fc1.weight` 指向 S3；reshard 用 storage resize(0) 归还 S3 显存但保留对象别名，供下次 `alloc_storage` 复用。
