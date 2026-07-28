# FSDP2：前向隐式预取的多流 overlap 与显存安全

> 配套文档：`FSDP2_Unshard的copy_in与copy_out实例详解.md`、`前向预取.md`、`FSDP2_fully_shard机制详解.md`、`FSDP2_反向预取_多流overlap与显存安全.md`
>
> 源码：
> - `_fsdp_param_group.py`：`FSDPCommContext.lazy_init`、`get_all_gather_streams`、`unshard` / `wait_for_unshard`、`AllGatherState`
> - `_fsdp_collectives.py`：`foreach_all_gather`（copy-in + all-gather 集体通信）、`foreach_all_gather_copy_out`
>
> 本文解读 `_fsdp_param_group.py` 中 `[Note: Overlapping all-gather copy-in and all-gather]`（源码顶部注释），并用一个**具体例子 + 从左到右的三流时序图**说明 overlap 如何达成、显存如何安全。

---

## 〇、原文 Note

```text
[Note: Overlapping all-gather copy-in and all-gather]
For implicit forward prefetching, we want to overlap the next copy-in with the
current all-gather. We do so using a separate copy-in stream. However, since
we have the all-gather input as a view into the output, we must make sure to
copy into different memory from the current all-gather's output. Thus, we keep
a reference to the current all-gather's output and have the next FSDP parameter
group free it after its copy-in. Finally, we have the last FSDP state flush the
reference to avoid holding onto memory after forward.
```

---

## 一、背景：五条 GPU stream 与一次 unshard 的三步

### 1. 五条 GPU stream

`FSDPCommContext.lazy_init`（`_fsdp_param_group.py:40-78`）创建了四条显式 stream，加上默认的计算流，FSDP2 共使用五条 GPU stream。下表列出全部五条 stream 的全称与作用：

| stream 字段名 | 优先级 | 作用 |
| ------ | ------ | ------ |
| `all_gather_copy_in_stream` | 高优先级 | 把本 rank 的参数分片 copy 到 all-gather 输出缓冲中本 rank 对应的 view 段（前向隐式预取时使用） |
| `all_gather_stream` | 高优先级 | 执行 all-gather 集体通信，将各 rank 的分片汇聚为完整参数 |
| `reduce_scatter_stream` | 高优先级 | 执行 reduce-scatter 集体通信及反向梯度的 pre/post-division 等后处理逻辑 |
| `all_reduce_stream` | 默认优先级 | 执行 HSDP 的 all-reduce 集体通信（与 all-gather/reduce-scatter 使用不同网络资源，可并行） |
| `current_stream`（计算流） | 默认优先级 | 执行 copy-out（all-gather 输出 → 参数）和前向算子（matmul 等），即 PyTorch 默认的当前流 |

> 本文聚焦前向隐式预取，涉及前三条 stream 中与 all-gather 相关的三条：`all_gather_copy_in_stream`（all-gather copy-in 流）、`all_gather_stream`（all-gather 流）、`current_stream`（计算流）。`reduce_scatter_stream` 和 `all_reduce_stream` 用于反向，本文不展开。

### 2. 关键：前向隐式预取走两条独立 stream

`get_all_gather_streams`（`_fsdp_param_group.py:80-90`）：

```python
def get_all_gather_streams(self, async_op, training_state):
    if not async_op and training_state in (FORWARD, PRE_BACKWARD):
        # async_op=False 且前向/反向预取：copy-in 和 all-gather 分别用独立 stream，才能 overlap
        return self.all_gather_copy_in_stream, self.all_gather_stream
    current_stream = self.device_handle.current_stream()
    return current_stream, current_stream   # async_op=True：同一条 stream（用 overlap 换避免多流显存碎片）
```

只要 **`async_op=False`（默认）且 `training_state ∈ {FORWARD, PRE_BACKWARD}`**，就返回 `all_gather_copy_in_stream` 和 `all_gather_stream` 两条独立 stream——这涵盖前向（`FORWARD`）和反向预取（`PRE_BACKWARD`）。只有当用户主动设置 `unshard_async_op=True` 时（`_fully_shard.py` 中 `set_unshard_async_op`），`async_op` 才为 True，两条返回的都是 `current_stream`（计算流），copy-in 和 all-gather 串行排在同一条计算流上——这是用 overlap 换取避免多流显存碎片的权衡，需配合显式预取使用。

### 3. 一次 unshard 在 stream 上的三步展开

`foreach_all_gather`（`_fsdp_collectives.py:323-372`）把 unshard 的前两步编排到 stream 上：

```text
all-gather copy-in 流:  copy 参数分片 → all-gather输出缓冲[本rank的view]   # 注意：input 是 output 的 view！
all-gather 流:          wait_stream(all-gather copy-in流); all-gather(输出缓冲); 记录 all-gather完成事件
```

随后 `wait_for_unshard`（`_fsdp_param_group.py:348-430`）在计算流上执行第三步：

```text
计算流:                  wait_event(all-gather完成事件); copy-out(输出缓冲 → 参数); 记录 copy-out完成事件
```

**重点**：`_fsdp_collectives.py:346` 调用的 `torch.ops.fsdp.all_gather_copy_in(...)` 返回的 `all_gather_input` 不是新内存，而是 **all-gather 输出缓冲里本 rank 那一段的 view**（`_fsdp_collectives.py:267-269` 的 `all_gather_output.narrow(...)`）。也就是说 copy-in 写入的内存 == all-gather 即将读/写的输出缓冲本身。这一点是后面显存安全的根源。

---

## 二、具体例子设定与记号说明

### 1. 例子设定

- 模型：2 个 Transformer block，各自用 `fully_shard` 包了一层 → 两个 FSDP 参数组。
- `world_size = 2`，参数按 FSDP 切分。
- 前向、隐式预取（`async_op=False`，默认）。

程序（Python/CPU）顺序编排到 stream 上：

```text
block0.pre_forward → block0.forward → block1.pre_forward → block1.forward → ...
```

其中 `pre_forward`（`_fsdp_param_group.py:443-452`）内部依次调用 `unshard()` + `wait_for_unshard()`。

### 2. 记号说明

为了在时序图中简洁表达，本文使用以下记号。**这些不是缩写，而是带编号的全称简写形式**，每个记号的含义如下：

| 记号 | 全称含义 | 来源 |
| ------ | ------ | ------ |
| **参数组0** | 第 0 个 FSDP 参数组，对应 Transformer block0（`fully_shard(block0)` 包裹的参数组） | `FSDPParamGroup` 实例 |
| **参数组1** | 第 1 个 FSDP 参数组，对应 Transformer block1（`fully_shard(block1)` 包裹的参数组） | `FSDPParamGroup` 实例 |
| **参数分片0** | 本 rank 持有的参数组0 的分片数据（sharded parameter） | `fsdp_param._sharded_param_data` |
| **参数分片1** | 本 rank 持有的参数组1 的分片数据（sharded parameter） | `fsdp_param._sharded_param_data` |
| **输出缓冲0** | 参数组0 的 all-gather 输出缓冲，由 `all_gather_comm.allocate(...)` 分配，存放 all-gather 后的完整参数 | `AllGatherResult.all_gather_output` |
| **输出缓冲1** | 参数组1 的 all-gather 输出缓冲，由 `all_gather_comm.allocate(...)` 分配，存放 all-gather 后的完整参数 | `AllGatherResult.all_gather_output` |
| **参数0** | 参数组0 unshard 后的实际参数张量（unsharded param），供前向算子使用 | `fsdp_param._unsharded_param` |
| **参数1** | 参数组1 unshard 后的实际参数张量（unsharded param），供前向算子使用 | `fsdp_param._unsharded_param` |
| **all-gather 完成事件0** | 参数组0 的 all-gather 操作完成时在 all-gather 流上记录的 CUDA event | `AllGatherResult.all_gather_event` |
| **all-gather 完成事件1** | 参数组1 的 all-gather 操作完成时在 all-gather 流上记录的 CUDA event | `AllGatherResult.all_gather_event` |
| **copy-out 完成事件0** | 参数组0 的 copy-out 操作完成时在计算流上记录的 CUDA event | `all_gather_copy_out_event` |
| **copy-out 完成事件1** | 参数组1 的 copy-out 操作完成时在计算流上记录的 CUDA event | `all_gather_copy_out_event` |
| **copy-in** | 将本 rank 的参数分片拷贝到 all-gather 输出缓冲中本 rank 对应的 view 段 | `torch.ops.fsdp.all_gather_copy_in` |
| **all-gather** | 执行 all-gather 集体通信，将所有 rank 的分片汇聚到输出缓冲 | `dist.all_gather_into_tensor` |
| **copy-out** | 将 all-gather 输出缓冲的数据拆分拷贝到参数组实际的参数张量 | `foreach_all_gather_copy_out` |

> **注意**：输出缓冲0 和输出缓冲1 是两块**不同的物理显存**。正是因为这个区别，copy-in 和 all-gather 才能安全并行——这正是后文要详细解释的核心。

---

## 三、从左到右的三流时序图（前向）

下图时间轴**从左到右**流动，三条 stream 各占一行。为清晰起见，同步原语做如下约定：

- `wait_stream(A→B)`：B 流等待 A 流完成（`B.wait_stream(A)`）。
- `wait_event(E)`：当前流等待事件 E 发生（`stream.wait_event(E)`）。
- `[E]`：在当前流上记录事件 E（`stream.record_event()`）。

```text
时间 ──────────────────────────────────────────────────────────────────────────────────────────→

                ┌─ copy-in0 ─────┐  ┌─ copy-in1 ─────┐
all-gather      │ copy 参数分片0 │  │ copy 参数分片1 │  wait_event(copy-out-event0)
copy-in 流      │ → 输出缓冲0    │  │ → 输出缓冲1    │  │ 约束后续复用输出缓冲0 的内存
                │ (参数组0)      │  │ (参数组1)      │  │
                └───────┬────────┘  └───────┬────────┘  └──────────────────────────────
                        │ wait_stream       │ wait_stream
                        ↓ (copy-in→all-gather)↓ (copy-in→all-gather)
all-gather              ┌─ all-gather0 ──┐    ┌─ all-gather1 ──┐
流                      │ (输出缓冲0)     │    │ (输出缓冲1)     │  wait_event(copy-out-event0)
                        │ [AG-event0]    │    │ [AG-event1]    │  │ 约束后续复用输出缓冲0 的内存
                        └───────┬────────┘    └───────┬────────┘  └──────────────────────────────
                                │ wait_event        │ wait_event
                                ↓ (AG-event0)       ↓ (AG-event1)
计算流                          ┌─ copy-out0 ────┐  ┌─ block0 ──┐  ┌─ copy-out1 ────┐  ┌─ block1 ──┐
                                │ 输出缓冲0      │  │  算子     │  │ 输出缓冲1      │  │  算子     │
                                │ → 参数0        │  │ (matmul等)│  │ → 参数1        │  │ (matmul等)│
                                │ [CO-event0]    │  │           │  │ [CO-event1]    │  │           │
                                └────────────────┘  └───────────┘  └────────────────┘  └───────────┘

         ◄──── overlap 1 ────►              ◄────────── overlap 2 ──────────►
         copy-in1(参数分片1→输出缓冲1)
         ∥ all-gather0(输出缓冲0)              all-gather1(输出缓冲1)
                                                 ∥ copy-out0(输出缓冲0→参数0) + block0 算子
```

### 图中发生的事情，逐步说明

1. **参数组0 的 copy-in**（all-gather copy-in 流）：`foreach_all_gather` 在 `all_gather_copy_in_stream` 上分配输出缓冲0，并将本 rank 的参数分片0 拷贝到输出缓冲0 中本 rank 对应的 view 段。由于 all-gather 的 input 是 output 的 view，这一步实际上是在写输出缓冲0 本身。

2. **参数组0 的 all-gather**（all-gather 流）：`all_gather_stream` 通过 `wait_stream(all_gather_copy_in_stream)`（`_fsdp_collectives.py:355`）等待 copy-in0 完成后，执行 all-gather0，将各 rank 的分片汇聚到输出缓冲0。完成后在 `all_gather_stream` 上记录 **all-gather 完成事件0**（`AG-event0`）。

3. **参数组1 的 copy-in**（all-gather copy-in 流）：copy-in0 一完成，`all_gather_copy_in_stream` 立即接上 copy-in1——分配输出缓冲1（此时输出缓冲0 仍被 `all_gather_state` 持有，allocator 不会复用它的内存，所以输出缓冲1 ≠ 输出缓冲0），将参数分片1 拷贝到输出缓冲1。**copy-in1 与 all-gather0 并行**——这就是 **overlap 1**。

4. **参数组0 的 copy-out + block0 算子**（计算流）：`current_stream` 通过 `wait_event(AG-event0)` 等待 all-gather0 完成后，执行 copy-out0（输出缓冲0 → 参数0），记录 **copy-out 完成事件0**（`CO-event0`），然后执行 block0 的前向算子（matmul 等）。

5. **参数组1 的 all-gather**（all-gather 流）：copy-in1 完成后，`all_gather_stream` 执行 all-gather1，记录 `AG-event1`。**all-gather1 与 block0 算子并行**——这就是 **overlap 2**。

6. **参数组1 的 copy-out + block1 算子**（计算流）：block0 算子完成后，`current_stream` 等待 `AG-event1`，执行 copy-out1（输出缓冲1 → 参数1），记录 `CO-event1`，然后执行 block1 的前向算子。

7. **延迟释放 + 插入后续等待**：在参数组1 的 `wait_for_unshard` 开头（`_fsdp_param_group.py:360-363`），先清除 `comm_ctx.all_gather_state`，释放对输出缓冲0 的引用。此时 copy-in1 已经入队、输出缓冲1 已经分配好，所以释放输出缓冲0 不会导致 allocator 把同一块内存分给输出缓冲1。随后 `_wait_all_gather_streams_on_event`（`_fsdp_param_group.py:423-428`）在 `all_gather_copy_in_stream` 和 `all_gather_stream` 的末尾排入 `wait_event(CO-event0)`，保证**后续**任何复用输出缓冲0 内存的操作必须等 copy-out0 读完后才能执行。

8. **末尾 flush**：参数组1 是最后一个参数组，后面没有"下一个参数组"来替它清 `all_gather_state`。如果不 flush，输出缓冲1 这块 unsharded 缓冲会一直挂到下一次前向才开始，白白占显存。因此前向收尾时由 **root FSDPState 的 `_post_forward`**（`_fsdp_state.py:315-324`）检查并清空 `comm_ctx.all_gather_state`，释放输出缓冲1。

### overlap 在哪里？

1. **copy-in(参数分片1)（all-gather copy-in 流） ∥ all-gather(输出缓冲0)（all-gather 流）**：all-gather copy-in 流和 all-gather 流是两条独立 stream。参数组0 的 copy-in 一完成，all-gather copy-in 流立刻接参数组1 的 copy-in，all-gather 流同时开始参数组0 的 all-gather。→ 这就是 Note 说的 "overlap the next copy-in with the current all-gather"。

2. **all-gather1(输出缓冲1)（all-gather 流） ∥ block0 算子（计算流）**：copy-out0 完成后计算流跑 block0 算子，`all_gather_stream` 上参数组1 的 all-gather1 同时进行。→ `wait_for_unshard` docstring（`_fsdp_param_group.py:349-356`）说的 "overlap the current copy-out with the next all-gather"。

---

## 四、结合 Note 逐句说明

### ① "overlap the next copy-in with the current all-gather"

对应时序图 **overlap 1** 区域：

- **current all-gather** = 参数组0 的 all-gather（all-gather 流上 `all-gather(输出缓冲0)`）
- **next copy-in** = 参数组1 的 copy-in（all-gather copy-in 流上 `copy 参数分片1 → 输出缓冲1`）

因为 all-gather copy-in 流和 all-gather 流是两条独立 stream，参数组0 的 copy-in 一完成，all-gather copy-in 流立刻接参数组1 的 copy-in，而 all-gather 流同时开始参数组0 的 all-gather。两者**并行**。这就是 `get_all_gather_streams` 在前向返回两条独立 stream 的目的（`_fsdp_param_group.py:83-88`）。

### ② "all-gather input as a view into the output"

`foreach_all_gather` 里 `_fsdp_collectives.py:346` 调用 `torch.ops.fsdp.all_gather_copy_in(...)`，其实现（`_fsdp_collectives.py:267-269`）通过 `all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)` 返回 `all_gather_input`——它不是新内存，而是 **all-gather 输出缓冲里本 rank 那一段的 view**。也就是说 copy-in 写入的内存 == all-gather 即将读/写的输出缓冲本身。

### ③ "must copy into different memory from the current all-gather's output"

**危险场景**：如果输出缓冲0 在参数组1 的 copy-in 之前就被释放掉，那么 caching allocator 在参数组1 的 `all_gather_comm.allocate(...)`（`_fsdp_collectives.py:344`）时，**很可能把同一块物理显存分给输出缓冲1**。于是：

- all-gather copy-in 流上参数组1 的 copy-in 正在写输出缓冲1（== 输出缓冲0 那块内存）；
- all-gather 流上参数组0 的 all-gather 可能还没写完输出缓冲0。

→ 两条 stream 对同一块显存并发写，**数据竞争**，all-gather 结果损坏。

### ④ "keep a reference to the current all-gather's output … have the next group free it after its copy-in"

**解决办法**（`wait_for_unshard`，`_fsdp_param_group.py:408-417`）：参数组0 的 copy-out 完成后**不立刻释放输出缓冲0**，而是把 `(AllGatherResult, copy-out 完成事件)` 存到全局 `comm_ctx.all_gather_state`：

```python
# _fsdp_param_group.py:408-417
if (
    not async_op
    and self._training_state == TrainingState.FORWARD
    and world_size > 1
):
    # Defer free to allow for overlap of this copy-out with next
    # all-gather collective
    self.comm_ctx.all_gather_state = AllGatherState(
        self._all_gather_result, all_gather_copy_out_event
    )
else:
    self._wait_all_gather_streams_on_event(all_gather_copy_out_event)

self._all_gather_result = None  # free unless saved in `all_gather_state`
```

只要这个引用还在，allocator 就**不会把输出缓冲0 的内存分给输出缓冲1**，保证输出缓冲1 ≠ 输出缓冲0 → 安全 overlap。

等到参数组1 的 `wait_for_unshard`（`_fsdp_param_group.py:360-363`）：

```python
if self._training_state == TrainingState.FORWARD:  # implicit prefetch
    if prev_all_gather_state := self.comm_ctx.all_gather_state:
        self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
        self.comm_ctx.all_gather_state = None  # free the all-gather result
```

**关键**：这个清除发生在参数组1 的 `unshard()` **之后**（`pre_forward` 里先 `unshard` 再 `wait_for_unshard`，`_fsdp_param_group.py:449-450`）。也就是说，参数组1 的 copy-in 已经入队、输出缓冲1 已经分配好之后，才丢掉输出缓冲0 引用。

同时 `_wait_all_gather_streams_on_event`（`_fsdp_param_group.py:423-428`）在 all-gather copy-in 流和 all-gather 流末尾排一个 `wait_event(CO-event0)`，保证之后任何复用输出缓冲0 内存的操作必须等参数组0 的 copy-out 读完：

```python
def _wait_all_gather_streams_on_event(self, event: torch.Event | None):
    if hasattr(self.comm_ctx, "all_gather_copy_in_stream") and event is not None:
        self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
    if hasattr(self.comm_ctx, "all_gather_stream") and event is not None:
        self.comm_ctx.all_gather_stream.wait_event(event)
```

注意：这个 wait 是排在参数组1 自己的 copy-in / all-gather **之后**的（CPU 入队顺序），所以**不会挡住**已经发起的参数组1 copy-in，只约束后续对输出缓冲0 内存的复用。

### ⑤ "the last FSDP state flush the reference to avoid holding onto memory after forward"

参数组1 是最后一个参数组，后面没有下一个参数组来替它清 `all_gather_state`。如果不 flush，输出缓冲1 这块 unsharded 缓冲会一直挂到下一次前向才开始，白白占显存。因此前向收尾时由 **root FSDPState 的 `_post_forward`**（`_fsdp_state.py:315-324`）检查并清空 `comm_ctx.all_gather_state`，释放输出缓冲1。

> 反向不走延迟释放这条路：延迟释放的条件是 `not async_op AND training_state == FORWARD AND world_size > 1`（`_fsdp_param_group.py:408-411`），反向的 `training_state` 是 `PRE_BACKWARD`，不满足条件，走 `else` 分支立即 `wait_event(copy-out 完成事件)`，不写 `all_gather_state`。但反向**仍然使用独立流**（`PRE_BACKWARD` 在 `get_all_gather_streams` 的条件中），只是不需要延迟释放——因为反向的 overlap 对象是 reduce-scatter（不同流、不同缓冲），而非下一个 all-gather（同类型缓冲），无显存安全问题。详见配套文档 `FSDP2_反向预取_多流overlap与显存安全.md`。

---

## 五、从左到右的浓缩图：overlap 与显存生命周期

```text
时间 ──────────────────────────────────────────────────────────────────────────→

                参数组0                                参数组1
                ────────                                ────────

all-gather      ████████████████████                                            
copy-in 流      copy-in(参数分片0→输出缓冲0)        ████████████████████
                                                    copy-in(参数分片1→输出缓冲1)

all-gather                  ████████████████████                        ████████████████████
流                          all-gather(输出缓冲0)                        all-gather(输出缓冲1)

计算流                                ████████████████████  ████████████████████  ████████████████████  ████████████████████
                                      copy-out(输出缓冲0)   block0 算子           copy-out(输出缓冲1)   block1 算子

                ◄── overlap 1 ──►              ◄──── overlap 2 ────►

输出缓冲0       [分配]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ←──── 被 all_gather_state 持有 ────→                  ┃
                                                          被参数组1.wait_for_unshard 释放 ─→

输出缓冲1                         [分配]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                                                                                末尾 flush 释放 ─→
```

**核心收益**：参数组1 的 copy-in 不必等参数组0 的 all-gather / copy-out / block0 算子完成，而是搭便车在 all-gather copy-in 流上和它们并行跑；参数组0 的输出缓冲通过"延迟一拍释放"既不挡 overlap、又不被复用导致写冲突；最后一个参数组自己收尾避免显存悬挂。

---

## 六、关键代码索引

| 关注点 | 位置 |
| -------- | ------ |
| 创建 all-gather copy-in 流 / all-gather 流（高优先级） | `_fsdp_param_group.py:71-81`（`FSDPCommContext.lazy_init`） |
| 创建 reduce-scatter 流 / all-reduce 流 | `_fsdp_param_group.py:82-92`（`FSDPCommContext.lazy_init`） |
| 前向返回两条独立 stream | `_fsdp_param_group.py:80-90`（`get_all_gather_streams`） |
| copy-in 写入 all-gather 输出缓冲的 view | `_fsdp_collectives.py:346`（`all_gather_copy_in`） |
| all-gather 输出缓冲的 view 实现 | `_fsdp_collectives.py:267-269`（`all_gather_output.narrow`） |
| all-gather 流等 all-gather copy-in 流 | `_fsdp_collectives.py:355`（`all_gather_stream.wait_stream`） |
| copy-in → all-gather → copy-out 编排 | `_fsdp_param_group.py:314-430`（`unshard` / `wait_for_unshard`） |
| 延迟释放：存 all-gather state | `_fsdp_param_group.py:408-417` |
| 下一个参数组释放上一个输出缓冲 | `_fsdp_param_group.py:360-363` |
| all-gather copy-in 流 / all-gather 流排 wait(copy-out 完成事件) | `_fsdp_param_group.py:423-428`（`_wait_all_gather_streams_on_event`） |
| AllGatherState 定义 | `_fsdp_param_group.py:116-118` |
| 末尾 flush `all_gather_state` | `_fsdp_state.py:315-324`（`_post_forward`） |

---

## 七、一句话总结

**前向隐式预取**用 all-gather copy-in 流和 all-gather 流两条独立 stream 让"下一个参数组的 copy-in"与"当前参数组的 all-gather"并行；但因为 all-gather 的 input 是 output 的 view，必须**持有当前参数组的输出缓冲引用直到下一个参数组的 copy-in 已分配新内存**，由下一个参数组在 `wait_for_unshard` 里释放，最后一个参数组由末尾 flush 收尾——既拿到 overlap，又不泄漏显存、不产生写冲突。
