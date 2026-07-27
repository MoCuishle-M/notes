# FSDP2：反向预取的多流 overlap 与显存安全

> 配套文档：`FSDP2_前向隐式预取_多流overlap与显存安全.md`、`反向预取.md`、`FSDP2_fully_shard机制详解.md`
>
> 源码：
> - `_fsdp_param_group.py`：`pre_backward`、`_backward_prefetch`、`_prefetch_unshard`、`post_backward`、`wait_for_unshard`
> - `_fsdp_collectives.py`：`foreach_all_gather`（copy-in + all-gather 集体通信）、`foreach_reduce`（reduce-scatter copy-in + reduce-scatter 集体通信）
> - `_fsdp_state.py`：`_pre_backward`（隐式/显式预取分发）、`_post_forward`（末尾 flush）
>
> 本文解读 `wait_for_unshard` docstring（`_fsdp_param_group.py:382-388`）关于反向的第 2 点，并用一个**具体例子 + 从左到右的四流时序图**说明反向预取如何达成**三路并行 overlap**、为何不需要延迟释放。

---

## 〇、原文 docstring

```python
def wait_for_unshard(self):
    """
    1. In forward with implicit prefetching, to overlap the current copy-out
    with the next all-gather, we save a reference to the current all-gather
    result to free after the next copy-out.
    2. Otherwise (explicit prefetching or in backward), we free the
    all-gather result immediately after the current copy-out since we can
    already overlap the current copy-out with the previous reduce-scatter.
    """
```

本文聚焦第 2 点中**反向**的部分。

> **注意**：docstring 中的 "explicit prefetching" 实际指的是 `unshard_async_op=True` 的场景（此时 `async_op=True`，延迟释放条件 `not async_op` 不满足）。反向（`PRE_BACKWARD`）即使 `async_op=False`（默认），也因 `training_state != FORWARD` 走立即释放分支。详见后文第五节。

---

## 一、背景：反向执行顺序与四条流

### 1. 反向执行顺序

PyTorch autograd 按前向计算图的**逆拓扑序**执行反向。对于 3 个 Transformer block 的顺序模型：

```text
前向: block0.forward → block1.forward → block2.forward → loss
反向: block2.backward → block1.backward → block0.backward
```

每个 block 的反向涉及三个阶段（由 FSDP hook 驱动）：

| 阶段 | 触发方式 | 做什么 |
| ------ | ------ | ------ |
| **pre_backward** | 输出张量的 `register_hook` | unshard（all-gather 参数）+ 预取下一个反向模块的 unshard |
| **backward compute** | autograd 自动 | 用 unsharded 参数计算梯度 |
| **post_backward** | `RegisterPostBackwardFunction`（输入张量上的自定义 autograd Function） | reduce-scatter 梯度 |

### 2. 四条流

反向预取涉及四条 GPU stream（均为 `FSDPCommContext.lazy_init` 创建，`_fsdp_param_group.py:72-100`）：

| stream 字段名 | 优先级 | 反向中的作用 |
| ------ | ------ | ------ |
| `all_gather_copy_in_stream` | 高优先级 | copy-in：本 rank 参数分片 → all-gather 输出缓冲 |
| `all_gather_stream` | 高优先级 | all-gather 集体通信 |
| `current_stream`（计算流） | 默认优先级 | copy-out（all-gather 输出 → 参数）、反向算子、reduce-scatter copy-in（梯度 → reduce-scatter 输入缓冲） |
| `reduce_scatter_stream` | 高优先级 | reduce-scatter 集体通信及梯度后处理 |

> 与前向相比，反向多了 `reduce_scatter_stream`。前向的三流 overlap 是 all-gather copy-in 流 ∥ all-gather 流 ∥ 计算流；反向的四流 overlap 多了 reduce-scatter 流，可实现**三路并行**。

### 3. 反向预取的关键：pre_backward 中发起下一个 all-gather

`pre_backward`（`_fsdp_param_group.py:505-514`）：

```python
def pre_backward(self, default_prefetch, *unused):
    ...
    self._training_state = TrainingState.PRE_BACKWARD
    self.unshard(self.unshard_async_op)  # no-op if prefetched
    self.wait_for_unshard()
    if default_prefetch:
        self._backward_prefetch()  # ← 预取下一个反向模块的 all-gather
```

`_backward_prefetch`（`_fsdp_param_group.py:657-692`）在当前模块的 `wait_for_unshard` **之后**调用，预取前向顺序中**前一个**模块（= 反向顺序中**下一个**模块）的 all-gather：

```python
def _backward_prefetch(self):
    ...
    curr_index = self._post_forward_indices.pop()
    ...
    elif curr_index > 0:
        target = self.comm_ctx.post_forward_order[curr_index - 1]
        self._prefetch_unshard(target, "backward")
```

`_prefetch_unshard`（`_fsdp_param_group.py:694-710`）将目标模块的 `training_state` 临时设为 `PRE_BACKWARD`，然后调用 `unshard(async_op)`——由于 `PRE_BACKWARD` 在 `get_all_gather_streams` 的条件中，**反向预取也使用独立流**。

---

## 二、具体例子设定与记号说明

### 1. 例子设定

- 模型：3 个 Transformer block，各自用 `fully_shard` 包了一层 → 三个 FSDP 参数组。
- `world_size = 2`，参数按 FSDP 切分。
- 反向、隐式预取（`async_op=False`，默认）。
- 前向顺序：block0 → block1 → block2，反向顺序：block2 → block1 → block0。

为简洁起见，时序图以 2 个 block（block1、block0）为主展示一个完整的 overlap 周期；3 个 block 时模式重复，后文附浓缩流水线图。

### 2. 记号说明

| 记号 | 全称含义 | 来源 |
| ------ | ------ | ------ |
| **参数组N** | 第 N 个 FSDP 参数组，对应 Transformer blockN | `FSDPParamGroup` 实例 |
| **参数分片N** | 本 rank 持有的参数组N 的分片数据 | `fsdp_param._sharded_param_data` |
| **输出缓冲N** | 参数组N 的 all-gather 输出缓冲 | `AllGatherResult.all_gather_output` |
| **参数N** | 参数组N unshard 后的实际参数张量 | `fsdp_param._unsharded_param` |
| **all-gather 完成事件N** | 参数组N 的 all-gather 完成时在 all-gather 流上记录的 CUDA event | `AllGatherResult.all_gather_event` |
| **copy-out 完成事件N** | 参数组N 的 copy-out 完成时在计算流上记录的 CUDA event | `all_gather_copy_out_event` |
| **RS 输入缓冲N** | 参数组N 的 reduce-scatter 输入缓冲，存放待 reduce-scatter 的梯度 | `reduce_scatter_input` |
| **RS 完成事件N** | 参数组N 的 reduce-scatter 完成时在 reduce-scatter 流上记录的 CUDA event | `reduce_scatter_event` |
| **copy-in** | 将本 rank 的参数分片拷贝到 all-gather 输出缓冲中本 rank 的 view 段 | `torch.ops.fsdp.all_gather_copy_in` |
| **all-gather** | all-gather 集体通信 | `dist.all_gather_into_tensor` |
| **copy-out** | 将 all-gather 输出缓冲拆分拷贝到参数张量 | `foreach_all_gather_copy_out` |
| **RS copy-in** | 将反向梯度拷贝到 reduce-scatter 输入缓冲 | `foreach_reduce_scatter_copy_in` |
| **reduce-scatter** | reduce-scatter 集体通信 | `dist.reduce_scatter_tensor` |

---

## 三、从左到右的四流时序图（反向）

下图时间轴**从左到右**流动，四条 stream 各占一行。以 block1 和 block0 为例（block2 的模式与 block1 相同）。箭头 `↓ wait` 表示下游 stream 等待上游 stream 完成。

```text
时间 ──────────────────────────────────────────────────────────────────────────────────────────→

                ┌─ copy-in ──────┐  ┌─ copy-in ──────┐
all-gather      │ copy 参数分片1 │  │ copy 参数分片0 │  ← 预取:在 block1 的 pre_backward 中发起
copy-in 流      │ → 输出缓冲1    │  │ → 输出缓冲0    │
                │ (参数组1)      │  │ (参数组0)      │
                └───────┬────────┘  └───────┬────────┘
                        │ wait               │ wait
                        ↓                    ↓
all-gather            ┌─ all-gather ──┐    ┌─ all-gather ──┐
流                    │ (输出缓冲1)    │    │ (输出缓冲0)    │
                      │ record         │    │ record         │
                      │ all-gather     │    │ all-gather     │
                      │ 完成事件1      │    │ 完成事件0      │
                      └───────┬────────┘    └───────┬────────┘
                              │ wait               │ wait
                              ↓                    ↓
计算流                      ┌─ copy-out ────┐  ┌─ block1 ──┐  ┌─ RS copy-in ─┐  ┌─ copy-out ────┐  ┌─ block0 ──┐  ┌─ RS copy-in ─┐
                            │ 输出缓冲1      │  │ 反向算子  │  │ 梯度1 →      │  │ 输出缓冲0      │  │ 反向算子  │  │ 梯度0 →      │
                            │ → 参数1        │  │           │  │ RS输入缓冲1  │  │ → 参数0        │  │           │  │ RS输入缓冲0  │
                            └────────────────┘  └───────────┘  └──────┬───────┘  └────────────────┘  └───────────┘  └──────┬───────┘
                                                                    │ wait                                    │ wait
                                                                    ↓                                         ↓
reduce-scatter                                                                            ┌─ reduce-scatter1 ─┐  ┌─ reduce-scatter0 ─┐
流                                                                                         │ (RS输入缓冲1)      │  │ (RS输入缓冲0)      │
                                                                                          │ record RS完成事件1 │  │ record RS完成事件0 │
                                                                                          └────────────────────┘  └────────────────────┘

         ◄──────── overlap ────────►
         all-gather(输出缓冲0) ∥ block1反向算子 ∥ reduce-scatter1
         三条独立流上的三路并行
```

### 3 个 block 的浓缩流水线图

```text
时间 ──────────────────────────────────────────────────────────────────────────────────────────→

all-gather      [copy-in2]  [copy-in1(预取)]              [copy-in0(预取)]
copy-in 流

all-gather      [all-gather2] [all-gather1(预取)]          [all-gather0(预取)]
流

计算流          [copy-out2][block2算子][RS-cin2]  [copy-out1][block1算子][RS-cin1]  [copy-out0][block0算子][RS-cin0]

reduce-scatter                          [reduce-scatter2]          [reduce-scatter1]          [reduce-scatter0]
流

                            ◄── overlap 1 ──►              ◄── overlap 2 ──►
                            all-gather1 ∥ block2算子 ∥ RS2  all-gather0 ∥ block1算子 ∥ RS1
```

---

## 四、图中发生的事情，逐步说明

以 2 个 block（block1、block0）为例，反向顺序：block1 → block0。

### 阶段 1：block1.pre_backward

1. **block1 的 unshard**（all-gather copy-in 流 + all-gather 流）：`foreach_all_gather` 在 all-gather copy-in 流上分配输出缓冲1，将参数分片1 拷贝到输出缓冲1 中本 rank 的 view 段。all-gather 流通过 `wait_stream(all_gather_copy_in_stream)`（`_fsdp_collectives.py:362`）等待 copy-in 完成后，执行 all-gather 集体通信，记录 all-gather 完成事件1。

2. **block1 的 wait_for_unshard**（计算流）：计算流通过 `wait_event(all-gather 完成事件1)` 等待 all-gather 完成后，执行 copy-out（输出缓冲1 → 参数1），记录 copy-out 完成事件1。由于 `training_state == PRE_BACKWARD`（不是 `FORWARD`），**走立即释放分支**（`_fsdp_param_group.py:451-452`）：调用 `_wait_all_gather_streams_on_event(copy-out 完成事件1)`，在 all-gather copy-in 流和 all-gather 流上排入 `wait(copy-out 完成事件1)`，然后释放输出缓冲1 的引用。

3. **block1 预取 block0**（all-gather copy-in 流 + all-gather 流）：`_backward_prefetch`（`_fsdp_param_group.py:690-692`）调用 `_prefetch_unshard(block0, "backward")`，将 block0 的 `training_state` 临时设为 `PRE_BACKWARD`，调用 `block0.unshard(False)`。由于 `async_op=False` 且 `training_state=PRE_BACKWARD`，`get_all_gather_streams` 返回独立流。在 all-gather copy-in 流上分配输出缓冲0（此时输出缓冲1 已释放，allocator **可能**复用其内存，但 all-gather copy-in 流上已排入 `wait(copy-out 完成事件1)`，所以 copy-in0 必须等 copy-out1 读完才能开始），将参数分片0 拷贝到输出缓冲0。all-gather 流执行 block0 的 all-gather，记录 all-gather 完成事件0。

### 阶段 2：block1 反向算子

4. **block1 反向算子**（计算流）：使用参数1 计算梯度。**此时 all-gather 流上 block0 的 all-gather 正在并行执行**——这就是 overlap。

### 阶段 3：block1.post_backward（reduce-scatter）

5. **RS copy-in**（计算流）：`foreach_reduce`（`_fsdp_collectives.py:604`）在计算流上将 block1 的反向梯度拷贝到 RS 输入缓冲1。

6. **reduce-scatter**（reduce-scatter 流）：`reduce_scatter_stream.wait_stream(current_stream)`（`_fsdp_collectives.py:608`）确保 RS 流等待 RS copy-in 完成后，执行 reduce-scatter 集体通信，记录 RS 完成事件1。**此时 all-gather 流上 block0 的 all-gather 仍在并行执行**——三路并行达成。

### 阶段 4：block0.pre_backward

7. **block0 的 unshard**：返回空操作（已在步骤 3 预取，`_fsdp_param_group.py:338-339` 的 early return）。

8. **block0 的 wait_for_unshard**（计算流）：计算流通过 `wait_event(all-gather 完成事件0)` 等待 block0 的 all-gather 完成后，执行 copy-out（输出缓冲0 → 参数0），记录 copy-out 完成事件0，立即释放输出缓冲0。block0 是最后一个反向模块，没有预取目标（`curr_index == 0`，`_fsdp_param_group.py:690` 的条件不满足）。

### 阶段 5-6：block0 反向算子 + post_backward

9. **block0 反向算子**（计算流）。

10. **block0 的 reduce-scatter**（reduce-scatter 流）。

### 末尾收尾

11. **`_root_post_backward_final_callback`**（`_fsdp_state.py:348-373`）：在最后一个 block 的 post_backward 之后，清空 `reduce_scatter_states`，等待所有未完成的 RS 事件。

---

## 五、overlap 在哪里？

反向预取实现了**三条独立流上的三路并行**：

```
all-gather 流:    block0 的 all-gather(输出缓冲0)    ← 预取
计算流:           block1 的反向算子                  ← 当前模块计算
reduce-scatter 流: block1 的 reduce-scatter(RS输入缓冲1) ← 当前模块梯度归约
```

三者分别使用**不同的流**和**不同的缓冲**（输出缓冲0 / 参数1 / RS 输入缓冲1），互不冲突，完全并行。

> 对比前向：前向的 overlap 是两路——copy-in(N+1)（all-gather copy-in 流）∥ all-gather(N)（all-gather 流），以及 all-gather(N+1)（all-gather 流）∥ blockN 算子（计算流）。反向多了 reduce-scatter 流，实现三路并行，overlap 程度更高。

---

## 六、为什么反向不需要延迟释放？

这是反向与前向最关键的区别。延迟释放（`all_gather_state`）的条件是（`_fsdp_param_group.py:441-444`）：

```python
if (
    not async_op
    and self._training_state == TrainingState.FORWARD
    and world_size > 1
):
    # 延迟释放
```

反向的 `training_state == PRE_BACKWARD`，不满足 `== FORWARD`，走 `else` 分支立即释放。根本原因在于**前向和反向的 overlap 对象不同，导致显存安全约束不同**：

### 前向：copy-in(N+1) ∥ all-gather(N) → 并发访问同类型缓冲 → 需延迟释放

前向的 overlap 是"下一个模块的 copy-in"与"当前模块的 all-gather"并行。两者都在 all-gather 相关流上，且都涉及 all-gather 输出缓冲。由于 all-gather 的 input 是 output 的 view（`_fsdp_collectives.py:270-272`），如果输出缓冲(N+1) 复用了输出缓冲(N) 的内存，copy-in(N+1) 正在写而 all-gather(N) 可能还没写完 → **数据竞争**。

前向的解法是延迟释放：持有输出缓冲(N) 的引用直到输出缓冲(N+1) 分配完毕，确保两者是**不同的物理内存**。

关键时序：前向中 copy-in(N+1) 在 `unshard(N+1)` 中发起，此时 `wait(copy-out 完成事件N)` 尚未排入 all-gather copy-in 流（它被延迟到 `wait_for_unshard(N+1)` 才排入），所以 copy-in(N+1) **不等** all-gather(N) → 并行 → 需要不同内存。

### 反向：copy-in(N-1) 在 all-gather(N) 之后 → 串行化 → 可复用内存

反向的 overlap 是"下一个反向模块的 all-gather"与"当前模块的反向算子 + reduce-scatter"并行。all-gather(N-1) 在 all-gather 流上，反向算子(N) 在计算流上，reduce-scatter(N) 在 reduce-scatter 流上——**三条流、三种缓冲**，互不冲突。

更关键的是，反向的预取（copy-in(N-1)）在 `wait_for_unshard(N)` **之后**发起（`pre_backward` 里先 `wait_for_unshard` 再 `_backward_prefetch`，`_fsdp_param_group.py:511-514`）。而 `wait_for_unshard(N)` 走立即释放分支，在 all-gather copy-in 流和 all-gather 流上排入 `wait(copy-out 完成事件N)`（`_fsdp_param_group.py:451-452`），**然后** copy-in(N-1) 才入队。因此：

- copy-out(N) 在 all-gather(N) 之后（`wait_event(all-gather 完成事件N)`）
- copy-in(N-1) 在 copy-out(N) 之后（`wait_event(copy-out 完成事件N)` 排在 copy-in(N-1) 前面）
- → copy-in(N-1) 在 all-gather(N) **之后**，不存在并发

即使 allocator 把输出缓冲(N) 的内存分给输出缓冲(N-1)，`wait(copy-out 完成事件N)` 也保证了 copy-in(N-1) 等 all-gather(N) + copy-out(N) 全部完成后才开始写 → **安全**，不需要延迟释放。

### 对比总结

| | 前向隐式预取 | 反向隐式预取 |
| ------ | ------ | ------ |
| `training_state` | `FORWARD` | `PRE_BACKWARD` |
| 用独立流？ | 是 | 是 |
| 预取目标 | 下一个前向模块 | 上一个前向模块（= 下一个反向模块） |
| 预取发起时机 | 下一个模块的 `pre_forward` | 当前模块的 `pre_backward`（`wait_for_unshard` 之后） |
| overlap 对象 | copy-in(N+1) ∥ all-gather(N) | all-gather(N-1) ∥ compute(N) ∥ reduce-scatter(N) |
| copy-in(预取) 与 all-gather(当前) 的关系 | **并发**（不同流，wait 延迟排入） | **串行**（wait 先排入，copy-in 后发起） |
| 延迟释放？ | **是**（`all_gather_state`） | **否**（立即释放 + `wait_event`） |
| 为什么？ | 并发访问同类型 all-gather 缓冲，需不同内存 | 串行化保证无并发，可安全复用内存 |

---

## 七、显式反向预取

`set_modules_to_backward_prefetch`（`_fully_shard.py:481-499`）允许用户手动指定预取目标：

```python
def set_modules_to_backward_prefetch(self, modules):
    """
    Sets the FSDP modules for which this FSDP module should explicitly
    prefetch all-gathers in backward. This overrides the default backward
    prefetching implementation that prefetches the next FSDP module based
    on the reverse post-forward order.
    """
```

显式与隐式的区别在于**预取目标的选择**，不在于流的使用：

- **隐式预取**（默认，`default_prefetch=True`）：`_backward_prefetch` 根据 `post_forward_order` 自动选择前一个前向模块作为预取目标。
- **显式预取**（`set_modules_to_backward_prefetch`，`default_prefetch=False`）：用户指定预取目标，`_pre_backward`（`_fsdp_state.py:340-344`）直接调用 `_prefetch_unshard`。

两者**都使用独立流**（默认 `unshard_async_op=False` 时），都走立即释放分支（`training_state=PRE_BACKWARD`）。显式预取的意义在于：当模型结构复杂（如跨层共享、跳跃连接）时，隐式预取的"前一个前向模块"可能不是最优预取目标，用户可以手动指定更合适的目标。

---

## 八、reduce-scatter 的流间状态管理

反向的 reduce-scatter 也有类似前向 all-gather 的"跨流引用"问题，但管理方式不同：

`foreach_reduce`（`_fsdp_collectives.py:616-618`）将 `(RS 输入缓冲, RS 完成事件)` 存入 `comm_ctx.reduce_scatter_states`：

```python
self.comm_ctx.reduce_scatter_states.append(
    ReduceScatterState(reduce_scatter_input, reduce_scatter_event)
)
```

`RS 输入缓冲`在 reduce-scatter 流上分配和使用，但 RS copy-in 在计算流上完成。持有引用是为了防止 allocator 在 reduce-scatter 完成前复用 RS 输入缓冲的内存。

清理发生在两个地方：
1. **最后一个参数组的 `post_backward`**（`_fsdp_param_group.py:553-561`）：等待所有 RS 事件后清空 `reduce_scatter_states`。
2. **`_root_post_backward_final_callback`**（`_fsdp_state.py:367-372`）：捕获最后一个模块的 RS 状态（没有后续模块来清理）。

---

## 九、关键代码索引

| 关注点 | 位置 |
| -------- | ------ |
| `pre_backward`：unshard + wait + 预取 | `_fsdp_param_group.py:505-514` |
| `_backward_prefetch`：隐式反向预取目标选择 | `_fsdp_param_group.py:657-692` |
| `_prefetch_unshard`：预取执行（设 PRE_BACKWARD + unshard） | `_fsdp_param_group.py:694-710` |
| `post_backward`：reduce-scatter 编排 | `_fsdp_param_group.py:517-628` |
| 立即释放（`else` 分支）：`_wait_all_gather_streams_on_event` | `_fsdp_param_group.py:451-452` |
| `_wait_all_gather_streams_on_event`：all-gather 流排 wait | `_fsdp_param_group.py:456-461` |
| `foreach_reduce`：RS copy-in + reduce-scatter + record event | `_fsdp_collectives.py:523-747` |
| RS copy-in 在计算流上 | `_fsdp_collectives.py:604` |
| reduce-scatter 流等计算流 | `_fsdp_collectives.py:608` |
| reduce-scatter 在 reduce-scatter 流上 + record event | `_fsdp_collectives.py:612-633` |
| 存 `reduce_scatter_states` | `_fsdp_param_group.py:616-618` |
| 最后一个参数组清理 RS states | `_fsdp_param_group.py:553-561` |
| 末尾 `_root_post_backward_final_callback` 清理 | `_fsdp_state.py:348-373` |
| `_pre_backward`：隐式/显式预取分发 | `_fsdp_state.py:334-345` |
| `set_modules_to_backward_prefetch`：显式预取 API | `_fully_shard.py:481-499` |

---

## 十、一句话总结

**反向隐式预取**在当前模块的 `pre_backward` 中（`wait_for_unshard` 之后）预取下一个反向模块的 all-gather，利用 all-gather 流、计算流、reduce-scatter 流三条独立流实现**三路并行**（all-gather ∥ 反向算子 ∥ reduce-scatter）；由于预取的 copy-in 在当前模块的 all-gather + copy-out 之后才发起（`wait_event` 串行化），不存在并发访问同类型缓冲的问题，因此**不需要延迟释放**，立即释放 + `wait_event` 即可保证显存安全——这是与前向隐式预取最本质的区别。
