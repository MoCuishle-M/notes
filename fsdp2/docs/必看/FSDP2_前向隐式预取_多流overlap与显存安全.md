# FSDP2：前向隐式预取的多流 overlap 与显存安全

> 配套文档：`FSDP2_Unshard的copy_in与copy_out实例详解.md`、`前向预取.md`、`FSDP2_fully_shard机制详解.md`
>
> 源码：
> - `_fsdp_param_group.py`：`FSDPCommContext.lazy_init`、`get_all_gather_streams`、`unshard` / `wait_for_unshard`、`AllGatherState`
> - `_fsdp_collectives.py`：`foreach_all_gather`（copy-in + NCCL）、`foreach_all_gather_copy_out`
>
> 本文解读 `_fsdp_param_group.py:57-66` 的 `[Note: Overlapping all-gather copy-in and all-gather]`，并用一个**具体例子 + 三流时序图**说明 overlap 如何达成、显存如何安全。

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

## 一、背景:三条 stream 与一次 unshard 的两步

### 1. 三条 GPU stream

`FSDPCommContext.lazy_init`(`_fsdp_param_group.py:72-93`)创建:

| 简写 | stream 字段 | 作用 |
| ------ | ------------ | ------ |
| **CI** | `all_gather_copy_in_stream` | 把本 rank 的分片 copy 到 AG 输入缓冲(高优先级) |
| **AG** | `all_gather_stream` | 跑 all-gather 集体通信(高优先级) |
| **CMP** | current/compute stream | copy-out(AG 输出→参数)+ 前向算子(matmul 等) |

另有 `reduce_scatter_stream` / `all_reduce_stream` 用于反向,本文不展开。

### 2. 关键:前向隐式预取走两条独立 stream

`get_all_gather_streams`(`_fsdp_param_group.py:102-112`):

```python
def get_all_gather_streams(self, async_op, training_state):
    if not async_op and training_state in (FORWARD, PRE_BACKWARD):
        # 隐式预取:copy-in 和 all-gather 分别用独立 stream,才能 overlap
        return self.all_gather_copy_in_stream, self.all_gather_stream
    current_stream = self.device_handle.current_stream()
    return current_stream, current_stream   # 显式预取/反向:同一条 stream
```

只有**前向 + `async_op=False`(默认)** 才用两条独立 stream,这是隐式预取 overlap 的前提。

### 3. 一次 unshard 在 stream 上的展开

`foreach_all_gather`(`_fsdp_collectives.py:336-370`)把 unshard 拆成两段:

```text
CI stream:  copy 分片 → out_N[本rank的view]      # 注意:input 是 output 的 view!
AG stream:  wait(CI); all-gather(out_N); 记录 AG_evt_N
```

随后 `wait_for_unshard`(`_fsdp_param_group.py:381-454`)在 CMP 上:

```text
CMP stream: wait(AG_evt_N); copy-out(out_N → params_N); 记录 copy_out_evt_N
```

**重点**:`_fsdp_collectives.py:354` 调用的 `torch.ops.fsdp.all_gather_copy_in(...)` 返回的 `all_gather_input` 不是新内存,而是 **all-gather output 缓冲里本 rank 那一段的 view**。这一点是后面显存安全的根源。

---

## 二、具体例子设定

- 模型:2 个 Transformer block,各自用 `fully_shard` 包了一层 → 两个参数组 **G0(block0)** 和 **G1(block1)**。
- `world_size = 2`,参数按 FSDP 切分。
- 前向、隐式预取(`async_op=False`,默认)。

程序(Python/CPU)顺序编排到 stream 上:

```text
block0.pre_forward → block0.forward → block1.pre_forward → block1.forward → ...
```

其中 `pre_forward`(`_fsdp_param_group.py:475-484`)内部依次调用 `unshard()` + `wait_for_unshard()`。

---

## 三、三流时序图(前向)

```text
        CI stream                AG stream                 CMP stream
        ─────────                ─────────                 ─────────
                                                         ┌─ block0.pre_forward:
t0  ┌─ G0 copy-in ──┐         │                        │  unshard(G0)
    │  shard0→out0  │         │                        │
t1  └────────────────┘        │                        │
    ┌─ G1 copy-in ──┐ ──┐     │ wait(CI)                │
    │  shard1→out1  │   │    ┌─ G0 all-gather(out0) ─┐  │   ← ★ G1 copy-in 与
t2  │ (新分配的     │   └───►│                       │  │     G0 AG 并行重叠
    │  out1≠out0!)  │        │                       │  │
t3  └────────────────┘        │                       │  │
                              └── AG_evt_0 ───────────┤─ wait(AG_evt_0)
                                                       │  ┌─ G0 copy-out ──┐
                              ┌─ G1 all-gather(out1)─┐ │  │ out0→params0   │
                              │ wait(CI)             │ │  └── copy_out_evt_0
t4                           │                      │ │  ┌─ block0 compute┐ │ ← ★ G1 AG 与
                              │                      │─┼─►│ matmul...      │ │   block0 compute
t5                           │                      │ │  │ (长算子)        │ │   并行重叠
                              └── AG_evt_1 ──────────┤ │  └────────────────┘ │
                                                       │  (此时 G0 的 out0 已被
                                                       │   all_gather_state 持有,
                                                       │   还没释放)
                                                       │
                              ┌─ (CI/AG).wait ────────┤─ block1.pre_forward:
                              │  (copy_out_evt_0)     │  unshard(G1) 已入队
                              │  → 现在安全释放 out0  │  → wait_for_unshard(G1):
                              └───────────────────────┤    ① 清掉 all_gather_state
                                                       │      (out0 真正被释放)
t6                                                     │  wait(AG_evt_1)
                                                       │  ┌─ G1 copy-out ──┐
                                                       │  │ out1→params1   │
                                                       │  └── copy_out_evt_1
                                                       │  ┌─ block1 compute┐
t7                                                     │  │ matmul...      │
                                                       │  └────────────────┘
                                                       │
                       前向结束                         │  最后一个 group 没有
                                                       │  "下一个"替它释放 →
                                                       │  由末尾 flush 清掉
                                                       │  all_gather_state(out1)
```

### overlap 在哪里?

1. **G1 copy-in(CI) ∥ G0 all-gather(AG)**:CI 和 AG 是两条独立 stream。G0 copy-in 一完成,CI 立刻接 G1 copy-in,AG 同时开始 G0 的 all-gather。→ 这就是 Note 说的 "overlap the next copy-in with the current all-gather"。
2. **G1 all-gather(AG) ∥ block0 compute(CMP)**:G0 copy-out 完成后 CMP 跑 block0 算子,AG stream 上 G1 的 all-gather 同时进行。→ `wait_for_unshard` docstring(`_fsdp_param_group.py:382-385`)说的 "overlap the current copy-out with the next all-gather"。

---

## 四、结合 Note 逐句说明

### ① "overlap the next copy-in with the current all-gather"

对应时序图 **t1–t2**:

- **current all-gather** = G0 的 AG(AG stream 上 `G0 all-gather(out0)`)
- **next copy-in** = G1 的 copy-in(CI stream 上 `G1 copy-in → out1`)

因为 CI 和 AG 是两条独立 stream,G0 的 copy-in 一完成,CI stream 立刻接 G1 的 copy-in,而 AG stream 同时开始 G0 的 all-gather。两者**并行**。这就是 `get_all_gather_streams` 在前向返回两条独立 stream 的目的(`_fsdp_param_group.py:105-110`)。

### ② "all-gather input as a view into the output"

`foreach_all_gather` 里 `_fsdp_collectives.py:354` 调用 `torch.ops.fsdp.all_gather_copy_in(...)`,返回的 `all_gather_input` 不是新内存,而是 **all-gather output 缓冲里本 rank 那一段的 view**。也就是说 copy-in 写入的内存 == AG 即将读/写的 output 缓冲本身。

### ③ "must copy into different memory from the current all-gather's output"

**危险场景**:如果 G0 的 `out0` 在 G1 copy-in 之前就被释放掉,那么 caching allocator 在 G1 的 `all_gather_comm.allocate(...)`(`_fsdp_collectives.py:351`)时,**很可能把同一块物理显存分给 `out1`**。于是:

- CI stream 上 G1 的 copy-in 正在写 `out1`(== `out0` 那块内存);
- AG stream 上 G0 的 all-gather 可能还没写完 `out0`。

→ 两条 stream 对同一块显存并发写,**数据竞争**,AG 结果损坏。

### ④ "keep a reference to the current all-gather's output … have the next group free it after its copy-in"

**解决办法**(`wait_for_unshard`,`_fsdp_param_group.py:441-454`):G0 的 copy-out 完成后**不立刻释放 `out0`**,而是把 `(AllGatherResult, copy_out_event)` 存到全局 `comm_ctx.all_gather_state`:

```python
# _fsdp_param_group.py:441-450
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

只要这个引用还在,allocator 就**不会把 `out0` 的内存分给 `out1`**,保证 `out1 ≠ out0` → 安全 overlap。

等到 G1 的 `wait_for_unshard`(`_fsdp_param_group.py:393-396`):

```python
if self._training_state == TrainingState.FORWARD:  # implicit prefetch
    if prev_all_gather_state := self.comm_ctx.all_gather_state:
        self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
        self.comm_ctx.all_gather_state = None  # free the all-gather result
```

**关键**:这个清除发生在 G1 的 `unshard()` **之后**(`pre_forward` 里先 `unshard` 再 `wait_for_unshard`,`_fsdp_param_group.py:481-482`)。也就是说,G1 的 copy-in 已经入队、`out1` 已经分配好之后,才丢掉 `out0` 引用——对应时序图 t5→t6。

同时 `_wait_all_gather_streams_on_event`(`_fsdp_param_group.py:456-461`)在 CI/AG stream 末尾排一个 `wait(copy_out_evt_0)`,保证之后任何复用 `out0` 内存的操作必须等 G0 copy-out 读完:

```python
def _wait_all_gather_streams_on_event(self, event):
    if hasattr(self.comm_ctx, "all_gather_copy_in_stream") and event is not None:
        self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
    if hasattr(self.comm_ctx, "all_gather_stream") and event is not None:
        self.comm_ctx.all_gather_stream.wait_event(event)
```

注意:这个 wait 是排在 G1 自己的 copy-in/AG **之后**的(CPU 入队顺序),所以**不会挡住**已经发起的 G1 copy-in,只约束后续对 `out0` 内存的复用。

### ⑤ "the last FSDP state flush the reference to avoid holding onto memory after forward"

G1 是最后一个 group,后面没有 G2 来替它清 `all_gather_state`。如果不 flush,`out1` 这块 unsharded 缓冲会一直挂到下一次前向才开始,白白占显存。所以前向收尾时由最后一个 state 把 `all_gather_state` 清掉(对应时序图末"末尾 flush")。

> 反向不走这条路:`get_all_gather_streams` 在反向返回 `(current_stream, current_stream)`,走 `else` 分支立即 `wait(copy_out_event)`,不延迟释放,也不写 `all_gather_state`。延迟释放仅在前向隐式预取时发生。

---

## 五、浓缩图:overlap 与显存生命周期

```text
            G0 copy-in   G0 AG          G0 copy-out   block0 compute
CI  stream: ████         ················              ·············
AG  stream: ·····  ████  (G0 AG)                       ·············
                               G1 copy-in  G1 AG ──────┐ (与 compute 重叠)
CI  stream:               ████                          │
AG  stream:                         ████  (G1 AG)      │
CMP stream:                                wait→ copy-out  block1 compute

out0 生命周期:  [分配]━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                                              ┃ 被 all_gather_state 持有
                                              ┗━━━┛ ← G1.wait_for_unshard 释放
out1 生命周期:        [分配]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                                                            ┗━ ← 末尾 flush 释放
```

**核心收益**:G1 的 copy-in 不必等 G0 的 AG/copy-out/block0 compute 完成,而是搭便车在 CI stream 上和它们并行跑;G0 的 output 缓冲通过"延迟一拍释放"既不挡 overlap、又不被复用导致写冲突;最后一个 group 自己收尾避免显存悬挂。

---

## 六、关键代码索引

| 关注点 | 位置 |
| -------- | ------ |
| 创建 CI/AG 两条独立 stream | `_fsdp_param_group.py:81-86`(`FSDPCommContext.lazy_init`) |
| 前向返回两条独立 stream | `_fsdp_param_group.py:102-112`(`get_all_gather_streams`) |
| copy-in 写入 output 的 view | `_fsdp_collectives.py:354`(`all_gather_copy_in`) |
| AG stream 等 CI stream | `_fsdp_collectives.py:362`(`all_gather_stream.wait_stream`) |
| copy-in → AG → copy-out 编排 | `_fsdp_param_group.py:337-454`(`unshard` / `wait_for_unshard`) |
| 延迟释放:存 all_gather_state | `_fsdp_param_group.py:441-450` |
| 下一个 group 释放上一个 out | `_fsdp_param_group.py:393-396` |
| CI/AG stream 排 wait(copy_out_evt) | `_fsdp_param_group.py:456-461`(`_wait_all_gather_streams_on_event`) |
| AllGatherState 定义 | `_fsdp_param_group.py:116-118` |

---

## 七、一句话总结

**前向隐式预取**用 CI/AG 两条独立 stream 让"下一个 group 的 copy-in"与"当前 group 的 all-gather"并行;但因为 all-gather 的 input 是 output 的 view,必须**持有当前 group 的 output 引用直到下一个 group 的 copy-in 已分配新内存**,由下一个 group 在 `wait_for_unshard` 里释放,最后一个 group 由末尾 flush 收尾——既拿到 overlap,又不泄漏显存、不产生写冲突。
