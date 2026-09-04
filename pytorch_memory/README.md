# PyTorch 内存管理模型与 Memory Snapshot 阅读指南

本文只从 PyTorch 自身的内存抽象和原生设备缓存分配器出发，重点解释 GPU/NPU 训练中经常看到的 `segment`、`block`、内存池、`allocated`、`reserved` 和碎片。

需要先说清范围：普通 CPU Tensor 最终使用的是 CPU allocator，CUDA Tensor 默认使用 CUDA caching allocator；`segment`、`block` 和 Memory Snapshot 主要描述后者这类**设备缓存分配器**。本文以 PyTorch 官方 CUDA 实现为准，不把某个模型的参数、激活、梯度或优化器策略当成“PyTorch 分配器规则”。文末再单独用给定的 `torch_npu` 快照说明这些概念如何迁移到 NPU。

本文核对资料的日期为 2026-09-04。分配阈值、配置项和实验 API 会随 PyTorch 版本变化；分析具体环境前，应先记录 `torch.__version__` 和 allocator backend。

## 1. 先记住一张层级图

```text
Python Tensor
    │  shape / stride / dtype / storage_offset
    │  多个 Tensor 可以共享同一个 Storage
    ▼
Storage
    │  持有 DataPtr，决定底层数据何时可以释放
    ▼
一次 allocator 请求：requested_size
    │  分配器按对齐规则向上舍入
    ▼
Block：分配器交给这个 Storage 使用的一段连续地址
    │  Block 可以 active，也可以 inactive 等待复用
    │  相邻 inactive Block 可以合并
    ▼
Segment：分配器向设备运行时申请并纳入缓存管理的一大片地址
    │  一个 Segment 可被拆成多个 Block
    │  普通模式下，不同 Segment 的空闲 Block 不能跨边界合并
    ▼
设备运行时 / driver
    │
    ▼
GPU/NPU 物理内存
```

内存池不处在这条链上的某一个额外地址层级。它更像是分配器给 Block/Segment 建的“分类货架”：分配请求只去符合设备、stream、尺寸类别和私有池身份的货架里找可复用空间。

最容易混淆的三句话是：

- Tensor 不必独占一块内存。切片、`view()`、很多 `reshape()` 只创建新的 Tensor 元数据并共享 Storage。
- 一次 Tensor 数据分配通常对应一个 active Block，但 Block 的 `size` 可能大于 Tensor 真正请求的 `requested_size`。
- Segment 不是 Tensor；它是分配器为了减少昂贵的设备分配调用而缓存的大块内存，里面可以同时住很多 Tensor 对应的 Block 和空闲 Block。

## 2. Tensor、Storage 与“是否真的分配了内存”

PyTorch 用户直接操作的是 Tensor。Tensor 记录 shape、stride、dtype、device 和 storage offset，真正的数据由 Storage 持有。

```python
x = torch.empty((1024, 1024), device="cuda", dtype=torch.float16)
y = x.view(512, 2048)       # 通常共享 x 的 Storage，不新增数据分配
z = x.clone()                # 新 Storage，通常触发新的 allocator 请求
w = x.t().contiguous()       # 为得到连续布局，通常触发新分配
```

因此看到源码中的 Tensor 对象数量，不能直接推导 Block 数量或显存量。分析 snapshot 时，悬停提示定位的是**底层分配发生的调用栈**，不一定能告诉你当前有哪些 Python 变量共享它。

一个 Storage 能否释放取决于是否仍被引用。引用可能来自普通 Python 变量、容器、autograd 为 backward 保存的 Tensor、梯度、优化器状态或异步算子。`del tensor` 只是减少一个引用，并不承诺这个 Storage 当场变成可复用状态。

## 3. 为什么 PyTorch 要做缓存分配

如果每次 `torch.empty(..., device="cuda")` 都直接调用 `cudaMalloc`，每次 Tensor 消失又直接 `cudaFree`，高频小分配会很贵，释放还可能引入设备同步。PyTorch 原生 caching allocator 的目标是：

1. 向设备运行时批量申请较大的 Segment；
2. 把 Segment 切成 Block 交给 Tensor/Storage；
3. Tensor 释放后，先把 Block 留在 PyTorch 缓存中；
4. 后续相似请求直接复用 Block，减少进入 driver 的次数；
5. 只有在 `empty_cache()`、内存压力或其它释放路径中，才把满足条件的缓存空间还给设备运行时。

这就是为什么 Tensor 已经删除后，`nvidia-smi` 中的进程显存可能不下降：内存已不再由 Tensor 占用，但仍由 PyTorch allocator 保留，以便本进程快速复用。

## 4. Segment：向设备申请并缓存的“大地块”

在经典 `backend:native`、非 expandable 模式下，一次底层 `cudaMalloc` 形成一个 Segment。Snapshot 中：

- `address`：Segment 起始地址；
- `total_size`：Segment 总大小；所有 Segment 的和就是 allocator 的 reserved memory；
- `stream`：这个 Segment/其中 Block 所属的 stream 语境；
- `segment_type`：来自 small pool 还是 large pool；
- `segment_pool_id`：默认池或某个私有内存池的身份；
- `blocks`：Segment 被切分后的布局；
- `allocated_size`：其中 `active_allocated` Block 的大小之和；
- `active_size`：`active_allocated` 与 `active_awaiting_free` Block 的大小之和。

这里的 `segment_type="small"` 表示它服务于 small-pool 请求，不表示整个 Segment 一定很小。特别是在 expandable 模式里，一个 small-pool Segment 可以持续长大到几十或几百 MiB。

### 4.1 普通 Segment 为什么会产生边界碎片

假设先后获得两个独立 Segment：

```text
Segment A: [16 MiB inactive]
Segment B: [16 MiB inactive]
```

总空闲量是 32 MiB，但它们来自两个独立的底层分配，不能拼成一个连续的 32 MiB Block。此时请求 32 MiB，allocator 仍可能不得不再申请一个 Segment。

### 4.2 Expandable Segment

`expandable_segments` 使用设备虚拟内存 API，先保留很大的连续虚拟地址空间，再按需映射物理页。它试图让同一 pool/stream 的分配在可增长的地址区间中排列，从而减少动态 batch 或动态序列长度造成的许多尾部碎片。

在 Snapshot 中常见：

- Segment 有 `is_expandable: true`；
- trace 中出现 `segment_map`/`segment_unmap`，表示物理页的映射和解除映射；
- 一个逻辑 Segment 的 `total_size` 可以很大；
- 仍然可能被存活 Block 截断，不能把两侧空洞跨过 active Block 合并。

Expandable Segment 改善的是部分分配形状导致的外部碎片，不会消灭引用未释放、跨 stream 等待、请求尺寸舍入等问题。

## 5. Block：真正参与复用、拆分和合并的单元

一个 Snapshot Block 的核心字段是：

| 字段 | 含义 |
|---|---|
| `address` | Block 的起始地址 |
| `requested_size` | 上层最初请求的字节数 |
| `size` | allocator 舍入、拆分后实际给这个请求的 Block 大小 |
| `state` | 当前是否在用、等待释放或已经可复用 |
| `frames` | 这次分配对应的调用栈 |

Block 有三种关键状态：

| 状态 | 含义 | 能否立即复用 |
|---|---|---|
| `active_allocated` | 仍被 Tensor/Storage 使用 | 不能 |
| `active_awaiting_free` | 上层已请求释放，但其它 stream 上的相关工作尚未完成 | 不能 |
| `inactive` | 当前没有活跃使用，已经回到缓存 | 可以，但仍受 pool、stream、尺寸和连续性约束 |

不要把 `inactive` 翻译成“泄漏”。它首先表示可复用缓存。只有结合请求形状、最大连续空闲块、OOM 和多轮趋势，才能判断缓存是否形成了有害碎片。

### 5.1 `requested_size` 与 `size`：内部碎片

设备 allocator 会为了对齐和复用向上舍入请求。例如给定快照中可以看到：

```text
requested_size = 4 bytes
block.size     = 512 bytes
```

多出来的 508 bytes 属于这个 active Block，别的请求不能使用。这类差额可视作内部舍入开销：

```text
rounding overhead = sum(active block.size - active block.requested_size)
```

它和 inactive Block 不是同一种“空闲”。前者被包在已分配 Block 内，后者已经回到 pool 等待复用。

### 5.2 拆分与合并：外部碎片

allocator 使用接近 best-fit 的策略：在合适的 pool 和 stream 中找“能容纳请求的最小空闲 Block”。找到更大的 Block 后，满足拆分条件时把它切成两部分：

```text
拆分前: [                 20 MiB inactive                 ]
拆分后: [6 MiB active][            14 MiB inactive        ]
```

6 MiB Block 释放后，如果它与 14 MiB Block 相邻、状态和所属关系允许，就能重新合并：

```text
合并后: [                 20 MiB inactive                 ]
```

如果中间还有一个 active Block，则两侧空闲块无法跨过去合并：

```text
[8 MiB inactive][1 MiB active][8 MiB inactive]
```

这里总共有 16 MiB 空闲，但没有 16 MiB 连续 Block。这才是阅读 allocator state 时最值得关注的外部碎片形态。

## 6. 内存池到底有几层含义

“内存池”在讨论中经常混指不同东西，最好拆开：

### 6.1 small pool 与 large pool

原生 allocator 把小请求和大请求分开放，避免大量小对象把适合大 Tensor 的空间切碎。按当前官方说明，约 1 MiB 是两类请求的边界；这个阈值和精确比较方式属于版本实现细节。

经典 native 默认策略中：

- 小于约 1 MiB 的请求进入 small pool，缺少缓存时通常申请 2 MiB 的 Segment/页再拆分；
- 约 1 MiB 到 10 MiB 的请求进入 large pool，缺少缓存时通常申请 20 MiB 再拆分；
- 更大的请求通常按 2 MiB 粒度向上扩展底层申请。

这些是理解布局的默认模型，不应该脱离 PyTorch 版本、backend 和 `PYTORCH_ALLOC_CONF` 配置当成永久常量。

### 6.2 默认池与私有池

`segment_pool_id` 描述 Segment 属于哪个 allocator pool。普通 eager 分配通常在默认池；CUDA Graph 为保证 replay 时地址稳定，会使用 graph-private pool。`torch.cuda.MemPool` 也可以让一个代码区域使用特定 allocator/pool。

不同私有池即使都有空闲空间，也不能假设可以任意相互借用。Snapshot 中必须先看 `segment_pool_id`，再比较空闲量。

### 6.3 stream 也是复用约束

原生 allocator 的 Block 与 stream 关联。同一 stream 有严格顺序，刚释放的地址可以安全地排在先前 kernel 之后复用。不同 stream 没有这种天然顺序。

当一个 Tensor 在创建 stream 之外被使用时，应让 allocator 知道这次使用，例如调用 `tensor.record_stream(user_stream)`。释放请求到来后，allocator 会等待相关 stream event 完成，期间 Block 显示为 `active_awaiting_free`。如果这种状态很多且持续很久，问题更像 stream 生命周期/同步，而不是普通缓存。

## 7. 一次分配请求怎样走完整条路径

以原生 caching allocator 的经典流程概括：

```text
Tensor/算子请求 N bytes
    │
    ├─ 1. 对 requested size 做对齐/舍入
    ├─ 2. 根据尺寸选择 small pool 或 large pool
    ├─ 3. 限定 device、stream、private pool 等身份
    ├─ 4. 在缓存中 best-fit 查找最小可用 Block
    │      ├─ 找到：必要时 split，返回 active Block
    │      └─ 找不到：向设备申请/映射新 Segment
    ├─ 5. 新 Segment 也可能被 split，余量留作 inactive Block
    └─ 6. 如果底层申请 OOM：尝试释放合适的缓存并重试，最终仍失败才抛 OOM
```

释放路径则是：

```text
Storage 最后一个相关引用消失
    │
    ├─ 没有跨 stream 未完成使用：active_allocated -> inactive
    └─ 仍有跨 stream 工作：active_allocated -> active_awaiting_free -> inactive

inactive Block
    ├─ 与同 Segment 中合适的相邻 inactive Block 合并
    └─ 留在缓存，供之后的相容请求复用
```

## 8. `allocated`、`active`、`reserved` 和设备监控值

建议把指标按层看：

| 指标 | 大致回答的问题 |
|---|---|
| `requested_bytes` | 上层调用实际想要多少字节 |
| `allocated_bytes` / `memory_allocated()` | 当前 Tensor 使用的 allocator Block 有多少 |
| `active_bytes` | allocated 加上等待其它 stream 完成释放的 Block 有多少 |
| `reserved_bytes` / `memory_reserved()` | PyTorch allocator 当前管理的全部 Segment 有多少 |
| `device_memory_used()` 或 `nvidia-smi` | 整个进程/设备实际使用量，包括上下文及不经过 PyTorch allocator 的分配 |

在 native allocator 的典型快照中，可用下面的关系建立直觉：

```text
requested bytes <= allocated bytes <= active bytes <= reserved bytes
```

设备监控值通常还会更大，因为 cuBLAS、cuDNN、NCCL、设备上下文、第三方扩展或直接调用设备 API 的内存不一定进入 PyTorch Snapshot。反过来，不能只凭这几个数的差值就断言泄漏；必须确认采样边界和 backend。

最实用的现场采样代码：

```python
import torch

print("backend:", torch.cuda.get_allocator_backend())
print("allocated:", torch.cuda.memory_allocated())
print("reserved:", torch.cuda.memory_reserved())
print(torch.cuda.memory_summary())

stats = torch.cuda.memory_stats()
for key in (
    "allocated_bytes.all.current",
    "active_bytes.all.current",
    "reserved_bytes.all.current",
    "inactive_split_bytes.all.current",
    "num_alloc_retries",
    "num_ooms",
):
    print(key, stats.get(key))
```

使用 `backend:cudaMallocAsync` 时，一部分 native allocator 统计没有意义或固定为零，不能照搬 native 指标解释。

## 9. 碎片怎么判断，`empty_cache()` 又做什么

### 9.1 两类碎片要分开

- 内部舍入开销：active Block 的 `size - requested_size`；
- 外部碎片：总 inactive 很多，但被 Segment 边界或 active Block 切成许多不能满足目标请求的小块。

判断外部碎片至少要同时看：

1. OOM 想申请的 `size`；
2. 当时全部 inactive bytes；
3. 最大的相容、连续 inactive Block；
4. 这些 Block 是否在相同 pool/stream；
5. 是否被 active Block 隔开；
6. 是否存在大量 `inactive_split_bytes`；
7. allocator 是普通 Segment 还是 expandable Segment。

“reserved 远大于 allocated”只是调查入口，不是碎片结论。如果空闲空间是一整块很大的连续尾部，它通常很好复用；如果被切成几百个小洞，才更可疑。

### 9.2 `empty_cache()` 的边界

`torch.cuda.empty_cache()` 会把 allocator 当前能够释放的未占用缓存还给设备运行时，使其它进程可用，也可能改变后续布局。它不会：

- 释放仍被 Tensor/Storage 引用的 active Block；
- 神奇地增加当前 PyTorch 进程可用于 active Tensor 的理论上限；
- 跨过 active Block 合并两侧空洞；
- 让 Snapshot 看见 NCCL 或第三方直接分配的内存。

普通非 expandable Segment 通常需要整个 Segment 都 inactive 才能返还。Expandable Segment 还可能按物理页解除映射，具体行为要看当前 backend 和版本。

生产优化时，不要把每步调用 `empty_cache()` 当成默认解法。它可能破坏稳态缓存并增加设备分配开销。应先用 Snapshot 证明问题来自缓存布局，再决定是否调整分配形状、对象生命周期或 allocator 配置。

## 10. 怎样采集一个可读的 Memory Snapshot

### 10.1 推荐采集代码

```python
import torch

torch.cuda.memory._record_memory_history(
    enabled="all",
    context="all",
    stacks="all",
    max_entries=100_000,
)

try:
    run_the_region_you_want_to_debug()
except torch.cuda.OutOfMemoryError:
    torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")
    raise
else:
    torch.cuda.memory._dump_snapshot("normal_snapshot.pickle")
finally:
    torch.cuda.memory._record_memory_history(enabled=None)
```

采集时注意：

- 在目标区间前开启，而不是程序启动后很久才开启；否则早期存量只能以当前状态出现，缺少完整出生历史。
- `max_entries` 是环形历史上限。运行很久且事件很多时，旧事件会被覆盖。
- 每条事件及调用栈会增加 Host 内存和 snapshot 文件体积；官方说明长 trace 很容易达到 GB 级。
- 若只记录 `enabled="state"`，能看到最终存活分配及调用栈，但没有完整 alloc/free 时间线。
- 当前 PyTorch main 还能选择记录 pinned Host memory，但在线 `memory_viz` 暂不能渲染 Host 部分；旧版本也可能没有对应参数。
- `_record_memory_history`、`_dump_snapshot` 以下划线开头，属于实验/内部 API，升级 PyTorch 后应重新核对签名。

### 10.2 采样边界比“多采一点”更重要

分析训练时建议至少分别采：

```text
模型构造完成
第一次 forward 前
forward 峰值附近
backward 后
第一次 optimizer.step 后
第二个稳定 step 后
```

第一次 optimizer step 往往才创建 optimizer state，因此 step 0 和稳态 step 的基线不同。只看单个终点 snapshot，很难区分“常驻基线”“本轮临时峰值”和“逐轮增长”。

## 11. 怎样打开与阅读 Snapshot

### 11.1 最直接：官方交互式查看器

1. 打开 [PyTorch Memory Visualizer](https://pytorch.org/memory_viz)。
2. 把 `.pickle` 文件拖入页面。
3. 先选择正确 device。
4. 在 `Active Memory Timeline` 和 `Allocator State History` 之间切换。

官方说明这个查看器是浏览器本地 JavaScript 应用，拖入的 snapshot 不会上传。不过 snapshot 含本地源码路径和调用栈；在有额外保密要求的环境里，仍应使用离线页面或本地脚本。

如果文件很大，先把 detail 调低，再放大目标峰值；否则浏览器一次画出几十万条分配会很慢。

### 11.2 Active Memory Timeline 看什么

它回答的是：“随着分配事件推进，哪些 Block 还活着，它们共同构成了怎样的峰值？”

建议按下面顺序：

1. 看是否有规则重复的 forward 上升、backward 回落；
2. 比较每轮结束后的基线是否稳定；
3. 找全局峰值和 OOM 前最后一个峰；
4. 放大峰顶，悬停最大的色块；
5. 从调用栈中先找最靠近业务源码的 Python frame，再向内核/ATen frame 追；
6. 记录地址标签，跨视图搜索同一个 Block。

地址标签类似 `b7f064c000000_0`：前半段是 Block 地址，`_0` 表示该地址第 0 次被分配。同一地址会被缓存反复复用，因此地址相同不等于同一个 Tensor 生命周期。

时间线主要表达**事件顺序和活跃区间**。旧格式不一定携带可靠时间戳，不要直接把横向宽度当成 kernel 耗时；性能耗时应结合 PyTorch Profiler。

### 11.3 Allocator State History 看什么

它回答的是：“选中某个事件时，reserved memory 被哪些 Segment 占有，每个 Segment 又被切成了哪些 active/inactive Block？”

阅读步骤：

1. 左侧先选峰值、`oom`、`segment_alloc` 或可疑的 `free` 事件；
2. 右侧看 Segment 总数、大小、pool id、stream 和 expandable 状态；
3. 在每个 Segment 内看 active 与 inactive 的交错；
4. 比较 OOM 请求大小与最大连续空闲 Block；
5. 悬停 Block 看 `size`、`requested_size` 和分配调用栈；
6. 悬停事件看 alloc/free/OOM 发生位置；
7. 用地址标签回到 Active Memory Timeline，确认它跨越了哪些阶段。

典型布局的含义：

| 看到的模式 | 优先怀疑 |
|---|---|
| 每轮峰值相似、基线稳定 | 正常稳态缓存 |
| 每轮结束基线单调上升 | 长生命周期引用、容器累积、autograd graph 或 optimizer state 初始化尚未结束 |
| inactive 很多且被 active 小块密集隔开 | 外部碎片或长尾存活 Block 钉住 Segment |
| 大量 `active_awaiting_free` | 跨 stream 使用、`record_stream` 或异步工作未完成 |
| 频繁 `segment_alloc`/`segment_free` | 未达到缓存稳态、形状抖动或频繁 `empty_cache()` |
| OOM 时 reserved 很大但最大空闲块小于请求 | 碎片是强嫌疑，但仍需检查 pool/stream |
| 设备监控远高于 Snapshot reserved | NCCL、设备上下文、第三方扩展或其它非 PyTorch allocator 分配 |
| 大色块悬停只有 C++/扩展栈 | Python 栈未记录、扩展直接分配，或调用栈在算子内部丢失 |

## 12. 给定快照的实战判读

### 12.1 实际文件

用户给出的路径按各段分开的写法不存在；在本机定位到的实际目录是：

```text
C:\Users\zhangxubin\PycharmProjects\Profile\layers12_mbs4_coco_910B3-memory_snapshot-step-0-3
```

WSL 路径和具体文件为：

```text
/mnt/c/Users/zhangxubin/PycharmProjects/Profile/layers12_mbs4_coco_910B3-memory_snapshot-step-0-3/snapshot_2026-09-03-03-44_0.pickle
```

文件大小为 11,669,389 bytes，顶层只有 `segments` 与 `device_traces`。调用栈包含 `libtorch_npu.so` 和 `torch_npu`，所以它是 Ascend NPU allocator 生成的 PyTorch 风格快照，不是 CUDA 原生快照。

以下结论只使用通用字段；`workspace_snapshot` 是这份 NPU trace 的额外事件，不能拿上游 CUDA 文档生搬硬套。

### 12.2 终点 allocator 状态

| 项目 | 实际值 |
|---|---:|
| device | 0 |
| Segment 数 | 10 |
| large / small Segment | 6 / 4 |
| expandable Segment | 8 |
| 不同 stream | 4 |
| `segment_pool_id` | 全部为 `(0, 0)` |
| reserved，即 `sum(total_size)` | 33.943 GiB |
| `active_allocated` | 15.638 GiB，1027 Blocks |
| `active_awaiting_free` | 0 |
| inactive | 18.305 GiB，216 Blocks |
| active Block 内部舍入差额 | 0.488 MiB，约占 active 的 0.003% |
| 完全 inactive 的 Segment | 8 个，共 7.615 GiB |

第一眼会看到 inactive 占 reserved 的 53.93%，但这**不能直接判成严重碎片**。最主要的 large expandable Segment 是：

```text
total          26.230 GiB
active         15.557 GiB
inactive       10.673 GiB
largest free   10.667 GiB
```

也就是说，这个 Segment 内几乎所有 inactive 空间都集中成一个约 10.667 GiB 的连续大块，而不是碎成许多小洞。small expandable Segment 也类似：总计 100 MiB，active 83.178 MiB，最大连续空闲块 16.764 MiB。

所以当前终点更准确的表述是：

> allocator 保留了较多可复用缓存，其中相当一部分是完整空闲 Segment 或很大的连续空闲尾部；单凭终点布局，没有看到“空闲很多但没有大块”的典型严重碎片证据。

8 个完全 inactive Segment 是缓存释放候选；部分 active 的 expandable Segment 中空闲物理页能否被解除映射，要以当前 `torch_npu` 版本行为为准，不能直接套用 CUDA `empty_cache()` 结论。

### 12.3 事件历史

`device_traces` 有 8 个 device 槽位，只有 device 0 有事件：

| action | 数量 |
|---|---:|
| `alloc` | 17,989 |
| `free_requested` | 16,960 |
| `free_completed` | 16,960 |
| `segment_map` | 153 |
| `segment_alloc` | 2 |
| `workspace_snapshot` | 2 |
| `oom` | 0 |

这段历史没有 OOM，也没有积压的 `active_awaiting_free`。`free_requested` 与 `free_completed` 数量相同，说明在采样终点前，已经请求释放的普通 Block 都完成了释放状态迁移。

按 trace 的请求大小重放，峰值出现在事件索引 34,993 附近（从 0 计数），约为 28.131 GiB。这里包含开始处两个 NPU `workspace_snapshot` 对应的 2 MiB 和 1008 MiB 合成分配；这些初始 workspace 没有普通 free 事件，终点 trace 重放值会比 `segments` 的当前 active 真值高约 0.986 GiB。因此：

- 找峰和调用栈时可以使用这个事件位置；
- 判断终点占用时应以 `segments` 字段的 15.638 GiB 为准；
- 不应把 NPU 的 `workspace_snapshot` 当成上游 PyTorch 标准 action。

### 12.4 峰值和终点是谁占的

按每个 Block 调用栈中最靠近 MindSpeed-MM 的 Python 分配 frame 聚合，峰值附近最大的来源是：

| 分配位置 | 峰值附近仍活跃的请求量 |
|---|---:|
| `mindspeed_mm/fsdp/optimizer/optimizer.py:89 step` | 10.423 GiB |
| `mindspeed_mm/fsdp/utils/utils.py:69 _replace_tensor` | 5.211 GiB |
| 只有 `libtorch_npu.so` 栈、无法归到 Python 源码 | 4.736 GiB |
| `chunkloss.py:220 calculate_lm_loss` | 1.895 GiB |
| `chunkloss.py:127 fixed_cross_entropy` | 1.895 GiB |
| `fully_shard.py:440 _pre_forward` | 0.947 GiB |
| `fully_shard.py:581 param_group_wait_for_unshard_pt29` | 0.947 GiB |
| `chunkloss.py:53 forward` | 0.947 GiB |

到 snapshot 终点，主要存活来源变成：

| 分配位置 | 终点 active Block 总量 |
|---|---:|
| `optimizer.py:89 step` | 10.423 GiB |
| `utils.py:69 _replace_tensor` | 5.211 GiB |
| `flash_attn.py:36 get_attn_mask` | 4 MiB |

这个差异很有信息量：峰值时 ChunkLoss、FSDP unshard 和 NPU 内核临时量同时存在；终点留下的主要是 optimizer step 创建的长期状态和模型构造/替换时创建的长期 Tensor。它更像“约 15.6 GiB 常驻基线 + 约 12.5 GiB 临时峰值”，不是这些临时量在终点全部泄漏。

但调用栈聚合有边界：它按**分配发生的位置**归因，不等价于 Tensor 的语义类别。例如 `optimizer.py:89` 很可能对应 Adam 状态初始化，但要确认具体是参数、梯度还是 optimizer state，仍需把悬停地址、dtype/shape 日志和源码结合起来。

### 12.5 在官方 viewer 中具体点哪里

把这份 pickle 拖入 viewer 后，建议：

1. 选 device 0，其它 device trace 是空的；
2. 在 Active Memory Timeline 中跳到事件索引约 34,993 的位置；
3. 依次悬停约 10.4 GiB、5.2 GiB 的长期色带和峰顶新出现的大块；
4. 搜索调用栈中的 `optimizer.py:89`、`utils.py:69`、`chunkloss.py:220`；
5. 切到 Allocator State History，看 26.230 GiB large expandable Segment；
6. 确认其约 10.667 GiB 空闲是否显示为连续尾部；
7. 再看峰值位置上临时 Block 如何把这个 Segment 切开；
8. 如果 upstream viewer 不认识 `workspace_snapshot`，改用 `torch_npu` 环境随包提供的 viewer，或用下一节脚本检查原始字典。

这份快照没有 `oom` 事件，所以它适合研究峰值组成和缓存布局，不足以回答“某次 OOM 为什么失败”。要回答 OOM，需要在真正抛异常时 dump，并保留 `device_free`、失败请求 `size` 以及当时 allocator state。

## 13. 不依赖图片，怎样程序化检查 pickle

pickle 反序列化可以执行代码。只对自己生成或确认可信的 snapshot 使用 `pickle.load()`，不要加载来源不明的文件。

下面的脚本只汇总最终 Segment/Block 状态：

```python
import collections
import pickle

MiB = 1024 ** 2
GiB = 1024 ** 3
path = "snapshot.pickle"

with open(path, "rb") as f:
    snapshot = pickle.load(f)

segments = snapshot["segments"]
state_bytes = collections.Counter()
state_blocks = collections.Counter()

for segment in segments:
    for block in segment["blocks"]:
        state_bytes[block["state"]] += block["size"]
        state_blocks[block["state"]] += 1

reserved = sum(segment["total_size"] for segment in segments)
allocated = sum(segment["allocated_size"] for segment in segments)
active = sum(segment["active_size"] for segment in segments)

print("segments:", len(segments))
print("reserved GiB:", reserved / GiB)
print("allocated GiB:", allocated / GiB)
print("active GiB:", active / GiB)

for state in sorted(state_bytes):
    print(state, state_blocks[state], state_bytes[state] / GiB, "GiB")

for index, segment in enumerate(segments):
    inactive = [
        block["size"]
        for block in segment["blocks"]
        if block["state"] == "inactive"
    ]
    print(
        index,
        segment["segment_type"],
        "expandable=", segment.get("is_expandable"),
        "total=", segment["total_size"] / GiB, "GiB",
        "largest_free=", max(inactive, default=0) / GiB, "GiB",
    )
```

找终点最大的 active Block 及调用栈：

```python
active_blocks = []

for segment in segments:
    for block in segment["blocks"]:
        if block["state"] == "active_allocated":
            active_blocks.append(block)

for block in sorted(active_blocks, key=lambda item: item["size"], reverse=True)[:20]:
    print(
        "address=", hex(block["address"]),
        "size MiB=", block["size"] / MiB,
        "requested MiB=", block["requested_size"] / MiB,
    )
    for frame in block.get("frames", []):
        if frame.get("filename", "").endswith(".py"):
            print("  ", frame["filename"], frame["line"], frame["name"])
```

这类脚本适合验证 viewer 中的直觉，但不要自行把所有 `alloc - free` 相加就当作最终真值。环形历史可能丢掉早期事件，扩展 backend 也可能插入合成事件；最终状态优先使用 `segments`，历史用于解释过程。

## 14. 一套可复用的 Snapshot 排查清单

```text
[ ] 记录 torch 版本、设备 backend、allocator backend 和分配器配置
[ ] 确认 snapshot 从目标区间之前就开始记录
[ ] 选择正确 device / rank
[ ] 判断历史是否被 max_entries 截断
[ ] 先看终点 allocated / active / reserved
[ ] 再看全局峰值、稳定步基线和 OOM 事件
[ ] 区分 active_allocated、active_awaiting_free、inactive
[ ] 比较 requested_size 与 block.size，判断内部舍入开销
[ ] 比较总 inactive 与最大连续相容 inactive Block
[ ] 检查 Segment 边界、pool id、stream 和 is_expandable
[ ] 用地址标签把 Timeline 与 Allocator State History 对上
[ ] 调用栈先找最靠近业务代码的 Python frame
[ ] 将 Snapshot reserved 与设备总占用比较，识别不可见内存
[ ] 对可疑结论再采一份下一稳定 step 的 snapshot 做趋势验证
```

## 15. 推荐阅读与可直接打开的示例

下面按“先会看图，再懂实现”的顺序排列：

1. [PyTorch 官方：Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)：snapshot 采集、两个 viewer 视图、地址标签、不可见内存和数据结构定义。
2. [PyTorch Memory Visualizer](https://pytorch.org/memory_viz)：直接拖入可信的 `.pickle` 文件。
3. [PyTorch 官方博客：Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/)：用实际时间线讲正常训练、梯度未清理和 optimizer state；文中有可交互 snapshot 示例。
4. [PyTorch 官方博客：Understanding GPU Memory 2](https://pytorch.org/blog/understanding-gpu-memory-2/)：用 snapshot 定位 Python 引用环导致的周期性显存增长。
5. [PyTorch CUDA Semantics：Memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)：`allocated`、`reserved`、`empty_cache()`、allocator backend 和 `PYTORCH_ALLOC_CONF`。
6. [PyTorch `memory_stats()` 官方文档](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.memory_stats.html)：所有核心统计量和 `inactive_split_bytes` 的正式定义。
7. [PyTorch allocator 源码 `CUDACachingAllocator.cpp`](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)：best-fit、small/large pool、拆分、stream、OOM 重试和 expandable segment 的最终依据。
8. [PyTorch DevLog：When does fragmentation occur in the CUDA caching allocator?](https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/)：用具体 Block 布局演示 Segment 边界、分配顺序与 expandable segment。
9. [Zach DeVito：A guide to PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)：非常易读的伪代码解释。它描述的是 2022 年实现，适合建立直觉，具体配置名与当前行为要回到官方文档/当前源码核对。

## 16. 最后压缩成六句话

1. Tensor 是视图和元数据，Storage 才持有底层数据；多个 Tensor 可以共享一个 Storage。
2. Storage 的设备分配请求被 allocator 舍入成 Block，`requested_size` 不一定等于 `size`。
3. Segment 是 allocator 向设备运行时获得并缓存的大块空间，一个 Segment 会被拆成多个 Block。
4. Pool、stream 和 private-pool id 决定空闲 Block 是否真的能服务某个新请求。
5. `inactive` 首先是可复用缓存；只有“总空闲很多、最大相容连续块却小于请求”才像有害碎片。
6. Timeline 用来找“谁在什么时候活着”，Allocator State History 用来解释“当时 Segment 被怎样切开”；两个视图必须通过地址和调用栈一起读。
