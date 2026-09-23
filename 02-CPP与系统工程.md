# 02 · C++ 与系统工程（必背）

> 本文所有答案都对应 `~/inferserving` 里你自己写的代码。
> 面试官会顺着简历往下挖实现细节，这些是最容易被挖到的点。

---

## 一、动态 Batch 引擎（`infer_serving/src/batch_engine.hpp`）

### Q1. 为什么用无锁队列？和加锁队列差在哪？

> 场景是多个网络 IO 线程往队列里塞请求、多个推理 worker 线程批量取，
> 典型的 MPMC。用 mutex + condition_variable 的话，高 QPS 下所有生产者争同一把锁，
> 锁竞争会成为瓶颈，而且被唤醒的线程要做上下文切换，延迟抖动大。
>
> 我用的是 moodycamel 的 `BlockingConcurrentQueue`，它内部给每个生产者分配独立的
> sub-queue，生产侧基本无竞争；消费侧支持 `wait_dequeue_bulk` **批量出队**，
> 一次系统调用拿一批，这对我的凑批场景正好——我本来就要成批处理，
> 批量出队把"取 N 次"压成"取 1 次"。
>
> 代价是内存占用更高（每个生产者一套结构）、不保证严格 FIFO、
> 而且调试比加锁队列难得多。所以不是无脑用无锁，是这个场景刚好匹配。

**追问：无锁队列是不是就没有等待了？**
> 不是。无锁保证的是"不会因为某个线程挂起导致整体阻塞"，不是"没有开销"。
> 高竞争下 CAS 重试一样耗 CPU。而且我用的是 Blocking 版本，
> 队列空时消费者仍然会挂起等待——那部分还是有信号量的。

---

### Q2. 你的凑批策略具体是什么？为什么要分段续取？

**照着代码说**：
```
1. wait_dequeue_bulk(tasks, max_batch_size)          // 先尽量取满
2. while (count < min_batch_size && elapsed < qtimeout)
       wait_dequeue_bulk_timed(..., max-count, qtimeout/3)   // 不足则分段续取
3. 交给 worker 推理
```

> 第一步尽量一次取满，流量高时直接就够了，零额外延迟。
> 不够 `min_batch_size` 才进等待，而且**不是一次等满 qtimeout，是按 qtimeout/3 分三段**。
>
> 分段的意义是：如果中途请求来齐了，第二段的 timed 调用会立刻返回，批马上就能走，
> 不用白等到超时。一次等满的话，即使第 1ms 就凑齐了，也要傻等到 10ms。
> 这是延迟和吞吐之间一个很便宜的优化。

**追问：`max_batch_size` 和 `qtimeout` 怎么定的？**
> 压测定的。固定输入分布，扫 batch size 看吞吐拐点——
> 到某个点后 GPU 已经打满，再加 batch 只涨延迟不涨吞吐，那个点就是上限。
> `qtimeout` 按业务的延迟预算倒推：总预算减去推理耗时和网络开销，剩下的才是能等的时间。
> 这两个参数在我们框架里是每个模型独立配置的，因为模型的推理耗时差异很大。

**追问：低流量时是不是每个请求都要白等 qtimeout？**⚠️ 这是个真实的弱点，别硬撑
> 会。这是这套静态凑批的固有缺陷——低峰期 batch 凑不满，请求要等超时才走。
> 缓解办法是把 `min_batch_size` 配成 1，退化成来一个算一个；
> 更好的做法是按队列长度自适应调整 timeout，我们当时没做到这一步。

---

### Q3. 代码里 `running_` 用了 `memory_order_acquire/release`，为什么不用默认的 seq_cst？

> `running_` 是个停止标志：`start()` 里 release 写入，worker 循环里 acquire 读。
> acquire/release 保证的是"release 之前的所有写，对看到这个 release 的 acquire 之后可见"——
> 对标志位场景这就够了，它需要的只是单个变量的可见性和它前后的顺序。
>
> seq_cst 会额外要求**全局单一修改顺序**，在 x86 上表现为写入要加
> `mfence`（或 `xchg`），有实际开销。这个标志在 worker 的热循环里每轮都读一次，
> 省下来是值得的。

⚠️ **这题容易被反杀**。如果面试官追问"你确定你的代码里没有别的地方依赖 seq_cst"，
诚实回答：
> 这个标志是独立的，不和其它原子变量组合使用，所以 acquire/release 足够。
> 说实话在 x86 上这两者的读侧生成的代码是一样的，主要差别在写侧；
> 我这么写更多是表达意图，真正的性能收益在 x86 上有限，在 ARM 上才明显。

---

## 二、自研 Tensor（`infer_common/src/refcount.h`、`tensor.h`）

### Q4. 为什么自己写引用计数，不直接用 `shared_ptr`？

> 三个原因：
> 1. **对象头开销**：`shared_ptr` 是两个指针（对象 + 控制块）16 字节，
>    还有一次额外的控制块分配（除非 make_shared）。侵入式引用计数把计数放进对象本身，
>    传递时就是一个裸指针 8 字节，对要频繁在插件和框架之间传递的 Tensor 更合适。
> 2. **跨边界传递**：Tensor 要在 C++ 框架、插件 .so、Python worker 之间流转，
>    裸指针 + 显式 Ref/Unref 的语义在 ABI 边界上更可控。
> 3. 这套实现思路参考了 TensorFlow 的 `core/lib/core/refcount.h`，
>    是被验证过的成熟做法。
>
> 代价很明确：**手动管理，漏 Unref 就泄漏，多 Unref 就 double free**。
> 所以我们在外层封了 RAII 的 Tensor 包装类，业务侧不直接碰 Ref/Unref。

**⚠️ 深度追问：`Unref()` 里的快速路径线程安全吗？**
```cpp
if (RefCountIsOne() || ref_.fetch_sub(1) == 1) { delete this; }
```
这题很刁，但你要能答：
> 快速路径是：如果计数已经是 1，说明当前线程是**唯一持有者**，
> 不可能有别的线程再来 Ref（没有别的线程持有这个指针，也就没法增加计数），
> 所以可以跳过原子减直接删。这是 TensorFlow 原版的优化。
>
> 前提是**调用方必须保证不存在"弱引用突然升级"的场景**。
> 如果需要类似 weak_ptr 的语义，这个快速路径就是错的，必须老老实实走 fetch_sub。
> 我们的使用场景是纯强引用，没有这个问题。
>
> 另外 `Ref()` 用 relaxed 是对的——增加引用时调用方本来就持有有效引用，
> 不需要同步任何数据。而 `Unref()` 的递减必须至少是 release，
> 并在真正 delete 前有 acquire 语义，否则可能在其它线程的写还没可见时就析构了。

**这是个加分回答**：你能指出"Ref 用 relaxed 是安全的，Unref 的递减需要 release +
delete 前需要 acquire"，说明你真懂 memory order，不是背的。

---

### Q5. Allocator 为什么要抽象一层？

> 为了后续能替换分配策略而不改调用方。当前默认实现是普通的堆分配，
> 但接口留出了空间：GPU 侧可以换成 pinned memory（锁页内存能让 H2D/D2H 走 DMA，
> 带宽更高且能异步），高频小张量可以换成内存池避免反复 malloc。
>
> ⚠️ 实话是我们目前只用了默认实现，这层抽象是为了以后。

---

## 三、Python 插件（`infer_interface/plugin/py_infer_connector.*`）⭐ 高频深挖

### Q6. 为什么用独立进程 + Protobuf，而不是 pybind11 把 Python 嵌进来？

**短答**：为了隔离 GIL 和崩溃。

**展开**：
> pybind11 嵌入式方案里，Python 解释器活在 C++ 服务进程内，
> 所有调进 Python 的线程都要抢**同一把 GIL**。我的推理服务是多 worker 线程的，
> 前后处理一旦走 Python，这些线程在 Python 段就完全串行了——
> C++ 这边辛辛苦苦做的多线程并发，到插件这里被 GIL 掐成单线程。
>
> 第二个原因更现实：**插件是算法同学写的**。Python 代码抛异常、死循环、
> 甚至 numpy 的段错误，在嵌入式方案里会直接带崩整个推理服务。
> 独立进程的话，worker 挂了只影响它自己，主服务能检测到并重启它。
>
> 代价是多了一次序列化和 IPC 往返。所以我们用 Protobuf 序列化 Tensor
> （`infer_plugin.proto` 里的 `TensorProto`：dtype + dims + bytes），
> 走 Unix domain socket，这个开销相对模型推理本身是可以接受的。

### Q7. 具体怎么实现的？

**完整讲一遍，这段细节很能体现工程能力**：

> 1. **建立通道**：`socketpair(AF_UNIX, SOCK_STREAM)` 创建一对全双工的匿名 socket，
>    然后 `fork()`，子进程 `execlp("python3", worker.py, sv[1], conf)`，
>    把 fd 作为命令行参数传给 Python worker。
>    用 socketpair 而不是 TCP 是因为同机通信，走 Unix domain socket 没有协议栈开销；
>    用匿名 pair 而不是命名 socket 文件，省掉了路径管理和权限问题。
>
> 2. **消息分帧**：SOCK_STREAM 是字节流，有粘包问题。
>    我用 **4 字节长度前缀 + Protobuf body**，长度用 `htonl` 转网络字节序，
>    读的时候先读满 4 字节拿长度，再按长度读满 body。
>
> 3. **超时控制**：读写都用 `select` 加超时包了一层，
>    避免 Python 侧卡死时 C++ 线程被无限阻塞。
>
> 4. **进程保活**：单独起一个 monitor 线程定期检查 worker 存活，
>    挂了就重建 socketpair 重新 fork，并记录 `restart_count`，
>    重启太频繁说明插件本身有问题，会告警。
>
> 5. **请求响应匹配**：维护 `request_id` / `response_id`，
>    保证请求和响应能对上，防止超时重试后收到上一次的响应。

**追问：为什么用 select 不用 epoll？**
> 这里每个连接是独占的 socketpair，一次只 poll 一个 fd，
> select 的 O(n) 扫描和 1024 fd 上限都不构成问题，而且 select 的超时接口更直接。
> epoll 的优势在于管理大量 fd，这个场景用不上，反而是额外复杂度。

**追问：一个 worker 会不会成为瓶颈？**
> 会，所以是 worker 池（`begin_worker_num_` / `end_worker_num_`），
> 多个 Python 进程并行处理，每个进程有独立的 GIL，这才是真正绕开 GIL 的地方。
> 池子大小按插件的 CPU 耗时和 QPS 配。

**追问：fork 之后 exec 之前，有没有考虑过父进程的锁状态？**（高阶）
> 有风险。fork 出的子进程只保留调用线程，如果 fork 那一刻别的线程正持有
> malloc 的内部锁，子进程里那把锁就永远是锁着的状态，
> 在 exec 之前调用任何可能 malloc 的函数都可能死锁。
> 我的实现里 fork 和 execlp 之间只做了 fd 处理和 execlp 本身，
> 保持了 async-signal-safe，这是必须注意的。

---

## 四、工程升级与线上问题

### Q8. TensorRT 7 升到 10.13、gcc 4.9 升到 13.1，踩了什么坑？

> 最主要的是 **`_GLIBCXX_USE_CXX11_ABI`**。gcc5 之后 `std::string` 和 `std::list`
> 换了新 ABI，老库是用 `_GLIBCXX_USE_CXX11_ABI=0` 编的，
> 新代码默认是 1，两边一混就是链接期找不到符号（符号里带 `__cxx11` 标记），
> 或者更糟——链上了但运行时内存布局不一致，随机崩溃。
> 处理办法是把所有依赖库统一到同一个 ABI 档位重编，不能重编的用旧 ABI 编译单元隔离。
>
> TRT 侧的变化：7 到 10 大量 API 废弃，
> 隐式 batch 模式被移除必须全部改成显式 batch + optimization profile，
> `IPluginV2` 系列的插件接口换代，序列化格式不兼容所以**所有 engine 必须重新构建**。
>
> 最麻烦的不是改代码，是**验证**：200 多个模型要逐个重建 engine 并做精度对拍，
> 确认新版本的 kernel 选择和层融合没有改变数值结果。我们是按业务重要性分批灰度迁移的。

**追问：怎么保证精度没变？**
> 固定一批线上真实请求做基线，新旧 engine 各跑一遍，对比输出的最大绝对误差和相对误差。
> FP16 下有微小差异是正常的（层融合顺序变了），关键是看下游业务指标有没有偏移，
> 排序模型看 NDCG，分类模型看在阈值附近的翻转比例。

### Q9. 为什么禁用 tcmalloc？

> 我们遇到过 tcmalloc 和 CUDA / TRT 的显存管理在某些场景下冲突的问题【补：具体现象】。
> tcmalloc 的线程缓存策略对大块分配和跨线程释放的处理，
> 和框架内部的内存管理叠加后出现异常，禁用后恢复正常。
> 这类问题的排查思路是先二分定位——换回 ptmalloc 看是否复现，确认是分配器的问题。

⚠️ **这条如果你记不清当时的具体现象，面试时别主动提**。
说不清的细节主动提出来只会给自己挖坑。

### Q10. 线上 coredump 和性能问题怎么排查？

**这是 C++ 岗的标配题，要有方法论**：

> **coredump**：`gdb 程序 core` 进去先 `bt` 看栈。
> 栈是好的就直接定位；栈烂了（`??` 一堆）说明栈被踩了，
> 那就看寄存器、`info registers` 结合反汇编往回推，或者用 `x/` 检查栈内存找返回地址。
> 常见的几类：野指针/use-after-free（Tensor 引用计数用错就是这类）、
> 缓冲区越界、std 容器并发读写、栈溢出。
> 线下复现不了的话上 ASan / Valgrind 跑压测。
> 还要注意 `ulimit -c` 和 core 文件路径配置，线上很多机器默认不生成 core。
>
> **性能问题**：先分清是 CPU 密集还是等待。
> `top` 看 CPU 是否打满，`perf top` 看热点函数，`perf record` + 火焰图看调用栈占比。
> CPU 不高但延迟高就是在等——用 `strace` 看系统调用、看是不是锁竞争
> （`perf` 能看到 futex），或者下游依赖慢。
> GPU 服务额外看 `nvidia-smi` 的利用率和显存，利用率低但延迟高，
> 大概率是 batch 没凑起来或者有同步点——我在 GR 那个项目里就是这么定位到 D2H 的。

---

## 五、推理代理层（`infer_proxy`）

### Q11. 推理结果为什么能缓存？key 怎么设计？

> 因为模型推理是**确定性**的：同样的输入、同样的 engine，输出一定一样。
> embedding 类模型尤其适合——线上 query 有明显的长尾重复，
> 热门 query 反复算是纯浪费。
>
> key 用请求内容的 **MD5**（加上模型名和版本），value 存序列化后的输出张量。
> 模型版本必须进 key，否则模型更新后会读到旧结果——这是最容易出事的地方。
>
> 实现上是 Redis/Pika 的**读写分离双实例**，读走一个、写走另一个；
> 全程异步，基于 Future 编排，缓存查询不阻塞推理链路；
> 支持请求级的 `disable_cache=1` 参数，方便调试和压测时绕过缓存。
> TTL 可配置。

**追问：缓存击穿、雪崩考虑了吗？**
> 我们的场景相对缓和——缓存没命中就是走一次正常推理，不会打垮后端，
> 所以没做互斥重建。TTL 是统一配置的，没做随机抖动，
> 这块如果 QPS 再上一个量级是需要补的。

⚠️ 诚实承认没做的部分，比编一套完整的防护方案安全。

### Q12. K8s 多副本抢着 build engine 的竞争是怎么解决的？

> 现象是滚动更新时多个 Pod 同时启动，发现没有 engine cache 就各自开始 build，
> 一个 engine 构建很吃 CPU 和显存，几个 Pod 一起 build 会把节点打爆，启动时间还特别长。
>
> 解决思路是把构建从"第一次请求触发"提前到"启动时按配置确定性地构建"，
> 并配合 shape 配置让构建行为可预测，避免多副本重复竞争【补：你实际用的是
> 文件锁 / init container 预构建 / 还是构建产物外置到共享存储——按真实情况说】。

---

## 六、如果面 C++ 后端岗，额外准备

这些和推理无关，但纯 C++ 岗必问，简历里的项目也支撑得住：

| 题目 | 你的答题素材 |
|---|---|
| 智能指针 / 移动语义 | Tensor 的引用计数设计，为什么没用 shared_ptr |
| 多线程与 memory order | BatchEngine 的 running_ 标志、RefCounted 的 relaxed/release |
| 模板与类型萃取 | BatchEngine 用了 `std::is_invocable_v` 做编译期校验 |
| 进程间通信 | Python 插件的 socketpair + 分帧 + 保活 |
| epoll / Reactor | 基于公司 UCS 框架，能讲清楚 Reactor 模型和你在上面做了什么 |
| ZooKeeper / 分布式 | 数据通道的 Leader 选举、节点监听、灰度锁（见 03） |
| 构建与发布 | CMake + qmodule 依赖管理、RPM 打包、Docker 镜像、x86/ARM 双架构 |

⚠️ **UCS 是公司内部框架，面试官不知道它是什么。**
提到时要补一句"类似 brpc 的 Reactor 模型 RPC 框架"，
并且准备好回答"如果让你在开源框架上重做一遍你会怎么做"——
这题在问你有没有可迁移的能力，还是只会用内部轮子。
