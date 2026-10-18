# Channel

基于图拓扑解耦的 SpinalHDL 硬件建模与连接框架。

详细架构设计、API 手册与实现规范请参考：[doc/Channel.wiki.md](doc/Channel.wiki.md)。

## 快速开始

```bash
# 编译工程
mill channel.compile

# 运行全拓扑能力演示检查
mill channel.runMain channel.DemoTopologyCheck

# 生成演示 Verilog (输出至 gen/demo/)
mill channel.runMain demo.DemoGen
```

## 测试套件

```bash
# 槽位规划与不变量回归检查
mill channel.runMain channel.ChannelSlotCheck

# 字段级强类型连线 DSL 回归检查
mill channel.runMain channel.ChannelFieldConnectCheck

# 值流与参数推导回归检查
mill channel.runMain channel.ChannelVarFlowCheck
mill channel.runMain channel.ChannelVarExportCheck

# 默认连线与边界检查
mill channel.runMain channel.DefaultConnectCheck
mill channel.runMain channel.EnvBoundaryCheck
```

## Reward Town 演示

`town_demo/` 是一个宝可梦风格的实时小游戏：玩家在 Tuxemon 的 Cotton Town 地图上走动，8 个 NPC 的每个活动和每句
对话都由 `Skywork-Reward-V2-Qwen3-0.6B` 在本地 GPU 上实时从预写候选中打分取最优。NPC 在咖啡馆、超市、学院、
家中和公园里使用实体工具（咖啡机、烤箱、配药计数器、钢琴、扩音器等）工作：模型为每种工具的 6 种用法打分，
使用者选出自己会的最佳用法；在场的人会评价成果（星级，影响之后谁会被请来帮忙），也可能看会新用法，或加入
帮忙。玩家可以使用同样的工具、帮 NPC 干活、被 NPC 请去帮忙；玩家的用法与台词都是盲选，选完才揭示模型的分数。
对话在地图上的对话框中进行；玩家转告的派对邀请会改变 NPC 在本局剩余时间里的知识。
公园南门上方的情人节爱心拱门分 5 个阶段搭建（锯、锤子、油漆刷、串灯、浇水壶），全程不靠说话：没有人请人帮忙，
看到半成品拱门的 NPC 会自己拿起下一阶段的工具，路过的人会加入正在干活的人，并通过观看学会新用法。玩家面对拱门按
A 可以查看进度、帮忙，或在学会该工具的 3 种用法后亲自完成下一阶段（`GET /api/site`、`POST /api/build`）。

```bash
./startup.sh                # 缺少时创建 .venv-gpu（CUDA torch）并下载素材，然后在 8765 端口启动游戏服务器
./startup.sh --cpu          # 使用仅 CPU 的 .venv 运行奖励模型
./startup.sh --port 9000    # 换端口
```

游戏地址：`http://127.0.0.1:8765/`。操作：方向键/WASD 移动，Z/空格/回车 = A，X/Esc = B，Shift 跑步，
L 对话记录，J 工作记录，K 技能，T 切换速度，P 暂停。GPU 环境单独放在 `.venv-gpu`，不改动 `.venv`（`prepare_real_gemm.py`
的缓存绑定 torch 版本）。

- `town_demo/engine.py`：模拟引擎（生成器形式的逐拍循环，玩家需要选择时暂停）。
- `town_demo/server.py`：游戏服务器（静态文件 + `/api/*`，单一全局会话）。
- `town_demo/simulate.py`：无玩家的整日模拟，用作引擎回归/性能基准，输出 `out/town-demo/replay.json`。
- `town_demo/fetch_assets.py`：从 Tuxemon（固定提交）下载素材到 `town_demo/game/assets/tuxemon/` 并把地图转换为
  `town.json`；素材许可见该目录下的 `ATTRIBUTIONS.md`（多为 CC BY-SA 4.0）与 `LICENSE`（GPLv3）。

## 真实模型的分层 Verilator 验证

`test_real_scenarios.py` 是 PyTorch 评分脚本，**不运行 RTL**。下面的流程将离线模型前向中
Linear 的真实输入和转置权重送入默认 `GemmTop(16,32,3072)`；NumPy INT64 矩阵乘法提供独立金标，
逐元素检查 INT32 值、坐标、`last`、反压保持和任务间隔离。RTL 当前仅实现 INT8 GEMM，
不是完整 Qwen3/Reward 模型：Attention、归一化、SiLU、量化重定标及模型评分仍由软件执行。

### 环境与最短路径

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install 'transformers>=4.52,<5' numpy safetensors

# 离线准备第 0 层 q/down 投影：默认各选一个 16×32 输出 tile。
# 默认只取聊天前 32 tokens；manifest 明确记录是否截断，256 可容纳完整数学示例。
.venv/bin/python prepare_real_gemm.py --max-tokens 256

# 每次最多两个未完成 tile；180 秒包含 Mill/Verilator 构建，超时会终止整个子进程组。
python3 run_real_gemm.py --stages smoke --max-tasks 2
python3 run_real_gemm.py --stages tile --max-tasks 2
python3 run_real_gemm.py --status
```

实测依赖：CPU PyTorch 2.14.1、Transformers 4.57.6、NumPy 2.5.4、Verilator 4.228、
Mill 1.1.8。源码默认配置/构建缓存沿用 `ChannelSim`；不生成波形。

仿真线程与构建并发独立设置：`--threads 1 --build-jobs 6` 为默认，两个参数均只接受 1–6。
当前 Verilator 4.228 对此小核在 2/6 仿真线程下均报 `UNOPTTHREADS`，因此使用单线程仿真、
最多六路 C++ 编译，不屏蔽告警或假报多线程提速。更大的可并行 RTL 可显式选择 `--threads 6`。
`ChannelSim` 通过 `SIM_VERILATOR_THREADS` 将选项加入 `simulatorFlags`；
不能仅通过 `SPINAL_VERILATOR_FLAGS` 传线程数：SpinalHDL 1.14.2 的缓存键遗漏该环境变量，
可能误用旧二进制。构建 wrapper 会将 SpinalHDL 自动传入的宿主 `-j` 覆盖为明确的并发上限。

### 分级覆盖与层间续跑

| 级别 | 准备/运行选项 | 验证范围 |
| --- | --- | --- |
| 快速链路 | `--stages smoke` | 真实输入、权重的前 32 个 K 分量；不能代替完整点积 |
| 完整点积 | `--stages tile` | 所选输出 tile 的完整 K：q/k/v/gate/up 为 1024，o 为 2048，down 为 3072 |
| 完整投影 | 准备时 `--stages tile --tiles-per-projection all` | 所选序列的全部 token 行块与输出列块，含尾块 |
| 跨 Transformer 层 | 准备时 `--layers all --projections all`；运行时 `--layers 0` 等 | 先固定计划，再逐层/逐批运行；四种场景用不同输出目录 |

```bash
# 先生成固定计划，避免每增加一层就改变 manifest、使旧断点失效。
.venv/bin/python prepare_real_gemm.py --output out/real-gemm-math \
  --scenario math --max-tokens 256 --layers 0,14,27 --projections all

python3 run_real_gemm.py --manifest out/real-gemm-math/manifest.json \
  --layers 0 --stages smoke --max-tasks 2
# 重复同一命令：跳过 PASS，继续下一批；完成一层后切换 --layers 14 或 27。
python3 run_real_gemm.py --manifest out/real-gemm-math/manifest.json \
  --layers 0 --stages tile --max-tasks 2

# 单独观察供数和消费能力；与无反压结果使用不同断点，不能混计。
python3 run_real_gemm.py --manifest out/real-gemm-math/manifest.json \
  --layers 0 --projections q_proj --stages tile \
  --input-gap 1 --output-stall 1 --max-tasks 1
```

`--scenario` 支持 `math/code/architecture/safety`；`--layers` 支持逗号列表或 `all`，
`--projections` 支持七种投影名或 `all`。数字 `--tiles-per-projection N` 选择行优先的前 N 块；
只有 `all` 且所有任务通过才能称为完整投影覆盖。默认单块不代表全层通过。
同一输出目录不要并发执行数据准备和仿真；不同场景应使用独立目录。

每个 tile 成功后原子保存 TSV；中断时仅重跑未完成 tile，不保存半个 tile 的仿真内部状态。
准备器缓存真实前向激活；相同配置和数据完整时直接复用，不重新前向。
模型/tokenizer 文件、准备器及依赖身份绑定准备缓存；manifest、RTL/驱动源码、
Verilator 版本、仿真线程/构建并发、供数/反压策略绑定仿真断点。源码或数据变化会创建新运行目录，不复用旧 PASS。
`--max-tasks` 是本次增量工作上限，不是总覆盖上限；返回成功不等于所有任务完成，应查看 `pending`。

产物位于 `<output>/runs/<fingerprint>/`：`checkpoint.json`、`report.json`、
`results/*.tsv` 和每次 `attempt-*.log`。日志区分构建与运行；TSV 的 `wall_seconds`
计 tile 首个归属周期至最后操作数接受的驱动时间，批次最后 tile 还包含结果尾部，不含编译。
失败/超时返回非零；已成功的 tile 保留。重复 `--status` 会校验 fixture 和结果摘要，不启动 Verilator。

### 性能指标的边界

周期类别互斥：命令、操作数接受、输入缺数、结果接受、输出反压、其他。周期归属于当前发命令/供数的 tile；
批次最后 tile 还拥有结果尾部。`result_fires` 独立记录该 tile 实际输出数，不加入互斥周期总和。
结果 bank 允许累加与前一 tile 输出重叠，结果端仍每拍输出一个 INT32；最后操作数等待 bank 排空。
无空拍/反压时，每 tile 周期为 `1 + max(K, prev_m*prev_n)`，批次首 tile 的前驱输出数取 0，
批次最后 tile 额外加自身 `m*n`。因此单 tile 周期仍为 `1 + K + m*n`。
周期依赖批次组成；比较利用率时应保持 `--max-tasks` 和任务顺序一致，该选项不参与运行身份。
`useful_mac_utilization = m*n*K / (16*32*cycles)` 同时包含阶段占空比和小矩阵的空闲 lane，
不是宿主仿真速度。量化误差单列于 manifest 的 `error_vs_fp`，不与 RTL 的整数精确比对混淆；
FP Reward 只是软件参考，不能据此宣称 INT8 端到端评分精度。
该测试没有 DDR/DMA/片上缓存模型，不能据此判断真实系统内存瓶颈或给出可信 MHz/TOPS。

### 串行实现基线记录（2026-10-16，重叠优化前）

- 数学场景完整 161 tokens：全部 28 层 × 七种投影各抽一个完整 K tile，`196/196 PASS`。
  这是跨层抽样，不是所有层的全部输出覆盖。
- 同一场景第 0 层 K 投影的完整 `161×1024` 输出：352 个 tile（包括 `m=1` 尾块），
  全部 164864 个整数输出逐元素一致，`352/352 PASS`。
- 第 0/14/27 层七种投影的 smoke/full-K 计划 `42/42 PASS`；
  代码、体系结构、安全三个完整聊天场景分别为 204/179/98 tokens，
  各检查第 0/27 层 q/down 投影，均为 `4/4 PASS`。
- 最终配置为仿真 1 线程、构建最多六路。28 层按每批两层（14 tile）运行，每批约 4.87–5.77 秒；
  完整投影每批 32 tile，约 5.30–6.40 秒。此为本机缓存构建后的宿主耗时，不是硬件延迟。

| 实测任务 | m×n | K | RTL 周期 | 串行结果输出占比 | 有效 MAC 利用率 |
| --- | --- | --- | --- | --- | --- |
| 短 K 链路冒烟 | 16×32 | 32 | 545 | 93.94% | 5.87% |
| q/k/v/gate/up 完整点积 | 16×32 | 1024 | 1537 | 33.31% | 66.62% |
| o 完整点积 | 16×32 | 2048 | 2561 | 19.99% | 79.97% |
| down 完整点积 | 16×32 | 3072 | 3585 | 14.28% | 85.69% |
| K 投影最后一行尾块 | 1×32 | 1024 | 1057 | 3.03% | 6.05% |

基线瓶颈随形状变化：短 K 被逐元素串行输出主导；完整 1024-K 点积约三分之一周期无法计算，
3072-K 的计算阶段占比更高。当前结果 bank 已实现输出/计算重叠；上述数值仅代表串行基线。
尾块仅激活 16 行中的一行，属于物理 lane 利用不足，不能仅靠扩大结果带宽解决。
未测综合面积或时钟频率。

对 q tile 每两个输入 beat 间插入一拍、每个输出反压一拍后，测得 3072 周期：
包含输入缺数 1023 拍、输出反压 512 拍；值、坐标、`last` 和反压保持均通过。
同时验证了实际硬超时退出 124 后续跑、连续小批跳过 PASS、仅状态查询不重跑、
前向激活缓存复用、损坏 fixture 提前拒绝、错误金标被 RTL 结果比较检出且不写 PASS，
以及原有 `GemmCheck smoke`。

主要证据（本地工件；断点同时绑定 `JAVA_HOME` 等环境，换环境会建立新目录）：

- 全部层抽样：`out/real-gemm-all-layers/runs/0505fcfbfb06b448ab51/report.json`
- 完整投影：`out/real-gemm-full-projection/runs/95229ff810c6abb06dbb/report.json`
- 额外场景：`out/real-gemm-{math,code,architecture,safety}/runs/*/report.json`

### 结果 bank 重叠优化实测（2026-10-16）

默认 `GemmTop(16,32,3072)` 使用 27 位累加器、寄存行列掩码和移位结果 bank；
数据寄存器无复位，首 beat 加载乘积，端口及 INT32 输出格式不变。未增加操作数或乘积流水寄存器。

同一完整 K 投影 manifest、每批 32 tile：

| 实现 | 通过 tile | 总周期 | 有效 MAC 利用率 |
| --- | --- | --- | --- |
| 串行基线 | 352/352 | 525664 | 0.6273 |
| 累加/输出重叠 | 352/352 | 365952 | 0.9010 |

- `mill channel.runMain npu.sim.GemmCheck`：signed、partial、handshake、reset、limits、random、stream 七套全部通过。
- 四任务 stream 断言 21 周期（串行需要 31）；handshake 验证输出反压期间下一命令被接受、下一任务最后 beat 等待 bank 排空。
- 同一真实投影另外以 `--input-gap 1 --output-stall 1 --max-tasks 4` 验证四个 tile，全部逐元素通过。
- 生成 Verilog 的 `acc_0_0`、`bank_0_0` 均为 `[26:0]`，原 `selectedValue` 选择链已移除；面积和 Fmax 未测量。

本地证据：

- 完整投影：`out/real-gemm-full-projection/runs/0ed7dd2702d15c78388a/report.json`
- 供数/输出反压：`out/real-gemm-full-projection/runs/41c85a8e6e2ba7415955/report.json`

## VU19P 板级实测（2026-10-17）

`GemmTop(16,32,3072)` 经 XDMA（Gen3 x8）接入 VU19P，片上存储供数，无 DDR：

- `src/npu/fpga/GemmFpgaTop.scala`：板级封装。`s_axi_data`（256 位，核心时钟）读写操作数 beat
  （32768×64 B，URAM）与结果记录（32768×8 B，BRAM）；`s_axi_lite`（BAR0，XDMA `axi_aclk`）为
  CMD 表与控制/状态寄存器。寄存器表见文件头注释。GEMM 内部未增加复位或流水寄存器，端口与载荷格式不变。
- `fpga/vu19p/`：Block Design（xdma_0 → `axi_data_cc` → gemm_0；M_AXI_LITE 直连 gemm_0）、约束、
  构建 Tcl。`run.sh build|status|fetch` 在 Vivado 主机上远程构建，产物取回 `gen/fpga/artifacts/<tag>/`。
- `software/fpga-host/npugemm.c`：板端主机程序（`npugemm status` / `npugemm run <list> [--batch N]
  [--max-tasks N] [--tsv F] [--trace]`），批次划分、填充与 TSV 列和仿真一致。
- `npu.sim.GemmFpgaCheck`：Verilator 按主机方式驱动封装（双时钟、posted MMIO、XDMA 式并发 4 KiB
  突发），并验证核心单独复位期间 BAR 仍可读、复位不误触发启动。

```bash
mill channel.runMain npu.sim.GemmFpgaCheck --list out/real-gemm-full-projection/fixtures.list \
  --batch 32 --output out/gemm-fpga/sim-full-projection.tsv
CORE_MHZ=100 fpga/vu19p/run.sh build      # 之后 status / fetch；仅 WNS/WHS ≥ 0 且 0 失败端点的比特流上板
# 板端：aeprog-native 下载 .bin → sudo reboot（重新分配 BAR）→ sudo vu19p-load-xdma.sh
./npugemm run out/real-gemm-full-projection/fixtures.list --batch 32 --tsv runs/board-full-projection.tsv
cmp runs/board-full-projection.tsv out/gemm-fpga/sim-full-projection.tsv
```

### PCIe 卡死修复

早期比特流中 AXI-Lite 从机位于核心时钟域，经 `axi_ctrl_cc` 跨时钟。板上曾出现 STATUS 读回
`0xffffffff`（PCIe 完成超时），随后 DMA 使 ThinkStation 整机卡死，需人工重启。BAR 读必须在任何
核心状态下完成，因此改为：

- AXI-Lite 从机整体运行于 XDMA `axi_aclk`，删除 `axi_ctrl_cc`。控制以 toggle 跨入核心域，
  NUM_TILES/TOTAL_BEATS 在同步后的启动沿采样；状态每 8 个核心周期整体快照回传。toggle、序号与快照
  位于无复位（上电初始化）域，任一侧单独复位不会伪造启动或使握手失步。
- 新增健康寄存器 `CORE_TICKS`/`SNAP_AGE`/`CORE_RESET`，ID 改为 `NPG2`。核心停钟或处于复位时 BAR
  照常应答；`npugemm` 在每次 DMA 前检查，异常即以退出码 2 停止，不再继续 DMA。
- 去掉 pcie/sys 时钟组的全局 `set_clock_groups`：它覆盖了 `axi_data_cc` 同步器自带的 max-delay
  约束（methodology TIMING-24）。跨时钟改由 `constraints/cdc.xdc` 限制为 4 ns datapath-only，
  IP 自带的限定约束优先。修复后 TIMING-24 为 0，bus skew 最小裕量 3.549 ns。
  `report_cdc` 仍报 2 个 CDC-1，为 `busy/done` 由异步 NUM_TILES 组合判零后在启动沿捕获：
  启动沿到达前该值已稳定至少 2 个 ctrl 周期加 2 个核心周期，路径受 4 ns 约束。

旧路径中究竟是哪个环节停止应答，并未单独复现定位；修复去掉了 BAR 读对核心时钟、核心复位及
`axi_ctrl_cc` 的依赖。修复后每次运行 `CORE_RESET` 计数始终为 1（仅上电复位），`SNAP_AGE` 为 1–19。

### 板上结果（比特流 `gemm_100mhz_e2301a753b03`，核心 100 MHz）

| 负载 | tile | 批次 | 板上周期 | 仿真周期 | 有效 MAC 利用率 | GOPS |
| --- | --- | --- | --- | --- | --- | --- |
| 第 0 层 K 投影完整输出 | 352/352 | 11 | 365974 | 365974 | 0.9010 | 92.3 |
| 28 层 × 7 投影抽样 | 196/196 | 10 | 292056 | 292056 | 0.9817 | 100.5 |

- 每个结果的值、行、列、`last` 均与 goldC 一致；板上 TSV 与 Verilator TSV 逐字节相同（`cmp`）。
  封装相对理想流水每批多 2 个周期（启动与完成握手）。
- 峰值为 512 MAC × 2 × 100 MHz = 102.4 GOPS。GOPS 以板上 CYCLES 计时，不含 DMA。
  DMA 实测：h2c 约 1.7 GB/s、c2h 约 0.55 GB/s，传输较小，含驱动开销。
- 两负载交替各连续运行 25 次，50/50 结果一致；此前卡死的 4 tile/2 批与 32 tile 批均通过，
  AER fatal/nonfatal/correctable 计数均为 0。
- 时序：WNS 0.576 ns、WHS 0.010 ns、0 失败端点；核心时钟 WNS 2.761 ns（10 ns 周期）。
  早期 200 MHz 构建 `gemm_200mhz_aad1703ec775` 有 21 个失败端点，全部位于封装的 BRAM/URAM 读路径；
  进入 GEMM 的最差路径裕量 +0.055 ns，GEMM 内部最差 +0.299 ns。即 GEMM 核本身满足 200 MHz，
  封装存储输出需加寄存器后才能整体上 200 MHz。

| 资源（VU19P） | 全设计 | gemm_0 封装 | 其中 GemmTop |
| --- | --- | --- | --- |
| LUT | 96645 | 73559 | 72355 |
| FF | 57958 | 30380 | 27780 |
| BRAM（RAMB36 等效） | 88.5 | 38.5 | 0 |
| URAM | 48 | 48 | 0 |
| DSP | 0 | 0 | 0 |

本地证据：`gen/fpga/artifacts/gemm_100mhz_e2301a753b03/`（timing.txt、reports/）、
`out/gemm-fpga/{sim,board}-*.tsv` 与 `board-*.log`。
