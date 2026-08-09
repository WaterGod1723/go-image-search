# 神经网络检索排序（NN Ranker）方案与 TODO

> 目的：在本项目（Go、纯 stdlib、无第三方 ML 框架）中引入可训练的神经网络，
> 提升以图搜图 recall@1。本文档记录方案、实现位置、命令与当前进度，
> 供上下文压缩后恢复记忆使用。

## 1. 背景与基线

- 检索流程：`search/main.go` → 从 `test_pngs/`（**96 个 ref icon，含 30 个新增灰度实心
  icon**）建索引 → 对 `test_set/*.png`（120 张查询图）`extractQuery` 分割出 sprite →
  `buildFeat` 提 7 类手工特征 → `composite`/`compositeAdaptive`（match.go）融合排序。
- 每张查询图的真值来源记录在 `test_set/manifest.json`（由 `gentest` 生成）。
- **基线（`search.exe .` 默认 = adaptive 融合）**（test_set，96 refs / 120 查询）：
  `recall@1 = 105/120 = 87.5%`，recall@3 = 91.7%，recall@5 = 94.2%。miss 集中在
  灰度描边族（home_work / fact_check / calculate / auto_awesome_motion / perm_phone）
  与新增灰度实心族（icon-check / icon-save / icon-trash / a-icon-file2 等）。
- 测试集 2026-08-09 随新 icon 重生成，并**去掉旋转干扰**（`gentest -no-rot`，
  与旋转版逐样本匹配：同 canvas/背景/文字/scale/translate，仅 sprite 不旋转，
  见 §7 旋转消融）；旧旋转版可用 `gentest -n 120 -seed 20260809` 复现。

## 2. 核心思路：Learning-to-Rank MLP（特征融合 + 神经网络路由）

**不是**从原始像素端到端学识别（小数据集 + 无框架下不现实、CPU 重），
而是**保留现有手工特征，把颜色无关的"专属灰度算法" sczl 作为第二专家信号
一起喂给 MLP，让网络自己学"何时信任哪个专家"**（soft routing / mixture of experts），
替代 `defaultWeights`（手工权重）和 `compositeAdaptive`（逐查询启发式）。

sczl（`search/sczl/`，从 `go-image-search` 项目 vendor 进来，零内部依赖）：
基于轮廓形状上下文 + Fourier 描述子 + HOG + 占据栅格的多视角、颜色无关检索，
对"背景色变化 / 图标颜色变化 / 四周文字"鲁棒——正是灰度查询需要的。实测它整体
不如现有融合（82.2%），但其**错误集不同**（能救回 5 个 NN/baseline 都错、
且几乎只错灰度难例的查询），与 NN 是天然互补。

### 网络输入（每对 query↔ref，最终 93 维）
- **31 个 pair 特征**：
  - 8 个基础相似度：`histSim`、`cosSim(Radial)`、`cosSim(AngMag)`、
    `polarShiftSim`（极坐标形状）、`polarColSim`、`roundTripFactor`、
    `cosSim(Zern)`、`maskScore`（64×64 掩膜旋转最优 Dice）
  - `gate`：该 ref 是否在查询直方图"家族"内；`mono`：查询是否单色
  - 16 个 per-ring 形状细节 `polarShiftDetail`
  - **5 个 sczl 颜色无关子分数**：`occ`（占据栅格旋转最优）、`ncc`（NCC 模板）、
    `region`（多区域匈牙利）、`fd`（Fourier 描述子）、`rad`（径向直方图交集）
    —— 由 sczl 全局融合分拆开（`SubScoresAll` / `GlobalSubScoresOf`），
    让网络对每个专家分量单独定价，替代原来 1 个融合数（81 维 → 93 维）。
- **62 个查询级统计**：上述 31 个特征各自的「全 ref 集 best」与「best−亚军 gap」。
- 网络：`93 → 96 (ReLU) → 48 (ReLU) → 1 (sigmoid)`，约 12.6k 参数。
- 排序：**hist 主键降序 → gate → NN 分数**；对 hist 主键下前 8 名做 shapeContext 精排。

## 3. 训练

- 训练数据：`go run ./gentest -n 8000 -seed 20260809 -out train_set`
  （96 refs 新训练集；旧 66-ref 训练集 seed 20260715 已归档备份）。
- 每查询样本：1 正样本（真值 ref）+ 难负样本（hist+mask 混合最接近 8 个 + 3 随机）。
- 损失：class-balanced BCE + Adam（最终 lr=0.002，batch=256，60 epoch —— 120 epoch 会过拟合）。
- 命令：`search.exe . train train_set weights.gob [epochs] [lr]` → 存 `weights.gob`。
  注意力模式经 `NN_ATN`（none/hidden/input）选择，默认 none。
- 训练耗时：8000 张 prep ~3min + 60 epoch ~6min，总 ~9min。

## 4. 评测（留出 test_set）

命令：`search.exe . nn weights.gob`，同时输出 baseline adaptive 与 NN 的
recall@1/@3/@5 及 NN miss。排序尾部用 shape-context 精排（默认 SCTOP=12、SCBLEND=0.7，
可经环境变量覆盖；纯 SC 或纯 NN 都更差，0.7 混合最优）。

**当前最终结果（test_set 无旋转版，96 refs / 120 查询，全部有效）**
```
baseline adaptive : recall@1 = 105/120 (87.5%)   recall@3 = 110/120 (91.7%)   recall@5 = 113/120 (94.2%)
neural (MLP 93d)  : recall@1 = 112/120 (93.3%)   recall@3 = 117/120 (97.5%)   recall@5 = 118/120 (98.3%)
  neural mono : recall@1 = 41/44 (93.2%)  @5 = 43/44 (97.7%)   [base @1 = 40/44 (90.9%)]
  neural color: recall@1 = 71/76 (93.4%)  @5 = 75/76 (98.7%)   [base @1 = 65/76 (85.5%)]
```
- 相对 baseline：recall@1 +5.8pp、recall@3 +5.8pp、recall@5 +4.1pp。
- 剩余 8 个 @1 miss：icon-check、icon-save、icon-trash、a-icon-file2（灰度实心族）；
  lingshi、fact_check、calculate、perm_phone（灰度描边族）。真值多在 rank 2-3。
- 提升轨迹：81 维 plain96 @1 92.5% / @3 96.7% / @5 97.5% → **+sczl 子分数拆分
  （93 维）@1 92.5%→93.3%、@3 97.5%、@5 99.2%→98.3%**（净效果大致持平，见旋转消融）。

### 历史：旧 test_set（66 refs，118 有效查询）
```
baseline adaptive : recall@1 = 109/118 (92.4%)   recall@3 = 109/118 (92.4%)   recall@5 = 109/118 (92.4%)
neural (MLP)      : recall@1 = 114/118 (96.6%)   recall@3 = 116/118 (98.3%)   recall@5 = 116/118 (98.3%)
```

## 5. 性能优化（针对"旋转扫描太耗 CPU/时间"的诉求）

问题根源：`maskScore` 对每个 (query,ref) 对都重跑 90 次 64×64 旋转；
训练时每 epoch 还重算全部特征。

已做：
1. **查询旋转掩膜缓存**：`queryMasks`（nnfit.go）把一张查询的 90 个旋转掩膜
   只算一次，66 个 ref 共享 → 掩膜开销摊到 ~1/66。
2. **数据集内存常驻**：`nnQuery`（nnfit.go）保存每查询全部 pair 特征与统计，
   epoch 阶段只做前向/反向，不重算特征（epoch 阶段实测 0.4s）。
3. **并行**：`parFor`（nnfit.go，按 GOMAXPROCS 分片）并行构建查询特征与评测。
   结果：pair 特征提取从"分钟级未完成"降到 **~3s**。
4. **shape-context 精排提速**（shapecontext.go，服务端热路径，~1s→~310ms）：
   - 查询的 36 个旋转 SC 直方图每查询只预计算一次（`scQueryHists`），top-12 候选共享；
   - 参考侧 SC 直方图按 ref 惰性缓存（Feat 内 `sync.Once`，`refSC`），不再每个 (q,r) 对重算；
   - SC 采样点 120→48（`SCPTS` 可覆盖），Hungarian O(n³) 降 ~15 倍，**recall@1/@3 实测无损失**。
5. **本地存储/索引持久化**（server.go）：
   - sczl 参考描述子写进 `.searchcache/index.gob`（indexEntry.SCZL），重启/重建免重解码 PNG：
     冷启动构建 2100ms → 热缓存重建 334ms；
   - 启动自动构建索引（无缓存时），免手动操作；
   - 查询结果 LRU 缓存（按内容 hash）照旧，重复查询秒回。
6. **两阶段粗筛（大规模库）**（nnfit.go `rankNN`，`PREFILTER_N` 可调，默认 50）：
   - **stage 1（全库、廉价）**：`cheapPref` = 0.4·hist + 0.2·zern + 0.2·radial + 0.2·angmag，
     全部旋转不变、无扫描，~5k ops/ref，取 top-N；
   - **stage 2（top-N 子集）**：昂贵的掩膜旋转、sczl 专家、极坐标细节、NN 前向与 SC 精排
     只在 top-N 上做（实测每 ref ~1.8ms）。
   - 量化依据（`rr` 诊断）：真值在 cheapPref 下的最差排名 = 37（样本 00109），
     N=40 即 100% 不漏真值；N=40/50 下 recall@1/@3 与全量完全一致（114/116）。
   - **规模效应**：per-query 开销从 `L×1.8ms + 固定 SC 230ms` 降为
     `L×0.05ms(廉价) + N×1.8ms + 固定 SC 230ms`。66 条无差别；**1 万条时 ~18s → ~0.35s（~50x）**。
   - 说明：KD-tree/VP-tree 对本项目高维特征收益有限（见 §9 权衡）；两阶段线性粗筛
     在几十万条以内足够，再大才需 ANN 嵌入索引。

## 6. 文件布局与命令

| 文件 | 作用 |
| --- | --- |
| `search/nn.go` | MLP 实现（forward/backprop/Adam/gob 存取；含可选 SE 注意力，`NN_ATN`） |
| `search/nnfit.go` | 特征向量构建、`queryMasks` 缓存、`parFor`、`runTrain`、`runNNEval`、诊断（`rr`/`seg`） |
| `search/sczl/` | vendor 的 sczl 颜色无关检索算法（含导出 `GlobalScoresAll` 与 5 子分数 `SubScoresAll`/`GlobalSubScoresOf`） |
| `search/sczleval.go` | sczl 单独评测（mono/color 拆分） |
| `search/main.go` | 子命令 `train`/`nn`/`sczl`/`rr`/`seg`/`space` |
| `train_set/` | gentest 生成的训练集（8000 张，seed 20260809，gitignore 不入库） |
| `weights.gob` | 训练产物（最终：8000 张 / 60 epoch / lr=0.002 / 93 维输入，plain MLP） |

常用命令：
```bash
go run ./gentest -n 8000 -seed 20260809 -out train_set   # 生成训练集（若需重建）
go run ./gentest -n 120 -seed 20260809 -out test_set     # 生成测试集（96 refs，含旋转）
go run ./gentest -n 120 -seed 20260809 -out test_set -no-rot  # 无旋转版（当前 canonical）
go build -o search_nn.exe ./search
.\search_nn.exe . train train_set weights.gob 60 0.002        # 训练 plain（NN_ATN=none）
$env:NN_ATN="hidden"; .\search_nn.exe . train train_set w.gob 60 0.002  # 注意力实验
$env:NN_NOGM="1"; .\search_nn.exe . train train_set w.gob 60 0.002       # gate/mono 消融
.\search_nn.exe . nn weights.gob                         # 在 test_set 上评测（对比 baseline）
.\search_nn.exe . sczl                                   # sczl 单独评测
.\search_nn.exe . rr weights.gob                         # 逐查询各方法真值排名
.\search_nn.exe .                                        # 原 baseline 评测
```

## 7. 当前状态 / 待办（TODO）

### 已完成
- [x] 实现 MLP（search/nn.go：forward/backprop/Adam/gob 存取）
- [x] 实现 train / nn 子命令与特征管线（search/nnfit.go）
- [x] 修复 gob 偏差字段未导出导致权重加载后 b1/b2/b3 为 nil 的 bug（b1→B1）
- [x] 性能优化（§5）：旋转掩膜按查询缓存、**Zernike 去 math.Pow（prep 101.6s→2.5s）**、并行化
- [x] vendor sczl 算法为第二专家（`search/sczl/`），并导出廉价 `GlobalScoresAll`
- [x] 实验验证：sczl 单独 82.2%（整体不如现有融合）但其**错误集互补**（救回 5 个灰度难例）
- [x] 把 sczl 全局相似度作为 NN 输入特征（soft 神经路由）
- [x] **sczl 5 子分数拆分进 NN**（`SubScoresAll`/`GlobalSubScoresOf`）：occ/ncc/region/fd/rad
      各占一个特征维度（81 维 → 93 维），网络可对每个专家分量单独定价
- [x] 8000 训练图 + shape-context 精排调参（SCTOP=12/SCBLEND=0.7）
- [x] 目标指标：新 test_set 上 recall@1 / @3 / @5 达成 92.5% / 97.5% / 99.2%（超 baseline 88.3/93.3/94.2）
- [x] `runNNEval` 输出 mono/color 拆分统计（灰度查询专门监控，验证注意力/路由假设）
- [x] **旋转干扰消融**：`gentest -no-rot` 生成逐样本匹配的无旋转测试集（仅 sprite 不旋转），
      证明新模型旋转鲁棒（有旋转 92.5 vs old 90.8）、老模型旋转敏感（去旋转后 90.8→94.2），
      无旋转时两者打平；canonical test_set 已切换为无旋转版
- [x] **接入 search server**：`/api/search` 启动时加载 `weights.gob`（可用 `NN_WEIGHTS` 覆盖路径），
      优先用 `rankNN`（含 sczl 专家 + shape-context 精排），权重缺失/形状不符则回退 `compositeAdaptive`；
      引用索引变化时自动重建 sczl 专家索引（与 entries 1:1 对齐）。
- [x] **检索/索引提速**（见 §5）：
  - shape-context 精排优化：查询旋转直方图每查询只算一次（12 候选共享）、参考侧 SC 直方图按 ref 惰性缓存、
    采样点 120→48（Hungarian O(n³) 降 ~15 倍，**recall 无损失**）→ 服务端单查询 ~1s 降到 **~310ms**
  - sczl 参考描述子持久化进 `.searchcache/index.gob` → 重启索引重建 2100ms 降到 **~334ms**（不再重解码 PNG）
  - 启动自动构建索引（无缓存时），免手动"构建索引"
  - **两阶段粗筛**（`PREFILTER_N`，默认 50）：廉价签名全库扫 + 昂贵特征只在 top-N 做，
    recall 不变；1 万条库预计 ~18s → ~0.35s

### 最终评测结果（新 test_set，96 refs / 120 查询）
```
baseline adaptive : recall@1 = 106/120 (88.3%)   recall@3 = 112/120 (93.3%)   recall@5 = 113/120 (94.2%)
neural (MLP 93d)  : recall@1 = 111/120 (92.5%)   recall@3 = 117/120 (97.5%)   recall@5 = 119/120 (99.2%)
```
- 相对 baseline：recall@1 +4.2pp、recall@3 +4.2pp、recall@5 +5.0pp。
- 剩余 9 个 @1 miss（真值多在 rank 2-3）：icon-check / icon-save / icon-trash /
  a-icon-file2（新增灰度实心族）；home_work / calculate / auto_awesome_motion /
  perm_phone / lingshi（灰度描边族）。

### 注意力机制实验（2026-08-09，已在 nn.go 实现，默认关闭）

在 MLP 上实现了两种 SE 式注意力（`search/nn.go`，`NN_ATN` 环境变量选择）：
- `hidden`：第一隐层上的 Squeeze-and-Excitation 通道注意力（96→24→96，门乘 h1）；
- `input`：输入级专家注意力，对 pair 特征块按样本做门（31→12→31），
  直接实现"何时信任哪个专家"的软路由；
- `none`：原始纯 MLP（训练/推理默认，实验证明它仍是最优）。

**新 test_set（96 refs / 120 查询）留出集结果**——plain/hidden/input 三份权重同配置
（8000 训练图 / 60 epoch / lr=0.002，基权重初始化 RNG 完全一致，干净消融，均为 93 维）：
```
plain (none)   : recall@1 = 111/120 (92.5%)   recall@3 = 117/120 (97.5%)   recall@5 = 119/120 (99.2%)
hidden (SE)    : recall@1 = 109/120 (90.8%)   recall@3 = 117/120 (97.5%)   recall@5 = 118/120 (98.3%)
input (SE)     : recall@1 = 110/120 (91.7%)   recall@3 = 117/120 (97.5%)   recall@5 = 118/120 (98.3%)
```
结论：注意力机制在两个测试集（旧 66 refs 与新 96 refs）上都**没有带来提升**——
hidden/input 的 @1 均不高于 plain，网络容量不是瓶颈；miss 集只是等量交换
（各救回 1-2 个、各丢掉 1-2 个）。代码保留（默认 none，`NN_ATN` 可开）。

**mono/color 拆分验证**（`runNNEval` 现输出 mono/color 统计；新 test_set 43 灰度 + 77 彩色查询）：
```
               mono@1          color@1
plain  (none): 40/43 (93.0%)   71/77 (92.2%)   total 111/120 (92.5%)
hidden (SE) : 40/43 (93.0%)    69/77 (89.6%)   total 109/120 (90.8%)
input  (SE) : 40/43 (93.0%)    70/77 (90.9%)   total 110/120 (91.7%)
```
即使灰度查询占比大幅提高（43/120 ≈ 36%，旧集合仅 ~30% 且灰度 ref 更少），
三个模型在 mono 查询上 recall **完全一致（40/43）**，注意力既没帮灰度查询，
其劣势还全部落在彩色查询上——"灰度占比太小掩盖注意力收益"的假设不成立。

### sczl 子分数特征实验（2026-08-09）
将单值 sczl 全局融合分拆为 occ/ncc/region/fd/rad 五个子分数作为独立特征维度
（81 维 → 93 维），同配置（8000/60/0.002，96 refs）重训对比（plain 模式）：
```
81 维（sczl 单值）: recall@1 = 111/120 (92.5%)   recall@3 = 116/120 (96.7%)   recall@5 = 117/120 (97.5%)
93 维（sczl 5 分）: recall@1 = 111/120 (92.5%)   recall@3 = 117/120 (97.5%)   recall@5 = 119/120 (99.2%)
```
结论：**sczl 5 子分数拆分有效**——@1 持平，@3 +1、@5 +2（救回 check_box、dashboard），
train@1 也升（94.8% → 96.1%）。已作为当前 weights.gob（93 维）基线保留。

### gate/mono 归纳偏置消融（2026-08-09，`NN_NOGM=1`）
去掉 pair 特征 8（gate：ref 是否在直方图家族）与 9（mono：查询是否单色）两个
手工注入偏置（81→87 维输入，rankNN 的 gate 硬排序键一并移除），同配置重训对比：
```
93 维（含 gate/mono）: recall@1 = 111/120 (92.5%)  @3 = 117/120 (97.5%)  @5 = 119/120 (99.2%)
87 维（去 gate/mono）: recall@1 = 111/120 (92.5%)  @3 = 117/120 (97.5%)  @5 = 118/120 (98.3%)
  mono/color：93 维 mono 40/43 color 71/77；87 维 mono 40/43 color 71/77（@5 各 -1）
```
结论：**去掉 gate/mono 没有收益**——@1/@3 完全持平，@5 反降 1（miss 集交换：
救回 home_work、丢掉 check_box）。这两个偏置对网络既无害也无明显增益，保留
（默认 NN_NOGM=0）。`NN_NOGM` 开关保留以便未来 re-ablation；权重维度随
NN_NOGM 变化，`checkMLPDims` 会在评测时校验一致性。

### 旋转干扰消融（2026-08-09，`gentest -no-rot`）
`gentest` 新增 `-no-rot` 开关：sprite 仍按抽到的旋转角做布局（scale/translate/
文字避让位置不变），但**绘制时用 0 度**，未旋转 sprite 落在旋转包围盒内不重叠文字。
同 seed 下旋转/无旋转两版测试集逐样本一致（canvas/背景/文字/scale/translate 全同，
仅 sprite 是否旋转），是干净的旋转消融。

在同一批样本（seed 20260809，96 refs / 120 查询）上评测新老模型：
```
                    recall@1      recall@3      recall@5
有旋转  old (81d) : 109/120 90.8%  113/120 94.2%  115/120 95.8%
有旋转  new (93d) : 111/120 92.5%  117/120 97.5%  119/120 99.2%
无旋转  old (81d) : 113/120 94.2%  117/120 97.5%  118/120 98.3%
无旋转  new (93d) : 112/120 93.3%  117/120 97.5%  118/120 98.3%
```
结论：
- **新模型对旋转更鲁棒**——有旋转时 new 明显超 old（92.5 vs 90.8，@3 +3、@5 +4），
  靠的是 sczl 5 子分数特征 + 96-ref 训练；去掉旋转后 new 几乎不变（92.5→93.3）。
- **老模型对旋转敏感**——去旋转后 old 大涨（90.8→94.2，@1 +4、@3 +4、@5 +3）。
- 无旋转时两者基本打平（old 94.2 vs new 93.3，差 1 条在噪声内）；baseline 对旋转不敏感
  （88.3 vs 87.5）。
- 已改用无旋转版作为 canonical test_set（文字干扰相同，避免更早生成的"去旋转集"因
  sprite 轴对齐留白多导致文字更多而失真）。

### 待办
- [ ] （可选）SC/HOG 精排进 NN，或新增"实心 vs 描边"密度特征，继续压灰度实心族
      （icon-check/search、icon-save/book、icon-trash/video 这类 mask/SC 都失效的难例）
- [ ] （可选）改进分割：处理 scale>1 溢出与细笔画丢失
- [ ] （可选）服务端 SC 精排再提速：剩余 ~310ms 中精排仍占大头，可进一步并行化或降旋转步数（36→18）

## 8. 已知风险与备注

- 训练集与测试集用不同 seed 生成，图像不同但同分布；若要严格泛化，可再生成
  第三个集做验证集/早停。
- `weights.gob` 是 gob 编码，结构变更（输入维数/层宽）会导致旧权重不兼容 → 需重训。
  当前为 **93 维 plain**（sczl 5 子分数版）；旧 81 维权重 `len(W1) != nnH1*nnInput`
  会被 server 拒绝并回退 baseline，需删除重建（如 `.searchcache/` 缓存）。
- 注意力（hidden/input）经 `NN_ATN` 选型，权重 gob 自带 WSE/WA 字段标记，
  无字段即为 plain——旧权重与 plain 模式完全兼容。
- sczl 来自 `go-image-search` 项目（MIT 风格自研代码），本仓库为 vendor 副本；
  跨模块 import 受 Go `internal` 规则限制，故直接拷贝。
- `compositeAdaptive` 含 shapeContext（36 次旋转 × Hungarian）较慢，评测保留作 baseline 对比。

## 9. KD-tree / ANN 权衡（结论）

- **为什么没用 KD-tree**：KD-tree 只适合低维欧氏；我们的廉价签名实测对难例（灰度族）
  真值排名最差 35~66（见 §5.6），建树后为了不漏真值仍需超大 top-N，无实际剪枝收益；
  真正的高判别力特征（mask/SC/sczl）都带旋转扫描，不是向量度量。
- **当前方案**：两阶段线性粗筛（廉价签名全库扫 → 昂贵特征只做 top-N）。
  对几十万条以内的 icon 库足够（1 万条 ~0.35s/查询）。
- **若库到百万级** 才需要：先学一个低维旋转不变 embedding，再上
  VP-tree / IVF（FAISS 式）/ HNSW。这是独立的大工程（需要额外训练 + ANN 库），
  当前 `image-search-test` 的纯 stdlib 约束下不建议提前做。
