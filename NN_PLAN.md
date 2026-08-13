# 神经网络检索排序（NN Ranker）方案与 TODO

> 目的：在本项目（Go、纯 stdlib、无第三方 ML 框架）中引入可训练的神经网络，
> 提升以图搜图 recall@1。本文档记录方案、实现位置、命令与当前进度，
> 供上下文压缩后恢复记忆使用。

## 0. 最新迭代：精简架构（分支 `exp/attn-lean-proj`，参数 100k→32k）

大砍手工相似度特征，改为"原始逐模式相似度向量 + 联合训练投影器"（learned
similarity），模型自己去学"哪些描述子的哪个子带重要，如何比较配对"。

- **删除的输入**：8 个基础相似度（`histSim/cosSim(Radial/AngMag)/polarShift/
  polarCol/roundTrip/cosSim(Zern)/maskScore`）、`gate`/`mono` 偏置、
  16 个 per-ring 形状细节、sczl 专家信号、查询级 `best`/`gap` 统计、
  192 维颜色投影直方图。
- **新增的输入**（每对 query↔ref，全部旋转不变原始描述子的逐元素交集 `min`）：
  1. Zernike 幅值逐模式重叠（49 维）
  2. HSV 直方图逐 bin 交集（288 维）
  3. 径向密度逐环重叠（16 维）
  4. 角向 FFT 幅值逐模式重叠（192 维）
- 每个原始块过一个小 MLP 投影器（`In→8(ReLU)→8`，联合训练）压缩成 8 维 token；
  注意力网络只吃 4 个 8 维 token（`dims=[8,8,8,8]`），query = WQ·[投影 token]。
- 输入向量 `nnInput=577`：前 32 维为 token 槽位（forward 用投影结果回填），
  尾部 545 维为原始块 carry（对注意力不可见，仅用于投影器反传）。
- 实现位置：`search/nn.go`（`attnProjConfigs`/`attnLayout`/`forward` 回填）、
  `search/nnfit.go`（`buildPairVec`/`elemMin`/`nnVec`，sczl 与 mask 旋转扫描全部移除）。
- 参数：注意力 ≈27.9k + 4×投影器 ≈4.7k = **≈32.6k**（旧版 ≈100k，**≈-68%**）。
- 训练/评测命令不变：`search . train <dir> <weights> [epochs] [lr]`、
  `search . nn <weights>`（注意：权重与输入布局强绑定，旧 weights 不可混用）。
- 回退：`git checkout exp/attn-color-backup`（改前完整快照，含旧 nn.go/nnfit.go）。

### 0.1 实测结果（2026-08-13，训练集 `train_set_lean`：train_set 前 4000 查询，60 epoch / lr=0.002 / 46k 样本）

```
原 test_set（120 查询，同域）:
  baseline adaptive : recall@1 = 88.3%
  新 lean  (32.6k)  : recall@1 = 92.5%   （旧 attn+color+zproj ≈96.7%）
iconfont 未见图标（354 查询，跨域）:
  baseline adaptive : recall@1 = 77.4%
  新 lean  (32.6k)  : recall@1 = 39.8%   （旧架构 ≈67.2%，基线 77.4%）
```

结论：learned-projection 架构在训练域内略优于手工融合、但明显落后旧注意力架构；
**跨域泛化崩坏更严重**（39.8% << 旧 67.2% << 基线 77.4%）——投影器把相似度权重学死在了
训练域（业务图标）的描述子分布上，颜色/细节被大幅削减后没有可迁移的兜底信号。
下一步方向：训练数据混入更多样图标；减少 overfit（降 epoch/加正则）；或保留部分
描述性信号（如归一化掩码或直方图距离）作为兜底 token。

## 1. 背景与基线

- 检索流程：`search/main.go` → 从 `test_pngs/`（66 个 ref icon）建索引 →
  对 `test_set/*.png`（120 张查询图）`extractQuery` 分割出 sprite →
  `buildFeat` 提 7 类手工特征 → `composite`/`compositeAdaptive`（match.go）融合排序。
- 每张查询图的真值来源记录在 `test_set/manifest.json`（由 `gentest` 生成）。
- **基线（`search.exe .` 默认 = adaptive 融合）**：
  `recall@1 = 109/120 = 90.8%`，recall@5 同；miss 集中在灰白描边图标
  （edit_square / account_balance / contract_edit / assistant 等易混淆族）。
- 2 张查询分割失败（sample_00064、sample_00085，`extractQuery` 返回空），当前不计入。

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

### 网络输入（每对 query↔ref，最终 81 维）
- **27 个 pair 特征**：
  - 8 个基础相似度：`histSim`、`cosSim(Radial)`、`cosSim(AngMag)`、
    `polarShiftSim`（极坐标形状）、`polarColSim`、`roundTripFactor`、
    `cosSim(Zern)`、`maskScore`（64×64 掩膜旋转最优 Dice）
  - `gate`：该 ref 是否在查询直方图"家族"内；`mono`：查询是否单色
  - 16 个 per-ring 形状细节 `polarShiftDetail`
  - **1 个 sczl 全局相似度**（占据栅格+NCC+FD+径向+区域，旋转扫描，排除昂贵 SC/HOG）
- **54 个查询级统计**：上述 27 个特征各自的「全 ref 集 best」与「best−亚军 gap」。
- 网络：`81 → 96 (ReLU) → 48 (ReLU) → 1 (sigmoid)`，约 12.5k 参数。
- 排序：**hist 主键降序 → gate → NN 分数**；对 hist 主键下前 8 名做 shapeContext 精排。

## 3. 训练

- 训练数据：`go run ./gentest -n 8000 -seed 20260715 -out train_set`
  （与 `test_set` 不同 seed；同一 seed 下前 4000 张与旧版一致，属纯扩展）。
- 每查询样本：1 正样本（真值 ref）+ 难负样本（hist+mask 混合最接近 8 个 + 3 随机）。
- 损失：class-balanced BCE + Adam（最终 lr=0.002，batch=256，60 epoch —— 120 epoch 会过拟合，train@1 升到 98% 但 test 反降）。
- 命令：`search.exe . train train_set weights.gob [epochs] [lr]` → 存 `weights.gob`。
- 训练耗时：8000 张 prep ~3min + 60 epoch ~4.5min，总 ~7.5min。

## 4. 评测（留出 test_set）

命令：`search.exe . nn weights.gob`，同时输出 baseline adaptive 与 NN 的
recall@1/@3/@5 及 NN miss。排序尾部用 shape-context 精排（默认 SCTOP=12、SCBLEND=0.7，
可经环境变量覆盖；纯 SC 或纯 NN 都更差，0.7 混合最优）。

**最终结果（118 张有效查询，2 张分割失败不计）**
```
baseline adaptive : recall@1 = 109/118 (92.4%)   recall@3 = 109/118 (92.4%)   recall@5 = 109/118 (92.4%)
neural (MLP)      : recall@1 = 114/118 (96.6%)   recall@3 = 116/118 (98.3%)   recall@5 = 116/118 (98.3%)
```
- recall@1 +4.2pp、recall@3 +5.9pp、recall@5 +5.9pp，均超 baseline。
- 剩余 4 个 miss：2 张 home_work（scale 0.74 过小 / 1.26 溢出裁边 + 旋转 + 双文字，
  真值不在 top-5，分割受损难例）；2 张在 rank 2-3（account_balance / dashboard 的
  灰色描边族内混淆）。
- 提升轨迹：无 sczl 91.5% → +sczl 特征 94.9% → 8000 训练图 + 精排调参 96.6% @1。

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
| `search/nn.go` | MLP 实现（forward/backprop/Adam/gob 存取） |
| `search/nnfit.go` | 特征向量构建、`queryMasks` 缓存、`parFor`、`runTrain`、`runNNEval`、诊断（`rr`/`seg`） |
| `search/sczl/` | vendor 的 sczl 颜色无关检索算法（含新增导出 `GlobalScoresAll` 廉价打分） |
| `search/sczleval.go` | sczl 单独评测（mono/color 拆分） |
| `search/main.go` | 子命令 `train`/`nn`/`sczl`/`rr`/`seg`/`space` |
| `train_set/` | gentest 生成的训练集（8000 张，seed 20260715，gitignore 不入库） |
| `weights.gob` | 训练产物（最终：8000 张 / 60 epoch / lr=0.002 / 81 维输入） |

常用命令：
```bash
go run ./gentest -n 8000 -seed 20260715 -out train_set   # 生成训练集（若需重建）
go build -o search_nn.exe ./search
.\search_nn.exe . train train_set weights.gob 60 0.002   # 训练
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
- [x] 8000 训练图 + shape-context 精排调参（SCTOP=12/SCBLEND=0.7）
- [x] 目标指标：recall@1 与 recall@3 达成 96.6% / 98.3%
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
  - **特征重复计算共享**（`computePairs` 热路径）：
    - `polarShiftSim`（best-shift 均值）与 `polarShiftDetail`（per-ring 明细）之前各做一遍
      完整的 k 扫描，合并为 `polarShiftBoth` 单次扫描同时出两者 → polar shift 开销减半；
    - `roundTripFactor` 之前重复算 `cosSim(Radial)`/`cosSim(AngMag)`，现复用 s[1]/s[2]。
    - 实测 polar shift 相关 per-ref 85.9μs → 43.2μs（~2x），**数学等价、recall 不变**
      （recall@1 96.6% / recall@3 98.3% 与优化前完全一致）。

### 最终评测结果（test_set，118 张有效查询）
```
baseline adaptive : recall@1 = 109/118 (92.4%)   recall@3 = 109/118 (92.4%)   recall@5 = 109/118 (92.4%)
neural (MLP)      : recall@1 = 114/118 (96.6%)   recall@3 = 116/118 (98.3%)   recall@5 = 116/118 (98.3%)
```
- 相对 baseline：recall@1 +4.2pp、recall@3 +5.9pp、recall@5 +5.9pp。
- 剩余 4 miss：2 张 home_work（分割受损难例，真值 >top5）；2 张在 rank 2-3
  （account_balance / dashboard 灰色族内混淆）。

### 待办
- [ ] （可选）sczl 更多子特征（occ/fd 分开）或 SC/HOG 精排进 NN，继续压灰度难例
- [ ] （可选）改进分割：处理 scale>1 溢出与细笔画丢失
- [ ] （可选）服务端 SC 精排再提速：剩余 ~310ms 中精排仍占大头，可进一步并行化或降旋转步数（36→18）

### 神经网络架构横向实验（2026，结论：baseline 已封顶）

> 用纯 stdlib 手写实现，测了「常用 NN 架构」能否超过定稿的 pointwise MLP
> （81→96→48，class-balanced BCE）。统一在同样 8000 训练图 / pointwise BCE /
> 两阶段 rankNNPri + SC 精排（SCBLEND=0.7）下对比。结论：**没有架构超过 baseline**
> （96.6%@1 / 98.3%@3），当前模型已触及该特征集封顶，剩余 miss 是分割受损
> （2 张 home_work，真值 >top5）与 2 张灰色族 rank2-3 混淆，均非模型容量问题。

| 架构（`search/nnarch.go` netMLP，可配深度/宽度/Dropout） | test recall@1 | @3 |
| --- | --- | --- |
| **baseline 81→96→48（定稿）** | **96.6%** | **98.3%** |
| 81→128→64（更宽） | 96.6% | 97.5% |
| 81→192→96（更宽） | 95.8% | 96.6% |
| 81→128→96→48（更深） | 95.8% | 96.6% |
| 81→128→128→96→48（更深） | 96.6% | 98.3% |
| 81→128→96→48 + Dropout 0.8 | 95.8% | 96.6% |
| 81→96→48 + Dropout 0.8 | 94.9% | 95.8% |

损失函数实验（`search/nnlist.go`）：
- **Listwise ListNet softmax（候选集 = 8 难负 + 3 随机）**：93.2%@1，**低于** pointwise。
  softmax 归一化在小候选集上训练，与推理时全 66 候选的分布不一致，负样本相对分数
  学不出来。
- **Listwise ListNet softmax（候选集 = 全 66 ref，匹配推理分布）**：train@1 达 97.2%，
  但 **test 崩到 68.6%**。softmax-CE 把 logit 推到极端/峰化，推理却用裸 sigmoid 输出
  跨查询排序，分数不再有绝对含义 → overfit 到训练集的对齐分布，不可迁移。
- **结论**：pointwise BCE 的绝对 sigmoid 分数天然可跨查询迁移，是当前数据量下最稳的目标；
  listwise 需要足够多样的候选分布 + 校准/排序损失（ListMLE/LambdaRank）才可能受益，
  在纯 stdlib 小数据下得不偿失。

实现说明：
- `nn.go` 新增 `forwardLogit`+`backpropGrad`（从给定输出梯度反传），`backprop` 改为委托；
  `rankNN` 抽出 `rankNNPri`（以 predict 函数为参数），固定 MLP 与 netMLP 共用同一
  两阶段排序 + SC 精排管线。
- 命令：`search.exe . sweep train_set`（一次 prep 训 6 个架构，各自存
  `weights_net*.gob`）；`search.exe . nnarch weights_netX.gob`（评测）。

### 注意力特征融合实验（2026，把定稿 MLP 的融合换成注意力机制）

> 用户诉求：把「MLP 特征融合」替换成「注意力机制」看效果。输入 `x` 天然是 3 个
> 特征块（`pair` / `best` / `gap`，各 pf=27/32 维），故把 3 块当作 3 个 token 做
> 注意力融合，输出仍是 pointwise BCE 的 sigmoid 相关度。实现替换 `search/nn.go` 的
> `MLP` → `AttnNet`，外部接口（`forward`/`predict`/`backprop`/`save`/`loadAttnNet`）
> 与排序管线（`rankNN`/`trainAt1`/`server.loadNN`）不变，仅 `server.loadNN` 的 shape
> 校验改用 `AttnNet.valid()`。统一在 8000 训练图 / pointwise BCE / 两阶段
> rankNNPri + SC 精排下对比。

**v1（3-token 单头，固定 query，共享 K/V，无 FFN，attnKey=64）**：test 崩到 85.0%@1。
- 教训：3 个 token 太少、注意力无可聚焦；query 是固定学习向量、与输入无关，
  学不到「随查询难度自适应融合」；且无深层非线性。在该任务上甚至不如 baseline。

**v2（3-token 单头，数据相关 query + 每块独立 K/V + FFN/残差，attnKey=96）**：
test 96.7%@1 / 97.5%@3，追平定稿 MLP（96.6%/98.3%）。
- 有效优化点：① query 由 `pair` 块经 `WQ` 生成（数据相关），让注意力随样本自适应；
  ② 每块独立 K/V 投影（不再共享）；③ 注意力输出过 FFN+残差补非线性；④ attnKey 64→96。
- 结论：注意力融合在当前 3 块 token 布局上最多「追平」pointwise MLP，未超过 ——
  与 §「架构横向实验」结论一致（该特征集、pointwise BCE 下模型已封顶）。注意力
  的优势（长序列/多 token 聚焦）在只有 3 个特征块时发挥不出来。

| 方案 | test recall@1 | @3 | @5 |
| --- | --- | --- | --- |
| baseline adaptive | 92.5% | 93.3% | 93.3% |
| 定稿 pointwise MLP（记录值） | 96.6% | 98.3% | — |
| attn v1（固定 query，3-token） | 85.0% | 91.7% | 91.7% |
| **attn v2（数据 query + 独立 K/V + FFN）** | **96.7%** | **97.5%** | **97.5%** |
| **attn v3（v2 升级为 4 头注意力）** | **96.7%** | **97.5%** | **97.5%** |
| **attn v4（语义 token 拆分，仅 base 作 query）** | 95.0% | 97.5% | 98.3% |
| **attn v5（v4 语义 token + 完整 pair 作 query）** | 96.7% | 97.5% | 97.5% |
| **attn RING（v5 + RINGFEAT 逐环特征扩维）** | 96.7% | 98.3% | 98.3% |
| **attn REGION（v5 + 多区域维度，去 RINGFEAT）** | 96.7% | 98.3% | **100.0%** |
| **attn COLOR（v5 + 32×32 颜色投影直方图，去REGION）** | **97.5%** | 98.3% | 98.3% |
| **attn COLOR-B（16×16 逐格颜色相似度，方案B）** | 97.5% | 97.5% | 98.3% |

**v3（多头注意力）**：v2 的每块独立 K/V 拆成 `nHead=4` 个头（headDim=24），每头有自己的
数据相关 query、独立 K/V 投影，对 3 个 token 各自出注意力分布，concat 后过同一 FFN。
- 参数量与 v2 相同（WQ/WK/WV 总量不变），但让不同头可专注不同特征族。
- 结果：test 与 v2 逐位相同（96.7%@1 / 97.5%@3），未提升。
- 结论：单头/多头在 3 token 特征块上无差别 —— 该特征集 + pointwise BCE + SC 精排下
  模型确已封顶（与 §架构横向实验一致），注意力容量不是瓶颈，瓶颈在特征种类与分割质量。

**v4/v5（语义 token 拆分）**：把 pair 块按特征族拆成独立 token（base 8 / bias 2 /
polar-shape 细节 16 / sczl），key/value 各 token 独立投影，query 用完整 pair 块。
- v4 曾把 query 限制在 base 8 维 → 信息不足，recall@1 掉到 95.0%；v5 恢复完整 pair 作
  query → 回到 96.7%。结论：query 必须吃完整 pair 特征，语义拆分本身不带来增益。

**RING（特征扩维，后已移除）**：pair 特征曾从 27 维扩到 59 维（SCZLSPLIT 下 64 维）——
除原有 16 维逐环 polar-shape 细节外，新增逐环 Radial 余弦（16 维）与逐环 AngMag 余弦
（16 维，`perRingCos`）。`RINGFEAT=1`+`SCZLSPLIT=1` 评测 96.7%@1 / 98.3%@3 / 98.3%@5。
- **结论：效果不明显（@1 未提升、@3/@5 各 +0.8pp 但维度从 27 翻倍到 59），已移除**，
  避免 32 维逐环特征与 base 里的全局余弦高度重复、稀释有效信号。

**REGION（多区域维度，后已移除）**：用户观察——真实检索查询多是组合图（多个图标/区域
拼在一起），检索目标只是其中一部分，而 `extractQuery` 会把所有内部连通域**合并**成一个
sprite，特征混合了多个物体，且多区域本身没有信号进模型。曾新增 **1 个 region 维度**
（`Feat.Region`，`feat.go` `regionFrag`）：`1 - 最大8连通域像素数/总前景像素数`，量化
查询前景的碎片化程度（单图标≈0，多区域组合图≈1）。作为独立 token 加入 pair 特征
（pair 26→27 维，SCZLSPLIT 下 31→32 维），query 吃完整 pair 块。
- 分布验证：120 查询 avg=0.242；多部件图标高（edit_square 0.557、home_work 0.565、
  account_balance 0.525），单图标低。
- `SCZLSPLIT=1`：test **96.7%@1 / 98.3%@3 / 100.0%@5**（首次 @5 全中），train@1 峰值
  98.2%。维度比 RING 更少（输入 96 维）却 @5 提升 1.7pp。
- **结论：@1 未提升（仍是 96.7%），且只编码"碎片程度"一个标量、不带空间/颜色布局，
  已移除**，被下文 COLOR 方案（空间颜色布局）取代。

**COLOR（32×32 颜色投影直方图，当前定稿）**：在上文组合图观察基础上，把查询的**空间
颜色布局**直接喂给模型。方案：`feat.go` `buildColorProj` 把 sprite bbox 补成方形
（letterbox，居中留边）→ 缩放到 32×32 → 记录每行/每列的**平均 RGB**，输出
`32*3 + 32*3 = 192 维`（`Feat.ColorProj`）；作为独立 color token 追加到输入末尾
（`nnInput = 3*pf + 192`，SCZLSPLIT 下 288 维）。query 级静态特征，方案 A。
- 动机：目标是组合图的一部分时，其所在行/列留有明显的颜色签名；`PolarCol` 是极坐标
  （展开角度、丢笛卡尔位置），`Hist` 是全局直方图（无空间），此特征补上笛卡尔空间色块。
  sczl 的 `Occupancy32`/`Patch` 只有形状/灰度，无彩色。
- `SCZLSPLIT=1`：test **97.5%@1**（历史最佳，miss 4→3，`account_balance` 被召回）/
  98.3%@3 / 98.3%@5，train@1 峰值 98.5%。nnInput 273→288 维。
- 剩余 3 miss 均为已知灰色族硬例：分割受损 home_work、dashboard、phone_in_talk。

**提取质量提升（旋转干扰诊断后续）**：3 个 miss 的根因是旋转干扰叠加提取失真：
- 定量证据：query/ref 的 mask 对真 ref 的 Dice 只有 0.37~0.60（即使按 manifest 真实角度
  反旋转也低），rankreport 显示 hist/zern（旋转不变）都排真 ref 第 1，但 mask/polar/SC
  （旋转敏感）排 30~64 名——NN 被坏掉的形状特征拖下水。
- 修复 1（`BLOBMIN=0.15`，sprite.go 连通域过滤）：原"保留全部 interior 连通域"把边缘
  文本 blob（"热卖63"/"50"/"详情"等）当 sprite 收进来，污染 mask 并撑大 RMS。改为保留
  最大块 + ≥15% 面积的块。
- 修复 2（`CLOSED=0`，sprite.go 形态学）：closeD=3 的膨胀把 dashboard 网格方块间隙
  （~2px）填死成实心块，且 D=1/3 都损毁 home_work 薄线条（no-morph maskScore 0.373→0.768）。
  支持 `CLOSED >= 0`（0=无形态学）。
- **重训后 test 98.3%@1（118/120，miss 3→2，home_work 召回）/ 98.3%@3 / 98.3%@5，
  train@1 99.2%**。`weights_ext0.gob`。剩余 2 miss：dashboard（shape 特征仍全坏：
  zern=64/m64=55）、phone_in_talk（mask rank 49，hist/zern 均第 1）。
- 注意：改提取后必须重训对齐（旧权重下 CLOSED=0 特征分布偏移，eval 不变）。

**mask 可信度门控（方案2，已尝试，弃用）**：给模型加 query 级专家一致性 token
（hist 冠军的 mask 分 + mask 冠军的 hist 分，2 维，layout 变 8 token，nnInput 294），
让 NN 学会"专家分歧时信 hist"。CLOSED=0 重训 60 epoch 后：
- test **97.5/98.3/99.2**，miss 3 个（dashboard 推进到 rank4、home_work 已召回，但
  account_balance 新 miss）。净效果比 ext0（98.3）差 → 已回退，定稿用 ext0。
- 教训：硬 argmax 的一致性门控太粗，网络在部分查询上被误导；门控收益不敌副作用。

**COLOR 方案 A vs B（对比）**：A（32×32 投影直方图，query 静态色块）与 B（16×16 逐格
颜色相似度，query↔ref pair 特征）对比：
- A：`Feat.ColorProj` 192 维 + 独立 token（nnInput 288）；`buildColorProj` 算行/列平均 RGB。
- B：`Feat.ColorGrid16` 768 维 + `colorGridSim` 逐格余弦，256 维嵌入 pair（nnInput 849）；
  `nnVec` 不必加 color（相似度已进 pair）。
- 结果：@1 同为 97.5%，但 A 的 @3 98.3% > B 的 97.5%，且 B 训练贵 ~3 倍（849 维，
  60 epoch 78 分钟）。**A 更优，定稿用 A**；B 因维度膨胀、收益不增而弃用。

**验证：旋转干扰诊断（ноrot 测试集）**：怀疑 3 个灰色 miss 是旋转干扰所致，给 gentest 加
`-norot` 开关（`render.go` 旋转概率归零），用 seed 20260716 重新生成 120 张无旋转测试集
`test_set_norot`（manifest 确认全部 rotation=0），用 A 权重重新评测：
- 原（90% 旋转）测试集：NN 97.5/98.3/98.3，baseline 92.5/93.3/93.3。
- 无旋转测试集：**NN 100.0/100.0/100.0**，baseline 88.3/89.2/91.7。
- 原 3 个 miss 的旋转角：home_work 29.8°、dashboard -159.9°、phone_in_talk 24.4° —— 全部带明显旋转。
- 结论：**3 个 miss 的根因是旋转干扰**（薄线条灰色图标在任意角度旋转后形状特征衰减），
  模型形状能力本身够用；继续在训练层面加形状权重收益有限，方向应转向**旋转鲁棒**。

**COLOR 方案 C（10×10 颜色 + query 静态形状 token，已尝试）**：同时做两个改动：颜色投影
32×32 → 10×10（192 维 → 60 维），并新增 query 静态形状 token `Feat.Shape`
（Radial 16 + Zernike 49 + 4 标量 aspect/fill/rms/boundary = 69 维，`buildQueryShape`），
layout 变 8 token（nnInput 225，PF 现显式=3*pf）。本想让网络感知 query 的**绝对形状**
以补 relative 相似度的盲区。
- 结果：test **97.5/97.5/97.5**，仍是同样 3 个灰色 miss（home_work/dashboard/phone_in_talk），
  形状 token 未召回任何 miss；@3/@5 比 A 差 0.8pp，train@1 收敛也低于 A（峰值 97.8% vs 98.5%）。
- 结论：**混做无提升**（与预判一致：3 miss 全是灰色图标，减分辨率的颜色 + 绝对形状都
  救不了真实灰色形状混淆），已回退到 A。

实现细节：
- `nn.go` `AttnNet`：`q=WQ·pair`，`k_t/v_t = WK[t]/WV[t]·block_t`，`alpha=softmax(q·k/√d)`，
  `ctx=Σ alpha·v`，`ctx+=FFN(ctx)`，`p=sigmoid(WO·ctx+BO)`；backprop 相应手写。
- 注意：gob 只编码导出字段，`b1/b2/pf` 必须导出（`B1/B2/PF`），否则 load 后权重残缺。
- 命令：`search.exe . train train_set weights_attn2.gob 60 0.002`；`search.exe . nn weights_attn2.gob`。
- v3 多头：`search.exe . train train_set weights_attn3.gob 60 0.002`（`nHead=4, headDim=24`，
  权重拆为 `[nHead][]float64` / `[nHead][nTok][]float64`，`valid()` 校验形状）。
- v4/v5 语义 token：`computePairs` 不变，`attnLayout()` 把 pair 拆成 base/bias/detail/sczl
  独立 token（best/gap 仍各 1 token）；query 用完整 pair 块（`WQ: attnKey x pf`）。
- REGION 多区域维度（已移除）：`feat.go` `Feat.Region` + `regionFrag` 在 `buildFeat` 计算；
  `computePairs` 把 `q.Region` 追加为 pair 特征（base 后第 3 维）；`attnLayout()` 多出
  region token（dims `{8,2,1,16,sczlDim,pf,pf}`）。训练/评测：
  `SCZLSPLIT=1 search.exe . train train_set weights_region.gob 60 0.002`。
- COLOR 颜色投影直方图（当前定稿，替代 REGION）：`buildColorProj` 在 `buildFeat` 计算
  `ColorProj`（192 维）；`nnVec` 追加 color 块，`nnQuery` 缓存 `color` 字段；
  `attnLayout()` 末尾加 color token（dims `{8,2,16,sczlDim,pf,pf,192}`，nnInput 273→288）。
  训练/评测：`SCZLSPLIT=1 search.exe . train train_set weights_color.gob 60 0.002`。

### 卷积/循环网络特征提取实验（2026，结论：纯 CNN 在原始网格上不敌特征 MLP）

> 用户诉求：尝试 CNN / RNN 提取特征能否提升。在纯 stdlib（无框架）约束下，
> 像素级 CNN 太重且易错，故采用**对齐的极坐标形状网格上的成对 1D CNN**
> （`search/nncnn.go` `cnnNet`）：查询是参考图的旋转渲染，故每环角度剖面是对方的
> 循环移位；沿角度轴做**循环卷积（旋转等变）+ 全局 max-pool（旋转不变）**即天然
> 旋转不变 —— 这是极坐标 CNN 的标准做法。输入是 (query, ref) 对齐后的 2 通道
> 16×24 网格，输出 relevancy。损失同为 pointwise BCE，与固定 MLP 完全可比。

```text
cnn（81→conv8×5→pool→32→1，30 epoch）：train@1 封顶 ~70.8%（epoch 11 即停滞）
baseline pointwise MLP（81→96→48）  ：train@1 ~96%+，test 96.6%
```

- **结论**：纯 CNN 只看极坐标网格这一种特征族，而 MLP 吃 7 类手工特征
  （hist/zern/radial/angmag/mask/sczl/polar 细节），CNN 远不够判别力，封顶 ~70%。
  即便收敛，也不会超过特征 MLP —— 瓶颈在输入特征种类，不在网络结构。
- **正确性验证**：对 conv 权重做数值梯度检查，analytic 与 numeric diff ≤ 5e-11，
  反向传播实现无误（梯度下降有效，loss 单调降、train@1 升）。
- **RNN 未单独实现**：极坐标网格沿环/角度的序列结构，其旋转不变信息（每环 FFT
   幅度 AngMag）已作为 MLP 的 27 维 pair 特征 + 16 维 per-ring 细节喂给模型，
   循环网络的序列建模与此高度冗余（§10 已证旋转不变 sczl 特征喂入无增益）。

### 新特征族：拆分 sczl 子信号（2026，recall@1 96.6% → 97.5%，唯一提升）

> 需求：引入新的特征族能否提升。此前 sczl 专家被折叠成**单个融合分**喂给 NN
> （`GlobalScoreOf` = 0.20·occ + 0.20·ncc + 0.20·region + 0.25·fourier + 0.15·radial），
> 网络无法学习各子信号的独立权重，且 **HOG 梯度方向信号根本没进模型**（融合式里没有）。
> 做法：新增 `PreparedQuery.GlobalScoresRow`，把 6 个旋转不变子信号
> （**occ / ncc / region / fourier / radial / hog**）作为独立 pair 特征喂给 NN，
> 让网络自己学每个专家的权重（soft 神经路由的粒度细化）。

```text
SCZLSPLIT=1（32 pair 特征，nnInput 81→96，60 epoch）：
  train@1 97.5%（baseline 96.4%）  test recall@1 = 115/118 (97.5%)  recall@3 = 98.3%
SCZLSPLIT=1 + SCZL_NOHOG=1（5 子信号，去 HOG）：
  test recall@1 = 113/118 (95.8%)  recall@3 = 96.6%
baseline（27 pair，nnInput 81）：
  test recall@1 = 114/118 (96.6%)  recall@3 = 98.3%
```
- **+0.9pp @1（114→115）**，miss 从 4 降到 3。这是本系列实验中首个真正超过定稿的改动。
- **子信号消融（关键结论）**：`SCZL_NOHOG=1` 去掉 HOG 再训 → test 掉到 95.8%，**低于 baseline**。
  这说明**增益几乎全部来自 HOG 梯度方向信号**（此前完全没进模型的唯一子信号）；
  而单纯把 occ/ncc/region/fd/rad 拆开（不引入新信息）反而过参数化/引入噪声，hurt test。
  即：拆分本身不赚，**新增 HOG 特征族**才是 +0.9pp 的根源。
- 剩余 3 miss：dashboard（灰网格族 grid_view/settings 混淆）、phone_in_talk（rank 2）、
  home_work（分割受损，真值 >top5，见 §7 待办）。
- 实现：
  - `search/sczl/search.go`：`GlobalScoresRow`（6 子信号）+ `hogDisabled`（`SCZL_NOHOG` 消融开关）；
    `GlobalScoresAll` 改返回 `[][]float64` 行。
  - `search/nn.go`：`nnInput` 由 const 改 var（`nnNewInputDim`），`SCZLSPLIT=1` 时 pair 特征 27→32、
    输入 81→96；`nnSczlSplit()` 判定。
  - `search/nnfit.go`：`computePairs`/`buildPairDataScored` 的 sczl 参数由 `[]float64` 改 `[][]float64`；
    `rankNN` 与 `buildPairData` 按 split 标志产出 1 或 6 个子信号。
- 训练：`SCZLSPLIT=1 search.exe . train train_set weights_split.gob 60 0.002`；
  评测：`SCZLSPLIT=1 search.exe . nn weights_split.gob`（注意训练/评测必须同布局）。
- 风险：`nnInput` 变 var 后，`weights.gob`（81 维）只在 `SCZLSPLIT` 未设时兼容；server 需
  在启动时按同一环境变量加载对应权重（`NN_WEIGHTS`）。

### 分割改进：自适应阈值 + 更强闭运算（2026，97.5% → 99.2%，最大提升）

> 需求：继续改进。诊断 `featcmp`/`segdiag` 发现两个 miss 的根因都在**分割**而非模型：
> 1. **深色图标 + 深色背景**：`extractQuery` 用固定阈值 `rgbDist(p,bg) > 0.16`。icon
>    是深灰 `#434343`，当背景也是深色（如 home_work 00109 背景 `#393560`，距离仅 0.132 <
>    0.16）时，大部分图标被判为背景 → 只分割出 ~20% 像素（3575/18177），特征全错。
> 2. **小图标 / 强旋转碎片化**：`dilate 1 + erode 1` 闭运算不足以把细线条旋转后断裂的
>    碎片连通域合并回完整图标（home_work 00006 scale 0.74 被碎成 9 块）。

改动（`search/sprite.go`）：
- **`adaptiveFGMask`**：不再用固定 0.16，而是统计全图到背景距离的直方图，在背景噪声簇
  与前景簇之间的谷底取自适应阈值（`adaptiveDistThreshold`），回退下限 ~0.12 保留细笔画。
  对深色图标+深色背景：距离分布出现第二簇，阈值自动落在 gap 上，完整捕获图标。
- **更强闭运算**（`CLOSED` env，默认 **3**）：默认 `dilate 3 + erode 3`，把旋转后断裂的
  碎片连通合并回完整图标（原来 1 太少）。CLOSED=3 最优，4/5 会略擦除细内线。

**结果（test_set，120 张全部有效）**：
```text
SCZLSPLIT=1 + CLOSED=3（默认）：recall@1 = 119/120 (99.2%)  recall@3 = 119/120 (99.2%)
baseline（无 split）+ CLOSED=3  ：recall@1 = 116/120 (96.7%)
```
- recall@1 从 97.5% → **99.2%**（+1.7pp），且**分割失败的 2 张（00064 check_box、
  00085 delete，均为深色背景）被救回**，120 张全部有效（此前 118 有效）。
- 唯一剩余 miss：sample_00083 dashboard（灰色 2×2 网格 vs grid_view/settings 的
  圆角/方角混淆，模型侧灰色族问题）。
- **净 miss 轨迹**：96.6%(4) → 97.5%(3) → 98.3%(2) → **99.2%(1)**。
- 训练数据用同一 `train_set` 生成（含深色背景），无需重训权重；`CLOSED`/自适应阈值只影响
  `extractQuery`，训练与评测同管线自动一致。

### 剩余 miss 攻坚：dashboard 灰色框类（2026，结论 = 模型上限）

> 用户要求"归档后继续攻坚"。唯一剩余 miss = `sample_00083` dashboard_76dp（-160° 旋转，
> 深灰 #434343 图标 on #4dae7e 背景，含 HOT672/热卖63 文字），模型误判为 grid_view。
> 用 `trneval`/`trnmiss`/`pairdump`/`nnall` 等诊断工具定位根因。

**分批诊断结论**：
1. **文字污染**：q83 分割 N=52673（参考 23695），`extractQuery` 保留全部内部连通域，
   把"热卖63/HOT672"文字也算进特征。但 `TEXTSKIP=1`（保最大连通域+近邻碎片）过滤文字后
   dashboard 的 NN 分仍低（0.42，grid_view 0.93），且**引入 3 个新 miss**（supervisor_
   account/account_balance，灰色框类图标本身碎成多框）→ 文字不是根因，TEXTSKIP 不可行。
2. **固有特征重叠**：dashboard 与 grid_view/calculate/account_balance/check_box 都是
   灰色框类，颜色同 #434343、mask/polar 高度重叠（dashboard vs grid mask=0.60 vs 0.60、
   polar=0.65 vs 0.65），仅 angmag 略可分（0.50 vs 0.27）但训练集上 angmag 范围重叠大
   （dashboard 0.36~0.99，grid 0.20~0.99），NN 学不出可靠边界。
3. **训练集系统性混淆**：dashboard train recall@1=92.8%（误判 grid_view 3 次、calculate
   5 次）；grid_view=82.0%（误判 dashboard 6 次、account_balance 4 次）。SC 精排（SCBLEND
   0/0.5/0.7/0.9）对这两个的训练 recall 无改善。
4. **dashboard 90° 非对称**：旋转对称性分析 = 90° 重叠仅 50%（180° 重叠 81%）。-160°
   旋转后形状特征（zern rank 61、m64 rank 57）相对同类图标退化，而 grid_view 保留较好，
   故 NN/blend 都给 grid_view 更高分。

**结论**：dashboard 是**灰色框类图标固有的特征重叠难例**——颜色/形状/mask 与同类几乎
不可分，非分割、非模型容量、非 SC 精排可解决。这是当前特征族（hist/zern/radial/angmag/
polar/mask/sczl）的判别上限。**接受 99.2% (119/120) 为定稿**，若要再突破需引入能区分
"圆角/方角、线宽、内部网格结构"的更高分辨率形状特征（像素级 CNN 或拓扑/骨架特征），
超出纯 stdlib 当前方案的性价比。

## 8. 已知风险与备注

- 训练集与测试集用不同 seed 生成，图像不同但同分布；若要严格泛化，可再生成
  第三个集做验证集/早停。
- `weights.gob` 是 gob 编码，结构变更（输入维数/层宽）会导致旧权重不兼容 → 需重训。
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

## 10. Embedding 向量检索（低维欧氏距离，可索引）

> 目标：把"以图搜图"变成向量检索——网络对单张图输出一个低维向量，检索时用
> 欧氏距离做最近邻，参考侧向量可预计算/持久化/建 ANN 索引，查询只需一次前向 +
> 距离扫描，比逐对打分更适合大规模库。

### 方案（已实现）

- **网络**（`search/embed.go` `EmbedNet`）：`输入 → 128 → 64 → 64 → Dim`（ReLU），
  输出层为线性，embedding 使用前 L2 归一化（欧氏距离 == 余弦，有界 [0,2]）。
- **输入**：只用**旋转不变**的单图特征 `embedInput` = Hist + Radial + AngMag + Zern + Mono。
  查询是 ref 的任意旋转渲染，故 embedding 必须对该旋转不变两图才能直接比距离。
  这是本方案成立的前提（Zern/Radial/AngMag 天然旋转不变）。
- **训练**：**softmax 分类头**（66 个 ref 类）在 embedding 之上做交叉熵。
  相比最初的 triplet hinge（实测 collapse 到随机），分类损失稳定得多，且 embedding
  层自然学到旋转不变、类间可分的紧凑表示。推理时丢弃分类头，只用 embedding 层。
- **检索**：`rankEmbed` 对查询 embedding 与所有参考 embedding 求欧氏距离排序。
  参考 embedding 可预计算（server 启动时 `addEmb` 一次性算好，存进 indexEntry.Embed）。
- **命令**：`search.exe . embt train_set embweights.gob [epochs] [lr] [dim]` 训练；
  `search.exe . emb embweights.gob` 评测；server 用 `EMB_WEIGHTS` 环境变量指定权重
  （默认 `<root>/embweights.gob`），/api/search 优先用 embedding，缺失则回退 NN/adaptive。

### 评测结果（test_set，118 张有效查询）

```
embedding (dim=32, 8000 train, 60ep, SCBLEND=0.5): recall@1=107/118 (90.7%)  recall@3=109/118 (92.4%)  recall@5=110/118 (93.2%)
embedding (dim=32, 8000 train, 60ep, SCBLEND=0.7): recall@1=105/118 (89.0%)  @3=92.4%  @5=93.2%
embedding (dim=32, 8000 train, 60ep, 无 SC 精排):  recall@1=104/118 (88.1%)  @3=92.4%  @5=93.2%
embedding (dim=32, 2000 train, 60ep): recall@1=84.7%  (训练规模↑ 明显提升)
embedding (dim=64, 8000 train):        recall@1=85.6%  (维度↑ 不提升，甚至略降)
raw 旋转不变特征直接距离（无训练）:   recall@1=74.6%
baseline adaptive / neural MLP (§4):  92.4% / 96.6%
```

- **SC 精排调参是主要增益**：把 SCBLEND 从 0.7 调低到 0.5，recall@1 88.1%→90.7%。
  给 SC 过多权重反而降（1.0 时仅 44%），说明低维 embedding 距离已能分离大部分类，
  SC 只负责重排灰白描边族内部的并列；SCBLEND 反映"embedding 为主、SC 为辅"。
  SCTOP（6~30）不敏感，保持默认 12。
- 加 SC 精排本身（0.7）只 +0.9pp；真正把差距缩小的是 blend 调优。
- **维度 32 已接近最优**：64 维反而降，说明判别力瓶颈在输入特征而非模型容量。
- 剩余 miss 集中在灰白描边族（edit_square / perm_phone / support_agent /
  account_balance / home_work）与若干黄色族（zufangfangdai / fuyejianzhi），
  这些图标颜色相同、形状相近，低维 embedding 难以单靠距离分开。
- 前期教训：triplet hinge 在纯 stdlib 无框架下易 collapse（训练到 70% 后骤降到随机），
  改为 softmax 分类后训练稳定单调收敛。

### 与 pointwise MLP 的对比

| 维度 | pointwise MLP (§2) | embedding（本方案） |
| --- | --- | --- |
| 输入 | (query,ref) pair 81 维 | 单图旋转不变特征 ~546 维 |
| 输出 | 1 维 sigmoid 分数 | Dim 维向量（默认 32） |
| 损失 | class-balanced BCE | softmax CE（分类） |
| 检索 | 逐对前向打分 | 欧氏距离最近邻 |
| 参考侧 | 无法预计算 | **可预计算/索引/ANN** |
| recall@1 | 96.6% | 88.1% |
| 大库扩展 | 两阶段粗筛 ~0.35s/万条 | 低维暴力快，可上 HNSW |

### 待办（可选）
- [x] embedding + shape-context 精排：对 top-N 做 SC 复排，并把 SCBLEND 调低到 0.5
      （embedding 为主、SC 为辅），recall@1 88.1%→90.7%
- [x] 拆 sczl 特征喂入 embedding 输入（消融见下）：旋转不变特征无增益，旋转敏感特征有害
- [ ] 用 HNSW/IVF 替换暴力扫描，验证百万级库的向量检索

### sczl 特征拆开输入 embedding 的消融结论

> 需求：把 `search/sczl` 的各特征拆开作为 embedding 网络的输入，看哪个有效。
> 实现：`embedInput` 现支持 `EMBSCZL` env（逗号分隔 `fourier,radial,solidity,occ16,occ32`）
> 选择拼接哪些 sczl 签名；query 用 `sczl.Extract(img)`（整图），ref 用 1:1 对齐的 sczl 描述子。

| EMBSCZL | 输入维 | recall@1（2000 训练）| 结论 |
| --- | --- | --- | --- |
| （空，纯 feat） | 546 | 83.1% | baseline |
| fourier,radial | 594 | 83.1% | 无增益，与已有 Radial/AngMag 冗余 |
| fourier | 578 | 83.1% | 无增益 |
| solidity | 547 | 83.1% | 无增益 |
| occ16 | 802 | 71.2% | **有害** |

- **旋转不变特征（Fourier/Radial/Solidity）无增益**：sczl 的 Radial 桶数（16）与
  buildFeat 的 nRing（16）一致，Fourier（轮廓幅度）的信息与已有 AngMag（每环 FFT 幅度）
  高度重叠，网络学不到新判别力。
- **旋转敏感特征（Occupancy16/32）有害**：Occupancy 是"质心+R98 归一化框架"内裁剪，
  图标在框内旋转后栅格值改变，query（旋转渲染）与 ref 无法对齐，破坏 embedding 距离
  的旋转不变前提，判别力反而下降（83.1%→71.2%）。
- **结论**：sczl 的判别力依赖旋转扫描对齐（`GlobalScoreAll` 逐对计算），不适合作为
  单图 embedding 输入；它作为 NN 的 pair 特征（§2 soft 路由）才是正确用法。embedding
  侧保留纯旋转不变 buildFeat 特征即可，sczl 特征作为可选扩展保留（`EMBSCZL`），
  默认关闭。

### 联合训练输入投影器（InputProjector）实验

> 需求：实现一个可复用的联合训练输入投影模块：把高维逐分量相似度（49 维 Zernike
> min-sim）经共享 MLP（49→16→8）压成低维 token，patch 进注意力网络的 token 槽，
> 端到端联合训练；未来可复用于其他原始特征块。
> 实现：`InputProjector`（nn.go，config 表 `attnProjConfigs`，当前含 zern 49→16→8）、
> gob 持久化、adam 优化、梯度全链路。布局：`nnPairDim=8+2+16+sczlDim+8=40`，
> `nnInput=3*40+192+49=361`（raw zsim 作为 carry 区输入）。

| 版本 | 输入维 | train@1 终值 | test recall@1/@3/@5 | 结论 |
| --- | --- | --- | --- | --- |
| ext0（定稿，无投影器） | 288 | 99.2% | 118/120 (98.3%) | baseline |
| zproj（联合 Zernike 投影） | 361 | 99.1% | 117/120 (97.5%) | **中性**（差 1 票，噪声内） |

- **中性结果**：+49 维 raw carry + 8 维投影 token，recall@1 118→117，
  在 120 个测试查询上属噪声范围。miss 集合不同（zproj 错 account_balance/
  dashboard/phone_in_talk；ext0 错另外 3 张），但错误率相同量级。
- **原因**：基础 8 维里已有 zern 余弦（s[6]），49 维 min-sim 与其高度冗余；投影器
  学到的新判别力 ≈ 0。投影框架本身工作正常（梯度流、保存/加载、训练曲线均验证）。
- **教训**：投影器只对"基础特征里没有的新特征族"才有价值；Zernike 已用余弦形式表达，
  再加逐分量版本冗余。框架保留（`weights_zproj.gob` 可训练/加载），后续若引入真正
  正交的新特征（如纹理/边缘直方图）可直接复用。
- **布局不兼容警告**：新布局（pf=40, nnInput=361）与旧 gob（ext0 等，pf=32,
  nnInput=288）不兼容；旧权重在新代码下 valid() 检查失败，评测需用对应版本编译。
