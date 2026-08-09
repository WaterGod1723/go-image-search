# 神经网络检索排序（NN Ranker）方案与 TODO

> 目的：在本项目（Go、纯 stdlib、无第三方 ML 框架）中引入可训练的神经网络，
> 提升以图搜图 recall@1。本文档记录方案、实现位置、命令与当前进度，
> 供上下文压缩后恢复记忆使用。

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
- [x] **检索/索引提速**（见 §6）：
  - shape-context 精排优化：查询旋转直方图每查询只算一次（12 候选共享）、参考侧 SC 直方图按 ref 惰性缓存、
    采样点 120→48（Hungarian O(n³) 降 ~15 倍，**recall 无损失**）→ 服务端单查询 ~1s 降到 **~310ms**
  - sczl 参考描述子持久化进 `.searchcache/index.gob` → 重启索引重建 2100ms 降到 **~334ms**（不再重解码 PNG）
  - 启动自动构建索引（无缓存时），免手动"构建索引"

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

## 8. 已知风险与备注

- 训练集与测试集用不同 seed 生成，图像不同但同分布；若要严格泛化，可再生成
  第三个集做验证集/早停。
- `weights.gob` 是 gob 编码，结构变更（输入维数/层宽）会导致旧权重不兼容 → 需重训。
- sczl 来自 `go-image-search` 项目（MIT 风格自研代码），本仓库为 vendor 副本；
  跨模块 import 受 Go `internal` 规则限制，故直接拷贝。
- `compositeAdaptive` 含 shapeContext（36 次旋转 × Hungarian）较慢，评测保留作 baseline 对比。
