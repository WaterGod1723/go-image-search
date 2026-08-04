# 调优记录与下次分析指引

> 本文档记录真实数据集检索（`TestRealDatasetSearch`，15 个用例）的历次优化决策、
> 失败根因与下一步方向，便于后续会话快速接手。

## 当前状态（最近一次提交）

- **汇总：15/15 命中 top-3，全部 rank=#1**。
- `TestRealDatasetSearch` 已从 `internal/index`（区域感知哈希）方案切换至
  `internal/sczl`（SCZL：形状上下文 + Fourier + HOG + 自适应权重）方案。
- 自适应权重默认启用（`DefaultOptions().Adaptive=true`），CLI/web/测试全路径覆盖。
- 各用例通过情况见 `go test -run TestRealDatasetSearch -v -count=1` 输出。

## 核心机制（SCZL 算法）

- **前景提取**：边界泛洪去背景 + 连通域，颜色仅作弱辅证，主判别力来自颜色无关的形状与纹理签名。
- **描述子**（`internal/sczl/types.go`）：
  - 占据栅格（Occupancy 64/32/16）：归一化前景掩码多分辨率软栅格，直接刻画内部结构（网格十字、环 vs 实心、镂空等）。
  - Fourier 描述子（Fourier）：归一化 |FD|，平移/旋转/缩放/起点不变。
  - 径向距离直方图（Radial）：补充整体轮廓分布。
  - Shape Context（SC）：局部形状判别，nPoints × nBins 直方图。
  - NCC 归一化互相关（Patch）：去均值 + L2 归一化灰度块，亮度/对比度仿射不变。
  - HOG 梯度方向直方图：捕捉边缘方向空间分布，天然对填充色/背景色不变。
  - 多区域划分（Regions）：前景连通域逐个描述，检索时多区域匈牙利匹配。
  - 实度（Solidity）：低实度=镂空/碎片化，SC 置零避免噪声拖分。
- **检索流水线**（`internal/sczl/search.go`）：
  1. 粗排：`globalSim`（占据栅格 + NCC + 区域匹配 + FD + 径向）合并召回 top-200。
  2. 精排：对候选集计算 6 维得分（occ/ncc/reg/hog/sc/fd），自适应权重融合打分。
  3. 降序输出 top-K。
- **自适应权重**（`adaptiveWeights`）：
  - 对每个维度 d 计算 z-score 区分度：`disc(d) = (top1 - mean(top2..topK)) / std(top2..topK)`
  - softmax(disc/T) 得到自适应权重分布，与先验固定权重按 α 混合、上下界裁剪后归一化
  - `final_w(d) = clip(α·prior_norm(d) + (1-α)·softmax(disc/T)(d), lo, hi)`
  - 所有维度无正区分度时回退先验，保证退化安全
  - 仅在精排阶段使用，粗排保持 `globalSim` 不变（保护召回率）

## 关键参数（当前值）

- `internal/sczl/types.go`（`DefaultOptions`）：
  - `OccWeight=0.25, NccWeight=0.25, RegWeight=0.15, HogWeight=0.15, SCWeight=0.10, FDWeight=0.10`
  - `FDPreFilter=200`（粗排保留候选数），`FDCut=0.4`（粗排截断），`SCMaxPoints=48`
  - `Adaptive=true`（默认启用自适应权重）
  - `AdaptiveAlpha=0.5`（先验权重占比），`AdaptiveTemp=1.0`（softmax 温度）
  - `AdaptiveK=10`（计算区分度时取前 K 个候选）
  - `AdaptiveLo=0.05, AdaptiveHi=0.5`（单维度权重上下界）
- `internal/sczl/search.go`：
  - `globalSim` 粗排：`0.30*occ + 0.30*ncc + 0.15*reg + 0.15*fd + 0.10*rad`
  - `scSolidityThr=0.4`（实度低于此值 SC 置零）
- `internal/sczl/foreground.go`：`fillThr=0.5`（实度低于此值对占据栅格做孔洞填充）

## 自适应权重效果验证

真实图库 15 个查询，对比自适应 vs 固定权重：

| 指标 | 自适应权重 | 固定权重 |
|---|---|---|
| top1 命中率 | 15/15 | 15/15 |
| top1-top2 gap 更大 | **12/15** | 2/15 |
| 持平 | 1/15 | — |

自适应权重在 80% 的 case 上放大了正样本与混淆项的分数差距，让检索结果更"确定"。
top1 命中率持平（测试集偏简单），但 gap 增大意味着误判风险更低，对阈值截断场景更有价值。

## 旧方案（internal/index 区域感知哈希）历史

> 以下为 `internal/index` 方案的调优记录，`realdata_test.go` 已切换至 SCZL，
> 但 `internal/index` 包本身仍保留，可继续通过 `go-image-search build/query` 使用。

### 旧方案核心机制

- 索引/查询流水线：`segment.Segment` → `MergeSimilar` → `GravityMerge`（Additive 追加蓝框）
  → `FilterFullFrame`（边框过滤）→ `ShouldAddWholeAux`（绿框整图辅助）
  → `MergeContainedOverlapping`（蓝框相交合并）→ 红框转蓝 → 索引只入库"蓝框+绿框"。
- `main.go` 与 `internal/web/server.go` 各有一份 `hashImage`/`handleQuery`，**必须同步修改**。
- 检索：8×8-bit 分段倒排，`maxFlips=2` 保证召回 ≤16 汉明距离；`variantsN(key,n)` 生成变体。
- pHash 缩放到 32×32 → 灰度 → DCT 低频 8×8 → 与中位数比较；**透明像素视为白**。

### 旧方案 TEST11 根因与最终方案（已修复）

- 查询图 `test_pngs_target/TEST11_FROM_fushixiebao.png`（136x147，不透明深底 #191919）：
  **底部有英文文字，内容就是图片名称"fushixiebao.png"**。
- 图库透明底按白渲染，查询不透明深底 → 颜色/结构哈希整体"互补/反转"漂移；
  文字带 + 大片空底把整图低频结构进一步污染。
- 最终方案：`internal/imageproc/preprocess.go`，把"统一背景色 + 图标 + 附属内容带"
  的截图归一化为"白底 + 居中图标"，并只取整图辅助哈希作为额外证据。

### 旧方案曾试过但未采用的方向

1. 整图纯 shape 距离（`pairDist` 返回 `min(gd,sd)`）：放大假阳性，TEST10/14 翻车。
2. GlobalWeight 整图 3.0→2.0：TEST14/10 翻车。
3. 骨架 variant 不加整图 / 稀疏整图直接过滤（fill<0.1）：TEST14 翻车。
4. 图标提取（边框泛洪去背景 + 最大连通分量 + 归一化方块哈希）：TEST12 极好但 TEST11/3 无区分度。
5. cell 密度哈希 / 背景归一化为白：TEST14/15 有改善但 TEST11/3 无区分度。
6. 归一化图作为完整查询衍生图：碎片区域数量大压低 countRatio。**结论：归一化内容只应贡献整图辅助哈希**。
7. 裁剪紧贴主体内容带（不补白边）：图标贴画布边缘导致白底被断开。**必须补白边让内容浮于白底**。

## 最终生效的改动（本提交）

- **SCZL 自适应权重**（`internal/sczl/search.go`）：精排阶段基于各维度得分分布
  动态计算融合权重，取代写死的固定权重。核心算法：
  - z-score 区分度：`(top1 - mean(top2..topK)) / std(top2..topK)`（比相邻两点差值更抗噪声）
  - softmax 消除量纲差异 + 与先验 α 混合 + 上下界裁剪 + 归一化
  - 无区分度时回退先验（退化安全）
- **`realdata_test.go` 切换至 SCZL 算法**：从 `internal/index`（区域感知哈希）改为
  `internal/sczl`（形状上下文 + Fourier + HOG + 自适应权重），测试从有失败用例
  变为 15/15 全 rank=#1。
- **默认启用自适应**：`DefaultOptions().Adaptive=true`，CLI/web/测试全路径覆盖；
  CLI 提供 `-no-adaptive` 回退固定权重。

## 下一步候选方向

1. **验证改动后必须**：`go build ./...`、`go vet ./...`、`go test ./... -count=1`
2. **更大图库压测**：当前测试集仅 15 个查询 + 数十张图库，自适应权重在更大规模
   （数千~数万图库 + 更多混淆项）下的表现待验证。可用 `sczl-build -dir <大图库>`
   构建索引后批量跑 `sczl-query` 对比。
3. **自适应参数调优**：当前 `α=0.5, T=1.0, K=10, lo=0.05, hi=0.5` 为经验值，
   可在更大图库上做 grid search 优化。`T=0.5`（更锐利）或 `α=0.3`（更激进自适应）
   值得尝试。
4. **SCZL 粗排优化**：当前 `globalSim` 的权重仍写死（`0.30*occ + 0.30*ncc + ...`），
   可考虑对粗排也引入自适应，但需注意召回率风险。
5. **SCZL 索引加速**：当前线性遍历，图库规模到数万后可换 KD-tree / IVF 等近似检索。
