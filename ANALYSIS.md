# 调优记录与下次分析指引

> 本文档记录真实数据集检索（`TestRealDatasetSearch`，15 个用例）的历次优化决策、
> 失败根因与下一步方向，便于后续会话快速接手。

## 当前状态（最近一次提交）

- **汇总：14/15 命中 top-3**，唯一失败用例：`TEST11_FROM_fushixiebao.png`。
- 各用例通过情况见 `go test -run TestRealDatasetSearch -v -count=1` 输出。

## 核心机制

- 索引/查询流水线：`segment.Segment` → `MergeSimilar` → `GravityMerge`（Additive 追加蓝框）
  → `FilterFullFrame`（边框过滤）→ `ShouldAddWholeAux`（绿框整图辅助）
  → `MergeContainedOverlapping`（蓝框相交合并）→ 红框转蓝 → 索引只入库"蓝框+绿框"。
- `main.go` 与 `internal/web/server.go` 各有一份 `processRegions`/`hashImage`/`handleQuery`，**必须同步修改**。
- 检索：8×8-bit 分段倒排，`maxFlips=2` 保证召回 ≤16 汉明距离；`variantsN(key,n)` 生成变体。

## 关键参数（当前值）

- `internal/index/index.go`：
  - `maxFlips=2`，`maxWholeFlips=3`（仅整图区域用）。
  - `considerHit` 中整图对整图放宽距离门限：`q.Global>=2.5 && e.Global>=2.5 &&
    q.Fill>=0.15 && e.Fill>=0.15` 时 `limit=24`。
  - 探针翻转数同样受 `q.Fill>=0.15` 门限控制（防止稀疏骨架整图区域乱匹配）。
  - `idf[qi] = log(1+n)/log(1+df) * sqrt(area) * Global`，`df<1` 时按 1 计（防 countRatio 虚高）。
- `internal/segment/gravity.go`：`GlobalWeight` 整图=3.0，组合区域 `1+0.5*ln(n)`。
- 查询阶段（`runQuery`/web `handleQuery`）强制 `grav.CombineAlways=true`。

## 失败用例根因（TEST11）

- 查询图 `test_pngs_target/TEST11_FROM_fushixiebao.png`（136x147）语义上仍是"完整图标"，
  但叠加了文字干扰 + 背景色变化（不透明深色底，图库图标为透明底）。
- 现象：
  - 原图 variant（variant 0）与骨架 variant（variant 1）都无法把 `fushixiebao` 排进 top-3。
  - 整图对整图 shape 距离 ~24-30（全部图标都在这个量级，无区分度）。
  - 区域级匹配：查询 31 个区域与 fushixiebao 27 个区域，hash/shape ≤12 的命中数为 0。
- 关键矛盾：**图库图标透明底在 pHash 里按白底渲染，查询图是不透明深底，导致
  颜色哈希与结构哈希都在"互补/反转"方向漂移**。文字干扰进一步污染整图低频结构。

## 曾试过但未采用的方向（保留供参考）

1. **整图纯 shape 距离**（`pairDist` 对整图对整图返回 `min(gd,sd)`）：会放大假阳性，
   使 TEST10/14 翻车，整体掉到 10/15。已回退。
2. **GlobalWeight 整图 3.0→2.0**：TEST15/3 转绿但 TEST14/10 翻车（因 `>=2.5` 门限连带
   关闭了整图特殊探针/放宽），12/15。已回退为 3.0。
3. **骨架 variant 不加整图 / 稀疏整图直接过滤（fill<0.1）**：TEST3 转绿但 TEST14 翻车，
   12/15。已回退。
4. **图标提取（边框泛洪去背景 + 最大连通分量 + 归一化方块哈希）**：对 TEST12 极好
   （dist=4），但 TEST11/3 的归一化图标哈希仍无区分度（18~31）。未接入。
5. **cell 密度哈希 / 背景归一化为白**：TEST14/15 有改善但 TEST11/3 无区分度。未接入。

## 最终生效的改动（本提交）

- **整图探针 + 距离放宽增加 `Fill>=0.15` 门限**：把 TEST15（#4→#1）、TEST3（#7→#1）
  拉回 top-3，同时保住 TEST14/10。这是 12/15 → 14/15 的关键。
- 配套：`realdata_test.go` 查询区补传 `Global: h.Global`（此前漏传导致 TEST14 失真）。

## 下一步候选方向

1. **TEST11 专用**：其整图/区域信号全面失效，可能需要"抗背景反相的表示"：
   - 对整图区域计算"反转不变"的结构哈希（同时算 shape 与 ~shape，取 min 距离）；
   - 或按边框连通性识别查询图的深色底并归一化为白后再哈希（注意透明底在
     `imageproc.Grayscale` 中会被转成黑，需在 RGBA 层处理，见 `bgNormRGBA` 思路）。
2. **多哈希通道入库**：为整图区域额外入库一个"背景归一化/细胞密度"哈希作为第二通道，
   查询时任一通道命中即候选，扩大召回同时保持精度。
3. 验证改动后必须：`go build ./...`、`go vet ./...`、`go test ./... -count=1`
   （`-count=1` 防缓存）。结果有缓存现象，务必 `-count=1`。
