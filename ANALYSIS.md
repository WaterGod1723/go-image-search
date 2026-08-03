# 调优记录与下次分析指引

> 本文档记录真实数据集检索（`TestRealDatasetSearch`，15 个用例）的历次优化决策、
> 失败根因与下一步方向，便于后续会话快速接手。

## 当前状态（最近一次提交）

- **汇总：15/15 命中 top-3**。TEST11 已通过（rank #1），上一轮遗留的唯一失败用例已解决。
- 各用例通过情况见 `go test -run TestRealDatasetSearch -v -count=1` 输出。
- 仓库根目录无残留调试文件（本轮调试用的 `debug11_test.go` 已删除）。

## 核心机制

- 索引/查询流水线：`segment.Segment` → `MergeSimilar` → `GravityMerge`（Additive 追加蓝框）
  → `FilterFullFrame`（边框过滤）→ `ShouldAddWholeAux`（绿框整图辅助）
  → `MergeContainedOverlapping`（蓝框相交合并）→ 红框转蓝 → 索引只入库"蓝框+绿框"。
- `main.go` 与 `internal/web/server.go` 各有一份 `hashImage`/`handleQuery`，**必须同步修改**。
- 检索：8×8-bit 分段倒排，`maxFlips=2` 保证召回 ≤16 汉明距离；`variantsN(key,n)` 生成变体。
- pHash 缩放到 32×32 → 灰度 → DCT 低频 8×8 → 与中位数比较；**透明像素视为白（`lumAt` 中 `a==0` 返回 255）**。

## 关键参数（当前值）

- `internal/index/index.go`：
  - `maxFlips=2`，`maxWholeFlips=3`（仅整图区域用）。
  - `considerHit` 中整图对整图放宽距离门限：`q.Global>=2.5 && e.Global>=2.5 &&
    q.Fill>=0.15 && e.Fill>=0.15` 时 `limit=24`（整图哈希 dist≤24 即可成为候选）。
  - 探针翻转数同样受 `q.Fill>=0.15` 门限控制（防止稀疏骨架整图区域乱匹配）。
  - `idf[qi] = log(1+n)/log(1+df) * sqrt(area) * Global`，`df<1` 时按 1 计（防 countRatio 虚高）。
- `internal/segment/gravity.go`：`GlobalWeight` 整图=3.0，组合区域 `1+0.5*ln(n)`。
- 查询阶段（`runQuery`/web `handleQuery`）强制 `grav.CombineAlways=true`。

## TEST11 根因与最终方案（已修复）

- 查询图 `test_pngs_target/TEST11_FROM_fushixiebao.png`（136x147，不透明深底 #191919）：
  **底部有英文文字，内容就是图片名称"fushixiebao.png"**（此前视觉模型误判为"无文字"）。
- 实际版面（按行像素实测）：
  - 图标（毛衣+背包）位于 y≈25..85；其下 y≈86..121 是**大片空深色底**；
  - y≈122..133 是文字带（小号深色字形）；y≈134..146 又是空底。
- 图库 `fushixiebao.png`（300x300）为透明底图标，整图辅助区域 bbox=(0,66)-(300,234)，
  hash=`318d109e928eaeed`，图标占宽 100% × 高 56%（居中横带）。
- **关键矛盾**：图库透明底按白渲染，查询不透明深底 → 颜色/结构哈希整体"互补/反转"漂移；
  文字带 + 大片空底把整图低频结构进一步污染，且把查询整图 bbox 拉成接近正方形
  （aspect 1.01 vs 图库 1.79），导致整图哈希无法匹配（raw 整图 hash dist=46，任何区域
  都不进 top-9，fushixiebao 得 0 分）。

### 最终方案：查询侧"截图归一化 → 主体图标整图哈希"条件预处理（泛化版）

- 新增 `internal/imageproc/preprocess.go`，按通用规则把"统一背景色 + 图标 + 附属内容带"
  的截图归一化为与图库一致的"白底 + 居中图标"，并只取其整图辅助哈希作为额外证据：
  1. `dominantBorderColor`：边框像素量化直方图 → 主色，覆盖率相对**全部边框采样**
     （透明底图库天然不触发）；主色非近白（lum<240）才继续。
  2. `removeBackground`：泛洪填充边框连通且接近主色的像素为白（透明底、白底均无操作）。
  3. `splitContentBands`：按空带把内容切分为行带（先滤掉太薄的噪声带，再合并不足
     12% 高度的空隙）。
  4. `pickMainBand`：选内容量最大的带为主体，要求 ≥2× 次带（无明确主体则放弃）。
  5. `cropBand`：按主体内容包围盒裁剪 + 四周补白边（内容浮于白底），使白色背景
     聚合成完整边框区域被 `FilterFullFrame` 过滤，整图辅助包围盒=图标本体。
- `QueryNormalizedWholeHashes`：对归一化图跑流水线并**只返回整图辅助区域哈希**
  （Global≥2.5）。归一化内容的低分辨率碎片不可靠（这正是归一化的原因），只贡献
  "白底+主体"的整图低频结构证据，避免碎片带来的误匹配与布局惩罚。
- 效果：TEST11 整图哈希 dist 46 → **6**（aspect 1.77 vs 图库 1.79），fushixiebao
  0 分 → rank #1（score 0.432）。触发器还命中 TEST2/3/4/8/14（深底或红框 + 附属文字
  带），全部保持 top-3 且分数提升（如 TEST3 0.267→0.732、TEST8 0.352→1.043）。
  白底无文字（TEST1/5/6/9/10 等）、满画布图标（TEST13）、无附属带（TEST12/15）
  均不触发，行为不变。
- 三个查询路径（`realdata_test.go`/`main.go`/`web/server.go`）统一为：
  `QueryVariants`（原图+骨架）+ `QueryNormalizedWholeHashes`（条件整图证据）。
  索引/构建仍用 `QueryVariants`。

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
6. **归一化图作为完整查询衍生图**（把裁剪结果整张加入查询集合）：TEST11 反而不进
   top-3 —— 低分辨率碎片区域数量大（~107 个）压低 countRatio，且碎片与布局惩罚
   把正确图的分数拖低（0.067 vs 整图证据的 0.432）。**结论：归一化内容只应贡献整图
   辅助哈希，不能整张参与**。
7. **裁剪紧贴主体内容带**（不补白边）：图标贴画布边缘 → 白色背景被图标断开成两个
   半幅区域（非满框，不被 FilterFullFrame 过滤），整图辅助包围盒被拉成整画布
   （aspect 2.23、dist 22）。**必须补白边让内容浮于白底**。

## 最终生效的改动（本提交，15/15）

- **查询侧截图归一化 + 整图哈希证据**（`internal/imageproc/preprocess.go`）：
  通用地检测统一背景色与附属内容带，把"深底/彩框截图 + 底部文字"归一化为
  "白底 + 居中图标"，并只贡献整图辅助哈希。这是 TEST11 从 0 分 → rank#1 的关键，
  且未影响任何既有用例（其余触发用例分数普遍提升）。
- 配套：`realdata_test.go`/`main.go`/`web/server.go` 三个查询路径统一改为
  `QueryVariants` + `QueryNormalizedWholeHashes`（索引仍用 `QueryVariants`）。

## 下一步候选方向

1. **验证改动后必须**：`go build ./...`、`go vet ./...`、`go test ./... -count=1`
   （`-count=1` 防缓存）。结果有缓存现象，务必 `-count=1`。
2. TEST14（shoujikuandai 查询）当前 top1 是 huoche（期望图 rank#2 但分数 0.485 远超
   旧版）。这是"整图证据强"带来的高置信度误判，未影响 top-3 但值得后续关注：
   若继续出现可考虑对整图证据增加形状哈希校验，或在归一化带选择时收紧主带判定。
3. 归一化触发阈值（边框覆盖率 0.5、颜色距离 30、空带 12%、主带 2×）为通用参数，
   若新用例误触发/漏触发，优先调这些常量而非加特例。
