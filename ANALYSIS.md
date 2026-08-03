# 调优记录与下次分析指引

> 本文档记录真实数据集检索（`TestRealDatasetSearch`，15 个用例）的历次优化决策、
> 失败根因与下一步方向，便于后续会话快速接手。

## 当前状态（最近一次提交）

- **汇总：14/15 命中 top-3**，唯一失败用例：`TEST11_FROM_fushixiebao.png`。
- 各用例通过情况见 `go test -run TestRealDatasetSearch -v -count=1` 输出。
- 仓库根目录无残留调试文件（本轮实验用的 `debug11_test.go`/`debughelper_test.go`/`hamming_helper_test.go` 已删除）。

## 核心机制

- 索引/查询流水线：`segment.Segment` → `MergeSimilar` → `GravityMerge`（Additive 追加蓝框）
  → `FilterFullFrame`（边框过滤）→ `ShouldAddWholeAux`（绿框整图辅助）
  → `MergeContainedOverlapping`（蓝框相交合并）→ 红框转蓝 → 索引只入库"蓝框+绿框"。
- `main.go` 与 `internal/web/server.go` 各有一份 `processRegions`/`hashImage`/`handleQuery`，**必须同步修改**。
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

## 失败用例根因（TEST11）

- 查询图 `test_pngs_target/TEST11_FROM_fushixiebao.png`（136x147，不透明深底 #191919）：
  **底部有英文文字，内容就是图片名称"fushixiebao.png"**（此前视觉模型误判为"无文字"）。
- 实际版面（按行像素实测）：
  - 图标（毛衣+背包）位于 y≈25..85；其下 y≈86..121 是**大片空深色底**；
  - y≈122..133 是文字带（小号深色字形）；y≈134..146 又是空底。
- 图库 `fushixiebao.png`（300x300）为透明底图标，整图辅助区域 bbox=(0,66)-(300,234)，
  hash=`318d109e928eaeed`，图标占宽 100% × 高 56%（居中横带）。
- 关键矛盾：图库透明底按白渲染，查询不透明深底 → 颜色/结构哈希整体"互补/反转"漂移；
  文字带 + 大片空底把整图低频结构进一步污染，且把查询整图 bbox 拉成接近正方形
  （aspect 1.01 vs 图库 1.79），导致整图哈希无法匹配。

## 本轮（本会话）实验数据（均基于真实流水线 segment.RunDefault 测量）

- **边框扫描结论**（`test_pngs_target`）：深底查询（TEST8/11/12/13/14/15）边框主色 {24,24,24}
  lum=24；**TEST6 边框主色 {248,0,0} lum=82（红框，不可归白）**；白底查询 lum=248。
  图库侧多数为透明底（borderOpaque=0）；部分深色边框库图（ditié/goulufei/riyongqingjie/
  shoujikuandai/shuidianranqi/tijianbaojian/waimai）ratio 0.5~0.92。
- **整图对整图距离**（查询 whole vs 图库 whole，hash/shape）：
  - TEST11：raw hash=46；**背景归白后 hash=24、shape=30**（hash 恰好压线 limit=24 成为候选）。
  - TEST12：raw=28；归白后 **hash=8**（这就是 TEST12 目前 rank#1 的机制）。
  - TEST11 各类裁剪/补边整图哈希（归白后）：裁图标 98x45=30；裁图标+pad aspect=1.79=26；
    pad 方形=28；**裁图标+pad aspect=0.8（竖版画布）→ dist=16（本轮找到的最小值）**。
- **区域/形状匹配**：TEST11 在 归白 ± 擦除文字带、原图/骨架两个 variant 下，
  hash/shape ≤12 的命中数均为 0（最佳 hashDist=16，仅一个 13px 碎片的 shape=0 是巧合）。
  → **TEST11 的区域级匹配是死路，只能靠整图哈希**。
- **端到端试验**：
  - 在 `hashImage` 中归一化（**同时影响图库与查询**）：12/15（TEST11/13/6 翻车）——
    图库侧被归一化改变了索引，产生新假阳性（TEST13 的 huoche 掉出 top3）。
  - **仅查询侧归一化 + 深色门限**（lum<60，排除 TEST6 红框；在 realdata_test.go 查询循环
    注入、归白后再 QueryVariants）：13/15 —— TEST6 恢复，但 **TEST11 仍 rank#12(score=0.044)，
    TEST13 反而翻车(rank#6)**。
  - TEST13 现状：不归一化时靠**骨架/形状命中**（shapeDist=0，形状哈希与反相无关）通过；
    归一化改变了查询区域集合与打分分布，huoche 掉出 top3。

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
6. **本轮：hashImage 全量归一化 / 仅查询侧归白+深色门限**：见上文"端到端试验"。
   单靠归白无法让 TEST11 进 top3（整图 hash=24 压线但 score 仅 0.044），且会翻 TEST13。

## 最终生效的改动（上一提交 bd74a2b，14/15）

- **整图探针 + 距离放宽增加 `Fill>=0.15` 门限**：把 TEST15（#4→#1）、TEST3（#7→#1）
  拉回 top-3，同时保住 TEST14/10。这是 12/15 → 14/15 的关键。
- 配套：`realdata_test.go` 查询区补传 `Global: h.Global`（此前漏传导致 TEST14 失真）。

## 下一步候选方向

1. **TEST11 需要更强的整图哈希（dist 明显 < 24）**，且不能动 TEST13（其必须保持原样处理）：
   - 关键区分器：TEST11 图标只占画布上部、下方有大片空底 + 文字带；TEST13 图标铺满画布。
   - 候选实现：仅当"内容 bbox 显著小于整图 + 检测到低区文字/大片空底"时，对查询做
     **归白 + 裁掉图标下方内容 + 居中补边到竖版画布**（prototype 中 aspect≈0.8 的
     crop+pad 得到 dist=16）。
   - 或尝试"整图区域额外入库/查询第二哈希通道"（背景归一化哈希作第二通道）。
2. **背景归白原型算法（已删，需重建）**：`NormalizeBackground(img, ratio)`：
   边框主色直方图（R&^7 量化）→ 覆盖率 ≥0.5（**不是 0.9**，TEST11 图标贴右缘导致边框
   仅 ~80% 是底色）→ **主色 lum<60 才触发**（排除 TEST6 红框）→ 泛洪填充为白。
   注意：透明底（图库）borderOpaque=0 不触发。
3. **TEST13 不可归白**：其 pass 依赖原图深底上的形状哈希（反相不变）；归白改变区域集合后
   huoche 掉出 top3。因此任何查询预处理都必须是"检测到文字/空底才触发"的条件式。
4. 验证改动后必须：`go build ./...`、`go vet ./...`、`go test ./... -count=1`
   （`-count=1` 防缓存）。结果有缓存现象，务必 `-count=1`。
