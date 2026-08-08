# image-search-test 测试集说明

本项目用于制作 icon 图片反向检索（以图搜图）的测试集，并对检索算法做评测。
`gentest` 工具从 `test_pngs/` 的原图标随机生成带干扰的查询图，`test_set/manifest.json` 记录每张查询图的真实来源与所用变换，作为评测/分析失败用例的 ground truth。

## 目录结构

| 路径                     | 说明                                             |
| ------------------------ | ------------------------------------------------ |
| `test_pngs/`             | 原始 icon 图（透明背景，约 300×300）              |
| `test_set/`              | 生成的测试集：`sample_*.png` + `manifest.json`    |
| `gentest/`               | 生成器源码（`go run ./gentest`）                |
| `verify/`                | 校验工具：解码所有样本并检查文字尺寸上限         |

## 测试集组成

每张生成的 `sample_*.png` 由以下步骤得到：

1. **去透明边框**：用 `trimAlpha` 把原图中 alpha≤4 的像素裁掉，
   得到"紧致的 sprite"。      若不裁边，缩放/旋转会引入大面积背景，干扰检索；裁剪后各图标占据的实际内容比例才一致。
2. **画布**：边长为 320~560px 的随机矩形（正方形概率约 1/3），背景色随机（各通道 10%–90% 亮度）。
3. **Sprite 变换**（随机组合，概率见下）：
   - **旋转**：围绕 sprite 中心，-180°~+180°，触发概率 90%（双线性采样去锯齿）。
   - **缩放 + 平移**：sprite 尺寸取画布短边的 35%–85%（`-sprite-lo/-sprite-hi`），中心位置在画布内随机偏移 ±45%；偶尔（1/10）放大到画布短边的 1.0–1.5 倍，让 sprite 溢出画布。
   - **背景颜色变化**：每张都不同（生成时就换背景）。
4. **四周随机文字**：随机选 0–4 边（默认随机子集，`-all-sides` 强制四边），文字权重列表见 `文字词表`，字号严格 ≤ 画布高的 30%：
   - 上/下边：水平文字，贴着边缘随机位置；
   - 左/右边：旋转 90° 的垂直文字。
   文字颜色黑/白随机，保证与背景对比。

## manifest.json 字段

```json
[{
  "image":      "sample_00012.png",                 // 生成的文件名（键）
  "src":        "waimai.png",                       // 原始图文件名（基准真值）
  "crop":       [32, 0, 236, 300],                 // 裁剪框 [x, y, w, h]
  "canvas":     [547, 603],                        // 最终画布 [width, height]
  "bg_hex":     "#6fccc5",                         // 背景颜色
  "rotation":   79.02,                             // sprite 旋转角（度）
  "scale":      1.2295,                            // sprite 缩放因子
  "translate":  [-222, 28],                        // sprite 中心相对画布中心的偏移（px）
  "texts": [
    {"side":"bottom","text":"领券","size":94,"x":67,"y":518,"color":"#ffffff"}
  ]
}]
```

要点：
- 一张原图可被抽多次 → 一张 `src` 可能对应多条记录，属于正常。
- `Rotation` 为 0 且 `scale` 接近 1、`translate` 接近原点时，该样本几乎"与原件近似"，用于测最平缓的 case。
- `texts` 为空表示这张**无文字干扰**。

## 失效用例分析

当检索结果与 `src` 不符/召回失败时，优先核对：

- 会放大干扰的字段：
  - `rotation` 大（如 >45°）：旋转后像素采样模糊、区域重叠，对特征提取敏感。
  - `scale` > 1 造成 sprite 溢出 `canvas`：部分图标边缘被裁剪丢像素。
  - `translate` 偏移明显：图标离开画布中心，若对齐算法假设中心在原点会失效。
  - `texts` 数量多、字号接近 30% 上限：文字可能盖住 icon 特征或作为强特征误导。
- 失败模式与字段对应：
  - 总是召回错误：多因 `src` 图标本身形状相近（如临近 class），或 `crop`/画布差异大。
  - 特定 1-2 张失败：多为旋转接近 90°/180° 或文字与图标同色干扰。
- 复现：用生成时的 seed 重新 `go run ./gentest -n 120 -seed <seed>`，可得到完全一致的数据。

## 用法

```bash
go run ./gentest                   # 生成 120 张，随机种子
go run ./gentest -n 300 -seed 42   # 指定数量和可复现种子
go run ./gentest -all-sides        # 每张都四边加文字
go run ./gentest -sprite-lo 0.5 -sprite-hi 1.2   # 调整缩放范围
go run ./verify                    # 校验已生成的样本解码与字号上限
```

## 高级可调项（gentest flags）

```
-src <dir>        原始 icon 目录      (default "test_pngs")
-out <dir>        输出目录            (default "test_set")
-n <int>          样本数量            (default 120)
-seed <int64>     RNG 种子，0=时间      (default 0)
-min-canvas / -max-canvas   画布侧边范围 (320 / 560)
-sprite-lo / -sprite-hi     sprite 缩放占画布比例 (0.35 / 0.85)
```

## 提醒（供 AI 分析时注意）

- 检索算法评测时请**把 `test_pngs/` 目录作为底库（index），把 `test_set/*.png` 作为查询（query）**，
  以 `manifest.json` 验证命中是否正确，可计算 recall@k / mAP 等指标。
- 生成器用系统字体（Windows: `msyh.ttc` 等），文字内容、字形取决于本机字体，
  但 `manifest.json` 中记录了实际文字，无需依赖字体差异做判断。
- 若测试时追求与原文完全一致的结果，请固定 `-seed` 并保留 `test_pngs/` 不变。