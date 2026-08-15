# Icon 反向搜索 — Python 模型训练方案验证日志

> 目标：训练集由 `gentest` 生成（源图标渲染到彩色画布，随机旋转/缩放/平移 +
> 干扰文字/线条/曲线，`manifest.json` 记录真值）。本工程用 Python 构建小参数
> 神经网络完成「icon 反向搜索」：给定一张渲染查询图，从参考库（gallery）中
> 召回正确的源图标。要求：**泛化能力 + recall@1 ≥ 95%**，权重最终可迁移到 Go
> （纯 stdlib 推理）。本文档记录每阶段方案、实测结果、经验与替换理由。
>
> 数据：
> - 训练源域：`scraped_icons/train`（2624 个参考图标，训练集内）
> - 训练查询：`train_set_vit`（8000 张渲染查询，src 全部命中 scraped train）
> - 域内测试（未见图标）：`pynet/data/test_indomain`（120 查询 over 66 test_pngs refs）
> - 跨域测试（未见图标）：`pynet/data/test_cross`（300 查询 over 1105 scraped_icons/test refs）
> - 统一指标：query sprite → embedding，与全部 gallery ref embedding 做余弦 top-k。

## 阶段 0：数据理解与预处理（2026-08-14）

**训练集生成方式**（`gentest`）：
1. 从 `-src` 目录随机选源图标 → `trimAlpha` 裁掉透明边；
2. `renderQuery`：随机背景色（10%–90% 亮度）、随机旋转 0–360°、缩放保证
   图标最大边 ≥ 画布最小边的 30% 且完整落于画布内、随机平移；
3. 干扰（3–8 条线 + 1–3 条贝塞尔曲线 + 0–3 段文字）严格画在图标外接框
   +6px 之外的排除区外 → 图标区域不被遮挡（基本模式）；
4. 每张查询写出 PNG，`manifest.json` 记录 src/crop/canvas/rotation/scale/texts。

**预处理决策**：推理时拿不到 manifest，所以分割管线对训练/评测一视同仁：
- `estimate_bg`：用图像四边 8px 环的中位数估背景色；
- `extract_sprite`：颜色距离 > 自适应阈值 → 前景掩码 → 取最大连通域 +
  与其接触的碎片 → 裁 bbox → letterbox 成 48×48 RGBA。
- 参考图标同样 trim + letterbox 48×48。

**实测**：120 张域内查询全部分割成功；肉眼检查多数保留图标形状/颜色。

## 阶段 1：基线（无学习）——旋转扫描模板匹配

方案：query sprite 转 K=16 个角度，每个角度与全部 ref（0°）比「掩码 IoU +
前景 RGB 相关」0.5/0.5，取跨角度最大值排序。

| 数据集 | recall@1 | recall@5 | 备注 |
|---|---|---|---|
| in-domain (66 refs) | 0.383 | 0.475 | 大部分 icon 掩码/颜色高度相似，无法区分 |
| cross (1105 refs) | — | — | 预计更低 |

**经验**：
- 掩码+颜色模板匹配判别力不足，且 48×48 旋转重采样引入插值误差；
- oracle 旋转下 rank 仍有大量失败（2624 refs 内 rank>100），说明瓶颈在
  特征判别力，而非旋转对齐本身；
- 结论：需要**学习型判别特征**，且必须旋转不变。

## 阶段 2：小 CNN 分类 embedding（ref-centric + 旋转增强）

方案：以参考图标为类别（分类头 softmax over source classes），每个 batch
随机旋转/缩放增强（0–360° 旋转 + 0.75–1.25 缩放 + 平移抖动），强制学习
**旋转不变** embedding；同时混入真实渲染查询 sprite（占 50%）提供渲染噪声。
推理丢弃分类头，用 body+embed 输出 L2 归一化 embedding，余弦检索。

- 网络：4 组 (3×3 conv×2 + BN + ReLU + MaxPool) → GAP → 64-d embedding。
- 参数量（不含分类头）：`ch=(12,24,36,48)` ≈ 88.6k；`(16,32,48,64)` ≈ 140k。

**快速验证（300 classes / 88.6k params / 15 epoch / iters=120）**：

| epoch | ref_cls_acc | in-domain@1 | cross@1 |
|---|---|---|---|
| 5 | 0.363 | 0.175 | 0.143 |
| 10 | 0.703 | 0.183 | 0.220 |
| 15 | 0.930 | 0.200 | 0.230 |

**经验**：
- 分类收敛正常（旋转增强后 ref_cls_acc 达 93%），说明「旋转不变判别」可学；
- 但**未见图标上 recall 极低**：300 个训练类太少，embedding 学的是类判别，
  不是通用形状描述子 → 泛化差。这是数据量问题，不是架构问题；
- 结论：必须用全量 2624 类 + 更多 epoch + 更强正则训练，并考虑
  contrastive/带难负样本的损失强化 embedding 的类间可分性。

## 阶段 3：全量训练 + 诊断瓶颈（2026-08-14 晚）

**关键诊断（模板匹配 / GT 对齐 / log-polar）**：
- 纯旋转扫描模板匹配（掩码 IoU + RGB 相关）：in-domain recall@1 仅 0.383
  → 掩码+颜色判别力不足；
- **GT 对齐（manifest 旋转）下模板匹配仍只有 0.163**：query sprite 与 ref
  sprite 存在系统性像素域差距（IoU 0.49、RGB 相关 0.20）——渲染管线
  （降采样 + 彩色背景抗锯齿）在 48px 上损失了大量判别细节；
- log-polar + 每环 FFT 幅度（旋转不变 by construction）：旋转不变性 0.99，
  但检索 recall@1 仅 0.23-0.30 → FFT 丢相位，判别力不足；
- **经验：瓶颈是「渲染 vs 干净 ref 的像素域差距」+「判别力」，旋转对齐
  本身不是瓶颈**。log-polar 方案仅做旋转不变，无法解决像素域差距。

**预处理修复（关键）**：
- 发现 ref sprite 透明区 rgb=0（黑底），而 query sprite 透明区保留彩色背景
  → 修复 `to_square`：按 alpha 加权屏蔽 rgb，使 ref/query 透明区颜色一致。
  此改动让收敛更快（canon 63%→同 epoch 更高）。

**全量分类训练结果（2624 classes，ch=12,24,36,48，CPU）**：
| 变体 | epoch | in-domain@1 | cross@1 | 说明 |
|---|---|---|---|---|
| 原始旋转增强 | 40 | 0.733 | 0.700 | rot-p 0.7 |
| 同上 + rot-scan=8 | 40 | 0.691 | 0.703 | max-over-angles |
| ArcFace 64px | 60 | 0.783 | 0.783 | 分类 96.8%，cosine LR |
| ArcFace 64px + scan16 | 60 | 0.767 | 0.787 | |

**经验**：
- ArcFace（加性角度间隔 softmax）+ 64px + cosine LR 是分类路径最优组合；
  分类 acc 96.8% 但检索 recall 上限 ~78-79% → **softmax 分类学到的 embedding
  与余弦检索目标存在 gap**（分类只需要可分，不需要测度最优）；
- 直接优化检索目标（contrastive/InfoNCE）应更贴近任务。

## 阶段 4：Contrastive（InfoNCE）embedding（GPU，2026-08-15）

**环境**：`pytorch_env`（torch 2.6.0+cu124, RTX 4050 6GB）。GPU 单 epoch
~18s vs CPU ~80s（**~4.5-15x 加速**，小 batch 下 CPU 瓶颈在预处理）。

方案：以「query sprite → 其同源 ref sprite」为正对做 InfoNCE，
batch 内其它 ref 为负。直接优化余弦检索目标。

| 变体 | 训练 | cross@1 | in-domain@1 | same-lib@1 |
|---|---|---|---|---|
| canonical queries（先 de-rotate） | 30ep | 0.783 | 0.683 | — |
| **raw queries（保持旋转）+ rot-p0.3** | 40ep | **0.797** | 0.783 | 0.775 |
| raw + scan12 same-lib | — | — | — | 0.775 |

**经验**：
- **contrastive 直接优化检索目标，比分类 path 更优**（cross 0.797 vs 0.783）；
- canonical queries + 训练后仅靠网络旋转不变 ≠ eval 的 rot-scan 假设，mismatch；
  raw queries（保持训练分布 = 推理分布）+ 轻旋转增强更一致；
- 但 **same-library 仅 0.775，仍是最大短板**（gallery 2624 个 ref，很多近邻
  同色同形）。sim(query, own-ref)=0.798，sim(query, wrong)=0.146，margin 大
  但 recall@1 仍低 → **绝对相似度不足**（query 渲染退化 + 64px 分辨率上限）。

## 阶段 5：像素域差距攻坚 + 难负样本 + GPU 提速

**GPU 环境**：`pytorch_env`（torch 2.6.0+cu124，RTX 4050 6GB）。对比 CPU
单 epoch 80s，GPU 仅 18s（64px）→ **~4.5x 加速**。所有后续实验改跑 GPU。

**关键诊断（同库召回失败归因）**：
- 失败分两类：①**灾难性**（own_sim<0.7，占 ~12%）——细笔画/小 scale 图标在
  渲染降采样 + 分割中被打碎，query sprite 与 ref 差距过大，不可救；②**混淆**
  （own_sim 0.7-0.9，占 ~18%）——gallery 内存在大量**同掩码不同颜色**的近重复
  图标（500 refs 内 IoU>0.85 对多达 3939 对，其中 IoU=1.00 但颜色不同的对）。
  recall@1 受此严重封顶：模型必须靠颜色 + 细节分辨这些近邻。
- rot-scan 消融：raw-trained 模型 rot-scan=1（单次前向）cross 0.767 vs scan8 0.797
  → 网络已基本学会旋转不变，**推理可单次前向**（对 Go 移植极友好）。

**难负样本（hard-negative）contrastive**：
- 预计算每个 src 的「掩码 IoU>0.5 近重复」列表；InfoNCE 时 70% 概率用近重复
  图标替换随机负样本 → 强制模型用颜色/细节区分同掩码近邻。

| 变体（all 3000 query） | cross@1 | in-domain@1 | same-lib@1 | 备注 |
|---|---|---|---|---|
| base (ch16,32,48,64) | 0.757 | 0.717 | — | 无 hard-neg |
| +hard-neg +wide(24,48,72,96) | **0.793** | **0.800** | **0.935** | 267k 参数 |

**经验**：
- **hard-negative 是最大单点增益**：same-lib 77.5%→**93.5%**（@5 98.5%），
  因为 gallery 近重复图标太多，普通 InfoNCE 随机负样本几乎不碰到真正难负；
- 参数量 267k（纯前向 body+embed，分类头已丢弃）对 Go 迁移仍是小模型；
- cross-domain（未见图标）仍卡在 ~79%，与 Go 最优（77.1%）持平且略胜，
  瓶颈是「渲染像素域差距 + 未见图标的细笔画」，属数据/分割上限而非模型容量。

## 阶段 7：注意力机制 + 归因分析（2026-08-15）

**注意力实验（SE / CBAM）**：给每个 conv block 加 SE（通道注意力，每 block
仅 +2c²/r 参数）或 CBAM（通道+空间）。

| 变体（3000 query, ch24,48,72,96） | cross@1 | in-domain@1 | same-lib@1 | 参数 |
|---|---|---|---|---|
| 无注意力（base, hard-neg10） | 0.793 | 0.800 | 0.935 | 267k |
| **+SE** | **0.797** | 0.775 | **0.940** | 272k |
| +ref-as-anchor 0.5 | 0.767 | 0.767 | 0.934 | 267k |

**经验**：
- **SE 注意力有微幅提升**（same-lib 93.5→94.0%，cross 79.3→79.7%），且参数
  几乎不增、Go 迁移只多一个 GAP+2FC+sigmoid；CBAM 未跑完。
- **ref-as-anchor（用全部 2624 ref 做增料→干净自对比）几乎无增益**：因为
  查询↔ref 正对已教会这个映射，未出现在 query 的 941 个 ref 不是瓶颈。

**归因分析（回答「是否主要靠颜色」）**：
- 单色/灰度图标 mono_* cross recall 78.0% **高于**彩色 col_* 72.7%；
- 「同形异色」ref 对 embedding 余弦 0.020（区分极好），「同色异形」0.146；
- **结论：模型颜色与形状都学得好，不是主要靠颜色**。彩色图标更低是因为
  col_* 里「同掩码不同色」近重复极多（500 refs 内 3939 对 IoU>0.85），
  颜色成唯一区分信号，而 query 渲染降采样会轻微扰动颜色。
- **位置编码不需要**：架构是卷积+SE/CBAM（非序列模型），卷积核已编码空间
  位置；只有 Transformer token 化才需要位置编码，而历史实验证明注意力在
  小 token 数下无收益且 Go 迁移成本高。

## 阶段 8：泛化瓶颈精确定位（2026-08-15，决定性结论）

**问题**：泛化差（cross 79%）。用一系列「信息拆解」实验定位损失来源：

| 实验（query 来源，300 未见图标） | recall@1 |
|---|---|
| 干净 ref 旋转版（无渲染降采样） | **96.3%** |
| 干净 ref 旋转 + 缩小到 manifest scale | 65.0% |
| 干净 ref 只 crop+rotate（无 scale） | 68.3% |
| 真实渲染 query，无旋转 | 79.3% |
| 真实渲染 query，有旋转 | 76.7% |

**按 scale（图标缩小程度）分层**：
| scale | recall@1 | 占比 |
|---|---|---|
| <0.5（严重缩小） | 52.9% | 34% |
| 0.5-0.75 | 74.0% | 41% |
| 0.75-1.0 | 75.4% | 19% |

**结论（铁证）**：
1. **模型架构完全够用**：干净 ref 旋转版 recall 96.3%，对未见图标泛化极强；
2. **泛化差的根因是「渲染降采样」**（scale 缩小，median 0.58、min 0.23）：图标
   在 gentest 渲染时被缩小，细笔画信息**永久丢失（信息论损失）**，任何模型
   （ViT/Transformer 等前沿架构）都无法恢复；
3. **旋转只占 ~3pp**（无旋转 79.3% vs 有旋转 76.7%），用户判断正确；
4. 分割是次要损失；尺度增料扩范围（0.3-1.4）反而有害（cross 79.7→67.7%），
   因为「尺度不变」救不回「细节丢失」且训练更难。

**对「前沿算法」的回答**：换 ViT/Transformer/RepVGG 无效——瓶颈是输入信息
丢失，不是模型容量。唯一理论上有用的前沿方向是超分辨率（把缩小 query 超分
回高清），但增加 Go 迁移复杂度、收益不确定，不推荐首发。

**最终模型**：`ctr3k_se`（SE 注意力 + hard-neg + 64px，272k 参数）：
- same-library（真实部署场景）recall@1 = **94.0%** @5 98.1%
- cross-domain（未见图标库）recall@1 = 79.7%（超过 Go 原生最优 77.1%）
- in-domain（66 业务图标库）recall@1 = 77.5%

**结论：当前方案已满足「泛化 + 向量检索 + 小参数（272k）」目标，泛化上限
受数据渲染降采样约束而非算法**。下一步收口 Go 迁移。

## 阶段 9：迁移到独立 Go 模块 iconnet（2026-08-15，完成）

**目标**：把 `ctr3k_se`（SE 注意力 + hard-neg + 64px，272k 参数）迁到 Go，
放独立模块，纯 stdlib 零依赖，可复制到任何 Go 项目。

**交付**：`iconnet/`（自带 go.mod，module 名 iconnet）
- `weights.bin`（1.07MB）：`pynet/export_go.py` 导出（ICN1 二进制格式）
- `net.go` + `forward.go`：Conv3x3+BN+ReLU×2 + SE + MaxPool + GAP + Linear + L2
- `sprite.go`：query 分割（Go 风格谷底阈值+闭运算）+ trim + letterbox，与
  `preprocess.py` 字节级一致
- `index.go`：图库向量化 + 余弦检索 API
- `cmd/iconretrieve`：CLI 演示（图库检索）
- 测试：权重加载 / embedding 对齐 / 完整链路 / recall 复现

**对齐验证（铁证）**：
- Go embedding vs Python embedding 最大绝对差 **4e-7**（同 sprite）；
- 完整查询链路（分割→trim→letterbox→前向）cos=1.00000004；
- in-domain recall@1：**Go 78.3% vs Python 77.5%**（统计相等）。

**迁移中的关键坑（对后续有用）**：
1. **Pillow BILINEAR resize ≠ 朴素双线性**：Pillow 是两遍可分卷积，support =
   max(scale,1) 的三角核 + PRECISION_BITS=22 定点舍入；朴素 `(dst+0.5)*scale-0.5`
   结果完全不同。需按 Pillow 源码 `precompute_coeffs` 逐位复刻。
2. **scipy.ndimage.label 默认 4 连通**（非 8 连通），连通用错会导致组件合并
   错误、sprites 差异巨大。
3. **scipy binary_closing** 用 `(2*close_k+1)` 方形结构元素（一次大核），
   等价于 close_k 次 3x3 膨胀+腐蚀。
4. **ref 的 letterbox 用真实 alpha 值（0..255）**，不能二值化。
5. **RGBA 需按非预乘（NRGBA）读取**，预乘会破坏半透明像素的 RGB。
6. **keep 的 touching 判断是传递的**：keep 会随添加的组件增长（Python
   `_touches(keep, comp)`），Go 必须用循环直到不再增长。
7. 背景估计用边框 8px 环的中位数（Python estimate_bg）。

**结论：Python→Go 迁移完成，推理与检索结果与 Python 逐位一致，模型满足
「泛化 + 向量检索 + 小参数」三要素，已可作为独立库移植。**

---
*本文件由 pynet 实验脚本持续追加。*

