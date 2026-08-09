# go-image-search

> Reverse image search using a trained neural ranking engine (MLP fusing hand-crafted features + color-agnostic SCZL signatures with two-stage coarse pre-filter) — Go tool with CLI + desktop GUI (Wails) to find similar, cropped, thumbnail or duplicate images across a large local library. No external services.
>
> 完全本地化的图片反向搜索工具。指定一个图像目录即可构建可检索索引，随后用任意图片查询，找出视觉相似、旋转/缩放、裁剪、缩略图或重复的图片。排序引擎（`internal/nnengine`，复用 `image-search-test` 项目）是训练好的 learning-to-rank MLP：融合旋转不变手工特征（HSV 直方图、极坐标形状、角向 FFT、Zernike 矩、掩膜 Dice）与 sczl 颜色无关专家签名（占据栅格 + Fourier + 径向直方图 + 区域匹配），配合两阶段粗筛使检索成本随库规模近似亚线性（1 万张约 0.35s/查询）。提供命令行、本地 Web 界面与 Wails 桌面界面；索引、权重均为本地文件，无需任何外部服务。

Topics: `image-search` `neural-network` `reverse-image-search` `image-retrieval` `machine-learning` `go` `golang` `computer-vision`

支持命令行检索与 Wails 桌面 GUI（不带参数启动时默认打开界面，内置构建索引 / 检索 / 图库浏览）。

![alt text](image.png)
![alt text](image-1.png)

## 原理

### 神经网络排序引擎（`internal/nnengine`）

底层算法与索引来自 `image-search-test` 项目的检索引擎，核心是一个**学习排序（learning-to-rank）MLP**：

1. **特征提取**：对分割出的 icon sprite 计算一组手工描述子——HSV 直方图、极坐标形状（环×扇区）、角向 FFT 幅值、Zernike 矩、64×64 旋转对齐掩膜 Dice，以及逐环形状细节。
2. **第二专家（sczl）**：颜色无关签名（占据栅格 + Fourier 描述子 + 径向直方图 + 区域匹配），对"背景色变化 / 图标颜色变化 / 四周文字 / 填充区域"鲁棒——它的错误集与手工特征互补。
3. **MLP 融合**：把每个 (query, ref) 对的 27 维相似度 + 查询级统计喂给训练好的 `81 → 96 → 48 → 1` 网络，输出相关性并排序。网络由 `image-search-test` 在 `gentest` 生成的大规模带标签样本上训练得到（`weights.gob`）。
4. **两阶段粗筛**：先用廉价旋转不变签名（直方图+Zernike+径向+角向，纯余弦、无扫描）对全库取 top-N，再只对 top-N 计算昂贵的掩膜旋转 / sczl / NN / 形状上下文精排——检索成本随库规模近似亚线性，1 万条库预计 ~0.35s/查询。
5. **形状上下文精排**：对 top-12 候选做 48 点的形状上下文匈牙利匹配二次精排。

在 `image-search-test` 的 118 张留出测试集上：**recall@1 = 96.6%、recall@3 = 98.3%**（旧手工融合 baseline 为 92.4%）。

### 索引与本地存储

- 索引 = 引擎缓存文件（gob）：每张参考图同时保存手工 Feat 与 sczl 描述子，构建/重启时按 mtime 增量复用，免重复解码。
- 神经网络权重 `weights.gob` 需随程序放置（程序目录/当前目录，或用环境变量 `NN_WEIGHTS` 指定）；缺失时检索退化为廉价签名排序。

## 解决的实际问题

- **从文字描述中找不到图片**：当用户记不清图片文件名或内容，只能凭"大致长这样"的印象去检索时，用图像本身去搜，胜过手动翻目录。
- **旋转 / 缩放 / 背景色变化 / 文字干扰的 icon 检索**：传统整图哈希对旋转、换背景、四周加文字基本失效；本引擎的旋转不变描述子 + sczl 颜色无关专家对此鲁棒。
- **大图库的快速反向查重**：为海量图像批量建索引后，一次查询即可找出相似图片，常用于找重复/近似图标、素材去重。
- **无数据库的轻量部署**：索引与权重均为本地文件，命令行与本地 Web 界面即可完成建库与检索，无需任何外部服务。

## 构建

### 桌面 GUI（Wails，Windows）

```bash
wails build -tags wails     # 产物: build/bin/go-image-search.exe
```

不带参数运行 `go-image-search.exe` 即默认打开桌面界面；构建索引、以图搜图、图库浏览均可在界面内完成（目录/文件选择使用系统原生对话框）。请将 `weights.gob` 放在程序目录（或用 `NN_WEIGHTS` 指定）。

### 命令行工具

```bash
go build -o bin/go-image-search .
```

> 桌面界面代码位于 `gui.go`（`-tags wails` 时才编译），因此普通 `go build` / `go test ./...` 不受 Wails/WebView2 依赖影响，可正常跨平台构建。

## 使用

不带参数启动即进入桌面 GUI：

```
go-image-search             # 启动桌面界面
go-image-search help        # 查看命令行帮助
```

其余命令行子命令：

```
go-image-search build -dir <图像库目录> -out <索引文件>
go-image-search query -index <索引文件> -q <查询图像> [-top N]
go-image-search serve [-addr <host:port>] [-root <图像库目录>] [-index <索引文件>]  # Web 界面
```

### 示例

```bash
# 构建索引（66 张参考图示例）
go-image-search build -dir ./test_pngs -out index.bin

# 查询
go-image-search query -index index.bin -q query.png -top 5

# 启动 Web 界面（可带 root 浏览图库）
go-image-search serve -addr 127.0.0.1:8080 -root ./test_pngs -index index.bin

# 启动桌面界面
go-image-search
```

环境变量：

- `NN_WEIGHTS`：神经网络权重文件路径（默认依次查找程序目录 / 当前目录 `weights.gob`）
- `PREFILTER_N`：两阶段粗筛的 top-N 短名单大小（默认 50）
- `SCTOP` / `SCBLEND`：形状上下文精排的候选数 / 混合权重（默认 12 / 0.7）

## 目录结构

```
main.go            命令入口（无参数=GUI；build / query / serve）
gui.go             桌面 GUI 入口（-tags wails 构建；App 绑定 + Wails Run）
gui_cli.go         非 wails 构建时的 runGUI 占位（CLI 行为）
console_*.go       控制台编码处理（Windows 设置 UTF-8 代码页）
wails.json         Wails 工程配置
weights.gob        神经网络权重（训练产物，来自 image-search-test）
frontend/          桌面界面前端（index.html / src/app.js / src/styles.css）
cmd/gentest/       测试集生成器（与 image-search-test 同源）
internal/
  imageproc/       图像加载、格式解码与查询预处理
  nnengine/        神经网络排序引擎（特征提取 + MLP + 两阶段粗筛 + 索引缓存）
    sczl/          颜色无关专家签名（占据栅格 + Fourier + 区域匹配）
  web/             Web 界面与 Wails 绑定服务（server.go + assets + service.go：HTTP 与 GUI 共用）
```

## 测试

```bash
go test ./...
```
