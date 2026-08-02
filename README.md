# go-image-search

基于感知哈希（pHash）与图像区域划分的图片反向搜索工具。支持命令行检索与 iOS 风格 Web 界面。

## 原理

传统 pHash 对整张图计算指纹，难以处理局部相似或背景差异较大的情况。本项目先将图像划分为多个区域（自适应阈值 + 连通域分析），再对每个区域单独计算感知哈希；感知哈希相近且空间相邻的区域会被合并为组合区域。搜索时基于区域集合进行匹配，从而支持局部/局部衍生图的相似检索。

原始图像与其"骨架图"（衍生图）会分别建索引与检索，任一表示命中即视为相似。

## 构建

```bash
go build -o bin/go-image-search .
```

## 使用

```
go-image-search build -dir <图像库目录> -out <索引文件> [分段参数...]
go-image-search query -index <索引文件> -q <查询图像> [-top N] [-maxdist D]
go-image-search segments -img <图像> [-out <可视化png>]   # 调试：查看区域划分
go-image-search serve [-addr <host:port>] [-root <图像库目录>] [-index <索引文件>]
```

### 分段参数

- `-threshold-pct <0~1>` 相邻色差分位数
- `-factor <x>` 阈值因子
- `-min-area-ratio <r>` 噪声区域面积比例阈值
- `-median <k>` 预处理中值滤波核（0 关闭）
- `-connectivity <4|8>` 连通性

### 合并参数

- `-merge-dist <汉明距离阈值>` 合并的 pHash 汉明距离阈值
- `-merge-color <颜色阈值>` 合并的平均色归一化距离阈值
- `-no-merge` 关闭相似区域合并

### 示例

```bash
# 构建索引
go-image-search build -dir ./images -out index.bin

# 查询
go-image-search query -index index.bin -q query.png -top 5

# 查看某张图的区域划分
go-image-search segments -img query.png -out seg.png

# 启动 Web 界面
go-image-search serve -root ./images -index index.bin
```

## 目录结构

```
main.go            命令入口（build / query / segments / serve）
console_*.go       控制台编码处理（Windows 设置 UTF-8 代码页）
internal/
  imageproc/       图像加载、滤波与衍生图生成
  phash/           感知哈希
  segment/         区域划分与相似区域合并
  index/           索引序列化与检索
  web/              Web 界面（server.go + assets）
```

## 测试

```bash
go test ./...
```