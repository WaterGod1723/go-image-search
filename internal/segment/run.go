// run.go 区域划分统一入口：Run 运行任意策略组合，RunDefault 运行默认流水线。
package segment

import "image"

// Run 是区域划分的统一入口：按序运行一组区域划分策略，返回像素级划分结果与最终产出。
// strategies 可为 DefaultPipeline(...) 的展开，也可为自定义策略或二者的组合，
// 从而实现"不同划分策略可替换、可组合"的插件式扩展。
func Run(src image.Image, strategies ...Strategy) (*Result, Outcome, error) {
	ctx := &Context{Src: src}
	out, err := Pipeline(strategies).Run(ctx)
	return ctx.Res, out, err
}

// RunDefault 使用默认流水线（像素颜色划分 → 相似合并 → 引力聚合 → 辅助区域构建）
// 对图像执行区域划分，等价于历史 processRegions 行为。
func RunDefault(src image.Image, cfg Config, mergeCfg MergeConfig, grav GravityConfig) (*Result, Outcome, error) {
	return Run(src, DefaultPipeline(cfg, mergeCfg, grav)...)
}
