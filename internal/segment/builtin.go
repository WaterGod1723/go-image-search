// builtin.go 内置区域划分策略：每个策略实现 Strategy，可独立使用，也可经 Pipeline 组合。
// 策略是算法层（color/merge/gravity）的薄适配器；DefaultPipeline 返回与历史
// processRegions 一致的默认流水线。
package segment

import (
	"go-image-search/internal/phash"
)

// ColorSegment 像素颜色连通区域划分策略：基于颜色相似性与连通性划分出像素级区域
// （Context.Res），并计算每个区域的感知哈希特征（Context.Infos），产出初始划分区域集合。
// 可用自定义的像素级划分策略（网格/色相/模型分区等）替换。
func ColorSegment(cfg Config) Strategy {
	return named("color-segment", func(ctx *Context, in Outcome) (Outcome, error) {
		res, err := Segment(ctx.Src, cfg)
		if err != nil {
			return in, err
		}
		infos := make([]RegionInfo, 0, len(res.Regions))
		partition := make([]MergedRegion, 0, len(res.Regions))
		for _, reg := range res.Regions {
			crop := res.Crop(ctx.Src, reg.ID)
			if crop == nil {
				continue
			}
			info := RegionInfo{
				ID:    reg.ID,
				Hash:  phash.Hash(crop),
				Shape: phash.Hash(structuralMask(crop)),
				Area:  reg.Area,
				Color: reg.MeanColor,
				BBox:  reg.BBox,
			}
			infos = append(infos, info)
			partition = append(partition, MergedRegion{
				ID: info.ID, Members: []int{info.ID}, Hash: info.Hash, Shape: info.Shape,
				Area: info.Area, BBox: info.BBox, MeanColor: info.Color,
			})
		}
		ctx.Res = res
		ctx.Infos = infos
		out := in
		out.Partition = partition
		return out, nil
	})
}

// Merge 相似合并策略：将感知哈希相近且空间邻近/相邻的区域合并为组合区域。
// 需在 ColorSegment 之后运行（读取 ctx.Res 与 ctx.Infos）。
func Merge(cfg MergeConfig) Strategy {
	return named("merge", func(ctx *Context, in Outcome) (Outcome, error) {
		out := in
		out.Partition = MergeSimilar(ctx.Src, ctx.Res, ctx.Infos, cfg)
		return out, nil
	})
}

// Gravity 引力聚合策略：区域过多时按引力模型聚合。Additive 模式产出辅助组合区域
// （写入 in.Aux）；replace 模式用聚合后的完整集合替换划分区域。
func Gravity(cfg GravityConfig) Strategy {
	return named("gravity", func(ctx *Context, in Outcome) (Outcome, error) {
		out := in
		gravity := GravityMerge(ctx.Src, in.Partition, cfg)
		if cfg.Additive {
			out.Aux = gravity
		} else {
			out.Partition = gravity
		}
		return out, nil
	})
}

// FullFrameFilter 边框过滤策略：丢弃 bbox 覆盖图像宽、高均 ≥ ratio 的区域
// （可能为无效边框/背景）。ratio ≤ 0 时跳过。
func FullFrameFilter(ratio float64) Strategy {
	return named("filter-full-frame", func(ctx *Context, in Outcome) (Outcome, error) {
		if ratio <= 0 {
			return in, nil
		}
		b := ctx.Src.Bounds()
		out := in
		out.Partition = FilterFullFrame(in.Partition, b.Dx(), b.Dy(), ratio)
		return out, nil
	})
}

// WholeAux 整图辅助策略：按 ShouldAddWholeAux 判定是否追加"整图"辅助区域（绿框）。
// 判定的输入（partition 与 aux 数量）取判定时刻的值，故应置于 MergeContained /
// RedToBlue 之前；计算的整图区域暂存于 ctx.Whole，由 AppendWhole 在 RedToBlue 之后追加。
func WholeAux(cfg GravityConfig) Strategy {
	return named("whole-aux", func(ctx *Context, in Outcome) (Outcome, error) {
		out := in
		if ShouldAddWholeAux(cfg, out.Partition, out.Aux) {
			w := MergeAll(ctx.Src, out.Partition)
			ctx.Whole = &w
		}
		return out, nil
	})
}

// MergeContained 包含/相交合并策略：合并 bbox 相交或相互包含的辅助区域为新的
// 组合区域，提升每个索引区域的全局信息。
func MergeContained() Strategy {
	return named("merge-contained", func(ctx *Context, in Outcome) (Outcome, error) {
		out := in
		out.Aux = MergeContainedOverlapping(ctx.Src, in.Aux)
		return out, nil
	})
}

// RedToBlue 红框转蓝策略：把划分区域逐个标记为辅助区域（Whole=false）追加到 aux，
// 使索引只包含蓝框与绿框、红框不直接入库。
func RedToBlue(cfg GravityConfig) Strategy {
	return named("red-to-blue", func(ctx *Context, in Outcome) (Outcome, error) {
		if !cfg.Enabled || cfg.CombineFew <= 0 || len(in.Partition) < 2 {
			return in, nil
		}
		out := in
		for _, p := range in.Partition {
			p.Whole = false
			out.Aux = append(out.Aux, p)
		}
		return out, nil
	})
}

// AppendWhole 收尾策略：将 WholeAux 判定的"整图"辅助区域追加到 aux 末尾。
// 应置于 RedToBlue 之后、Relabel 之前，保证整图区域的 ID 排最后且不被
// MergeContained 合并。
func AppendWhole() Strategy {
	return named("append-whole", func(ctx *Context, in Outcome) (Outcome, error) {
		if ctx.Whole == nil {
			return in, nil
		}
		out := in
		out.Aux = append(out.Aux, *ctx.Whole)
		return out, nil
	})
}

// Relabel 收尾策略：对划分区域与辅助区域统一顺序编号（partition 1..P，
// aux P+1..P+A）。
func Relabel() Strategy {
	return named("relabel", func(ctx *Context, in Outcome) (Outcome, error) {
		out := in
		id := 0
		for i := range out.Partition {
			id++
			out.Partition[i].ID = id
		}
		for i := range out.Aux {
			id++
			out.Aux[i].ID = id
		}
		return out, nil
	})
}

// DefaultPipeline 返回与历史 processRegions 完全一致的默认流水线：
// 像素划分 → 相似合并 → 引力聚合 → 边框过滤 → 整图辅助判定 → 辅助合并 →
// 红框转蓝 → 追加整图辅助 → 统一编号。
// 返回的切片可直接展开传入统一入口：Run(src, DefaultPipeline(...)...)。
func DefaultPipeline(cfg Config, mergeCfg MergeConfig, grav GravityConfig) []Strategy {
	return []Strategy{
		ColorSegment(cfg),
		Merge(mergeCfg),
		Gravity(grav),
		FullFrameFilter(grav.FrameRatio),
		WholeAux(grav),
		MergeContained(),
		RedToBlue(grav),
		AppendWhole(),
		Relabel(),
	}
}
