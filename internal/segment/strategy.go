// strategy.go 策略层：Strategy 接口、共享上下文与结果类型、StrategyFunc 适配器及
// Pipeline 流水线编排。内置策略见 builtin.go，统一入口见 run.go。
package segment

import "image"

// Context 贯穿策略流水线的共享上下文，供各策略阶段读取/写入中间结果。
type Context struct {
	Src   image.Image   // 被划分的图像
	Res   *Result       // 像素级划分结果（标签图），由像素级策略阶段产出
	Infos []RegionInfo  // 各像素级区域的特征（哈希/平均色/bbox 等）
	Whole *MergedRegion // WholeAux 策略判定的"整图"辅助区域，由 AppendWhole 策略在 RedToBlue 之后追加
}

// Outcome 策略流水线运行后产出的区域集合。
type Outcome struct {
	Partition []MergedRegion // 空间不重叠的划分区域
	Aux       []MergedRegion // 辅助区域（追加，不参与划分）
}

// All 返回按 Partition → Aux 顺序连接、已统一编号的完整区域集合。
func (o Outcome) All() []MergedRegion {
	all := make([]MergedRegion, 0, len(o.Partition)+len(o.Aux))
	all = append(all, o.Partition...)
	all = append(all, o.Aux...)
	return all
}

// Strategy 区域划分策略：对图像执行区域划分（或对已有区域做后处理）。
//
// 每个策略代表一个可插拔的"划分算法/阶段"，可独立使用与测试，也可通过 Pipeline
// 按序组合成完整流水线。内置策略见 strategies.go（ColorSegment、Merge、Gravity…），
// 自定义策略只需实现本接口，或用 StrategyFunc 把普通函数适配为策略。
type Strategy interface {
	// Name 返回策略名称，便于日志与调试。
	Name() string
	// Apply 应用本策略，处理上一策略的产出 in，返回新的产出。
	// 需要像素级结果时读取 ctx.Res / ctx.Infos；in.Partition / in.Aux 为上一策略的产出。
	Apply(ctx *Context, in Outcome) (Outcome, error)
}

// StrategyFunc 将普通函数适配为 Strategy，便于测试与快速原型。
type StrategyFunc func(ctx *Context, in Outcome) (Outcome, error)

// Name 实现 Strategy。
func (f StrategyFunc) Name() string { return "custom" }

// Apply 实现 Strategy。
func (f StrategyFunc) Apply(ctx *Context, in Outcome) (Outcome, error) { return f(ctx, in) }

// namedStrategy 携带名称的函数式策略。
type namedStrategy struct {
	name string
	fn   func(ctx *Context, in Outcome) (Outcome, error)
}

// Name 实现 Strategy。
func (s namedStrategy) Name() string { return s.name }

// Apply 实现 Strategy。
func (s namedStrategy) Apply(ctx *Context, in Outcome) (Outcome, error) { return s.fn(ctx, in) }

// named 构造携带名称的函数式策略。
func named(name string, fn func(ctx *Context, in Outcome) (Outcome, error)) Strategy {
	return namedStrategy{name: name, fn: fn}
}

// Pipeline 区域划分策略流水线：按序运行一组策略，前一策略的产出作为后一策略的输入。
type Pipeline []Strategy

// Run 按声明顺序执行全部策略。ctx.Src 需预先设置；ctx.Res / ctx.Infos / ctx.Whole
// 由策略链中对应的阶段填充。
func (p Pipeline) Run(ctx *Context) (Outcome, error) {
	out := Outcome{}
	for _, s := range p {
		var err error
		out, err = s.Apply(ctx, out)
		if err != nil {
			return out, err
		}
	}
	return out, nil
}
