// 区域产出入库转换：将 segment 流水线的区域产出转换为待入库的 RegionHash。
package index

import (
	"go-image-search/internal/segment"
)

// RegionHashes 将区域划分产出的区域集合转换为待入库的 RegionHash 列表。
// 存在辅助区域（蓝框/绿框）时仅入库辅助区域（红框划分区域不直接入库，
// 避免碎片化索引影响召回）；否则入库划分区域（红框）。
func RegionHashes(res *segment.Result, out segment.Outcome) []RegionHash {
	idx := out.Partition
	if len(out.Aux) > 0 {
		idx = out.Aux
	}
	hashes := make([]RegionHash, 0, len(idx))
	for _, m := range idx {
		bw, bh := m.BBox.Dx(), m.BBox.Dy()
		fill, aspect := 0.0, 0.0
		if bw > 0 && bh > 0 {
			fill = float64(m.Area) / float64(bw*bh)
			aspect = float64(bw) / float64(bh)
		}
		nx, ny := 0.0, 0.0
		if res.Width > 0 {
			nx = float64(m.BBox.Min.X+m.BBox.Max.X) / (2 * float64(res.Width))
		}
		if res.Height > 0 {
			ny = float64(m.BBox.Min.Y+m.BBox.Max.Y) / (2 * float64(res.Height))
		}
		hashes = append(hashes, RegionHash{
			RegionID: m.ID,
			Hash:     m.Hash,
			Shape:    m.Shape,
			Area:     m.Area,
			BBox:     m.BBox,
			Color:    m.MeanColor,
			NX:       nx,
			NY:       ny,
			Fill:     fill,
			Aspect:   aspect,
			Global:   segment.GlobalWeight(m),
		})
	}
	return hashes
}
