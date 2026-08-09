package sczl

// hull.go 孔洞填充聚合占据栅格：对前景掩码做"孔洞填充"（淹没边框泛洪去外景，
// 剩余被前景包围的背景袋视为孔洞并填实），得到"补全镂空"后的实心占据。
//
// 与凸包（填到凸 envelope，过度膨胀）不同，孔洞填充只填"被前景包围的封闭袋"：
//  - fact_check 图库（框线+勾+线）框边围出封闭内部 → 填实为矩形，与 query 的 1 块对齐；
//  - dashboard 的 4 方块间隙对外开口（不封闭）→ 不填，保持 4 块结构；
//  - 实心图标无孔洞 → 填充后不变。
// 因此 filledOcc 是 occ 的严格改进（occ ∪ 封闭孔洞），直接替换 occ 即可，
// 无需 max/权重妥协。

// holeFill 对 64×64 软掩码做孔洞填充，返回填实后的 64×64 软掩码。
// 孔洞填充：从四条边对外景背景做 BFS 泛洪，未被泛洪到达的背景 = 封闭孔洞 → 填实。
func holeFill(maskSoft []float64) []float64 {
	if len(maskSoft) != normSize*normSize {
		return make([]float64, normSize*normSize)
	}
	filled := make([]float64, normSize*normSize)
	for i, v := range maskSoft {
		if v > 0.5 {
			filled[i] = v
		}
	}
	exterior := make([]bool, normSize*normSize)
	queue := make([]int, 0, 256)
	for x := 0; x < normSize; x++ {
		top := x
		bot := (normSize-1)*normSize + x
		if filled[top] <= 0 && !exterior[top] {
			exterior[top] = true
			queue = append(queue, top)
		}
		if filled[bot] <= 0 && !exterior[bot] {
			exterior[bot] = true
			queue = append(queue, bot)
		}
	}
	for y := 0; y < normSize; y++ {
		l := y * normSize
		r := y*normSize + normSize - 1
		if filled[l] <= 0 && !exterior[l] {
			exterior[l] = true
			queue = append(queue, l)
		}
		if filled[r] <= 0 && !exterior[r] {
			exterior[r] = true
			queue = append(queue, r)
		}
	}
	nbrs := [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for len(queue) > 0 {
		idx := queue[0]
		queue = queue[1:]
		y := idx / normSize
		x := idx % normSize
		for _, d := range nbrs {
			nx, ny := x+d[0], y+d[1]
			if nx < 0 || nx >= normSize || ny < 0 || ny >= normSize {
				continue
			}
			ni := ny*normSize + nx
			if filled[ni] <= 0 && !exterior[ni] {
				exterior[ni] = true
				queue = append(queue, ni)
			}
		}
	}
	for i := range filled {
		if filled[i] <= 0 && !exterior[i] {
			filled[i] = 1.0
		}
	}
	return filled
}

// filledOccupancy 对 64×64 软掩码做孔洞填充后下采样到 n×n 软占据（L2 归一化）。
func filledOccupancy(maskSoft []float64, n int) []float64 {
	return downsampleMask(holeFill(maskSoft), normSize, n)
}

// convexHull 保留供 solidity 计算使用。
func convexHull(pts [][2]float64) [][2]float64 {
	n := len(pts)
	if n == 0 {
		return nil
	}
	sorted := make([][2]float64, n)
	copy(sorted, pts)
	convexHullSort(sorted)
	j := 0
	for i := 0; i < n; i++ {
		if i > 0 && sorted[i] == sorted[i-1] {
			continue
		}
		sorted[j] = sorted[i]
		j++
	}
	sorted = sorted[:j]
	if j < 3 {
		return sorted
	}
	hull := make([][2]float64, 0, 2*j)
	for i := 0; i < j; i++ {
		for len(hull) >= 2 && convexHullCross(hull[len(hull)-2], hull[len(hull)-1], sorted[i]) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, sorted[i])
	}
	low := len(hull) + 1
	for i := j - 2; i >= 0; i-- {
		for len(hull) >= low && convexHullCross(hull[len(hull)-2], hull[len(hull)-1], sorted[i]) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, sorted[i])
	}
	if len(hull) > 1 {
		hull = hull[:len(hull)-1]
	}
	return hull
}

func convexHullSort(s [][2]float64) {
	for i := 1; i < len(s); i++ {
		for k := i; k > 0; k-- {
			a, b := s[k-1], s[k]
			if a[0] < b[0] || (a[0] == b[0] && a[1] <= b[1]) {
				break
			}
			s[k-1], s[k] = b, a
		}
	}
}

func convexHullCross(o, a, b [2]float64) float64 {
	return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
}

// solidity 实度 = 前景像素数 / 凸包面积 [0,1]，刻画轮廓的镂空/碎片化程度。
func solidity(maskSoft []float64) float64 {
	if len(maskSoft) != normSize*normSize {
		return 0
	}
	var pts [][2]float64
	fgArea := 0
	for y := 0; y < normSize; y++ {
		for x := 0; x < normSize; x++ {
			if maskSoft[y*normSize+x] > 0.1 {
				pts = append(pts, [2]float64{float64(x), float64(y)})
				fgArea++
			}
		}
	}
	hull := convexHull(pts)
	hullArea := polygonArea(hull)
	if hullArea <= 0 {
		return 0
	}
	s := float64(fgArea) / hullArea
	if s > 1 {
		s = 1
	}
	return s
}

// polygonArea 凸多边形面积（鞋带公式）。
func polygonArea(poly [][2]float64) float64 {
	n := len(poly)
	if n < 3 {
		return 0
	}
	s := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		s += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
	}
	if s < 0 {
		s = -s
	}
	return s / 2
}
