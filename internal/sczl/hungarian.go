package sczl

import "math"

// hungarian.go 最小成本二分图匹配（O(n^3)）。复用 internal/index 的实现思路，
// 用于 SC 点对点一对一分配，避免多个 query 点绑定同一 entry 点。
// 当矩阵规模大时，search.go 已通过 SCMaxPoints 下采样控制规模。

const hungarianPad = 1e7

// hungarian 求解 cost（r×c）的最小成本匹配，返回每行分配的列下标，未分配返回 -1。
func hungarian(cost [][]float64) []int {
	r := len(cost)
	res := make([]int, r)
	for i := range res {
		res[i] = -1
	}
	if r == 0 {
		return res
	}
	c := len(cost[0])
	if c == 0 {
		return res
	}
	n := r
	if c > n {
		n = c
	}
	a := make([][]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			a[i][j] = hungarianPad
		}
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if !math.IsInf(cost[i][j], 1) {
				a[i][j] = cost[i][j]
			}
		}
	}

	u := make([]float64, n+1)
	v := make([]float64, n+1)
	p := make([]int, n+1)
	way := make([]int, n+1)
	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		used := make([]bool, n+1)
		for j := 1; j <= n; j++ {
			minv[j] = math.Inf(1)
		}
		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := 0
			for j := 1; j <= n; j++ {
				if used[j] {
					continue
				}
				cur := a[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}
	for j := 1; j <= n; j++ {
		row := p[j] - 1
		if row >= 0 && row < r && j-1 < c && !math.IsInf(cost[row][j-1], 1) {
			res[row] = j - 1
		}
	}
	return res
}
