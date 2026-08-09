package sczl

import (
	"encoding/gob"
	"fmt"
	"os"
	"runtime"
	"sync"
)

// index.go SCZL 线性索引：构建/保存/加载。检索在 search.go。
// 库规模通常数千至数万，FD 为低维向量，暴力 L2/余弦足以；如需扩展可换 KD-tree。

// BuildProgress 单个文件构建结果（按输入顺序回调）。
type BuildProgress struct {
	Done int
	File string
	ID   string
	Err  error
}

// Index 线性描述子索引。
type Index struct {
	Entries []Descriptor
	Images   map[string]bool
}

// New 创建空索引。
func New() *Index {
	return &Index{Images: make(map[string]bool)}
}

// Add 追加一张图像的描述子。
func (ix *Index) Add(d Descriptor) {
	ix.Entries = append(ix.Entries, d)
	ix.Images[d.ImageID] = true
}

// AddImage 用 imageID 与描述子追加（描述子.ImageID 以传入 id 为准）。
func (ix *Index) AddImage(imageID string, d Descriptor) {
	d.ImageID = imageID
	ix.Add(d)
}

// Len 返回条目总数。
func (ix *Index) Len() int { return len(ix.Entries) }

// BuildParallel 并发构建索引。proc 对单个文件返回描述子（无效则跳过）。
// onProgress 可为 nil，按输入顺序回调。
func (ix *Index) BuildParallel(files []string, workers int, proc func(file string) (Descriptor, error), onProgress func(p BuildProgress)) int {
	if len(files) == 0 {
		return 0
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > len(files) {
		workers = len(files)
	}
	if workers <= 1 {
		added := 0
		for i, f := range files {
			d, err := proc(f)
			if err != nil || !d.Valid {
				if onProgress != nil {
					onProgress(BuildProgress{Done: i + 1, File: f, Err: err})
				}
				continue
			}
			d.ImageID = f
			ix.Add(d)
			added++
			if onProgress != nil {
				onProgress(BuildProgress{Done: i + 1, File: f, ID: f})
			}
		}
		return added
	}

	type result struct {
		idx int
		d   Descriptor
		err error
	}
	jobs := make(chan int)
	results := make(chan result, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for fi := range jobs {
				d, err := proc(files[fi])
				results <- result{idx: fi, d: d, err: err}
			}
		}()
	}
	go func() {
		for fi := range files {
			jobs <- fi
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	added := 0
	next := 0
	pending := make(map[int]result)
	for r := range results {
		pending[r.idx] = r
		for {
			cur, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			next++
			if cur.err != nil || !cur.d.Valid {
				if onProgress != nil {
					onProgress(BuildProgress{Done: next, File: files[cur.idx], Err: cur.err})
				}
				continue
			}
			cur.d.ImageID = files[cur.idx]
			ix.Add(cur.d)
			added++
			if onProgress != nil {
				onProgress(BuildProgress{Done: next, File: files[cur.idx], ID: files[cur.idx]})
			}
		}
	}
	return added
}

// Save 以 gob 编码写入文件。
func (ix *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if err := enc.Encode(ix); err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	return nil
}

// Load 从 gob 文件加载索引。
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	var ix Index
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&ix); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if ix.Images == nil {
		ix.Images = make(map[string]bool)
		for _, e := range ix.Entries {
			ix.Images[e.ImageID] = true
		}
	}
	return &ix, nil
}
