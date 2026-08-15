package iconnet

import (
	"math"
	"sort"
)

// Index is a pre-vectorized gallery: each reference icon is reduced to an
// L2-normalized embedding, enabling fast cosine retrieval.
type Index struct {
	Net       *Net
	Size      int      // sprite side
	Keys      []string // one per ref (file name / id)
	Embs      []float64 // flat Embs: len(Keys) x EmbDim, normalized
	EmbDim    int
	normConst float64 // unused placeholder kept simple
}

// NewIndex builds a gallery index from refs. ref is called per key; return the
// sprite (nil -> skip). embeddings are computed with the shared net.
func NewIndex(net *Net, keys []string, ref func(key string) (*Sprite, bool)) *Index {
	idx := &Index{Net: net, Size: net.InputSize, EmbDim: net.EmbDim}
	embs := make([]float64, 0, len(keys)*net.EmbDim)
	kept := make([]string, 0, len(keys))
	for _, k := range keys {
		sp, ok := ref(k)
		if !ok || sp == nil {
			continue
		}
		e := net.Embed(sp.Tensor())
		embs = append(embs, e...)
		kept = append(kept, k)
	}
	idx.Keys = kept
	idx.Embs = embs
	return idx
}

// Len returns the number of indexed refs.
func (idx *Index) Len() int { return len(idx.Keys) }

// EmbedSprite embeds a sprite (utility, e.g. for queries).
func (idx *Index) EmbedSprite(sp *Sprite) []float64 { return idx.Net.Embed(sp.Tensor()) }

// Query returns top-k refs by cosine similarity to query embedding q.
// Returns parallel slices of keys and similarities (descending).
func (idx *Index) Query(q []float64, topk int) ([]string, []float64) {
	n := len(idx.Keys)
	if n == 0 {
		return nil, nil
	}
	type kv struct {
		k string
		s float64
	}
	sims := make([]kv, n)
	emb := idx.Embs
	for i := 0; i < n; i++ {
		var s float64
		row := emb[i*idx.EmbDim : (i+1)*idx.EmbDim]
		for j := 0; j < idx.EmbDim; j++ {
			s += row[j] * q[j]
		}
		sims[i] = kv{idx.Keys[i], s}
	}
	sort.Slice(sims, func(a, b int) bool { return sims[a].s > sims[b].s })
	if topk > n {
		topk = n
	}
	keys := make([]string, topk)
	sc := make([]float64, topk)
	for i := 0; i < topk; i++ {
		keys[i] = sims[i].k
		sc[i] = sims[i].s
	}
	return keys, sc
}

// Cosine is the dot product of two L2-normalized embeddings.
func Cosine(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Norm returns the L2 norm.
func Norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}
