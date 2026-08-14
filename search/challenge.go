package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ChallengeEntry is one challenge query with its controlled transform type.
type ChallengeEntry struct {
	Image     string `json:"image"`
	Src       string `json:"src"`
	SourceID  string `json:"source_id,omitempty"`
	Type      string `json:"type"`       // position, scale, rotation, occlusion, composition, hard-negative
	Param     string `json:"param"`     // e.g. "left", "0.25", "45", "0.20"
	TrueRef   string `json:"true_ref"`  // filename of the target ref in the gallery
}

// runChallengeEval loads a challenge set from root/challenge/challenge_manifest.json
// (if it exists) and evaluates the model per challenge type. Challenge queries
// rank against the test-split gallery only.
func runChallengeEval(root string, m *AttnNet, refs []*Feat, refNames []string,
	refSrcIDs []string, trainSrcs, valSrcs, testSrcs map[string]bool,
	groups []sourceGroup) {
	chDir := filepath.Join(root, "challenge")
	manifestPath := filepath.Join(chDir, "challenge_manifest.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		fmt.Printf("\nChallenge: (no challenge set at %s)\n", manifestPath)
		return
	}
	var entries []ChallengeEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		fmt.Printf("\nChallenge: failed to parse manifest: %v\n", err)
		return
	}
	if len(entries) == 0 {
		fmt.Printf("\nChallenge: (empty manifest)\n")
		return
	}

	// build test-split gallery
	testRefIdxs := collectSplitRefs(groups, testSrcs)
	if len(testRefIdxs) == 0 {
		// fall back to val if test is empty
		testRefIdxs = collectSplitRefs(groups, valSrcs)
	}
	if len(testRefIdxs) == 0 {
		fmt.Printf("\nChallenge: (no test/val refs to rank against)\n")
		return
	}
	galleryRefs := makeGalleryRefs(refs, testRefIdxs)
	gallerySrcIDs := makeGallerySrcIDs(refSrcIDs, testRefIdxs)
	galleryMap := makeGalleryMap(testRefIdxs)

	// ensure gallery ref embeddings (evalAllEmb already ran in caller, but be safe)
	ensureStructEmb(galleryRefs, m.SN)

	// group entries by type
	byType := map[string][]ChallengeEntry{}
	for _, e := range entries {
		byType[e.Type] = append(byType[e.Type], e)
	}

	fmt.Printf("\nChallenge (n=%d, gallery=%d refs):\n", len(entries), len(testRefIdxs))
	for _, ctype := range []string{"position", "scale", "rotation", "occlusion", "composition", "hard-negative"} {
		es := byType[ctype]
		if len(es) == 0 {
			continue
		}
		mt := evalChallengeEntries(m, es, chDir, refs, refNames, galleryRefs, gallerySrcIDs, galleryMap)
		printMetrics("  "+ctype, mt)
	}
}

// evalChallengeEntries builds nnQueries from challenge entries and evaluates.
func evalChallengeEntries(m *AttnNet, entries []ChallengeEntry, chDir string,
	refs []*Feat, refNames []string, galleryRefs []*Feat, gallerySrcIDs []string,
	galleryMap map[int]int) rankMetrics {

	var qs []nnQuery
	for _, e := range entries {
		img, err := loadPNG(filepath.Join(chDir, e.Image))
		if err != nil {
			continue
		}
		px := extractQuery(img)
		if len(px) == 0 {
			continue
		}
		q := buildFeat(px)
		// find the true ref's global index
		trueIdx := -1
		for j, n := range refNames {
			if n == e.TrueRef || n == e.Src {
				trueIdx = j
				break
			}
		}
		if trueIdx < 0 {
			continue
		}
		if m.SN != nil {
			q.Emb = m.SN.embedEmbRot(q.Thumb16)
		}
		qs = append(qs, nnQuery{q: q, trueIdx: trueIdx})
	}
	return computeRankMetrics(m, qs, galleryRefs, gallerySrcIDs, galleryMap)
}
