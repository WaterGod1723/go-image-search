// Command iconretrieve demonstrates the iconnet module: build a gallery from a
// directory of reference icons and retrieve the closest match for a query
// image.
//
// Usage:
//
//	iconretrieve -weights iconnet/weights.bin -gallery scraped_icons/test -query query.png
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"iconnet"
)

func main() {
	weights := flag.String("weights", "weights.bin", "path to exported weights .bin")
	gallery := flag.String("gallery", "", "directory of reference icons (PNG)")
	query := flag.String("query", "", "query image path")
	topk := flag.Int("topk", 5, "number of results")
	flag.Parse()

	if *gallery == "" || *query == "" {
		flag.Usage()
		os.Exit(2)
	}

	net, err := iconnet.LoadFile(*weights)
	if err != nil {
		log.Fatalf("load weights: %v", err)
	}
	fmt.Printf("network: %d blocks, emb %d, %d input\n", len(net.Ch), net.EmbDim, net.InputSize)

	// build gallery
	refNames, err := pngsIn(*gallery)
	if err != nil {
		log.Fatal(err)
	}
	idx := iconnet.NewIndex(net, refNames, func(k string) (*iconnet.Sprite, bool) {
		return iconnet.RefSprite(filepath.Join(*gallery, k), net.InputSize)
	})
	fmt.Printf("indexed %d gallery icons\n", idx.Len())

	// embed query
	sp, ok := iconnet.QuerySprite(*query, net.InputSize)
	if !ok {
		log.Fatalf("could not extract sprite from %s", *query)
	}
	q := idx.EmbedSprite(sp)
	keys, sims := idx.Query(q, *topk)
	fmt.Println("top matches:")
	for i := range keys {
		fmt.Printf("  %2d. %s  (cos=%.4f)\n", i+1, keys[i], sims[i])
	}
}

func pngsIn(dir string) ([]string, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".png" {
			out = append(out, e.Name())
		}
	}
	return out, nil
}
