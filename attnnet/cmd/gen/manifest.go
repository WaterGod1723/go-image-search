package main

import (
	"encoding/json"
	"os"
)

// Sample is the ground-truth record for one generated sample.
type Sample struct {
	Image     string    `json:"image"`
	Src       string    `json:"src"`
	Crop      [4]int    `json:"crop"`
	Canvas    [2]int    `json:"canvas"`
	BGHex     string    `json:"bg_hex"`
	Rotation  float64   `json:"rotation"`
	Scale     float64   `json:"scale"`
	Translate [2]int    `json:"translate"`
	Texts     []TextOut `json:"texts"`
}

func writeManifest(path string, samples []Sample) error {
	data, err := json.MarshalIndent(samples, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
