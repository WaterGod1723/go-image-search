package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// writeManifest dumps the collected samples to manifest.json in the output dir.
func writeManifest() error {
	data, err := json.MarshalIndent(collected, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*outDir, "manifest.json"), data, 0o644)
}
