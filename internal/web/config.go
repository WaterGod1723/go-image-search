package web

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// AppConfig 持久化用户选择，避免每次启动重新配置。
type AppConfig struct {
	IndexPath        string  `json:"indexPath,omitempty"`
	Root             string  `json:"root,omitempty"`
	Algorithm        string  `json:"algorithm,omitempty"`
	SearchTop        int     `json:"searchTop,omitempty"`
	SearchMaxDist    int     `json:"searchMaxDist,omitempty"`
	SearchColorWeight float64 `json:"searchColorWeight,omitempty"`
}

// configDir 返回配置文件所在目录（~/.config/go-image-search/）。
func configDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".config", "go-image-search")
}

// configPath 返回配置文件完整路径。
func configPath() string {
	dir := configDir()
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, "config.json")
}

// LoadConfig 从磁盘读取配置文件，文件不存在时返回零值（不报错）。
func LoadConfig() AppConfig {
	p := configPath()
	if p == "" {
		return AppConfig{}
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return AppConfig{}
	}
	var cfg AppConfig
	_ = json.Unmarshal(data, &cfg)
	return cfg
}

// SaveConfig 将当前配置写入磁盘。
func SaveConfig(cfg AppConfig) error {
	p := configPath()
	if p == "" {
		return nil
	}
	dir := filepath.Dir(p)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(p, data, 0o644)
}
