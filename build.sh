#!/usr/bin/env bash
set -euo pipefail

# ---- 检测系统工具链 ----
is_wsl=0
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;                # Git Bash: 原生 Windows 工具链
  *)
    if uname -r | grep -qi microsoft; then # WSL
      is_wsl=1
    fi
    ;;
esac

# ---- 定位 go：支持原生 go 与 Windows 的 go.exe ----
GO_CMD="$(command -v go 2>/dev/null || command -v go.exe 2>/dev/null || true)"
if [ -z "$GO_CMD" ]; then
  echo "错误: 未找到 golang，请先安装 Go 并加入 PATH。"
  exit 1
fi
echo "使用 Go: $GO_CMD"

# ---- 在 PATH 中加入 GOPATH/bin（含 Windows 路径转 WSL 挂载路径）----
export GOSUMDB=off
gopath="$( "$GO_CMD" env GOPATH 2>/dev/null || true )"
if [ -n "$gopath" ]; then
  gopath="${gopath//\\//}"                       # C:\Users\x -> C:/Users/x
  if [ "$is_wsl" = 1 ] && echo "$gopath" | grep -qE '^[A-Za-z]:/'; then
    drive="$(printf '%s' "$gopath" | cut -c1)"
    gopath="/mnt/${drive,,}${gopath#?:}"          # C:/Users/x -> /mnt/c/Users/x
  fi
  export PATH="$PATH:$gopath/bin"
fi

cd "$(dirname "$0")"

NAME="go-image-search"
GOOS="$("$GO_CMD" env GOOS)"

# 将仓库根目录的图标（git 追踪）同步到 Wails 读取的 build/appicon.png
if [[ -f "$NAME.png" ]]; then
  cp "$NAME.png" build/appicon.png
fi

# ---- 前端依赖 ----
if command -v npm >/dev/null 2>&1; then
  echo "==> 安装前端依赖 ..."
  ( cd frontend && npm install ) || { echo "错误: 前端依赖安装失败。"; exit 1; }
else
  echo "警告: 未找到 npm，将跳过前端构建。"
fi

# ---- 安装/核对 wails CLI ----
echo "==> 安装/核对 wails CLI ..."
"$GO_CMD" install github.com/wailsapp/wails/v2/cmd/wails@v2.13.0 || true

WEXE="$(command -v wails 2>/dev/null || command -v wails.exe 2>/dev/null || true)"
if [ -z "$WEXE" ]; then
  echo "错误: 找不到 wails，请检查 GOPATH/bin 是否在 PATH。"
  exit 1
fi
echo "使用 wails: $WEXE"

# wails build 输出目录
OUT_DIR="build/bin"

# 带 all 参数：交叉编译各平台产物到 dist 目录；不带参数：构建当前平台到仓库根目录
if [[ "${1:-}" == "all" ]]; then
  TARGET_DIR="dist"
  PLATFORMS="darwin/arm64 darwin/amd64 windows/amd64"
else
  TARGET_DIR="."
  PLATFORMS="$("$GO_CMD" env GOOS)/$("$GO_CMD" env GOARCH)"
fi

mkdir -p "$TARGET_DIR"

# 逐平台构建；单个平台失败则跳过继续下一个
FAILED=()
for plat in $PLATFORMS; do
  os="${plat%%/*}"
  case "$os" in
    darwin)  ARTIFACT="$NAME.app" ;;
    windows) ARTIFACT="$NAME.exe" ;;
    *)       ARTIFACT="$NAME" ;;
  esac

  echo "==> wails build ($plat)"
  set +e
  "$WEXE" build -tags wails -platform "$plat" 2>&1 | tail -40
  exit_code=${PIPESTATUS[0]}
  set -e
  if [[ $exit_code -ne 0 ]]; then
    echo "WARN: build failed for $plat (exit $exit_code), skipping" >&2
    FAILED+=("$plat")
    continue
  fi

  if [[ ! -e "$OUT_DIR/$ARTIFACT" ]]; then
    echo "WARN: 产物 $OUT_DIR/$ARTIFACT 不存在，跳过 $plat" >&2
    FAILED+=("$plat")
    continue
  fi
  # all 模式：产物名加平台后缀，避免多架构同名覆盖
  if [[ "${1:-}" == "all" ]]; then
    suffix="${plat/\//-}"
    case "$os" in
      darwin)  DEST="$NAME-$suffix.app" ;;
      windows) DEST="$NAME-$suffix.exe" ;;
      *)       DEST="$NAME-$suffix" ;;
    esac
  else
    DEST="$ARTIFACT"
  fi
  rm -rf "$TARGET_DIR/$DEST"
  if [[ "$GOOS" == "darwin" ]]; then
    ditto "$OUT_DIR/$ARTIFACT" "$TARGET_DIR/$DEST"
  else
    cp -R "$OUT_DIR/$ARTIFACT" "$TARGET_DIR/$DEST"
  fi
  echo "==> 产物已输出到: $TARGET_DIR/$DEST"

  # macOS: 注册 .app 到 LaunchServices，使 Finder 图标立即生效
  if [[ "$os" == "darwin" && "$GOOS" == "darwin" ]]; then
    lsregister="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
    "$lsregister" -f "$TARGET_DIR/$DEST" 2>/dev/null || true
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "以下平台构建失败: ${FAILED[*]}" >&2
  exit 1
fi