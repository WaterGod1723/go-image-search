#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$PATH:$(go env GOPATH)/bin"
export GOSUMDB=off

NAME="go-image-search"
GOOS="$(go env GOOS)"

# 将仓库根目录的图标（git 追踪）同步到 Wails 读取的 build/appicon.png
if [[ -f "$NAME.png" ]]; then
  cp "$NAME.png" build/appicon.png
fi

# wails build 输出目录
OUT_DIR="build/bin"

# 带 all 参数：交叉编译各平台产物到 dist 目录；不带参数：构建当前平台到仓库根目录
if [[ "${1:-}" == "all" ]]; then
  TARGET_DIR="dist"
  PLATFORMS="darwin/arm64 darwin/amd64 windows/amd64"
else
  TARGET_DIR="."
  PLATFORMS="$(go env GOOS)/$(go env GOARCH)"
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
  wails build -tags wails -platform "$plat" 2>&1 | tail -40
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