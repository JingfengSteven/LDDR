#!/bin/bash
# 一键分析 CLIP 缓存中的 std 分布

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="/path/to/your/workspace/cache_center/clip-cache"
OUTPUT_DIR="./stats"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Analyzing Relevance Std Distribution from Cache"
echo "================================================"
echo "Cache dir: $CACHE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# 运行统计分析
python3 "$SCRIPT_DIR/analyze_std_from_cache.py" \
    --cache_dir "$CACHE_DIR" \
    --output "$OUTPUT_DIR/std_distribution.json"

# 生成可视化
python3 "$SCRIPT_DIR/visualize_std.py" \
    --input "$OUTPUT_DIR/std_distribution.json" \
    --output "$OUTPUT_DIR/std_distribution.png"

echo ""
echo "================================================"
echo "Done! Results:"
echo "  - Stats: $OUTPUT_DIR/std_distribution.json"
echo "  - Plot:  $OUTPUT_DIR/std_distribution.png"
echo "================================================"
