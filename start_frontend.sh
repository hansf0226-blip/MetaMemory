#!/bin/bash
# MetaMemory 前端启动脚本
# Linear Dark 风格仪表盘 — 纯静态 HTML，无需构建

FRONTEND_DIR="$(cd "$(dirname "$0")" && pwd)/frontend"
PORT="${1:-3002}"

echo "🚀 MetaMemory 前端"
echo "   风格: Linear Dark · 易经记忆系统"
echo "   端口: $PORT"
echo "   文件: $FRONTEND_DIR/index.html"
echo ""
echo "   启动方式:"
echo "   1. 直接打开: open $FRONTEND_DIR/index.html"
echo "   2. HTTP 服务 (推荐): python3 -m http.server $PORT -d $FRONTEND_DIR"
echo ""

# 直接打开浏览器
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$FRONTEND_DIR/index.html" 2>/dev/null && echo "✅ 已在浏览器中打开" || echo "⚠️  请手动打开: $FRONTEND_DIR/index.html"
else
    echo "📂 请手动打开: $FRONTEND_DIR/index.html"
fi

# 同时启动 HTTP 服务（用于开发调试）
echo ""
echo "📡 启动 HTTP 开发服务器 (Ctrl+C 停止)..."
cd "$FRONTEND_DIR" && python3 -m http.server "$PORT"
