#!/bin/bash
# 将 Jupyter notebooks 转换为 markdown 格式

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 需要转换的笔记本文件（按顺序）
NOTEBOOKS=(
    "1.quickstart.ipynb"
    "2.stategraph.ipynb"
    "3.middleware.ipynb"
    "4.human_in_the_loop.ipynb"
    "5.memory.ipynb"
    "6.context.ipynb"
    "7.mcp_server.ipynb"
    "8.supervisor.ipynb"
    "9.parallelization.ipynb"
    "10.rag.ipynb"
    "11.web_search.ipynb"
)

PROJECT_ROOT="$SCRIPT_DIR"
OUTPUT_DIR="$PROJECT_ROOT/skills/dive-into-langgraph/references"

# 清理旧的输出（可选）
# rm -rf "$OUTPUT_DIR"/*.md "$OUTPUT_DIR"/*_files 2>/dev/null

# 安装依赖（如未安装）
install_dependencies() {
    if ! python -c "import mdformat" 2>/dev/null; then
        echo "安装 mdformat..."
        pip install mdformat
    fi
}

# 转换单个 notebook 并处理图片引用
convert_notebook() {
    local nb_path="$1"
    local filename=$(basename "$nb_path" .ipynb)
    local output_file="$OUTPUT_DIR/${filename}.md"

    echo "转换: $(basename "$nb") -> ${filename}.md"

    # 先用默认方式转换（会生成 *_files 文件夹）
    if python -m jupyter nbconvert \
        --to markdown \
        --output-dir "$OUTPUT_DIR" \
        --output "${filename}.md" \
        "$nb_path" 2>&1; then
        echo "  ✅ 成功"
    else
        echo "  ❌ 失败"
        return 1
    fi

    # 将图片引用替换为注释（Bash 实现）
    if [ -f "$output_file" ]; then
        # 匹配 ![alt](xxx_files/image.png) 格式，替换为 <!-- IMAGE: xxx/image.png -->
        # 使用双引号，这样 ! 不需要转义
        sed -i '' -E "s/!\[([^]]*)\]\(([^_]+)_files\/([^)]+)\)/<!-- IMAGE: \2\/\3 -->/g" "$output_file"
    fi
}

# 主流程
main() {
    install_dependencies

    # 确保输出目录存在
    mkdir -p "$OUTPUT_DIR"

    # 转换每个 notebook
    for nb in "${NOTEBOOKS[@]}"; do
        nb_path="$PROJECT_ROOT/$nb"

        # 检查文件是否存在
        if [ ! -f "$nb_path" ]; then
            echo "⚠️  $nb 不存在，跳过"
            continue
        fi

        convert_notebook "$nb_path" || exit 1
    done

    # 删除所有 *_files 文件夹
    echo ""
    echo "清理 *_files 文件夹..."
    rm -rf "$OUTPUT_DIR"/*_files 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ 已删除所有 *_files 文件夹"
    fi

    echo ""
    echo "格式化 markdown 文件..."

    # 格式化所有生成的 markdown 文件
    shopt -s nullglob
    md_files=("$OUTPUT_DIR"/*.md)
    shopt -u nullglob
    if [ ${#md_files[@]} -gt 0 ]; then
        mdformat "${md_files[@]}"
    fi

    echo "🎉 全部完成! 输出目录: $OUTPUT_DIR"
}

main