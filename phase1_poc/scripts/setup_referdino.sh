#!/bin/bash
# ReferDINO 安装脚本
# 用法: bash scripts/setup_referdino.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/../weights"
REFERDINO_DIR="$WEIGHTS_DIR/referdino"

echo "=========================================="
echo "  ReferDINO 安装脚本"
echo "=========================================="

# 检查代理设置
if [ -z "$https_proxy" ]; then
    echo "[提示] 如果网络受限，请先设置代理:"
    echo "  export http_proxy=agent.baidu.com:8188"
    echo "  export https_proxy=agent.baidu.com:8188"
    echo ""
fi

# Step 1: 克隆 ReferDINO 仓库
echo ""
echo "[Step 1/4] 克隆 ReferDINO 仓库..."
if [ -d "$REFERDINO_DIR" ]; then
    echo "  目录已存在: $REFERDINO_DIR"
    echo "  跳过克隆，使用现有代码"
else
    git clone https://github.com/iSEE-Laboratory/ReferDINO.git "$REFERDINO_DIR"
    echo "  克隆完成: $REFERDINO_DIR"
fi

# Step 2: 安装 Python 依赖
echo ""
echo "[Step 2/4] 安装 Python 依赖..."
cd "$REFERDINO_DIR"
pip install -r requirements.txt
echo "  依赖安装完成"

# Step 3: 编译 MultiScaleDeformableAttention
echo ""
echo "[Step 3/4] 编译 MultiScaleDeformableAttention..."
cd "$REFERDINO_DIR/models/GroundingDINO/ops"
if [ -f "build/lib.linux-x86_64-cpython-*/MultiScaleDeformableAttention.cpython-*.so" ] 2>/dev/null; then
    echo "  已编译，跳过"
else
    python setup.py build install
    echo "  编译完成"
fi

# Step 4: 下载模型权重
echo ""
echo "[Step 4/4] 下载模型权重..."
cd "$REFERDINO_DIR"
mkdir -p ckpt pretrained

# 下载 GroundingDINO 预训练权重
if [ ! -f "pretrained/groundingdino_swinb_cogcoor.pth" ]; then
    echo "  下载 GroundingDINO Swin-B 权重..."
    wget -P pretrained https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
else
    echo "  GroundingDINO 权重已存在"
fi

# 下载 ReferDINO 权重（推荐使用 ryt_mevis_swinb.pth，在多个数据集上训练）
if [ ! -f "ckpt/ryt_mevis_swinb.pth" ]; then
    echo "  下载 ReferDINO 权重 (ryt_mevis_swinb.pth)..."
    # 从 HuggingFace 下载
    wget -P ckpt https://huggingface.co/liangtm/referdino/resolve/main/ryt_mevis_swinb.pth
else
    echo "  ReferDINO 权重已存在"
fi

# 验证安装
echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "目录结构:"
echo "  $REFERDINO_DIR/"
echo "  ├── configs/"
echo "  ├── models/"
echo "  ├── pretrained/"
echo "  │   └── groundingdino_swinb_cogcoor.pth"
echo "  └── ckpt/"
echo "      └── ryt_mevis_swinb.pth"
echo ""
echo "测试命令:"
echo "  cd $REFERDINO_DIR"
echo "  python demo_video.py <video_path> --text 'your description' -ckpt ckpt/ryt_mevis_swinb.pth"
echo ""
echo "在 T2V_Grounding 中使用:"
echo "  from visual_grounding.referdino_grounder import ReferDINOGrounder"
echo "  grounder = ReferDINOGrounder()"
echo ""
