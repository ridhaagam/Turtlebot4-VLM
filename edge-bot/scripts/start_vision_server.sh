#!/bin/bash
# Start llama.cpp server with SmolVLM2 for trash bin detection

LLAMA_DIR="/home/g3ubuntu/ROS/llama.cpp"
MODEL="${LLAMA_DIR}/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MMPROJ="${LLAMA_DIR}/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
PORT=8888

# Check if server is already running
if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "Vision server already running on port ${PORT}"
    exit 0
fi

# Check if models exist
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Download with:"
    echo "  cd ${LLAMA_DIR}/models"
    echo "  wget https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: MMProj not found: $MMPROJ"
    echo "Download with:"
    echo "  cd ${LLAMA_DIR}/models"
    echo "  wget https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
    exit 1
fi

echo "Starting SmolVLM2 vision server on port ${PORT}..."
echo "Model: SmolVLM2-500M-Video-Instruct-Q8_0"
echo ""

cd "${LLAMA_DIR}"
export LD_LIBRARY_PATH="${LLAMA_DIR}/build/bin:$LD_LIBRARY_PATH"

# Start server in background
./build/bin/llama-server \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --ctx-size 4096 \
    2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "Server is ready!"
        echo ""
        echo "To test: python3 /home/g3ubuntu/ROS/test_llama_vision.py"
        echo "To stop: pkill -f llama-server"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Server failed to start within 60 seconds"
kill $SERVER_PID 2>/dev/null
exit 1
