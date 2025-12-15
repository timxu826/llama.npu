adb shell " rm -rf /data/local/tmp/xujia/llama.cpp "
adb push pkg-adb/llama.cpp /data/local/tmp/xujia/

# wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf

#if the model file does not exist, upload it
if [ ! -f /data/local/tmp/xujia/llama.cpp/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf ]; then
    adb push Llama-3.2-1B-Instruct-Q4_0.gguf /data/local/tmp/xujia/llama.cpp/gguf
fi

