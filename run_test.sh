# adb shell    "cd /data/local/tmp/xujia/llama.cpp; ulimit -c unlimited;                      \
#         LD_LIBRARY_PATH=/data/local/tmp/xujia/llama.cpp/./lib                               \
#         ADSP_LIBRARY_PATH=/data/local/tmp/xujia/llama.cpp/./lib                             \
#         ././bin/llama-cli --no-mmap                                                         \
#         -m /data/local/tmp/xujia/llama.cpp/../gguf/Llama-3.2-1B-Instruct-Q4_0.gguf          \
#         -t 4 --ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on                    \
#         -ngl 99 --device HTP0                                                               \
#         -no-cnv -p \"what is the most popular cookie in the world?\""
set -x
persona=xujia
# Base directory on device
basedir=/data/local/tmp/$persona/llama.cpp-npu
htp_ops_lib_dir=/data/local/tmp/$persona/htp-ops-lib
hexagon_tool_dir=$htp_ops_lib_dir/hexagon_ReleaseG_toolv19_v75
android_release_dir=$htp_ops_lib_dir/android_ReleaseG_aarch64

htp_path=/data/local/tmp/$persona/llama.cpp/./lib

adb shell "cp $android_release_dir/*.so $htp_path/"
adb shell "cp $hexagon_tool_dir/* $htp_path/"
    
adb shell "touch $htp_path/htp_ops_test.farf"
adb shell "logcat -c"

adb shell    "cd /data/local/tmp/$persona/llama.cpp; ulimit -c unlimited;                   \
        REPACK_FOR_HVX=1                                                                    \
        GGML_HEXAGON_VERBOSE=1                                                              \    
        GGML_HEXAGON_EXPERIMENTAL=1                                                         \
        GGML_HEXAGON_ENABLE_HMX=1                                                           \    
        LD_LIBRARY_PATH=$htp_path                                                           \
        ADSP_LIBRARY_PATH=$htp_path                                                         \
        ././bin/llama-cli --no-mmap                                                         \
        -m $basedir/qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf                                       \
        -t 4 --ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on                    \
        -ngl 99 --device HTP0  -no-cnv                                                      \
        -p \"what is the most popular cookie in the world?\""