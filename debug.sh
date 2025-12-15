#!/bin/bash
adb forward tcp:9090 tcp:9090
# get the directory of the script
SCRIPT_DIR=$(cd $(dirname $0); pwd)
# echo $SCRIPT_DIR

# you need to first download lldb-server
adb shell "cd /data/local/tmp/mllm/bin && /data/local/tmp/lldb-server platform --server --listen '*:9090' &"
# gnome-terminal -- bash -c "adb shell 'cd /data/local/tmp && ./lldb-server platform --listen *:9090 --server'"