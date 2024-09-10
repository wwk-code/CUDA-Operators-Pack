#!/bin/bash

if [ $# -eq 0 ]; then
    arg=0
else
    # 获取脚本的第一个参数，kernel version
    arg=$1
fi

echo "$arg"

if [ "$arg" -eq 0 ];then
    nsys profile --stats=true ../build/top_k > ../profile_outputs/top_k_baseline.txt
elif [ "$arg" -eq 1 ];then
    nsys profile --stats=true ../build/top_k > ../profile_outputs/top_k_baseline_1.txt
else 
    echo "ERROR! there's no matched version kernel!"
fi