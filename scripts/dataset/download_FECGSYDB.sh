#!/bin/bash

trap 'kill -- -$$' EXIT
# 文件下载的基础 URL
BASE_URL="https://physionet.org/files/fecgsyndb/1.0.0/sub"

# 文件编号范围
START=0
END=4

# 下载目录
DOWNLOAD_DIR="../Data/FECGSYDB/"

# 如果下载目录不存在，则创建
mkdir -p "$DOWNLOAD_DIR"

get_files_to_download() {
    local i=$1
    local j=$2

    local FILE_URL="${BASE_URL}${i}/snr$(printf "%02d" $((j*3)))dB/"  # 拼接文件的 URL

    echo $FILE_URL

    # 获取目录内容（不下载），并过滤出符合 0001.* 格式的文件
    local OUTPUT_DIR="${DOWNLOAD_DIR}/sub${i}/snr$(printf "%02d" $((j*3)))dB/"
    mkdir -p "$OUTPUT_DIR"

    # 调用下载函数
    # echo "Downloading $FILE_URL ..."
    wget -q -r -N -c -np --reject "index.html*" -P "$OUTPUT_DIR" "$FILE_URL"
    if [ $? -ne 0 ]; then
        echo "Failed to download $FILE_URL"
    else
        echo "Successfully downloaded $FILE_URL $PID"
    fi
    PID=$!
    wait $PID
}

# 导出函数供并行调用
export -f get_files_to_download
export BASE_URL DOWNLOAD_DIR

# 使用并行下载（适当减少并发数量，避免过度并发导致问题）
for p_num in $(seq -f "%02g" 1 10); do
    seq $START $END | xargs -n 1 -P 5 -I {} bash -c 'get_files_to_download '$p_num' {}' &
done

wait
echo "All downloads are complete."
