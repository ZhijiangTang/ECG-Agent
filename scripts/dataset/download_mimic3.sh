#!/bin/bash
trap 'kill -- -$$' EXIT
# 文件下载的基础 URL
BASE_URL="https://physionet.org/files/mimic3wdb-matched/1.0/p"

# 文件编号范围
START=1
END=9999

# 下载目录
DOWNLOAD_DIR="../Data/mimic3_ecg_sub0002/"

# 如果下载目录不存在，则创建
mkdir -p "$DOWNLOAD_DIR"

# 获取目录文件列表，并过滤出符合 *0001.* 的文件
get_files_to_download() {
    export -f get_files_to_download
    local p_num=$1
    local i=$2
    local DIR_URL="${BASE_URL}${p_num}/p${p_num}$(printf "%04d" $i)/"  # 拼接文件的 URL

    # 获取目录内容（不下载），并过滤出符合 0001.* 格式的文件
    wget -q -O - "$DIR_URL" | grep -oP 'href="([^"]*0002\.[^"]*)"' | while read -r filename; do
        # 完整文件 URL
        FILE_URL="$DIR_URL$(echo "$filename" | sed 's/href="\([^"]*\)"/\1/')"
        local OUTPUT_DIR="${DOWNLOAD_DIR}/p${p_num}/p${p_num}$(printf "%04d" $i)"
        mkdir -p "$OUTPUT_DIR"

        PID=$!
        wait $PID
        # 调用下载函数
        # echo "Downloading $FILE_URL ..."
        wget -q -N -c -np -P "$OUTPUT_DIR" "$FILE_URL" &
        if [ $? -ne 0 ]; then
            echo "Failed to download $FILE_URL"
        else
            echo "Successfully downloaded $FILE_URL $PID"
        fi

    done
}

# 导出函数供并行调用
export -f get_files_to_download
export BASE_URL DOWNLOAD_DIR

# 使用并行下载（适当减少并发数量，避免过度并发导致问题）
for p_num in $(seq -f "%02g" 0 9); do
    seq $START $END | xargs -n 1 -P 10000 -I {} bash -c 'get_files_to_download '$p_num' {}'
done

echo "All downloads are complete."
