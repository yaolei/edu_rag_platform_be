#!/bin/bash
set -e

# 1. 把新包写进 requirements.in
#    例：echo "newpkg==1.2.3" >> requirements.in

# 2. 重新生成锁定文件
pip-compile requirements.in --output-file requirements.lock.txt

# 3. 把国内源 + 锁定内容合并成 Docker 用的 requirements.txt
cat > requirements.txt <<'EOF'
--index-url https://mirrors.aliyun.com/pypi/simple
--extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cpu
--trusted-host mirrors.aliyun.com
--trusted-host mirrors.tuna.tsinghua.edu.cn

EOF
cat requirements.lock.txt >> requirements.txt

# 4. 用新生成的 requirements.txt 安装到当前环境
pip install -r requirements.txt

echo "✅ 新包已安装、锁定文件已更新、Docker 文件已就绪"