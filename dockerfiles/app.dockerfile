# syntax=docker/dockerfile:1.3-labs

# Build a docker image for serving the CVML Debugger Streamlit app

FROM python:3.7-slim

ARG DEBIAN_FRONTEND="noninteractive"

ADD __code.tar.gz /app
WORKDIR /app

RUN <<EOF
set -e
apt-get update
apt-get install -y --no-install-recommends build-essential libssl-dev libasound2 wget
apt-get clean
date > build_date.txt
python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
EOF

CMD streamlit run src/app.py --theme.base light