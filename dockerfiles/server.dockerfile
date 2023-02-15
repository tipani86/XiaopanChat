# syntax=docker/dockerfile:1.3-labs

# Build a docker image for serving the CVML Debugger Streamlit app

FROM python:3.7-slim

ARG DEBIAN_FRONTEND="noninteractive"

ADD __code.tar.gz /server
WORKDIR /server

RUN <<EOF
set -e
date > build_date.txt
python3 -m pip install --no-cache-dir -r requirements_server.txt
EOF

CMD flask --app src/server.py run