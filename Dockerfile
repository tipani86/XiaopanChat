# syntax=docker/dockerfile:1.3-labs

# Build a docker image for serving the CVML Debugger Streamlit app

FROM python:3.7-slim

ARG DEBIAN_FRONTEND="noninteractive"

ADD __app.tar.gz /app

RUN <<EOF
set -e
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir -r /app/requirements.txt
EOF

WORKDIR /app
CMD streamlit run src/app.py