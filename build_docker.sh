#!/bin/bash

git archive -v -o __app.tar.gz --format=tar.gz HEAD
docker build -t tipani86/xiaopan-chat:latest -f Dockerfile .