#!/bin/bash

set -e

git archive -v -o __code.tar.gz --format=tar.gz HEAD
docker build -t tipani86/xiaopan-chat:latest -f dockerfiles/app.dockerfile .

# If supplied with --push argument, do a docker push

if [[ $1 == "--push" ]]; 
then
  docker push tipani86/xiaopan-chat:latest
fi
