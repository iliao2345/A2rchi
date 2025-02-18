#!/bin/bash

# create volume if it doesn't already exist
exists=`docker volume ls | awk '{print $2}' | grep a2rchi-prod-801-data`
if [[ $exists != 'a2rchi-prod-801-data' ]]; then
    docker volume create --name a2rchi-prod-801-data
fi

# build base image; try to reuse previously built image
cd A2rchi-prod-801/deploy/prod-801/
docker build -f ../dockerfiles/Dockerfile-base -t a2rchi-base:BASE_TAG ../..

# start services
echo "Starting docker compose"
docker compose -f prod-801-compose.yaml up -d --build --force-recreate --always-recreate-deps

# # secrets files are created by CI pipeline and destroyed here
# rm secrets/cleo_*.txt
# rm secrets/imap_*.txt
# rm secrets/sender_*.txt
# rm secrets/flask_uploader_app_secret_key.txt
# rm secrets/uploader_salt.txt
# rm secrets/openai_api_key.txt
# rm secrets/hf_token.txt
