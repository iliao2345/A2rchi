#!/bin/bash

echo "Stop running docker compose"
cd A2rchi-prod-cp/deploy/prod-cp/
docker compose -f prod-cp-compose.yaml down
