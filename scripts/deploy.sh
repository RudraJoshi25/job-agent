#!/usr/bin/env bash
# Usage: ./scripts/deploy.sh
# Requires: EC2_HOST, EC2_USER, EC2_KEY_PATH set in environment or exported before running.

set -euo pipefail

EC2_HOST="${EC2_HOST:?EC2_HOST not set}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_KEY_PATH="${EC2_KEY_PATH:?EC2_KEY_PATH not set}"
REPO_DIR="${REPO_DIR:-/home/ubuntu/job-agent}"

echo "Deploying to ${EC2_USER}@${EC2_HOST} ..."

ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" bash <<REMOTE
  set -euo pipefail
  cd "${REPO_DIR}"

  echo "--- pulling latest main ---"
  git fetch origin main
  git reset --hard origin/main

  echo "--- stopping containers ---"
  docker compose down --remove-orphans || true

  echo "--- building and starting ---"
  docker compose up --build -d

  echo "--- container status ---"
  docker compose ps

  echo "--- health check ---"
  sleep 5
  curl -sf http://localhost:8000/health && echo "Health OK" || echo "Health check FAILED"
REMOTE

echo "Deploy complete."
