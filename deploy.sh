#!/bin/bash

# Set variables
APP_NAME="gpt-stat"  # Replace with your actual Heroku app name

# Logging into Heroku container registry
echo "Logging into Heroku container registry..."
heroku container:login

# Building a new Docker image without using cache
echo "Building new Docker image..."
docker build --no-cache -t main:latest .

# Tagging Docker image for Heroku
echo "Tagging Docker image..."
docker tag main:latest registry.heroku.com/$APP_NAME/worker

# Pushing Docker image to Heroku
echo "Pushing Docker image to Heroku..."
docker push registry.heroku.com/$APP_NAME/worker

# Releasing Docker container on Heroku
echo "Releasing Docker container..."
heroku container:release worker -a $APP_NAME

# Checking process status on Heroku
echo "Checking process status..."
heroku ps -a $APP_NAME

# Tailing logs from Heroku
echo "Tailing logs..."
heroku logs --tail -a $APP_NAME
