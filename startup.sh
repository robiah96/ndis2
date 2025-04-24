#!/bin/bash
echo "🔧 Installing dependencies..."
apt-get update && apt-get install -y python3-pip
pip3 install --upgrade pip
pip3 install -r /home/site/wwwroot/requirements.txt

echo "🚀 Starting FastAPI app..."
python3 -m uvicorn rf_mission_api:app --host 0.0.0.0 --port 8000