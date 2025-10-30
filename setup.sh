#!/bin/bash
echo "🚀 Setting up Python virtual environment..."

# Update system packages
sudo yum update -y

# Install Python 3 and virtual environment tools
sudo yum install python3 python3-virtualenv -y

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
