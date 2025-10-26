#!/usr/bin/env python3
"""
Setup script for AI Customer Support Chatbot
This script automates the installation and setup process
"""

import subprocess
import sys
import os
import json

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    nltk_script = """
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded successfully!")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    exit(1)
"""
    return run_command(f'python -c "{nltk_script}"', "Downloading NLTK data")

def train_model():
    """Train the chatbot model"""
    print("\n🧠 Training AI model...")
    return run_command("python train_model.py", "Training the chatbot model")

def verify_setup():
    """Verify that everything is set up correctly"""
    print("\n🔍 Verifying setup...")
    
    # Check if model files exist
    model_files = ['chatbot_model.h5', 'words.pkl', 'classes.pkl']
    for file in model_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            return False
        print(f"✅ Found: {file}")
    
    # Check if intents.json exists
    if not os.path.exists('intents.json'):
        print("❌ Missing: intents.json")
        return False
    print("✅ Found: intents.json")
    
    # Test import of main modules
    try:
        import flask
        import tensorflow as tf
        import nltk
        print("✅ All required modules can be imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 AI Customer Support Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("\n❌ Failed to download NLTK data. Please check the error messages above.")
        sys.exit(1)
    
    # Train the model
    if not train_model():
        print("\n❌ Failed to train the model. Please check the error messages above.")
        sys.exit(1)
    
    # Verify setup
    if not verify_setup():
        print("\n❌ Setup verification failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: python app.py")
    print("2. Open your browser and go to: http://localhost:5000")
    print("3. Start chatting with your AI assistant!")
    print("\n💡 For more information, see README.md")

if __name__ == "__main__":
    main()
