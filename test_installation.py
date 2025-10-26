#!/usr/bin/env python3
"""
Test script to verify the AI chatbot installation
Run this script to check if everything is set up correctly
"""

import sys
import os
import json

def test_python_version():
    """Test if Python version is compatible"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible!")
        print("   Python 3.7 or higher is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n📦 Testing module imports...")
    
    modules = [
        ('flask', 'Flask'),
        ('tensorflow', 'TensorFlow'),
        ('nltk', 'NLTK'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('pickle', 'Pickle'),
        ('json', 'JSON')
    ]
    
    all_imported = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {display_name}: {e}")
            all_imported = False
    
    return all_imported

def test_nltk_data():
    """Test if NLTK data is available"""
    print("\n📚 Testing NLTK data...")
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Test tokenization
        test_text = "Hello world! This is a test."
        tokens = word_tokenize(test_text)
        print(f"✅ Tokenization works: {len(tokens)} tokens found")
        
        # Test lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized = lemmatizer.lemmatize("running")
        print(f"✅ Lemmatization works: 'running' -> '{lemmatized}'")
        
        return True
    except Exception as e:
        print(f"❌ NLTK data test failed: {e}")
        return False

def test_files():
    """Test if required files exist"""
    print("\n📁 Testing required files...")
    
    required_files = [
        'app.py',
        'train_model.py',
        'intents.json',
        'requirements.txt',
        'templates/index.html'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def test_model_files():
    """Test if model files exist (optional)"""
    print("\n🧠 Testing model files...")
    
    model_files = [
        'chatbot_model.h5',
        'words.pkl',
        'classes.pkl'
    ]
    
    all_exist = True
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"⚠️  Missing: {file_path} (will be created during training)")
            all_exist = False
    
    return all_exist

def test_intents_data():
    """Test if intents.json is valid"""
    print("\n📋 Testing intents data...")
    
    try:
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        
        if 'intents' not in intents:
            print("❌ Invalid intents.json: missing 'intents' key")
            return False
        
        intent_count = len(intents['intents'])
        print(f"✅ Found {intent_count} intents in intents.json")
        
        # Check structure of first intent
        if intent_count > 0:
            first_intent = intents['intents'][0]
            required_keys = ['tag', 'patterns', 'responses']
            for key in required_keys:
                if key not in first_intent:
                    print(f"❌ Invalid intent structure: missing '{key}' key")
                    return False
        
        print("✅ Intents data structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error reading intents.json: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI Chatbot Installation Test")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("Module Imports", test_imports),
        ("NLTK Data", test_nltk_data),
        ("Required Files", test_files),
        ("Model Files", test_model_files),
        ("Intents Data", test_intents_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\n📋 Next steps:")
        print("1. Train the model: python train_model.py")
        print("2. Run the app: python app.py")
        print("3. Open: http://localhost:5000")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\n💡 Common solutions:")
        print("- Install Python 3.7+: https://python.org/downloads/")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('wordnet')\"")

if __name__ == "__main__":
    main()
