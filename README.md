# AI Customer Support Chatbot

An intelligent chatbot leveraging natural language processing to provide customer support and automated responses. Built with Python, Flask, and TensorFlow.

## Features

- 🤖 **Intelligent Responses**: Uses TensorFlow neural network for natural language understanding
- 💬 **Real-time Chat Interface**: Modern, responsive web interface
- 🎯 **Intent Classification**: Automatically detects user intent and provides appropriate responses
- 📱 **Mobile Friendly**: Responsive design that works on all devices
- 🔄 **Quick Actions**: Pre-defined buttons for common queries
- 📊 **Confidence Scoring**: Shows confidence level for each response

## Technology Stack

- **Backend**: Python, Flask
- **AI/ML**: TensorFlow, NLTK, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript
- **NLP**: Natural Language Processing with lemmatization and tokenization

## Prerequisites

Before installing the chatbot, make sure you have:

- **Python 3.7 or higher** installed on your system
  - Download from [python.org](https://www.python.org/downloads/)
  - Make sure to check "Add Python to PATH" during installation
- **pip** (usually comes with Python)
- **Git** (optional, for cloning the repository)

## Installation

### Quick Setup (Recommended)

1. **Download the project**
   - Download and extract the project files to a folder
   - Or clone with: `git clone <repository-url>`

2. **Run the automated setup**
   ```bash
   python setup.py
   ```
   This will automatically:
   - Install all dependencies
   - Download NLTK data
   - Train the AI model
   - Verify everything is working

3. **Start the chatbot**
   ```bash
   python app.py
   ```
   Or on Windows, double-click `run.bat`

4. **Open your browser and go to**
   ```
   http://localhost:5000
   ```

### Manual Setup

If you prefer to set up manually:

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

3. **Train the AI model**
   ```bash
   python train_model.py
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```

### Windows Users

If you're on Windows and having issues:

1. Make sure Python is installed and added to PATH
2. Use `run.bat` for easy startup
3. If you get permission errors, run Command Prompt as Administrator

## Project Structure

```
Chatbot_project/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── intents.json          # Training data and responses
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Chat interface
├── chatbot_model.h5      # Trained model (generated)
├── words.pkl            # Vocabulary (generated)
├── classes.pkl          # Intent classes (generated)
└── README.md            # This file
```

## Usage

### Training the Model

The chatbot uses a neural network that needs to be trained before use:

```bash
python train_model.py
```

This will:
- Load training data from `intents.json`
- Preprocess text using NLTK
- Train a TensorFlow neural network
- Save the trained model and vocabulary

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send message and get response
- `GET /health` - Health check and model status

### Example API Usage

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, I need help with my order"}'
```

## Customization

### Adding New Intents

Edit `intents.json` to add new conversation patterns:

```json
{
  "tag": "new_intent",
  "patterns": [
    "Pattern 1",
    "Pattern 2"
  ],
  "responses": [
    "Response 1",
    "Response 2"
  ]
}
```

After adding new intents, retrain the model:

```bash
python train_model.py
```

### Modifying the Model

Edit `train_model.py` to:
- Change neural network architecture
- Adjust training parameters
- Modify preprocessing steps

### Styling the Interface

Edit `templates/index.html` to customize:
- Colors and themes
- Layout and spacing
- Additional features

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `train_model.py` has been run successfully
2. **NLTK errors**: Download required NLTK data manually
3. **TensorFlow issues**: Check Python and TensorFlow compatibility
4. **Port already in use**: Change port in `app.py` or kill existing process

### Dependencies Issues

If you encounter dependency conflicts:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Performance

- **Training Time**: ~2-5 minutes depending on hardware
- **Response Time**: <1 second for most queries
- **Memory Usage**: ~100-200MB for the model
- **Accuracy**: ~85-95% intent classification accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Open an issue in the repository

---

**Note**: This is a demonstration chatbot. For production use, consider additional features like:
- Database integration for conversation history
- User authentication
- Advanced NLP models (BERT, GPT)
- Integration with external APIs
- Analytics and monitoring
