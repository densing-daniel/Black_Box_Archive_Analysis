# Black Box Archive Analysis

<div align="center">

**Advanced AI-powered aviation incident analysis using Groq's LLM technology**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration) • [License](#-license)

</div>

## 📖 Description

Black Box Archive Analysis is an intelligent aviation safety analysis system that processes black box recordings and Air Traffic Control (ATC) communications to generate comprehensive incident investigation reports. Powered by Groq's state-of-the-art AI models (Whisper-large-v3 for transcription and Llama-3.1-8b for analysis), this tool helps aviation professionals, safety investigators, and researchers quickly analyze and document flight incidents.

### 🎯 Key Capabilities

- **Audio Transcription**: Converts MP3 audio recordings into accurate text using Groq's Whisper-large-v3 model
- **Content Validation**: Automatically detects aviation-related content to ensure relevant analysis
- **Intelligent Report Generation**: Creates structured incident reports with key details, safety concerns, and recommendations
- **Interactive Chat Assistant**: Ask questions about the generated report using natural language
- **Professional Export**: Download reports in DOCX format for official documentation

---

## ✨ Features

### 🎙️ Audio Processing
- Support for MP3 audio files
- Real-time audio preview
- High-accuracy transcription using Groq Whisper-large-v3
- Aviation content detection and validation

### 📋 Report Generation
- Structured incident reports with:
  - General information (date, aircraft, location, incident type)
  - Key event details
  - Safety concerns identification
  - Actionable recommendations
- Clean, professional formatting
- Markdown and DOCX export options

### 💬 AI Assistant
- Context-aware question answering
- Report-specific queries
- Concise, accurate responses
- User-friendly chat interface

### 🎨 User Interface
- Modern, responsive design
- Three-column layout for optimal workflow
- Real-time processing indicators
- Professional aviation-themed styling

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/))
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/black-box-archive-analysis.git
cd black-box-archive-analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## 📦 Dependencies
```
streamlit>=1.28.0
openai>=1.0.0
python-dotenv>=1.0.0
pydub>=0.25.1
python-docx>=1.0.0
```

---

## 💻 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

1. **Upload Audio File**
   - Click "Browse files" in the left panel
   - Select an MP3 file containing aviation communications
   - Preview the audio in the embedded player

2. **Analyze Content**
   - Click "🚀 Analyze Black Box Audio"
   - Wait for transcription and analysis to complete
   - View the generated report in the center panel

3. **Ask Questions**
   - Use the Report Assistant in the right panel
   - Type questions about the incident
   - Get instant, context-aware answers

4. **Export Report**
   - Click "📥 Download Report (.docx)"
   - Save the professional document for official use

### Example Questions for the Assistant
- "What was the primary incident type?"
- "List all safety recommendations"
- "What aircraft was involved?"
- "What were the key safety concerns?"
- "When did this incident occur?"

---

## ⚙️ Configuration

### Rate Limiting
The application includes intelligent rate limiting to manage API usage:
- Tracks token usage per minute
- Automatic retry with exponential backoff
- Jitter for distributed requests
- User-friendly wait notifications

### Customization
Edit `app.py` to customize:
- **Model Selection**: Change `llama-3.1-8b-instant` to other Groq models
- **Max Tokens**: Adjust `max_tokens` parameter for longer/shorter responses
- **Temperature**: Modify for more creative or deterministic outputs
- **Styling**: Update CSS in the Streamlit markdown sections

---

## 🏗️ Project Structure
```
black-box-archive-analysis/
│
├── app.py                 # Main application file
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
└── uploaded_file.mp3     # Temporary file (auto-generated)
```

---

## 🔒 Security Notes
- Never commit your `.env` file to version control
- Keep your Groq API key confidential
- The application processes files locally for privacy
- No audio data is stored permanently

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: GROQ_API_KEY not found  
**Solution**: Ensure `.env` file exists with correct API key

**Issue**: Rate limit errors  
**Solution**: Wait 60 seconds; the app handles this automatically

**Issue**: Non-aviation content detected  
**Solution**: Ensure audio contains flight/ATC communications

**Issue**: Transcription errors  
**Solution**: Check audio quality and format (MP3 required)

---

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments
- Groq for providing powerful AI models
- Streamlit for the excellent web framework
- OpenAI for the API interface standards
- Aviation safety professionals who inspired this project

---

## 📧 Contact
For questions, suggestions, or support:
- GitHub Issues: Create an issue
- Email: your.email@example.com

---

## 🔮 Future Enhancements
- Support for additional audio formats (WAV, M4A)
- Multi-language transcription support
- Batch processing for multiple files
- Advanced analytics and visualization
- Database integration for historical analysis
- PDF export option
- Real-time audio streaming analysis

---

<div align="center">

**Made with ❤️ for Aviation Safety**

⭐ Star this repository if you find it helpful!

</div>
