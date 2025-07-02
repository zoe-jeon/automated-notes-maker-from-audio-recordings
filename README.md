# 📓 Automatic Notes Maker from Audio Recording

An AI-powered Python application that lets you record speech or upload audio files to generate real-time notes.  
It supports transcription, multilingual translation, summarization, keyword extraction, text-to-speech, and export to multiple formats.

---

## 🚀 Features

- 🎙️ Live Speech Recording with instant transcription  
- 📂 Audio File Upload (.mp3 / .wav)  
- 🌐 Translate Text into 25+ languages  
- 🧠 Text Analysis: Keyword Extraction + Summarization  
- 🔊 Speak Out Translated Text using TTS  
- 💾 Export as `.txt`, `.docx`, `.pdf`, or `.md`  

---

## 🛠️ Tech Stack

- Python 3.x  
- CustomTkinter  
- SpeechRecognition  
- Googletrans  
- PyDub  
- NLTK  
- pyttsx3  
- reportlab  
- python-docx  
- soundfile  
- numpy  

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/automatic-notes-maker.git
cd automatic-notes-maker
pip install -r requirements.txt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

