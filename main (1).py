import customtkinter as ctk
import speech_recognition as sr
import threading
import queue
from pydub import AudioSegment
from googletrans import Translator
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from CTkMessagebox import CTkMessagebox
import pyttsx3
import os
import time
import tempfile
import wave
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import nltk
from docx import Document
import soundfile as sf
import numpy as np

# Set environment variable for ffmpeg
os.environ['PATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ.get('PATH', '')

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.key_phrases = ['in conclusion', 'to summarize', 'therefore', 'thus', 'hence', 'consequently',
                           'finally', 'in summary', 'as a result', 'overall', 'ultimately']
        
    def get_summary(self, text, num_sentences):
        if not text.strip():
            return ""
            
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        word_freq = FreqDist(words)
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            word_count = len([word for word in words_in_sentence if word.isalnum()])
            
            if word_count == 0:
                continue
                
            score = sum(word_freq[word.lower()] for word in words_in_sentence if word.lower() in word_freq)
            score = score / max(5, word_count)
            
            if i == 0 or i == len(sentences) - 1:
                score *= 1.25
            
            for phrase in self.key_phrases:
                if phrase in sentence.lower():
                    score *= 1.3
                    break
            
            if word_count < 5:
                score *= 0.8
            
            if word_count > 40:
                score *= 0.8
            
            sentence_scores[sentence] = score
        
        summary_sentences = nlargest(min(num_sentences, len(sentences)), 
                                   sentence_scores.items(), 
                                   key=lambda x: x[1])
        summary_sentences.sort(key=lambda x: text.find(x[0]))
        return " ".join([s[0] for s in summary_sentences])

class AutomaticNotesMaker:
    def __init__(self):
        try:
            nltk.download('punkt', download_dir='./nltk_data')
            nltk.download('stopwords', download_dir='./nltk_data')
            nltk.download('averaged_perceptron_tagger', download_dir='./nltk_data')
        except:
            pass
            
        self.setup_window()
        self.setup_variables()
        self.setup_mp3_support()
        self.setup_tts()
        self.create_gui()
        self.speaking = False
        self.summarizer = TextSummarizer()

    def setup_mp3_support(self):
        """Setup MP3 support using bundled ffmpeg"""
        try:
            from pydub.utils import mediainfo
            ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
            ffprobe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffprobe.exe")
            
            if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
                AudioSegment.converter = ffmpeg_path
                AudioSegment.ffmpeg = ffmpeg_path
                AudioSegment.ffprobe = ffprobe_path
        except Exception as e:
            print(f"MP3 support setup error: {str(e)}")

    def setup_window(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.geometry("1000x700")
        self.root.title("Automatic Notes Maker")

    def setup_variables(self):
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.is_listening = False
        self.languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja',
            'Korean': 'ko', 'Chinese (Simplified)': 'zh-cn', 'Hindi': 'hi',
            'Arabic': 'ar', 'Dutch': 'nl', 'Greek': 'el', 'Turkish': 'tr',
            'Vietnamese': 'vi', 'Thai': 'th', 'Polish': 'pl', 'Danish': 'da',
            'Finnish': 'fi', 'Czech': 'cs', 'Romanian': 'ro', 'Hungarian': 'hu',
            'Swedish': 'sv', 'Indonesian': 'id', 'Hebrew': 'he', 'Bengali': 'bn'
        }
        self.supported_formats = [
            ("Audio files", "*.wav;*.mp3"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
    
    def setup_tts(self):
        self.engine = None
        self.speaking = False
        self.speech_thread = None
        self.initialize_engine()
        
    def initialize_engine(self):
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1)


    def create_gui(self):
        self.container = ctk.CTkTabview(self.root)
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.container.add("Live Recording")
        self.container.add("File Processing")
        self.container.add("Analysis")
        
        self.setup_live_tab()
        self.setup_file_tab()
        self.setup_analysis_tab()
        
    def setup_live_tab(self):
        live_tab = self.container.tab("Live Recording")
        
        # Language selection frame
        lang_frame = ctk.CTkFrame(live_tab)
        lang_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(lang_frame, text="Translate to:").pack(side="left", padx=5)
        self.live_lang_var = ctk.StringVar(value="English")
        self.live_lang_combo = ctk.CTkOptionMenu(
            lang_frame, 
            values=sorted(list(self.languages.keys())),
            variable=self.live_lang_var,
            width=200
        )
        self.live_lang_combo.pack(side="left", padx=5)
        
        # Status label
        self.live_status = ctk.CTkLabel(live_tab, text="Ready", font=("Arial", 14))
        self.live_status.pack(pady=5)
        
        # Text areas frame
        text_frame = ctk.CTkFrame(live_tab)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Original text
        ctk.CTkLabel(text_frame, text="Original Text").pack(anchor="w", padx=5)
        self.live_text = ctk.CTkTextbox(text_frame, height=200)
        self.live_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Translated text
        ctk.CTkLabel(text_frame, text="Translated Text").pack(anchor="w", padx=5)
        self.live_translated = ctk.CTkTextbox(text_frame, height=200)
        self.live_translated.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(live_tab)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_btn = ctk.CTkButton(
            btn_frame, text="Start Recording",
            command=self.start_listening,
            fg_color="#4CAF50"
        )
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            btn_frame, text="Stop", 
            command=self.stop_listening,
            state="disabled",
            fg_color="#F44336"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Translate",
            command=self.translate_live_text,
            fg_color="#2196F3"
        ).pack(side="left", padx=5)
        
        self.speak_btn = ctk.CTkButton(
            btn_frame, 
            text="Speak Translation",
            command=lambda: self.speak_text(
                self.live_translated.get("1.0", "end").strip(), 
                self.live_lang_var.get()
            ),
            fg_color="#673AB7"
        )
        self.speak_btn.pack(side="left", padx=5)
        
        
        ctk.CTkButton(
            btn_frame, text="Save File",
            command=lambda: self.save_with_format('live'),
            fg_color="#9C27B0"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Clear",
            command=self.clear_live_text,
            fg_color="#607D8B"
        ).pack(side="left", padx=5)

    def setup_file_tab(self):
        file_tab = self.container.tab("File Processing")
        
        # Add supported formats info
        info_label = ctk.CTkLabel(
            file_tab, 
            text="Supported formats: WAV, MP3",
            font=("Arial", 12)
        )
        info_label.pack(pady=5)
        
        # Language selection
        lang_frame = ctk.CTkFrame(file_tab)
        lang_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(lang_frame, text="Translate to:").pack(side="left", padx=5)
        self.file_lang_var = ctk.StringVar(value="English")
        self.file_lang_combo = ctk.CTkOptionMenu(
            lang_frame,
            values=sorted(list(self.languages.keys())),
            variable=self.file_lang_var,
            width=200
        )
        self.file_lang_combo.pack(side="left", padx=5)
        
        # Status label
        self.file_status = ctk.CTkLabel(file_tab, text="Ready", font=("Arial", 14))
        self.file_status.pack(pady=5)
        
        # Text areas
        text_frame = ctk.CTkFrame(file_tab)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(text_frame, text="Original Text").pack(anchor="w", padx=5)
        self.file_text = ctk.CTkTextbox(text_frame, height=200)
        self.file_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(text_frame, text="Translated Text").pack(anchor="w", padx=5)
        self.file_translated = ctk.CTkTextbox(text_frame, height=200)
        self.file_translated.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buttons
        btn_frame = ctk.CTkFrame(file_tab)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            btn_frame, text="Load Audio",
            command=self.process_audio_file,
            fg_color="#FF9800"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Clean & Process",
            command=self.clean_and_process,
            fg_color="#009688"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Translate",
            command=self.translate_file_text,
            fg_color="#2196F3"
        ).pack(side="left", padx=5)
        
        self.speak_btn_file = ctk.CTkButton(
            btn_frame, 
            text="Speak Translation",
            command=lambda: self.speak_text(
                self.file_translated.get("1.0", "end").strip(), 
                self.file_lang_var.get()
            ),
            fg_color="#673AB7"
        )
        self.speak_btn_file.pack(side="left", padx=5)
        
        
        ctk.CTkButton(
            btn_frame, text="Save File",
            command=lambda: self.save_with_format('file'),
            fg_color="#9C27B0"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Clear",
            command=self.clear_file_text,
            fg_color="#607D8B"
        ).pack(side="left", padx=5)
    
    def setup_analysis_tab(self):
        analysis_tab = self.container.tab("Analysis")
        
        # Text input frame
        input_frame = ctk.CTkFrame(analysis_tab)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(input_frame, text="Input Text").pack(anchor="w", padx=5)
        self.analysis_input = ctk.CTkTextbox(input_frame, height=150)
        self.analysis_input.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Controls frame
        controls_frame = ctk.CTkFrame(analysis_tab)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Number of keywords/sentences selection
        ctk.CTkLabel(controls_frame, text="Number of keywords:").pack(side="left", padx=5)
        self.keyword_num = ctk.CTkEntry(controls_frame, width=50)
        self.keyword_num.insert(0, "5")
        self.keyword_num.pack(side="left", padx=5)
        
        ctk.CTkLabel(controls_frame, text="Summary sentences:").pack(side="left", padx=5)
        self.summary_num = ctk.CTkEntry(controls_frame, width=50)
        self.summary_num.insert(0, "3")
        self.summary_num.pack(side="left", padx=5)
        
        # Import buttons
        ctk.CTkButton(
            controls_frame,
            text="Import from Live",
            command=lambda: self.import_text('live'),
            fg_color="#FF9800"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            controls_frame,
            text="Import from File",
            command=lambda: self.import_text('file'),
            fg_color="#FF9800"
        ).pack(side="left", padx=5)
        
        # Results frame
        results_frame = ctk.CTkFrame(analysis_tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Keywords section
        ctk.CTkLabel(results_frame, text="Keywords").pack(anchor="w", padx=5)
        self.keywords_text = ctk.CTkTextbox(results_frame, height=100)
        self.keywords_text.pack(fill="x", padx=5, pady=5)
        
        # Summary section
        ctk.CTkLabel(results_frame, text="Summary").pack(anchor="w", padx=5)
        self.summary_text = ctk.CTkTextbox(results_frame, height=150)
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Action buttons
        btn_frame = ctk.CTkFrame(analysis_tab)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            btn_frame,
            text="Extract Keywords",
            command=self.extract_keywords,
            fg_color="#2196F3"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Generate Summary",
            command=self.generate_summary,
            fg_color="#4CAF50"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Clear All",
            command=self.clear_analysis,
            fg_color="#607D8B"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Save Results",
            command=self.save_analysis,
            fg_color="#9C27B0"
        ).pack(side="left", padx=5)

    def speak_text(self, text, lang):
        if self.speaking:
            self.stop_speaking()
            time.sleep(0.1)  # Brief pause to ensure cleanup
            return

        def speak():
            try:
                self.speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
            finally:
                self.speaking = False
                self.speech_thread = None

        self.speech_thread = threading.Thread(target=speak, daemon=True)
        self.speech_thread.start()

    def stop_speaking(self):
        if self.speaking:
            self.speaking = False
            if self.engine:
                try:
                    self.engine.endLoop()
                except:
                    pass
                try:
                    self.engine.stop()
                except:
                    pass
            self.initialize_engine()
            if self.speech_thread:
                self.speech_thread.join(timeout=0.5)


    def update_status(self, message, is_error=False, tab='live'):
        label = self.live_status if tab == 'live' else self.file_status
        label.configure(text=message, text_color="red" if is_error else "white")
        self.root.update()

    def start_listening(self):
        self.is_listening = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self.listen_loop, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.update_status("Ready")

    def listen_loop(self):
        with sr.Microphone() as source:
            self.update_status("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_listening:
                try:
                    self.update_status("Listening...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=None)
                    self.update_status("Processing...")
                    
                    text = self.recognizer.recognize_google(audio)
                    self.live_text.insert("end", text + "\n")
                    self.live_text.see("end")
                    self.update_status("Ready")
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.update_status("Could not understand audio", True)
                except sr.RequestError as e:
                    self.update_status(f"Error: {str(e)}", True)

    def translate_live_text(self):
        self.translate_text(self.live_text, self.live_translated, self.live_lang_var.get())

    def translate_file_text(self):
        self.translate_text(self.file_text, self.file_translated, self.file_lang_var.get())

    def translate_text(self, source_widget, target_widget, target_language):
        text = source_widget.get("1.0", "end").strip()
        if text:
            try:
                target_lang = self.languages[target_language]
                translated = self.translator.translate(text, dest=target_lang)
                target_widget.delete("1.0", "end")
                target_widget.insert("1.0", translated.text)
            except Exception as e:
                CTkMessagebox(title="Error", message=str(e), icon="cancel")
    
    def process_audio_file(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=self.supported_formats)
        if file_path:
            self.update_status("Processing audio file...", tab='file')
            try:
                # Handle MP3 files
                if file_path.lower().endswith('.mp3'):
                    try:
                        # Convert MP3 to WAV
                        audio = AudioSegment.from_mp3(file_path)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
                            wav_path = tmp_wav.name
                            audio.export(wav_path, format='wav')
                        use_path = wav_path
                    except Exception as e:
                        CTkMessagebox(
                            title="Error",
                            message=f"Error converting MP3: {str(e)}\nTry installing ffmpeg or use WAV files.",
                            icon="cancel"
                        )
                        return
                else:
                    use_path = file_path
                
                # Process the audio file
                with sr.AudioFile(use_path) as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    
                    self.file_text.delete("1.0", "end")
                    self.file_text.insert("1.0", text)
                    self.update_status("Ready", tab='file')
                
                # Cleanup temporary file if created
                if file_path.lower().endswith('.mp3') and os.path.exists(use_path):
                    os.remove(use_path)
                    
            except Exception as e:
                self.update_status(f"Error: {str(e)}", True, 'file')
                CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def clean_and_process(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=self.supported_formats)
        if file_path:
            self.update_status("Cleaning audio...", tab='file')
            try:
                # Load the audio file
                if file_path.lower().endswith('.mp3'):
                    audio = AudioSegment.from_mp3(file_path)
                else:
                    audio = AudioSegment.from_wav(file_path)
                
                # Normalize audio
                normalized_audio = audio.normalize()
                
                # Export to temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
                    wav_path = tmp_wav.name
                    normalized_audio.export(wav_path, format='wav')
                
                # Process the cleaned audio
                with sr.AudioFile(wav_path) as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    
                    self.file_text.delete("1.0", "end")
                    self.file_text.insert("1.0", text)
                    self.update_status("Ready", tab='file')
                
                # Cleanup
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    
            except Exception as e:
                self.update_status(f"Error: {str(e)}", True, 'file')
                CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def clear_live_text(self):
        self.live_text.delete("1.0", "end")
        self.live_translated.delete("1.0", "end")

    def clear_file_text(self):
        self.file_text.delete("1.0", "end")
        self.file_translated.delete("1.0", "end")

    def clear_analysis(self):
        self.analysis_input.delete("1.0", "end")
        self.keywords_text.delete("1.0", "end")
        self.summary_text.delete("1.0", "end")

    def import_text(self, source):
        if source == 'live':
            text = self.live_text.get("1.0", "end").strip()
        else:
            text = self.file_text.get("1.0", "end").strip()
            
        self.analysis_input.delete("1.0", "end")
        self.analysis_input.insert("1.0", text)

    def extract_keywords(self):
        try:
            text = self.analysis_input.get("1.0", "end").strip()
            if not text:
                raise ValueError("Please enter some text to analyze")
                
            num_keywords = int(self.keyword_num.get())
            
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in stop_words]
            
            # Calculate word frequencies
            freq_dist = FreqDist(words)
            keywords = nlargest(num_keywords, freq_dist, key=freq_dist.get)
            
            # Display with frequencies
            result = "\n".join(f"{word} " for word in keywords)
            self.keywords_text.delete("1.0", "end")
            self.keywords_text.insert("1.0", result)
            
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def generate_summary(self):
        try:
            text = self.analysis_input.get("1.0", "end").strip()
            if not text:
                raise ValueError("Please enter some text to analyze")
                
            num_sentences = int(self.summary_num.get())
            summary = self.summarizer.get_summary(text, num_sentences)
            
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", summary)
            
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def save_file(self, text, file_path, file_type):
        try:
            if file_type == 'txt':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            elif file_type == 'docx':
                doc = Document()
                doc.add_paragraph(text)
                doc.save(file_path)
                
            elif file_type == 'pdf':
                doc = SimpleDocTemplate(file_path, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                story.append(Paragraph(text, styles['Normal']))
                doc.build(story)
                
            elif file_type == 'md':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Generated Document\n\n{text}")
                    
        except Exception as e:
            raise Exception(f"Error saving file: {str(e)}")

    def save_with_format(self, mode='live'):
        file_types = [
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
            ("Word documents", "*.docx"),
            ("Markdown files", "*.md"),
            ("All files", "*.*")
        ]
        
        try:
            file_path = ctk.filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=file_types,
                initialfile=f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if file_path:
                ext = os.path.splitext(file_path)[1][1:]
                
                if mode == 'live':
                    text = self.live_text.get("1.0", "end").strip()
                    translated = self.live_translated.get("1.0", "end").strip()
                else:
                    text = self.file_text.get("1.0", "end").strip()
                    translated = self.file_translated.get("1.0", "end").strip()
                
                full_text = f"Original Text:\n\n{text}\n\n"
                if translated:
                    full_text += f"Translated Text:\n\n{translated}"
                
                self.save_file(full_text, file_path, ext)
                CTkMessagebox(title="Success", message=f"File saved successfully as {ext.upper()}!", icon="check")
                
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def save_analysis(self):
        try:
            file_path = ctk.filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("Text files", "*.txt"),
                    ("PDF files", "*.pdf"),
                    ("Word documents", "*.docx"),
                    ("Markdown files", "*.md"),
                    ("All files", "*.*")
                ],
                initialfile=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if file_path:
                ext = os.path.splitext(file_path)[1][1:]
                text = f"""Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Original Text:
{self.analysis_input.get("1.0", "end").strip()}

Keywords:
{self.keywords_text.get("1.0", "end").strip()}

Summary:
{self.summary_text.get("1.0", "end").strip()}
"""
                
                self.save_file(text, file_path, ext)
                CTkMessagebox(title="Success", message=f"Analysis saved successfully as {ext.upper()}!", icon="check")
                
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AutomaticNotesMaker()
    app.run()