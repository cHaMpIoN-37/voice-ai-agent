# Voice-Controlled Local AI Agent

**Mem0 AI & Internshala Internship Assignment**

A sophisticated, production-ready voice AI agent that runs entirely on your local machine. No cloud dependencies, no API keys, no data leaving your computer.

##  Features

### Core Functionality
-  **Audio Input**: Real-time microphone recording + file upload (.wav, .mp3)
-  **Speech-to-Text**: Fast, accurate local transcription using **faster-whisper** (small model, 8-bit quantized)
-  **Intent Classification**: Local LLM-powered intent detection using **Ollama phi3:mini**
-  **Multi-Intent Support**:
  -  Create empty files
  -  Write code to files
  -  Summarize text
  -  General conversation
-  **Safety Guarantees**: All file operations strictly confined to `./output/` folder
-  **Clean UI**: Professional Streamlit interface with real-time feedback

### Bonus Features Implemented 
1. ** Graceful Error Handling**: Comprehensive try-catch blocks, input validation, sanitization
2. ** Input Validation**: Filename sanitization, path traversal prevention, content length limits

##  Quick Start

### Prerequisites
- Python 3.8+
- Ubuntu 22.04 LTS (or similar Linux/macOS/Windows)
- Ollama installed and running (phi3:mini model)
- ffmpeg (for audio processing)

### Installation

1. **Clone the repository** (or extract the project folder)
   ```bash
   cd ~/Mem0\ AI\ _\ Internshala/assignment/voice-ai-agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running**
   ```bash
   # In a separate terminal, start Ollama server
   ollama serve
   ```

5. **Pull the phi3:mini model (first time only)**
   ```bash
   ollama pull phi3:mini
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open in browser**
   ```
   http://localhost:8501
   ```

##  Architecture

```
Voice Input (Mic/Upload)
        ↓
    Audio File (WAV)
        ↓
    [Validation Layer]
    - File exists?
    - Non-empty?
    - Correct format?
        ↓
    Faster-Whisper STT
    (Local, CPU-optimized)
        ↓
    Transcribed Text
        ↓
    Ollama Intent Classifier
    (phi3:mini, Local)
        ↓
    Intent Data (JSON)
    ├─ intent (create_file|write_code|summarize|general_chat)
    ├─ filename (if applicable)
    ├─ content (if applicable)
    └─ response (LLM output)
        ↓
    [User Confirmation]
    (For file operations)
        ↓
    Action Executor
    (Strictly within ./output/)
        ↓
    Result → UI Display
```

### Components

| Component | Technology | Mode | Purpose |
|-----------|-----------|------|---------|
| **UI** | Streamlit | Web | Real-time user interface |
| **Audio Input** | sounddevice + scipy | Local | Microphone recording |
| **Audio Processing** | pydub + ffmpeg | Local | Format conversion |
| **Speech-to-Text** | faster-whisper | CPU | Audio transcription |
| **Intent Detection** | Ollama + phi3:mini | CPU | LLM-based classification |
| **File Operations** | Python pathlib | Local | Safe file management |

##  Why CPU Mode for Ollama?

The phi3:mini model runs on **CPU mode** due to a compatibility issue with Ollama's GPU runner on certain systems.

### Issue Encountered
```
ERROR: llama runner process has terminated unexpectedly
```

### Solution Applied
```python
options={"num_gpu_layers": 0}  # Force CPU computation
```

### Why This Works
- **phi3:mini** is a tiny 3.8B parameter model
- **Int8 quantization** reduces memory footprint significantly
- CPU execution is **deterministic and reliable**
- Trade-off: ~0.5-1 second slower per inference (acceptable for voice agent)

### Performance (Measured on Intel i7)
- Transcription (5s audio): ~2-3 seconds
- Intent detection: ~1-2 seconds
- Total pipeline: ~3-5 seconds

##  Security Features

1. **Path Sanitization**
   - Filenames validated against regex: `^[a-zA-Z0-9._\-]+$`
   - No directory traversal (`..`, `/` prefixes blocked)
   - All operations verified to be within `./output/`

2. **Input Validation**
   - Max filename length: 100 characters
   - Max content length: 50,000 characters
   - Audio file size checks

3. **Error Handling**
   - No broad exception catching; specific error types handled
   - Detailed logging for audit trails
   - Graceful fallbacks for all failure modes

4. **Human Oversight**
   - Confirmation dialog before file create/write operations
   - Full JSON intent display for transparency
   - File size display in output folder

##  Supported Intents

### 1. Create File
```
"Create a file called hello.py"
→ Intent: create_file
→ Filename: hello.py
→ Action: Touch empty file in output/
```

### 2. Write Code
```
"Write a Python script that prints hello world to test.py"
→ Intent: write_code
→ Filename: test.py
→ Content: [Generated Python code]
→ Action: Write code to output/test.py
```

### 3. Summarize
```
"Summarize the key points about machine learning"
→ Intent: summarize
→ Content: [LLM-generated summary]
→ Action: Display summary (no file creation)
```

### 4. General Chat
```
"Hello, how are you?"
→ Intent: general_chat
→ Response: [Conversational reply]
→ Action: Display response
```

##  Testing the Application

### Test Case 1: Create File
```
1. Click "Start Recording" or upload audio file
2. Say: "Create a file named test.txt"
3. Confirm the action
4. Check output/ folder for test.txt
```

### Test Case 2: Write Code
```
1. Record/upload audio
2. Say: "Write a Python function to add two numbers to math_func.py"
3. Confirm the action
4. Verify code in output/math_func.py
```

### Test Case 3: Summarize
```
1. Record/upload audio
2. Say: "Summarize the benefits of machine learning"
3. No confirmation needed (read-only operation)
4. View summary in output panel
```

### Test Case 4: General Chat
```
1. Record/upload audio
2. Say: "Hello, what can you do?"
3. View conversational response
```

##  Example Output

```
 Code written successfully to `fibonacci.py` (245 bytes)

File path: /home/hannan/Mem0 AI _ Internshala/assignment/voice-ai-agent/output/fibonacci.py

 Output Folder Contents
Found 1 file(s) in `output/` folder
1. `fibonacci.py` (245 bytes)
```

##  Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ollama'"
**Solution:**
```bash
pip install ollama
```

### Issue: "Ollama connection refused"
**Solution:** Ensure Ollama is running in a separate terminal:
```bash
ollama serve
```

### Issue: "No module named 'faster_whisper'"
**Solution:**
```bash
pip install faster-whisper
```

### Issue: "ffmpeg not found"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### Issue: Recording produces empty/silent output
**Solution:**
- Check microphone permissions
- Verify `sounddevice` can access audio: `python -m sounddevice`
- Increase recording duration in code

### Issue: Transcription returns empty text
**Solution:**
- Ensure audio is clear (minimal background noise)
- Speak clearly and at normal pace
- Try uploading a .wav file of better quality

##  Performance Optimization

For improved speed on slower machines:
1. Use `device="cuda"` in `load_whisper_model()` (requires NVIDIA GPU + CUDA)
2. Switch to `faster-whisper` tiny model: `WhisperModel("tiny")`
3. Increase `num_gpu_layers` in Ollama options (if using GPU)

##  Learning Outcomes

This project demonstrates:
-  Local LLM inference with Ollama
-  Speech recognition with faster-whisper
-  Intent classification and NLP
-  Real-time streaming UI with Streamlit
-  Security best practices (path sanitization, input validation)
-  Error handling and logging
-  State management in web applications
-  Hardware-aware optimization (CPU mode for compatibility)

##  Project Structure

```
voice-ai-agent/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── output/                # Safe directory for file operations
    ├── test.txt
    ├── script.py
    └── ...
```

##  Future Enhancements

- [ ] Support for compound commands ("Create a file AND write code to it")
- [ ] Multi-language support (Spanish, French, etc.)
- [ ] Voice output (text-to-speech feedback)
- [ ] Command history and favorites
- [ ] Advanced intent extraction with few-shot examples
- [ ] Integration with local vector database for memory/context
- [ ] Web API for remote access (with authentication)
- [ ] Docker containerization for easy deployment

##  License

This project is part of the Mem0 AI &  Internship Assigment.


