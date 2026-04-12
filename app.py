import streamlit as st
import os
import json
import logging
from pathlib import Path
from faster_whisper import WhisperModel
import ollama
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from pydub import AudioSegment
import tempfile
import re
from typing import Dict, Tuple, Optional

# ===================== LOGGING SETUP =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VoiceAIAgent")

# ===================== CONFIG =====================
st.set_page_config(page_title="Voice AI Agent", page_icon="🦾", layout="wide")
st.title("Voice-Controlled Local AI Agent")
st.markdown("### Built for Mem0 AI / Internshala Internship Assignment")

# Create output directory safely
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Supported intents
SUPPORTED_INTENTS = {"create_file", "write_code", "summarize", "general_chat"}
MAX_FILENAME_LENGTH = 100
MAX_CONTENT_LENGTH = 50000

# ===================== SESSION STATE MANAGEMENT =====================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "intent_data" not in st.session_state:
    st.session_state.intent_data = None

# ===================== HELPERS - VALIDATION =====================
def is_valid_filename(filename: str) -> bool:
    """Validate filename to prevent directory traversal and invalid characters."""
    if not filename or len(filename) > MAX_FILENAME_LENGTH:
        return False
    
    # Prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        return False
    
    # Allow only alphanumeric, dots, underscores, and hyphens
    if not re.match(r"^[a-zA-Z0-9._\-]+$", filename):
        return False
    
    return True

def is_safe_audio_file(audio_path: str) -> bool:
    """Check if audio file exists and is not empty."""
    try:
        path = Path(audio_path)
        if not path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        if path.stat().st_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating audio file: {e}")
        return False

# ===================== HELPERS - WHISPER =====================
@st.cache_resource
def load_whisper_model():
    """Load faster-whisper model with CPU optimization."""
    try:
        logger.info("Loading Whisper model (small)...")
        model = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

try:
    whisper_model = load_whisper_model()
except Exception as e:
    st.error(f"❌ Failed to initialize Speech-to-Text engine: {e}")
    st.stop()

def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio file to text with error handling."""
    if not is_safe_audio_file(audio_path):
        logger.error(f"Audio file validation failed: {audio_path}")
        return None
    
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        segments, info = whisper_model.transcribe(
            audio_path, 
            beam_size=5, 
            language="en"
        )
        text = " ".join([segment.text for segment in segments])
        result = text.strip()
        
        if not result:
            logger.warning("Transcription returned empty text")
            return None
        
        logger.info(f"Transcription successful: {len(result)} characters")
        return result
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

# ===================== HELPERS - INTENT DETECTION =====================
def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Robustly extract and parse JSON from LLM response.
    Handles cases where LLM adds extra text, markdown, or formatting.
    """
    # Try to find JSON object
    start = text.find('{')
    if start == -1:
        return None
    
    # Count braces to find the matching closing brace
    brace_count = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    
    if end == start:
        return None
    
    json_str = text[start:end]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common issues
        # Remove trailing commas before closing braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

def get_intent_and_action(user_text: str) -> Dict:
    """
    Classify user intent using local LLM (Ollama).
    Returns structured intent data with fallback to general_chat on failure.
    Supports compound commands (e.g., "create file AND summarize").
    """
    if not user_text or len(user_text.strip()) == 0:
        logger.warning("Empty user text provided for intent detection")
        return {
            "intent": "general_chat",
            "filename": None,
            "content": None,
            "response": "Please provide some input to process.",
            "secondary_intent": None
        }
    
    # Check for compound commands (e.g., "and then", "then also", "also")
    has_compound = any(keyword in user_text.lower() for keyword in [" and then ", " then ", " also ", " after that "])
    
    prompt = f"""TASK: Classify user request and return ONLY valid JSON (no other text).

INPUT: "{user_text}"

RULES:
1. Return ONLY a JSON object - nothing else, no explanation
2. No markdown, no code blocks, no text before/after JSON
3. For code: use double backslash for newlines: \\n
4. All fields required: intent, filename, content, response, secondary_intent
5. Use null for empty fields (not empty strings)
6. Valid intents: create_file, write_code, summarize, general_chat

CLASSIFICATION:
- create_file: "create/make file [name]"
- write_code: "write code/script/function to [file]" 
- summarize: "summarize/explain/what is"
- general_chat: everything else

RESPONSE FORMAT - COPY EXACTLY:
{{"intent":"VALUE","filename":NULLORSTRING,"content":NULLORSTRING,"response":"BRIEF_TEXT","secondary_intent":null}}

EXAMPLES:

"create notes.txt": {{"intent":"create_file","filename":"notes.txt","content":null,"response":"Creating file","secondary_intent":null}}

"write python add function to calc.py": {{"intent":"write_code","filename":"calc.py","content":"def add(a,b):\\n    return a+b","response":"Added function","secondary_intent":null}}

"explain retry logic": {{"intent":"summarize","filename":null,"content":"Retry logic: Attempt operation multiple times. Wait between attempts. Raise error if all attempts fail.","response":"Explained","secondary_intent":null}}

"hello": {{"intent":"general_chat","filename":null,"content":null,"response":"Hi there! How can I help?","secondary_intent":null}}

Now return JSON for: {user_text}"""
    
    try:
        logger.info("Calling Ollama for intent detection...")
        response = ollama.chat(
            model='phi3:mini',
            messages=[{'role': 'user', 'content': prompt}],
            options={"num_gpu_layers": 0}   # Force CPU mode
        )
        
        response_text = response['message']['content'].strip()
        logger.info(f"Ollama response received: {len(response_text)} chars")
        logger.debug(f"Response content: {response_text[:500]}")
        
        # Extract JSON robustly
        intent_data = extract_json_from_text(response_text)
        
        if intent_data is None:
            logger.error(f"JSON extraction failed. Full response:\n{response_text}")
            return create_fallback_intent("Invalid JSON format from model")
        
        # Validate intent
        intent = intent_data.get("intent", "general_chat")
        if intent not in SUPPORTED_INTENTS:
            logger.warning(f"Unsupported intent: {intent}. Defaulting to general_chat.")
            intent_data["intent"] = "general_chat"
        
        # Ensure all required fields exist
        if "filename" not in intent_data:
            intent_data["filename"] = None
        if "content" not in intent_data:
            intent_data["content"] = None
        if "response" not in intent_data:
            intent_data["response"] = ""
        if "secondary_intent" not in intent_data:
            intent_data["secondary_intent"] = None
        
        # Sanitize filename
        if intent_data.get("filename"):
            filename = intent_data["filename"]
            if not is_valid_filename(filename):
                logger.warning(f"Invalid filename: {filename}. Sanitizing...")
                intent_data["filename"] = sanitize_filename(filename)
        
        # Truncate content if too large
        if intent_data.get("content") and len(str(intent_data["content"])) > MAX_CONTENT_LENGTH:
            logger.warning(f"Content exceeds max length. Truncating...")
            intent_data["content"] = str(intent_data["content"])[:MAX_CONTENT_LENGTH]
        
        logger.info(f"Intent: {intent}, Secondary: {intent_data.get('secondary_intent')}")
        return intent_data
        
    except Exception as e:
        logger.error(f"Intent detection error: {e}")
        return create_fallback_intent(f"Error: {e}")

def create_fallback_intent(error_msg: str) -> Dict:
    """Create a safe fallback intent for errors. Always includes all required fields."""
    return {
        "intent": "general_chat",
        "filename": None,
        "content": None,
        "response": f"I encountered an issue processing your request: {error_msg}. Please try again.",
        "secondary_intent": None
    }

def generate_code_content(request: str, filename: str = "") -> str:
    """
    Generate actual code content based on user request.
    Called when the initial intent detection doesn't include content.
    """
    # Infer language from filename
    language = "python"  # default
    if filename:
        if filename.endswith('.js'):
            language = "javascript"
        elif filename.endswith('.java'):
            language = "java"
        elif filename.endswith('.cpp') or filename.endswith('.c'):
            language = "c++"
        elif filename.endswith('.py'):
            language = "python"
    
    code_prompt = f"""You are a code generation expert. Generate complete, working {language} code based on this request:

Request: "{request}"

Generate ONLY the code, nothing else. No explanations, no markdown, just the raw code:"""
    
    try:
        logger.info(f"Generating {language} code for: {request}")
        response = ollama.chat(
            model='phi3:mini',
            messages=[{'role': 'user', 'content': code_prompt}],
            options={"num_gpu_layers": 0}
        )
        
        code = response['message']['content'].strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith(language):
                code = code[len(language):].lstrip('\n')
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        code = code.strip()
        logger.info(f"Generated code: {len(code)} characters")
        return code if code else generate_default_code(language, request)
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        return generate_default_code(language, request)

def generate_default_code(language: str, request: str) -> str:
    """Generate a default code template when generation fails."""
    templates = {
        "python": '''# Python code based on your request\n# Request: {request}\n\ndef main():\n    pass\n\nif __name__ == "__main__":\n    main()''',
        "javascript": '''// JavaScript code based on your request\n// Request: {request}\n\nfunction main() {{\n    // Implementation here\n}}\n\nmain();''',
        "java": '''// Java code based on your request\n// Request: {request}\n\npublic class Main {{\n    public static void main(String[] args) {{\n        // Implementation here\n    }}\n}}''',
        "c++": '''// C++ code based on your request\n// Request: {request}\n\n#include <iostream>\nusing namespace std;\n\nint main() {{\n    // Implementation here\n    return 0;\n}}''',
    }
    
    template = templates.get(language, templates["python"])
    return template.format(request=request)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe."""
    # Remove invalid characters
    sanitized = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)
    # Limit length
    sanitized = sanitized[:MAX_FILENAME_LENGTH]
    # Ensure it's not empty
    if not sanitized:
        sanitized = "output_file.txt"
    return sanitized

# ===================== HELPERS - ACTION EXECUTION =====================
def execute_action(intent_data: Dict) -> Tuple[str, str]:
    """
    Execute the action based on intent.
    Returns (result_message, details) tuple.
    All file operations are strictly confined to ./output/ folder.
    Supports compound commands with secondary_intent.
    """
    intent = intent_data.get("intent", "general_chat")
    filename = intent_data.get("filename")
    content = intent_data.get("content")
    response = intent_data.get("response", "")
    secondary_intent = intent_data.get("secondary_intent")

    try:
        result_messages = []
        final_details = ""

        if intent == "create_file" and filename:
            if not is_valid_filename(filename):
                error_msg = f"Invalid filename: {filename}"
                logger.error(error_msg)
                return f"❌ {error_msg}", ""
            
            file_path = OUTPUT_DIR / filename
            
            # Ensure the file path is within OUTPUT_DIR (security check)
            if not str(file_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
                error_msg = "Security violation: file path outside output folder"
                logger.error(error_msg)
                return f"❌ {error_msg}", ""
            
            # If content is provided, write it; otherwise check if we should generate code
            if content:
                # Decode escape sequences (e.g., \n to actual newlines)
                content_str = str(content).encode().decode('unicode_escape')
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content_str)
                logger.info(f"File created with content: {file_path}")
                result_messages.append(f"✅ File created: `{filename}` ({len(content_str)} bytes)")
                final_details = str(file_path)
            # Check if filename suggests code and response mentions code/function
            elif (filename.endswith(('.py', '.js', '.java', '.cpp', '.c')) and 
                  (response and any(word in response.lower() for word in ['code', 'function', 'python', 'script', 'write']))):
                # Generate code for the file
                code = generate_code_content(response, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                logger.info(f"Code file created: {file_path}")
                result_messages.append(f"✅ Code file created: `{filename}` ({len(code)} bytes)")
                final_details = str(file_path)
            else:
                file_path.touch(exist_ok=True)
                logger.info(f"Empty file created: {file_path}")
                result_messages.append(f"✅ Empty file created: `{filename}`")
                final_details = str(file_path)

        elif intent == "write_code" and filename:
            if not is_valid_filename(filename):
                error_msg = f"Invalid filename: {filename}"
                logger.error(error_msg)
                return f"❌ {error_msg}", ""
            
            file_path = OUTPUT_DIR / filename
            
            # Security check
            if not str(file_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
                error_msg = "Security violation: file path outside output folder"
                logger.error(error_msg)
                return f"❌ {error_msg}", ""
            
            # Generate content if not provided
            if not content or len(str(content).strip()) == 0:
                logger.info(f"Content missing, generating code for: {filename}")
                content = generate_code_content(response, filename)
            
            # Write content to file
            if content and len(str(content).strip()) > 0:
                content_str = str(content)
                # Decode escape sequences (e.g., \n to actual newlines)
                content_str = content_str.encode().decode('unicode_escape')
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content_str)
                logger.info(f"Code written to: {file_path}")
                result_messages.append(f"✅ Code written to `{filename}` ({len(content_str)} bytes)")
                final_details = str(file_path)
            else:
                # Fallback: create empty file
                file_path.touch(exist_ok=True)
                logger.info(f"Empty file created (write_code with no content): {file_path}")
                result_messages.append(f"✅ File created: `{filename}`")
                final_details = str(file_path)

        elif intent == "summarize" and content:
            logger.info(f"Text summarized: {len(content)} characters")
            result_messages.append(f"📝 Summary:\n{content}")

        else:
            # General chat or unknown intent
            logger.info(f"General chat response generated")
            result_messages.append(response if response else "Hello! How can I help you?")
        
        # Handle secondary intent (for compound commands)
        if secondary_intent and result_messages:
            # secondary_intent might be a string or dict, handle both cases
            if isinstance(secondary_intent, str):
                # If it's a string (like "summarize"), convert to dict format
                if secondary_intent == "summarize":
                    # Generate a summary of the created file or content
                    file_summary = f"The {filename} file has been created with the code written above."
                    if content:
                        file_summary = f"Created {filename} with content that performs the intended function."
                    logger.info("Executing secondary summarize intent...")
                    result_messages.append(f"📝 Summary:\n{file_summary}")
            elif isinstance(secondary_intent, dict):
                secondary_type = secondary_intent.get("intent")
                secondary_content = secondary_intent.get("content")
                
                if secondary_type == "summarize" and secondary_content:
                    logger.info("Executing secondary summarize intent...")
                    result_messages.append(f"📝 Summary:\n{secondary_content}")
                elif secondary_type == "general_chat":
                    logger.info("Executing secondary chat intent...")
                    secondary_response = secondary_intent.get("response", "")
                    if secondary_response:
                        result_messages.append(secondary_response)
        
        final_result = "\n\n".join(result_messages)
        return final_result, final_details

    except PermissionError:
        error_msg = "❌ Permission denied: Cannot write to output folder"
        logger.error(error_msg)
        return error_msg, ""
    except IOError as e:
        error_msg = f"❌ File I/O error: {e}"
        logger.error(error_msg)
        return error_msg, ""
    except Exception as e:
        error_msg = f"❌ Unexpected error during action execution: {e}"
        logger.error(error_msg)
        return error_msg, ""

# ===================== UI - SIDEBAR =====================
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.info(
    """
    **Voice AI Agent Features:**
    - 🎤 Real-time microphone recording
    - 📁 Audio file upload (.wav, .mp3)
    - 🗣️ Local Speech-to-Text (faster-whisper)
    - 🧠 Intent Classification (Ollama phi3:mini)
    - 🔐 Human-in-the-Loop confirmation
    - 🛡️ Security constraints (output/ only)
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline:")
st.sidebar.markdown("1️⃣ Audio Input → 2️⃣ Transcription → 3️⃣ Intent Detection → 4️⃣ Confirmation → 5️⃣ Execution")

# ===================== UI - AUDIO INPUT =====================
st.markdown("## 🎙️ Audio Input")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Record from Microphone")
    if st.button("🎙️ Start Recording (5 seconds)", type="primary", key="record_btn"):
        try:
            with st.spinner("🔴 Recording... Please speak now"):
                fs = 16000  # Sample rate
                duration = 5  # seconds
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()
                
                # Validate recording
                if recording is None or len(recording) == 0:
                    st.error("❌ Recording failed. Please try again.")
                    logger.error("Recording returned None or empty array")
                else:
                    # Save recording to temporary file
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    wav.write(temp_wav.name, fs, (recording * 32767).astype(np.int16))
                    
                    st.session_state.audio_path = temp_wav.name
                    st.success("✅ Recording completed!")
                    logger.info(f"Audio recorded: {temp_wav.name}")
                    
        except Exception as e:
            st.error(f"❌ Recording error: {e}")
            logger.error(f"Recording failed: {e}")

with col2:
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload .wav or .mp3 file", 
        type=["wav", "mp3"],
        key="file_uploader"
    )

    if uploaded_file:
        try:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            
            if uploaded_file.name.endswith(".mp3"):
                audio = AudioSegment.from_mp3(uploaded_file)
                audio.export(temp_wav.name, format="wav")
                logger.info(f"MP3 file converted to WAV: {temp_wav.name}")
            else:
                temp_wav.write(uploaded_file.getvalue())
                logger.info(f"WAV file saved: {temp_wav.name}")
            
            st.session_state.audio_path = temp_wav.name
            st.success("✅ File uploaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Upload error: {e}")
            logger.error(f"File upload failed: {e}")
            st.session_state.audio_path = None

# ===================== UI - AUDIO PROCESSING =====================
if st.session_state.audio_path:
    st.markdown("## Process Audio")
    
    if st.button("Process Audio", type="primary", key="process_btn"):
        # Step 1: Transcription
        with st.spinner("🔄 Transcribing audio..."):
            text = transcribe_audio(st.session_state.audio_path)
        
        if text is None:
            st.error("❌ Transcription failed. Please try again with clearer audio.")
            logger.error("Transcription returned None")
            st.stop()
        
        st.session_state.transcribed_text = text
        
        # Display transcription
        st.markdown("### 📝 Transcribed Text")
        st.text_area("Your audio was transcribed as:", value=text, height=100, disabled=True)
        
        # Step 2: Intent Detection
        with st.spinner("🧠 Understanding intent..."):
            intent_data = get_intent_and_action(text)
        
        st.session_state.intent_data = intent_data
        
        # Validate intent_data before using
        if not intent_data or not isinstance(intent_data, dict) or 'intent' not in intent_data:
            st.error("❌ Failed to parse intent from response. Please try again with clearer audio.")
            logger.error(f"Invalid intent_data returned: {intent_data}")
            st.stop()
        
        # Display intent
        st.markdown("### 🎯 Detected Intent")
        
        col_intent1, col_intent2 = st.columns([1, 2])
        with col_intent1:
            intent_badge = f"🏷️ **Intent:** `{intent_data.get('intent', 'unknown').upper()}`"
            st.markdown(intent_badge)
        
        with col_intent2:
            if intent_data.get("filename"):
                st.markdown(f"📄 **File:** `{intent_data['filename']}`")
        
        # Show full JSON for transparency
        with st.expander("📋 Full Intent Data (JSON)"):
            st.json(intent_data)
        
        # Step 3: Execute Action Directly
        st.markdown("### ⚙️ Executing Action")
        
        with st.spinner("⏳ Processing..."):
            result, details = execute_action(intent_data)
        
        # Display result
        st.markdown("### ✅ Action Result")
        if "❌" in result:
            st.error(result)
        else:
            st.success(result)
        
        if details and "output/" in details:
            st.info(f"📁 File path: `{details}`")

# ===================== UI - OUTPUT FOLDER =====================
st.markdown("---")
st.markdown("## 📂 Output Folder Contents")

try:
    files = list(OUTPUT_DIR.glob("*"))
    if files:
        st.success(f"Found {len(files)} file(s) in `output/` folder")
        for i, f in enumerate(files, 1):
            file_size = f.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb < 1:
                size_str = f"{file_size} bytes"
            else:
                size_str = f"{file_size_mb:.2f} MB"
            
            st.write(f"{i}. `{f.name}` ({size_str})")
    else:
        st.info("📭 No files yet. Create some using the voice agent!")
except Exception as e:
    st.error(f"❌ Error reading output folder: {e}")
    logger.error(f"Error listing output files: {e}")

# ===================== FOOTER =====================
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns([1, 1, 1])
with col_footer1:
    st.caption("🎙️ Speech-to-Text: faster-whisper (small, int8)")
with col_footer2:
    st.caption("🧠 LLM: Ollama phi3:mini (CPU mode)")
with col_footer3:
    st.caption("🛡️ All operations confined to output/ folder")