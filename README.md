# RTT
Real-Time Translator
GUI Real-time speech-to-text + translation with:
- Selectable input/output audio devices.
- Optional capture-from-output (loopback) mode.
- Selectable source & target languages.
- Switchable LLM backend: Local (Ollama-style) or Online (OpenAI-compatible).
- Session summary with key points.

On Windows, "Capture from output (loopback)" will try to use the
selected output device (headphones / speakers) as an input source
via WASAPI loopback, so you can transcribe/translate YouTube audio
even when using a headset.

You can download vosk models from:
    https://alphacephei.com/vosk/models

Run:
    python realtime_translator_gui.py
    
Credit: Mohammad Reza Gerami - mr.gerami@gmail.com
