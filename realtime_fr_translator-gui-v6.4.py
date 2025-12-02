"""
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
"""

import json
import queue
import threading
import sys
import platform
from typing import Optional, Dict, List

import numpy as np
import sounddevice as sd
import requests
from vosk import Model, KaldiRecognizer

import tkinter as tk
from tkinter import ttk, messagebox


# --------------- CONFIG --------------- #

# Map each source language to its VOSK model path.
# Change these paths to match your environment.
SOURCE_LANG_MODELS: Dict[str, str] = {
    "French":  r"D:\Projects\RTT\models\vosk-fr",
    "English": r"D:\Projects\RTT\models\vosk-en",
    "Persian": r"D:\Projects\RTT\models\vosk-fa",  # if you have a FA model
}

SUPPORTED_SOURCE_LANGS = ["French", "English", "Persian"]
SUPPORTED_TARGET_LANGS = ["Persian", "English", "French"]

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.25  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Default local LLM settings (Ollama-style)
DEFAULT_LOCAL_URL = "http://localhost:11434/api/generate"
DEFAULT_LOCAL_MODEL = "gemma2:2b"

# Default online LLM settings (OpenAI-compatible)
DEFAULT_ONLINE_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ONLINE_MODEL = "gpt-4o-mini"


# --------------- AUDIO DEVICE HELPERS --------------- #

def list_input_devices() -> List[str]:
    """
    Return a list of display strings for input devices like:
        "0: Microphone (Realtek...)"
    Only include devices that support input (max_input_channels > 0).
    """
    devices = []
    try:
        devs = sd.query_devices()
    except Exception as e:
        print(f"[Audio] Could not query devices: {e}", file=sys.stderr)
        return devices

    for idx, dev in enumerate(devs):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", f"Device {idx}")
            devices.append(f"{idx}: {name}")
    return devices


def list_output_devices() -> List[str]:
    """
    Return a list of display strings for output devices like:
        "3: Speakers (Realtek...)"
    Only include devices that support output (max_output_channels > 0).
    """
    devices = []
    try:
        devs = sd.query_devices()
    except Exception as e:
        print(f"[Audio] Could not query devices: {e}", file=sys.stderr)
        return devices

    for idx, dev in enumerate(devs):
        if dev.get("max_output_channels", 0) > 0:
            name = dev.get("name", f"Device {idx}")
            devices.append(f"{idx}: {name}")
    return devices


def parse_device_index(display: str) -> Optional[int]:
    """
    Parse an integer device index from a display string like "3: Device name".
    Return None if parsing fails (which will use the system default device).
    """
    if not display:
        return None
    try:
        idx_str = display.split(":", 1)[0].strip()
        return int(idx_str)
    except Exception:
        return None


# --------------- LLM HELPERS --------------- #

def call_local_llm(prompt: str, model: str, url: str, timeout: float = 40.0) -> str:
    """
    Call a local LLM (Ollama-style) with a simple non-streaming API.

    Expected endpoint: POST /api/generate with:
        { "model": "...", "prompt": "...", "stream": false, ... }
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Ollama uses "response"; some other servers might use "text".
        return (data.get("response") or data.get("text") or "").strip()
    except Exception as e:
        print(f"[LLM-local] Error: {e}", file=sys.stderr)
        return ""


def call_online_llm(prompt: str, api_key: str, model: str, base_url: str, timeout: float = 40.0) -> str:
    """
    Call an online LLM with an OpenAI-compatible /v1/chat/completions API.
    """
    if not api_key:
        print("[LLM-online] Missing API key", file=sys.stderr)
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message", {}).get("content", "")
        return (msg or "").strip()
    except Exception as e:
        print(f"[LLM-online] Error: {e}", file=sys.stderr)
        return ""


def call_llm_with_settings(prompt: str, settings: Dict[str, str], timeout: float = 40.0) -> str:
    """
    Decide between local and online LLM based on settings["mode"].
    """
    mode = settings.get("mode", "local")
    if mode == "online":
        return call_online_llm(
            prompt=prompt,
            api_key=settings.get("online_api_key", ""),
            model=settings.get("online_model", DEFAULT_ONLINE_MODEL),
            base_url=settings.get("online_base_url", DEFAULT_ONLINE_BASE_URL),
            timeout=timeout,
        )
    # default: local
    return call_local_llm(
        prompt=prompt,
        model=settings.get("local_model", DEFAULT_LOCAL_MODEL),
        url=settings.get("local_url", DEFAULT_LOCAL_URL),
        timeout=timeout,
    )


def build_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """
    Build a translation prompt for the LLM.
    """
    return (
        f"You are a professional translator.\n"
        f"Translate the following {source_lang} text into {target_lang}.\n\n"
        f"Rules:\n"
        f"- Translate EVERYTHING, do NOT summarise.\n"
        f"- Return ONLY the translation in {target_lang}, without any explanations.\n"
        f"- Use simple, natural, spoken-style language.\n\n"
        f"Text:\n{text.strip()}\n\n"
        f"Translation:"
    )


def build_summary_prompt(full_text: str, source_lang: str, target_lang: str) -> str:
    """
    Build a prompt for summarizing the whole session.
    """
    return (
        f"You are an expert note-taker and meeting assistant.\n"
        f"The following text is a transcript of a session in {source_lang}.\n"
        f"Write a concise and accurate summary in {target_lang}, then list key points.\n\n"
        f"Rules:\n"
        f"- Do NOT invent facts.\n"
        f"- Use simple, natural language.\n"
        f"- First write 1–3 short paragraphs of summary.\n"
        f"- Then write a section titled 'Key Points:' with bullet points.\n\n"
        f"Transcript:\n{full_text.strip()}\n\n"
        f"Summary and key points ({target_lang}):"
    )


# --------------- ASR CLASS --------------- #

class RealtimeVoskASR:
    """
    Minimal real-time ASR using VOSK + sounddevice.

    It supports two modes:
    - Normal input device (microphone, loopback device, etc.)
    - Loopback from a selected output device (Windows WASAPI only)
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int,
        ui_queue: "queue.Queue[dict]",
        translate_queue: "queue.Queue[str]",
        input_device_index: Optional[int] = None,
        loopback_output_index: Optional[int] = None,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.ui_queue = ui_queue
        self.translate_queue = translate_queue

        # If loopback_output_index is not None, we try to capture from output.
        self.input_device_index = input_device_index
        self.loopback_output_index = loopback_output_index

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stop_flag = threading.Event()

        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None

        self.asr_thread: Optional[threading.Thread] = None
        self.stream: Optional[sd.InputStream] = None

    def load_model(self) -> None:
        """
        Load the VOSK model from disk.
        """
        self.ui_queue.put({"type": "status", "text": f"Loading VOSK model from: {self.model_path}"})
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        try:
            self.recognizer.SetWords(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        self.ui_queue.put({"type": "status", "text": "ASR model loaded."})

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """
        Sounddevice callback: push audio blocks into the internal queue.
        """
        if status:
            print(f"[Audio] {status}", file=sys.stderr)
        data = indata.copy()
        if data.ndim > 1:
            data = data.mean(axis=1)
        self.audio_queue.put(data.astype(np.float32))

    def _asr_worker(self) -> None:
        """
        Worker thread that runs VOSK on incoming audio chunks.
        """
        assert self.recognizer is not None
        self.ui_queue.put({"type": "status", "text": "ASR worker started."})

        while not self.stop_flag.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
            pcm_bytes = pcm16.tobytes()

            try:
                accept = self.recognizer.AcceptWaveform(pcm_bytes)
            except Exception as e:
                print(f"[ASR] AcceptWaveform error: {e}", file=sys.stderr)
            else:
                if accept:
                    try:
                        res = json.loads(self.recognizer.Result())
                    except Exception:
                        res = {}
                    text = (res.get("text") or "").strip()
                    if text:
                        self.ui_queue.put({"type": "final", "text": text})
                        self.translate_queue.put(text)
                else:
                    try:
                        pres = json.loads(self.recognizer.PartialResult())
                        partial = (pres.get("partial") or "").strip()
                    except Exception:
                        partial = ""
                    if partial:
                        self.ui_queue.put({"type": "partial", "text": partial})

        self.ui_queue.put({"type": "status", "text": "ASR worker stopped."})

    def _create_input_stream(self) -> sd.InputStream:
        """
        Create an InputStream either from a normal input device
        or as a loopback from an output device (Windows WASAPI).
        """
        extra_settings = None
        device_index: Optional[int] = self.input_device_index

        # If loopback_output_index is set, prefer that.
        if self.loopback_output_index is not None:
            device_index = self.loopback_output_index
            # On Windows, try WASAPI loopback mode.
            if platform.system() == "Windows":
                try:
                    dev_info = sd.query_devices(device_index)
                    hostapi_info = sd.query_hostapis()[dev_info["hostapi"]]
                    if hostapi_info["type"] == "Windows WASAPI":
                        extra_settings = sd.WasapiSettings(loopback=True)
                        self.ui_queue.put({"type": "status", "text": "Using WASAPI loopback from output device."})
                    else:
                        self.ui_queue.put({
                            "type": "status",
                            "text": "Selected output device is not WASAPI; loopback may not work.",
                        })
                except Exception as e:
                    print(f"[Audio] Failed to configure WASAPI loopback: {e}", file=sys.stderr)
                    self.ui_queue.put({
                        "type": "status",
                        "text": "WASAPI loopback configuration failed; falling back to normal input.",
                    })
                    device_index = self.input_device_index
                    extra_settings = None

        # If device_index is None, sounddevice will pick the default input.
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            device=device_index,
            callback=self._audio_callback,
            extra_settings=extra_settings,
        )

    def start(self) -> None:
        """
        Start the ASR worker and microphone/loopback stream.
        """
        self.stop_flag.clear()
        self.load_model()

        self.asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self.asr_thread.start()

        self.stream = self._create_input_stream()
        self.stream.start()
        self.ui_queue.put({"type": "status", "text": "Audio input started."})

    def stop(self) -> None:
        """
        Stop ASR and microphone stream.
        """
        self.stop_flag.set()
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        if self.asr_thread is not None:
            self.asr_thread.join(timeout=2.0)


# --------------- TRANSLATOR WORKER --------------- #

def translator_worker(
    translate_queue: "queue.Queue[str]",
    ui_queue: "queue.Queue[dict]",
    stop_flag: threading.Event,
    source_lang: str,
    target_lang: str,
    llm_settings: Dict[str, str],
) -> None:
    """
    Worker that translates final ASR segments using the configured LLM.
    """
    while not stop_flag.is_set():
        try:
            text = translate_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if text is None:
            break

        text = text.strip()
        if not text:
            continue

        prompt = build_translation_prompt(text, source_lang, target_lang)
        translation = call_llm_with_settings(prompt, llm_settings, timeout=40.0)

        ui_queue.put({
            "type": "translation",
            "source": text,
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })


# --------------- TKINTER GUI --------------- #

class TranslatorGUI(tk.Tk):
    """
    Main GUI window for real-time translator.
    """

    def __init__(self):
        super().__init__()

        self.title("Real-time Speech Translator (Devices + Local/Online LLM)")
        self.geometry("1200x700")

        # Queues
        self.ui_queue: "queue.Queue[dict]" = queue.Queue()
        self.translate_queue: "queue.Queue[str]" = queue.Queue()

        # Threads / state
        self.asr: Optional[RealtimeVoskASR] = None
        self.stop_flag = threading.Event()
        self.translator_thread: Optional[threading.Thread] = None

        self.session_segments: List[str] = []

        # Languages
        self.source_lang_var = tk.StringVar(value="French")
        self.target_lang_var = tk.StringVar(value="Persian")

        # Audio devices
        self.input_devices = list_input_devices()
        self.output_devices = list_output_devices()
        self.input_device_var = tk.StringVar(value=self.input_devices[0] if self.input_devices else "")
        self.output_device_var = tk.StringVar(value=self.output_devices[0] if self.output_devices else "")

        # If True, capture from selected output device via WASAPI loopback
        self.capture_output_var = tk.BooleanVar(value=False)

        # LLM configuration
        self.llm_mode_var = tk.StringVar(value="local")
        self.local_model_var = tk.StringVar(value=DEFAULT_LOCAL_MODEL)
        self.local_url_var = tk.StringVar(value=DEFAULT_LOCAL_URL)
        self.online_api_key_var = tk.StringVar(value="")
        self.online_model_var = tk.StringVar(value=DEFAULT_ONLINE_MODEL)
        self.online_base_url_var = tk.StringVar(value=DEFAULT_ONLINE_BASE_URL)

        self._session_llm_settings: Dict[str, str] = {}

        self._build_ui()
        self.after(50, self._process_events)

    # ---------- UI construction ---------- #

    def _build_ui(self) -> None:
        # Top: language + audio + start/stop + summary
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Source / target
        ttk.Label(top, text="Source language:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.source_lang_var,
            values=SUPPORTED_SOURCE_LANGS,
            state="readonly",
            width=10,
        ).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(top, text="Target language:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.target_lang_var,
            values=SUPPORTED_TARGET_LANGS,
            state="readonly",
            width=10,
        ).grid(row=0, column=3, padx=5, pady=2, sticky="w")

        # Input / output devices
        ttk.Label(top, text="Input device:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.input_device_var,
            values=self.input_devices,
            state="readonly",
            width=40,
        ).grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky="w")

        ttk.Label(top, text="Output device:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.output_device_var,
            values=self.output_devices,
            state="readonly",
            width=40,
        ).grid(row=2, column=1, columnspan=3, padx=5, pady=2, sticky="w")

        # Capture-from-output checkbox
        ttk.Checkbutton(
            top,
            text="Capture from output (loopback)",
            variable=self.capture_output_var,
        ).grid(row=3, column=1, columnspan=3, sticky="w", padx=5, pady=2)

        # Start / stop / summary
        self.btn_start = ttk.Button(top, text="Start", command=self.start_recognition)
        self.btn_start.grid(row=0, column=4, padx=10, pady=2)

        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop_recognition, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=5, padx=5, pady=2)

        self.btn_summary = ttk.Button(top, text="Summarize Session", command=self.summarize_session)
        self.btn_summary.grid(row=1, column=4, columnspan=2, padx=10, pady=2)

        # Status label
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var, foreground="gray").grid(
            row=2, column=4, columnspan=2, sticky="e"
        )

        # LLM settings
        llm_frame = ttk.LabelFrame(self, text="LLM Settings")
        llm_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(llm_frame, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(llm_frame, text="Local", variable=self.llm_mode_var, value="local").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Radiobutton(llm_frame, text="Online (API)", variable=self.llm_mode_var, value="online").grid(
            row=0, column=2, sticky="w"
        )

        ttk.Label(llm_frame, text="Local model:").grid(row=1, column=0, sticky="w")
        tk.Entry(llm_frame, textvariable=self.local_model_var, width=20).grid(
            row=1, column=1, padx=5, pady=2, sticky="w"
        )

        ttk.Label(llm_frame, text="Local URL:").grid(row=1, column=2, sticky="w")
        tk.Entry(llm_frame, textvariable=self.local_url_var, width=40).grid(
            row=1, column=3, padx=5, pady=2, sticky="w"
        )

        ttk.Label(llm_frame, text="API key:").grid(row=2, column=0, sticky="w")
        tk.Entry(llm_frame, textvariable=self.online_api_key_var, width=30, show="*").grid(
            row=2, column=1, padx=5, pady=2, sticky="w"
        )

        ttk.Label(llm_frame, text="Online model:").grid(row=2, column=2, sticky="w")
        tk.Entry(llm_frame, textvariable=self.online_model_var, width=20).grid(
            row=2, column=3, padx=5, pady=2, sticky="w"
        )

        ttk.Label(llm_frame, text="Base URL:").grid(row=3, column=0, sticky="w")
        tk.Entry(llm_frame, textvariable=self.online_base_url_var, width=50).grid(
            row=3, column=1, columnspan=3, padx=5, pady=2, sticky="w"
        )

        # Middle: text areas
        middle = ttk.Frame(self)
        middle.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.Frame(middle)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.lbl_source_title = ttk.Label(left_frame, text="Source")
        self.lbl_source_title.pack(anchor="w")

        self.txt_source = tk.Text(left_frame, wrap=tk.WORD)
        self.txt_source.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_left = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.txt_source.yview)
        scroll_left.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_source.config(yscrollcommand=scroll_left.set)

        right_frame = ttk.Frame(middle)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.lbl_target_title = ttk.Label(right_frame, text="Translation")
        self.lbl_target_title.pack(anchor="w")

        self.txt_translation = tk.Text(right_frame, wrap=tk.WORD)
        self.txt_translation.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- FIX: Configure tags for justification (RTL/LTR) ---
        self.txt_translation.tag_configure("rtl", justify=tk.RIGHT)
        self.txt_translation.tag_configure("ltr", justify=tk.LEFT)
        # -----------------------------------------------------
        
        scroll_right = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.txt_translation.yview)
        scroll_right.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_translation.config(yscrollcommand=scroll_right.set)

        # Bottom: partial line
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        ttk.Label(bottom, text="Live partial:").pack(side=tk.LEFT)
        self.partial_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.partial_var, foreground="blue").pack(side=tk.LEFT, padx=5)

    # ---------- helpers ---------- #

    def _snapshot_llm_settings(self) -> Dict[str, str]:
        mode = self.llm_mode_var.get()
        if mode == "online":
            return {
                "mode": "online",
                "online_api_key": self.online_api_key_var.get().strip(),
                "online_model": self.online_model_var.get().strip() or DEFAULT_ONLINE_MODEL,
                "online_base_url": self.online_base_url_var.get().strip() or DEFAULT_ONLINE_BASE_URL,
            }
        return {
            "mode": "local",
            "local_model": self.local_model_var.get().strip() or DEFAULT_LOCAL_MODEL,
            "local_url": self.local_url_var.get().strip() or DEFAULT_LOCAL_URL,
        }

    def get_source_lang(self) -> str:
        val = self.source_lang_var.get()
        if val not in SUPPORTED_SOURCE_LANGS:
            return "French"
        return val

    def get_target_lang(self) -> str:
        val = self.target_lang_var.get()
        if val not in SUPPORTED_TARGET_LANGS:
            return "Persian"
        return val

    # ---------- session control ---------- #

    def start_recognition(self) -> None:
        """
        Start ASR + translator workers and reset session.
        """
        self.txt_source.delete("1.0", tk.END)
        self.txt_translation.delete("1.0", tk.END)
        self.partial_var.set("")
        self.session_segments = []
        self.status_var.set("Starting...")

        self.stop_flag.clear()

        source_lang = self.get_source_lang()
        target_lang = self.get_target_lang()
        model_path = SOURCE_LANG_MODELS.get(source_lang)
        if not model_path:
            messagebox.showerror(
                "ASR Model Error",
                f"No VOSK model configured for source language '{source_lang}'.",
            )
            self.status_var.set("Error")
            return

        self.lbl_source_title.config(text=f"Source ({source_lang})")
        self.lbl_target_title.config(text=f"Translation ({target_lang})")
        
        # --- FIX: Removed the line causing the error: self.txt_translation.config(justify=...) ---
        # The justification is now handled correctly by the tags in _process_events.
        # -----------------------------------------------------------------------------------------
            
        # Decide which device to use
        loopback_output_index = None
        input_index = None
        if self.capture_output_var.get():
            loopback_output_index = parse_device_index(self.output_device_var.get())
        else:
            input_index = parse_device_index(self.input_device_var.get())

        self._session_llm_settings = self._snapshot_llm_settings()

        self.asr = RealtimeVoskASR(
            model_path=model_path,
            sample_rate=SAMPLE_RATE,
            ui_queue=self.ui_queue,
            translate_queue=self.translate_queue,
            input_device_index=input_index,
            loopback_output_index=loopback_output_index,
        )

        try:
            self.asr.start()
        except Exception as e:
            messagebox.showerror("ASR Error", f"Failed to start ASR: {e}")
            self.status_var.set("Error")
            return

        self.translator_thread = threading.Thread(
            target=translator_worker,
            args=(
                self.translate_queue,
                self.ui_queue,
                self.stop_flag,
                source_lang,
                target_lang,
                self._session_llm_settings,
            ),
            daemon=True,
        )
        self.translator_thread.start()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set("Running")

    def stop_recognition(self) -> None:
        """
        Stop ASR + translator workers.
        """
        self.status_var.set("Stopping...")
        self.stop_flag.set()

        if self.asr is not None:
            self.asr.stop()

        try:
            self.translate_queue.put_nowait(None)
        except Exception:
            pass

        if self.translator_thread is not None:
            self.translator_thread.join(timeout=2.0)

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

    # ---------- summary ---------- #

    def summarize_session(self) -> None:
        """
        Summarize entire session with the configured LLM.
        """
        if not self.session_segments:
            messagebox.showinfo("Summary", "No text to summarize yet.")
            return

        full_text = "\n".join(self.session_segments)
        source_lang = self.get_source_lang()
        target_lang = self.get_target_lang()
        self.status_var.set("Summarizing session...")

        llm_settings = self._snapshot_llm_settings()

        def worker():
            prompt = build_summary_prompt(full_text, source_lang, target_lang)
            summary = call_llm_with_settings(prompt, llm_settings, timeout=90.0)
            self.ui_queue.put({
                "type": "summary",
                "summary": summary or "<No summary returned>",
                "source_lang": source_lang,
                "target_lang": target_lang,
            })

        threading.Thread(target=worker, daemon=True).start()

    # ---------- event processing ---------- #

    def _process_events(self) -> None:
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                mtype = msg.get("type")
                if mtype == "status":
                    self.status_var.set(msg.get("text", ""))
                elif mtype == "partial":
                    self.partial_var.set(msg.get("text", ""))
                elif mtype == "final":
                    text = msg.get("text", "")
                    if text:
                        self.txt_source.insert(tk.END, text + "\n\n")
                        self.txt_source.see(tk.END)
                        self.partial_var.set("")
                        self.session_segments.append(text)
                
                # --- FIX: Use defined tags 'rtl'/'ltr' for justification during translation insert ---
                elif mtype == "translation":
                    tr = msg.get("translation")
                    
                    if self.target_lang_var.get() == "Persian":
                        # \u202E: RLE (Right-to-Left Embedding)
                        # \u202C: PDF (Pop Directional Formatting)
                        text_to_insert = "\u202E" + tr + "\u202C" + "\n\n"
                        tag_to_apply = "rtl" 
                    else:
                        text_to_insert = tr + "\n\n"
                        tag_to_apply = "ltr" 
                    
                    self.txt_translation.insert(tk.END, text_to_insert, tag_to_apply)
                    self.txt_translation.see(tk.END)
                # -------------------------------------------------------------------------------------
                    
                elif mtype == "summary":
                    self._show_summary_window(
                        msg.get("summary", ""),
                        msg.get("source_lang", ""),
                        msg.get("target_lang", ""),
                    )
                    self.status_var.set("Summary ready.")
        except queue.Empty:
            pass

        self.after(50, self._process_events)

    def _show_summary_window(self, summary: str, source_lang: str, target_lang: str) -> None:
        win = tk.Toplevel(self)
        win.title(f"Session Summary ({target_lang})")
        win.geometry("800x600")

        ttk.Label(
            win,
            text=f"Summary of session ({source_lang} -> {target_lang})",
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.TOP, anchor="w", padx=10, pady=(10, 5))

        txt = tk.Text(win, wrap=tk.WORD)
        txt.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        scroll = ttk.Scrollbar(win, orient=tk.VERTICAL, command=txt.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        txt.config(yscrollcommand=scroll.set)
        
        # --- FIX: Correctly define rtl_summary and apply justification ---
        if target_lang == "Persian":
            # Add unicode characters for RTL directionality
            summary_to_insert = "\u202E" + summary + "\u202C"
            # Apply RIGHT justification for better appearance
            txt.config(justify=tk.RIGHT)
        else:
            summary_to_insert = summary
            txt.config(justify=tk.LEFT)
        
        txt.insert("1.0", summary_to_insert)
        txt.see(tk.END)
        txt.config(state=tk.DISABLED)
        # ---------------------------------------------------------------


# --------------- MAIN --------------- #

def main():
    app = TranslatorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()