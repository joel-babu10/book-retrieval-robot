"""
input/query_input.py
----------------------------------------------------------------------------
Step 1: Reader Input Request - Unified Application (concurrent voice/text)

Mic is always listening in the background. At any point, pressing Enter
interrupts listening and switches to typed input instead - so voice is
"always on" but text is always one keypress away, no menu needed.

Importable by any script (main.py, spine_pipeline, scattered_pipeline):

    from input.query_input import get_book_query

    result = get_book_query()
    if result:
        title = result["title"]
        ...
"""

import json
import sys
import threading
from pathlib import Path
from difflib import SequenceMatcher

VOICE_AVAILABLE = False
try:
    try:
        import pyaudio
    except ImportError:
        try:
            import queue
            import sounddevice as sd

            class _PyAudioStreamShim:
                def __init__(self, rate=16000, channels=1, device=None, frames_per_buffer=1024):
                    self.queue = queue.Queue()
                    self.rate = rate
                    self.channels = channels
                    def callback(indata, frames, time_info, status):
                        self.queue.put(bytes(indata))
                    self.stream = sd.RawInputStream(
                        samplerate=rate,
                        channels=channels,
                        dtype='int16',
                        device=device,
                        blocksize=frames_per_buffer,
                        callback=callback
                    )
                    self.stream.start()

                def read(self, num_frames, exception_on_overflow=False):
                    bytes_needed = num_frames * self.channels * 2
                    data = bytearray()
                    while len(data) < bytes_needed:
                        data.extend(self.queue.get())
                    return bytes(data[:bytes_needed])

                def is_stopped(self):
                    return not self.stream.active

                def is_active(self):
                    return self.stream.active

                def stop_stream(self):
                    if self.stream.active:
                        self.stream.stop()

                def close(self):
                    self.stop_stream()
                    self.stream.close()

            class _PyAudioInstanceShim:
                def get_default_input_device_info(self):
                    default_in = sd.default.device[0]
                    if default_in is None or default_in < 0:
                        default_in = 0
                    info = sd.query_devices(default_in)
                    return {
                        'index': default_in,
                        'name': info['name'],
                        'defaultSampleRate': info['default_samplerate'],
                        'maxInputChannels': info['max_input_channels']
                    }

                def get_device_count(self):
                    return len(sd.query_devices())

                def get_device_info_by_index(self, index):
                    info = sd.query_devices(index)
                    return {
                        'index': index,
                        'name': info['name'],
                        'defaultSampleRate': info['default_samplerate'],
                        'maxInputChannels': info['max_input_channels']
                    }

                def open(self, rate=16000, channels=1, format=8, input=True, input_device_index=None, frames_per_buffer=1024, **kwargs):
                    return _PyAudioStreamShim(rate=rate, channels=channels, device=input_device_index, frames_per_buffer=frames_per_buffer)

                def get_sample_size(self, format):
                    return 2

                def terminate(self):
                    pass

            class _PyAudioModuleShim:
                paInt16 = 8
                paFloat32 = 1
                PyAudio = _PyAudioInstanceShim
                def get_sample_size(self, format):
                    return 2

            sys.modules['pyaudio'] = _PyAudioModuleShim()
        except ImportError:
            pass

    import speech_recognition as sr
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# Catalog paths (OS-agnostic resolution)
CATALOG_PATH = Path(__file__).resolve().parent.parent / "mock_library_catalog.json"
FALLBACK_CATALOG_PATH = Path(__file__).resolve().parent.parent / "library_catalog.json"


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------
# Catalog matching
# ---------------------------------------------------------------

def load_catalog(path=None):
    if path is None:
        if CATALOG_PATH.exists():
            path = CATALOG_PATH
        elif FALLBACK_CATALOG_PATH.exists():
            path = FALLBACK_CATALOG_PATH
        else:
            path = CATALOG_PATH

    path = Path(path)
    if not path.exists():
        log(f"WARNING: catalog file not found at: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_match(query, book):
    query_lower = query.lower().strip()
    book_id = book.get("book_id", "").lower()
    title_lower = book.get("title", "").lower()
    author_lower = book.get("author", "").lower()
    aliases = [a.lower() for a in book.get("aliases", [])]

    if query_lower == book_id:
        return 1.0

    if query_lower in title_lower or query_lower in author_lower:
        return 1.0

    for alias in aliases:
        if query_lower in alias or alias in query_lower:
            return 1.0

    query_words = [w for w in query_lower.split() if len(w) > 1]
    title_words = [w for w in title_lower.split() if len(w) > 1]
    author_words = [w for w in author_lower.split() if len(w) > 1]
    all_target_words = title_words + author_words
    for a in aliases:
        all_target_words.extend([w for w in a.split() if len(w) > 1])

    matched_words = 0
    for qw in query_words:
        if any(qw in tw or tw in qw for tw in all_target_words):
            matched_words += 1
    token_score = (matched_words / len(query_words)) if query_words else 0.0

    title_ratio = SequenceMatcher(None, query_lower, title_lower).ratio()
    author_ratio = SequenceMatcher(None, query_lower, author_lower).ratio()
    alias_ratio = max([SequenceMatcher(None, query_lower, a).ratio() for a in aliases], default=0.0)
    fuzzy_ratio = max(title_ratio, author_ratio, alias_ratio)

    if token_score == 0.0:
        return 0.0

    return max(token_score, fuzzy_ratio)


def match_query(query, catalog=None, threshold=0.50):
    if catalog is None:
        catalog = load_catalog()
    if not query or not query.strip():
        return None, 0.0
    if not catalog:
        return {"book_id": "CUSTOM", "title": query.strip(), "call_number": "N/A", "shelf_level": "1", "order_position": "1", "rfid_info": "N/A"}, 1.0

    scored = [(score_match(query, book), book) for book in catalog]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_book = scored[0]

    # If confidence in catalog match is low, preserve the raw query as title
    if best_score < threshold:
        return {
            "book_id": "CUSTOM",
            "title": query.strip(),
            "call_number": "N/A",
            "shelf_level": "1",
            "order_position": "1",
            "rfid_info": "N/A",
            "author": "",
            "edition": "",
            "aliases": [],
        }, 1.0

    output = {
        "book_id": best_book.get("book_id", "N/A"),
        "title": best_book.get("title", query.strip()),
        "author": best_book.get("author", ""),
        "edition": best_book.get("edition", ""),
        "aliases": best_book.get("aliases", []),
        "call_number": best_book.get("call_number", "N/A"),
        "shelf_level": best_book.get("shelf_level", "1"),
        "order_position": best_book.get("order_position", "1"),
        "rfid_info": best_book.get("rfid_tag", "N/A"),
    }
    return output, best_score


def print_result(result, score, heard_text=None):
    if heard_text is not None:
        log(f'\nHeard voice query: "{heard_text}"')
    if result is None:
        log(f"\nNo catalog match found (best score: {score:.2f}).")
        return
    log(f"\nMatch confidence: {score:.2f}")
    log("\n---- Step 1 Output (feeds into Step 2) ----")
    log(f"Book ID:         {result['book_id']}")
    log(f"Title:           {result['title']}")
    log(f"Call Number:     {result['call_number']}")
    log("\n-- Shelf Location --")
    log(f"Shelf Level:     {result['shelf_level']}")
    log(f"Order Position:  {result['order_position']}")
    log(f"RFID Info:       {result['rfid_info']}")


# ---------------------------------------------------------------
# Concurrent listen: mic thread + stdin thread, first one wins
# ---------------------------------------------------------------

class InputRace:
    """Runs mic listening and keyboard input concurrently. Whichever
    resolves first (a transcribed phrase, or the user hitting Enter to
    type) wins; the other is abandoned."""

    def __init__(self):
        self.result_text = None       # winning text, from either source
        self.source = None            # "voice" or "text"
        self.done = threading.Event()
        self.lock = threading.Lock()

    def _finish(self, text, source):
        with self.lock:
            if self.done.is_set():
                return  # already resolved by the other thread
            self.result_text = text
            self.source = source
            self.done.set()

    def _voice_worker(self):
        if not VOICE_AVAILABLE:
            return
        try:
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = False
            recognizer.energy_threshold = 300
            recognizer.pause_threshold = 1.0

            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                while not self.done.is_set():
                    try:
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    except sr.WaitTimeoutError:
                        continue  # nothing heard this window, keep listening
                    if self.done.is_set():
                        return
                    try:
                        text = recognizer.recognize_google(audio, language="en-US")
                        self._finish(text, "voice")
                        return
                    except sr.UnknownValueError:
                        continue  # heard noise, not speech - keep listening
                    except sr.RequestError:
                        return  # no internet / API issue - give up quietly, text can still win
        except OSError:
            return  # no mic available - text input remains the only path

    def _text_worker(self):
        try:
            typed = input()  # blocks until Enter; empty Enter still counts as "switch to text"
        except EOFError:
            return
        self._finish(typed.strip(), "text")

    def run(self):
        voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        text_thread = threading.Thread(target=self._text_worker, daemon=True)
        voice_thread.start()
        text_thread.start()
        self.done.wait()  # blocks until either thread calls _finish
        return self.result_text, self.source


def get_query():
    """Low-level: races voice vs text input, returns (raw_text, source)."""
    if VOICE_AVAILABLE:
        log("Listening... (speak anytime, or press Enter to type instead)")
    else:
        log("Voice unavailable - type your query and press Enter:")
    race = InputRace()
    return race.run()


# ---------------------------------------------------------------
# Public API — this is what other scripts should import
# ---------------------------------------------------------------

def get_book_query(catalog_path=None, threshold=0.35, verbose=True, allow_raw_fallback=True):
    """
    Full Step 1 flow as a single reusable call.

    Returns a dict:
        {
            "book_id": str,
            "title": str,
            "call_number": str,
            "shelf_level": str,
            "order_position": str,
            "rfid_info": str,
            "match_score": float,
            "input_source": "voice" | "text",
        }
    """
    catalog = load_catalog(catalog_path)
    query, source = get_query()

    if not query:
        if source == "text":
            if verbose:
                log("Enter book title, author, or book ID:")
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                query = ""
        if not query:
            if verbose:
                log("\nNo input captured. Try again.")
            return None

    result, score = match_query(query, catalog=catalog, threshold=threshold)

    if result is None and allow_raw_fallback:
        if verbose:
            log(f"\nNo catalog match for '{query}'. Using raw query as target title.")
        result = {
            "book_id": "CUSTOM",
            "title": query.strip(),
            "call_number": "N/A",
            "shelf_level": "1",
            "order_position": "1",
            "rfid_info": "N/A",
        }
        score = 1.0

    if verbose and result:
        heard_text = query if source == "voice" else None
        print_result(result, score, heard_text=heard_text)

    if result is None:
        return None

    result["match_score"] = score
    result["input_source"] = source
    return result


def get_mode_selection(default="spine", verbose=True):
    """
    Prompts the user to select the arrangement mode:
      1. spine     - stacked spines on shelf (default)
      2. scattered - books lying flat/scattered on surface
      3. mixed     - scan both and keep higher-confidence match

    Returns "spine", "scattered", or "mixed".
    """
    if verbose:
        log("\nSelect arrangement mode:")
        log("  [1] spine     - Spines aligned on shelf (default)")
        log("  [2] scattered - Books flat/scattered on surface")
        log("  [3] mixed     - Run both pipelines, pick best match")
        try:
            choice = input("Enter choice (1-3 or mode name, default 1): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = ""
    else:
        choice = ""

    if choice in ("1", "spine", "s"):
        mode = "spine"
    elif choice in ("2", "scattered", "sc"):
        mode = "scattered"
    elif choice in ("3", "mixed", "m"):
        mode = "mixed"
    elif not choice:
        mode = default
    else:
        if verbose:
            log(f"Invalid mode choice '{choice}'. Defaulting to '{default}'.")
        mode = default

    if verbose:
        log(f"Arrangement mode set to: {mode}\n")
    return mode


def run():
    log("---- Step 1: Reader Input Request ----")
    return get_book_query()


if __name__ == "__main__":
    run()