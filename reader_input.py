"""
Step 1: Reader Input Request - Unified Application (concurrent voice/text)
----------------------------------------------------------------------------
Mic is always listening in the background. At any point, pressing Enter
interrupts listening and switches to typed input instead - so voice is
"always on" but text is always one keypress away, no menu needed.
"""

import json
import sys
import threading
from pathlib import Path
from difflib import SequenceMatcher

try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

CATALOG_PATH = Path(__file__).resolve().parent / "mock_library_catalog.json"


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------
# Catalog matching
# ---------------------------------------------------------------

def load_catalog(path=CATALOG_PATH):
    if not Path(path).exists():
        log(f"ERROR: catalog file not found at: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_match(query, book):
    query_lower = query.lower().strip()
    if query_lower == book["book_id"].lower():
        return 1.0
    title_ratio = SequenceMatcher(None, query_lower, book["title"].lower()).ratio()
    author_ratio = SequenceMatcher(None, query_lower, book["author"].lower()).ratio()
    title_contains = 1.0 if query_lower in book["title"].lower() else 0.0
    author_contains = 1.0 if query_lower in book["author"].lower() else 0.0
    return max(title_ratio, author_ratio, title_contains, author_contains)


def match_query(query, catalog=None, threshold=0.35):
    if catalog is None:
        catalog = load_catalog()
    if not query or not query.strip():
        return None, 0.0
    scored = [(score_match(query, book), book) for book in catalog]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_book = scored[0]
    if best_score < threshold:
        return None, best_score
    output = {
        "book_id": best_book["book_id"],
        "call_number": best_book["call_number"],
        "shelf_level": best_book["shelf_level"],
        "order_position": best_book["order_position"],
        "rfid_info": best_book["rfid_tag"],
    }
    return output, best_score


def print_result(result, score, heard_text=None):
    if heard_text is not None:
        log(f'\nHeard: "{heard_text}"')
    if result is None:
        log(f"\nNo confident match found (best score: {score:.2f}).")
        log("Try a more specific title, author name, or exact Book ID.")
        return
    log(f"\nMatch confidence: {score:.2f}")
    log("\n---- Step 1 Output (feeds into Step 2) ----")
    log(f"Book ID:         {result['book_id']}")
    log(f"Call Number:     {result['call_number']}")
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
    if VOICE_AVAILABLE:
        log("Listening... (speak anytime, or press Enter to type instead)")
    else:
        log("Voice unavailable - type your query and press Enter:")
    race = InputRace()
    text, source = race.run()
    return text, source


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def run():
    log("---- Step 1: Reader Input Request ----")
    catalog = load_catalog()

    query, source = get_query()

    if not query:
        log("\nNo input captured. Try again.")
        return None

    if source == "voice":
        print_result(*match_query(query, catalog=catalog), heard_text=query)
    else:
        result, score = match_query(query, catalog=catalog)
        print_result(result, score)

    result, _ = match_query(query, catalog=catalog)
    return result


if __name__ == "__main__":
    run()