"""
Pipeline: Step 1 -> Step 5
---------------------------
Step 1 (reader_input.py): voice/text query -> matched catalog entry (title, etc.)
Step 5 (reverse_title_search.py): title -> visual localization on shelf (u,v, width, top-edge-center)

Run this instead of the two scripts separately once both are in the same folder.
"""

import sys
sys.path.append("bs")   # bs/ holds find_target.py + spine detection training data

from reader_input import get_query                        # your Step 1 file - raw input only, no catalog
from find_target import find_target                       # loaded from bs/ via sys.path

IMG_PATH = r"bs\test\images\test_img.png"   # <-- shelf photo to search, also inside bs/
MATCH_THRESHOLD = 0.40   # lowered from 0.55 - OCR on embossed/dark spines rarely clears 0.55 even after rotation fix
MAX_QUERY_RETRIES = 2    # voice misfires / empty input are common; don't abort on first miss


def run_pipeline():
    target_title = None
    for attempt in range(1, MAX_QUERY_RETRIES + 2):
        query, source = get_query()
        if query:
            target_title = query
            break
        if attempt <= MAX_QUERY_RETRIES:
            print(f"[Pipeline] No input captured (attempt {attempt}), try again...")
    if target_title is None:
        print("[Pipeline] No input captured after retries. Aborting.")
        return None

    print(f"\n[Pipeline] Reader wants: '{target_title}' -> searching shelf image...")

    match = find_target(IMG_PATH, target_title)
    if match is None or match["similarity"] < MATCH_THRESHOLD:
        best_note = f" (best similarity found: {match['similarity']:.2f})" if match else ""
        print(f"[Pipeline] '{target_title}' NOT found on shelf image.{best_note}")
        print("Possible causes: book missing, poor OCR read, or wrong shelf photo.")
        return None

    print(f"[Pipeline] Found on shelf (OCR read: '{match['text']}', similarity={match['similarity']:.2f})")

    result = {
        "query": target_title,
        "shelf_location": match,
        "ocr_confidence": match["similarity"],
    }

    print("\n---- Pipeline Output (feeds into Step 6) ----")
    print(f"Query:           {target_title}")
    print(f"Center (u, v):   ({match['u']:.1f}, {match['v']:.1f})")
    print(f"Width:           {match['width_px']:.1f}px")
    print(f"Top-edge center: ({match['top_edge_center_u']:.1f}, {match['top_edge_center_v']:.1f})")

    return result


if __name__ == "__main__":
    run_pipeline()