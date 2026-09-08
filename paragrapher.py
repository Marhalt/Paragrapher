#!/usr/bin/env python3
import sys
import os
import re
import argparse
import requests

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
CHUNK_TARGET = 1800


OPEN_QUOTES = '"“‘'
CLOSE_QUOTES = '"”’'
_TERMINAL_RE = re.compile(r'[.!?][)\]"”’\']*\s*$')
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "prof", "rev", "gen", "sgt",
    "lt", "col", "capt", "vs", "etc", "no", "fig", "e.g", "i.e",
}


def prepass(lines):
    """Join hard-wrapped lines back into paragraphs.

    A new paragraph begins on a blank line, or when the previous line ended a
    sentence and the next line opens with a quotation mark — the near-universal
    signal that a new speaker is taking the floor. Paragraphs are separated by a
    blank line in the returned text; everything downstream treats those breaks as
    fixed.
    """
    paragraphs = []
    buffer = []

    def flush():
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer.clear()

    for line in lines:
        stripped = re.sub(r'\s+', ' ', line.strip())
        if not stripped:
            flush()
            continue
        if buffer and stripped[0] in OPEN_QUOTES and _TERMINAL_RE.search(buffer[-1]):
            flush()
        buffer.append(stripped)

    flush()
    return "\n\n".join(paragraphs)


def split_into_sentences(text):
    """Split text into sentences, ignoring '.!?' that fall inside a quotation.

    Double quotes (straight or curly) are tracked by parity so a period inside
    a quoted line — `"Come on in. I didn't realize it was late."` — is not
    mistaken for a sentence end. A boundary is only taken when the terminator
    sits outside quotes and is followed by whitespace and a capital, digit, or
    opening quote (or the end of the text)."""
    text = text.strip()
    if not text:
        return []

    sentences = []
    start = 0
    in_quote = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == '"' or ch in '“”':
            if ch == '“':
                in_quote = True
            elif ch == '”':
                in_quote = False
            else:
                in_quote = not in_quote
        elif ch in '.!?' and not in_quote:
            j = i + 1
            while j < n and text[j] in '.!?' + CLOSE_QUOTES + ")]'":
                j += 1
            if j >= n:
                sentences.append(text[start:j].strip())
                start = j
                i = j
                continue
            if text[i] == '.':
                word = re.search(r'(\S+)$', text[start:i])
                if word and word.group(1).lower().rstrip('.') in _ABBREVIATIONS:
                    i += 1
                    continue
            if text[j].isspace():
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k < n and (text[k].isupper() or text[k].isdigit() or text[k] in OPEN_QUOTES):
                    sentences.append(text[start:j].strip())
                    start = k
                    i = k
                    continue
        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def segment_chunk(chunk):
    """Return (sentences, forced_breaks) for a chunk.

    `forced_breaks` is the set of 1-indexed sentence numbers after which a
    paragraph break already exists because the pre-pass put one there."""
    sentences = []
    forced = set()
    for para in (p for p in chunk.split("\n\n") if p.strip()):
        sentences.extend(split_into_sentences(para))
        forced.add(len(sentences))
    forced.discard(len(sentences))
    forced.discard(0)
    return sentences, forced


def chunk_text(text, target=CHUNK_TARGET):
    """Group whole paragraphs into chunks of ~target chars.

    Paragraph breaks from the pre-pass are never split across chunks."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        current.append(para)
        current_len += len(para) + 2
        if current_len >= target:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("\n\n".join(current))

    return chunks


SYSTEM_PROMPT = (
    "You are a paragraph formatter for prose fiction. You receive a numbered list "
    "of sentences. Some paragraph breaks are ALREADY in place (you will be told "
    "which sentence numbers they follow) — these come from the source text's own "
    "line breaks and are correct. Your job is to add the breaks that are still "
    "missing.\n"
    "\n"
    "Add a break after a sentence when:\n"
    "- The speaker changes and no break is there yet. Every speaker's turn of "
    "dialogue is its own paragraph.\n"
    "- A long run of narration with no dialogue should be split for readability "
    "(roughly every 3-6 sentences, at a shift in time, place, topic, or focus).\n"
    "- The narrative moves to a new scene, action, or subject.\n"
    "\n"
    "Do NOT add a break when:\n"
    "- It would separate a line of dialogue from its speech tag (\"he said\", "
    "\"she asked\") or from the action beat of the character speaking in that "
    "moment. A quote and its attribution stay in one paragraph.\n"
    "- The same character keeps speaking across consecutive sentences with no "
    "change of focus between them.\n"
    "- An action beat or bit of narration belongs to the speaker of the dialogue "
    "right next to it — keep it with that dialogue.\n"
    "\n"
    "Treat a sentence that starts with an opening quotation mark as a new "
    "speaker's turn unless it is clearly the same speaker continuing.\n"
    "\n"
    "Respond with ONLY a comma-separated list of sentence numbers after which a "
    "new paragraph should start (the ones you are ADDING; you may also repeat the "
    "existing ones, it does no harm). Example: 3, 7, 12\n"
    "If nothing needs to be added, respond with: none\n"
    "Output nothing else — no explanation, no commentary."
)


def get_llm_breaks(sentences, forced, url, model=None, debug=False):
    """Ask the LLM for paragraph breaks to ADD on top of the pre-pass breaks.
    Returns a set of 1-indexed sentence numbers."""
    if len(sentences) <= 1:
        return set()

    numbered = "\n".join(f"{i + 1}: {s}" for i, s in enumerate(sentences))
    existing = ", ".join(str(n) for n in sorted(forced)) or "none"

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Paragraph breaks already exist after sentences: {existing}\n\n{numbered}",
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    if model:
        payload["model"] = model

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()
    if debug:
        print(f"\n      RAW: {repr(raw)}")

    # Extract all valid integers from the response (must be 1 to N-1)
    return set(
        int(n) for n in re.findall(r'\d+', raw)
        if 1 <= int(n) <= len(sentences) - 1
    )


def apply_breaks(sentences, break_positions):
    """Rejoin sentences, inserting paragraph breaks at the specified positions."""
    parts = []
    for i, sentence in enumerate(sentences):
        parts.append(sentence)
        if i < len(sentences) - 1:
            parts.append("\n\n" if (i + 1) in break_positions else " ")
    return "".join(parts)


def process_file(input_file, args):
    input_dir = os.path.dirname(os.path.abspath(input_file))
    filename = os.path.basename(input_file)
    clean_dir = os.path.join(input_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    output_path = os.path.join(clean_dir, filename)

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_words = len(" ".join(l.strip() for l in lines).split())
    print(f"  {raw_words:,} words in raw file")

    print("Pre-pass: joining broken lines...")
    text = prepass(lines)
    prepass_words = len(text.split())
    print(f"  {prepass_words:,} words after pre-pass", end="")
    if prepass_words != raw_words:
        print(f" — WARNING: pre-pass lost {raw_words - prepass_words} words")
    else:
        print()

    print(f"Chunking (target >= {args.chunk_size} chars per chunk)...")
    chunks = chunk_text(text, target=args.chunk_size)
    chunk_words = sum(len(c.split()) for c in chunks)
    print(f"  {len(chunks)} chunks, {chunk_words:,} words total")
    if chunk_words != prepass_words:
        print(f"  WARNING: chunking lost {prepass_words - chunk_words} words")

    print(f"Sending to LM Studio ({args.url})...")
    reformatted = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {len(chunk)} chars ... ", end="", flush=True)
        sentences, forced = segment_chunk(chunk)
        try:
            added = get_llm_breaks(sentences, forced, url=args.url, model=args.model, debug=args.debug)
        except Exception as e:
            added = set()
            print(f"LLM failed ({e}) — pre-pass breaks only ... ", end="")
        positions = forced | added
        reformatted.append(apply_breaks(sentences, positions))
        pre_str = ", ".join(str(p) for p in sorted(forced)) or "none"
        add_str = ", ".join(str(p) for p in sorted(added - forced)) or "none"
        print(f"pre-pass: {pre_str}  |  added: {add_str}")

    output = "\n\n".join(reformatted)
    output_words = len(output.split())
    print(f"Writing {output_path}... ({output_words:,} words)")
    if output_words != prepass_words:
        print(f"  WARNING: final output has {prepass_words - output_words} fewer words than pre-pass")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Reformat a file with proper paragraphs using a local LLM."
    )
    parser.add_argument("input_file", help="Path to the input file or directory")
    parser.add_argument(
        "--model",
        default=None,
        help="LM Studio model identifier (optional; uses whatever is loaded if omitted)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_TARGET,
        help=f"Minimum characters per chunk before a split (default: {CHUNK_TARGET})",
    )
    parser.add_argument(
        "--url",
        default=LM_STUDIO_URL,
        help=f"LM Studio chat completions endpoint (default: {LM_STUDIO_URL})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the raw LLM response for each chunk",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(args.input_file):
        txt_files = sorted(
            f for f in os.listdir(args.input_file) if f.lower().endswith(".txt")
        )
        if not txt_files:
            print(f"Error: no .txt files found in {args.input_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(txt_files)} .txt file(s) in {args.input_file}")
        for i, fname in enumerate(txt_files, 1):
            print(f"\n=== [{i}/{len(txt_files)}] {fname} ===")
            process_file(os.path.join(args.input_file, fname), args)
    else:
        process_file(args.input_file, args)


if __name__ == "__main__":
    main()
