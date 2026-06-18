#!/usr/bin/env python3
import sys
import os
import re
import argparse
import requests

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
CHUNK_TARGET = 1800


def prepass(lines):
    """Join lines that don't end with sentence-terminating punctuation."""
    buffer = []
    sentences = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                sentences.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(stripped)
        if stripped[-1] in '.!?"':
            sentences.append(" ".join(buffer))
            buffer = []

    if buffer:
        sentences.append(" ".join(buffer))

    return " ".join(sentences)


def split_into_sentences(text):
    """Split text into sentences on punctuation boundaries."""
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


def chunk_text(text, target=CHUNK_TARGET):
    """Split text into chunks of ~target chars, always ending at a sentence boundary."""
    sentences = split_into_sentences(text)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence) + 1
        if current_len >= target:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def get_break_positions(chunk, url, model=None, debug=False):
    """Ask the LLM where paragraph breaks should go. Returns a set of 1-indexed
    sentence numbers after which a break should be inserted."""
    sentences = split_into_sentences(chunk)
    if len(sentences) <= 1:
        return set(), sentences

    numbered = "\n".join(f"{i + 1}: {s}" for i, s in enumerate(sentences))

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a paragraph formatter. You will receive a numbered list of sentences. "
                    "Decide where paragraph breaks should go to improve readability — at scene shifts, "
                    "dialogue turns, or topic changes. "
                    "Respond with ONLY a comma-separated list of sentence numbers after which a new "
                    "paragraph should start. Example: 3, 7, 12\n"
                    "If no breaks are needed, respond with: none\n"
                    "Output nothing else — no explanation, no commentary."
                ),
            },
            {"role": "user", "content": numbered},
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
    positions = set(
        int(n) for n in re.findall(r'\d+', raw)
        if 1 <= int(n) <= len(sentences) - 1
    )
    return positions, sentences


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
        try:
            positions, sentences = get_break_positions(chunk, url=args.url, model=args.model, debug=args.debug)
            result = apply_breaks(sentences, positions)
            reformatted.append(result)
            breaks_str = ", ".join(str(p) for p in sorted(positions)) if positions else "none"
            print(f"breaks after: {breaks_str}")
        except Exception as e:
            print(f"FAILED ({e}) — no breaks added")
            reformatted.append(chunk)

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
