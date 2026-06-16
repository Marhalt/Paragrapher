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


def chunk_text(text, target=CHUNK_TARGET):
    """Split text into chunks of ~target chars, always ending at a sentence boundary."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        current.append(sentence)
        current_len += len(sentence) + 1  # +1 for the space between sentences
        if current_len >= target:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def reformat_chunk(chunk, url, model=None):
    """Send a chunk to LM Studio and return the paragraphed version."""
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text formatter. Your ONLY task is to add paragraph breaks "
                    "to the provided text to improve readability. Do NOT change any words, "
                    "do NOT fix spelling, do NOT alter punctuation, do NOT add or remove "
                    "any content whatsoever. Only insert paragraph breaks (blank lines) "
                    "where natural breaks in the narrative occur — scene shifts, dialogue "
                    "turns, or topic changes. Return ONLY the reformatted text with no "
                    "explanation, preamble, or commentary."
                ),
            },
            {"role": "user", "content": chunk},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    if model:
        payload["model"] = model

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Reformat a file with proper paragraphs using a local LLM."
    )
    parser.add_argument("input_file", help="Path to the input file")
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
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    base, ext = os.path.splitext(args.input_file)
    output_path = base + "_clean" + (ext or ".txt")

    print(f"Reading {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print("Pre-pass: joining broken lines...")
    text = prepass(lines)
    print(f"  {len(text):,} characters after pre-pass")

    print(f"Chunking (target >= {args.chunk_size} chars per chunk)...")
    chunks = chunk_text(text, target=args.chunk_size)
    print(f"  {len(chunks)} chunks")

    print(f"Sending to LM Studio ({args.url})...")
    reformatted = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {len(chunk)} chars ... ", end="", flush=True)
        try:
            result = reformat_chunk(chunk, url=args.url, model=args.model)
            reformatted.append(result)
            print("ok")
        except Exception as e:
            print(f"FAILED ({e}) — keeping original")
            reformatted.append(chunk)

    print(f"Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reformatted))

    print("Done.")


if __name__ == "__main__":
    main()
