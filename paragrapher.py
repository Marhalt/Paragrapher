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


_LINE_SEPARATORS = ("\r\n", "\r", " ", "", "\x0b", "\x0c")


def read_source_lines(path):
    """Read a text file and split it into lines, tolerating separators other
    than plain '\\n'. Word processors and ebook tools often use U+2028 (line
    separator) or U+2029 (paragraph separator); Python's readlines() ignores
    both, which would hand the pre-pass a few enormous 'lines'."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        content = f.read()
    content = content.replace("\u2029", "\n\n")  # U+2029 paragraph sep -> blank line
    for sep in _LINE_SEPARATORS:
        content = content.replace(sep, "\n")
    return content.split("\n")


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


_QUOTE_CHARS = set(OPEN_QUOTES) | set(CLOSE_QUOTES)


def filter_added_breaks(sentences, added):
    """The pre-pass owns dialogue paragraphing — it keys off the source's own
    line breaks, which reliably start a new line for each new speaker. So the
    LLM is only trusted to split runs of narration: drop any added break that
    falls right after a sentence containing a quotation mark, which is where the
    model tends to wrongly fragment one speaker's turn."""
    return {
        i for i in added
        if not (_QUOTE_CHARS & set(sentences[i - 1]))
    }


def chunk_text(text, target=CHUNK_TARGET):
    """Break the text into chunks of ~target chars for the LLM.

    Each chunk is a dict:
      text          the chunk's text (may contain "\\n\\n" between whole paragraphs)
      new_paragraph how to rejoin it to the previous chunk — "\\n\\n" or " "
      trusted       True when the chunk is made of complete pre-pass paragraphs
                    (so their breaks are reliable); False when it is a slice of
                    one oversized paragraph the pre-pass could not divide.

    Whole paragraphs are packed together up to target. A single paragraph larger
    than a few times target — the sign that the source had no usable paragraph
    breaks in that stretch — is split on sentence boundaries so it never reaches
    the model as one giant blob."""
    oversize = max(target * 3, 6000)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append({"text": "\n\n".join(current), "new_paragraph": True, "trusted": True})
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > oversize:
            flush_current()
            buf, buf_len, first = [], 0, True
            for sentence in split_into_sentences(para):
                buf.append(sentence)
                buf_len += len(sentence) + 1
                if buf_len >= target:
                    chunks.append({"text": " ".join(buf), "new_paragraph": first, "trusted": False})
                    buf, buf_len, first = [], 0, False
            if buf:
                chunks.append({"text": " ".join(buf), "new_paragraph": first, "trusted": False})
            continue

        current.append(para)
        current_len += len(para) + 2
        if current_len >= target:
            flush_current()

    flush_current()
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
    "Example — these belong in ONE paragraph, do NOT break between them:\n"
    "  \"It's mine!\" he said.\n"
    "  \"I earned it!\" As he said it, he worked his arm free.\n"
    "because it is one man speaking, plus his own action. A break only comes "
    "when a different person speaks — e.g. before \"It's not yours!\" replied "
    "Steven.\n"
    "\n"
    "Respond with ONLY a comma-separated list of sentence numbers after which a "
    "new paragraph should start (the ones you are ADDING; you may also repeat the "
    "existing ones, it does no harm). Example: 3, 7, 12\n"
    "If nothing needs to be added, respond with: none\n"
    "Output nothing else — no explanation, no commentary."
)


_THINK_RE = re.compile(r'<(think|thought|reasoning)>.*?</\1>', re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r'<(think|thought|reasoning)>.*\Z', re.DOTALL | re.IGNORECASE)


def strip_reasoning(raw):
    """Remove <think>…</think> blocks. An unclosed <think> (the response was
    truncated mid-reasoning) leaves nothing usable, so drop it entirely."""
    raw = _THINK_RE.sub("", raw)
    raw = _OPEN_THINK_RE.sub("", raw)
    return raw.strip()


def parse_break_list(raw, max_valid):
    """Pull the answer — a comma-separated list of sentence numbers, or 'none' —
    out of the model's reply. Reads the last line that looks like an answer so
    stray numbers in any surrounding chatter are ignored."""
    text = strip_reasoning(raw)
    if not text:
        return set()

    for line in reversed(text.splitlines()):
        line = line.strip().strip(".")
        if not line:
            continue
        if re.fullmatch(r'(?i)none', line):
            return set()
        if re.fullmatch(r'[\d\s,]+', line):
            return set(
                n for n in (int(x) for x in re.findall(r'\d+', line))
                if 1 <= n <= max_valid
            )
    # Last resort: no clean answer line — scrape the whole (de-reasoned) reply.
    return set(
        n for n in (int(x) for x in re.findall(r'\d+', text))
        if 1 <= n <= max_valid
    )


def get_llm_breaks(sentences, forced, url, model=None, debug=False, think=True):
    """Ask the LLM for paragraph breaks to ADD on top of the pre-pass breaks.
    Returns a set of 1-indexed sentence numbers."""
    if len(sentences) <= 1:
        return set()

    numbered = "\n".join(f"{i + 1}: {s}" for i, s in enumerate(sentences))
    existing = ", ".join(str(n) for n in sorted(forced)) or "none"
    user_content = f"Paragraph breaks already exist after sentences: {existing}\n\n{numbered}"
    if not think:
        user_content += " /no_think"

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        # A reasoning model needs room for the <think> block *and* the answer;
        # 1024 truncates mid-reasoning and no answer is ever emitted.
        "max_tokens": 8192 if think else 1024,
    }
    if model:
        payload["model"] = model
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["reasoning_effort"] = "none"

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    raw = (choice["message"].get("content") or "").strip()
    if debug:
        print(f"\n      RAW: {repr(raw)}")
    if choice.get("finish_reason") == "length":
        print("      WARNING: response hit the token limit (truncated) ... ", end="")

    return parse_break_list(raw, len(sentences) - 1)


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
    lines = read_source_lines(input_file)

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
    chunk_words = sum(len(c["text"].split()) for c in chunks)
    split_paras = sum(1 for c in chunks if not c["trusted"])
    print(f"  {len(chunks)} chunks, {chunk_words:,} words total")
    if split_paras:
        print(f"  note: {split_paras} chunk(s) are slices of oversized paragraphs "
              "(source had no usable paragraph breaks there) — LLM will paragraph them freely")
    if chunk_words != prepass_words:
        print(f"  WARNING: chunking lost {prepass_words - chunk_words} words")

    print(f"Sending to LM Studio ({args.url})...")
    pieces = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {len(chunk['text'])} chars ... ", end="", flush=True)
        sentences, forced = segment_chunk(chunk["text"])
        try:
            added = get_llm_breaks(
                sentences, forced, url=args.url, model=args.model,
                debug=args.debug, think=not args.no_think,
            )
        except Exception as e:
            added = set()
            print(f"LLM failed ({e}) — pre-pass breaks only ... ", end="")
        if chunk["trusted"]:
            added = filter_added_breaks(sentences, added)
        added -= forced
        positions = forced | added
        piece = apply_breaks(sentences, positions)
        joiner = "" if i == 1 else ("\n\n" if chunk["new_paragraph"] else " ")
        pieces.append(joiner + piece)
        pre_str = ", ".join(str(p) for p in sorted(forced)) or "none"
        add_str = ", ".join(str(p) for p in sorted(added)) or "none"
        print(f"pre-pass: {pre_str}  |  added: {add_str}")

    output = "".join(pieces)
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
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Ask the model to skip its reasoning phase (faster; needed for some "
        "reasoning models that otherwise burn the whole token budget thinking)",
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
