"""Create token windows by slicing original characters, retaining provenance.

The original-document token window and a chunk's re-tokenized count can differ
at subword boundaries. Both are recorded. This utility never decodes token IDs
into replacement text. It cannot recover characters lost by an earlier decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunks(
    text: str,
    tokenizer: Any,
    *,
    source_id: str,
    document_id: str,
    tokenizer_id: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic original-text windows with character/token offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not all(isinstance(x, str) and x for x in (source_id, document_id, tokenizer_id)):
        raise ValueError("nonempty source, document and tokenizer identifiers required")
    if type(max_tokens) is not int or type(overlap_tokens) is not int:
        raise TypeError("window lengths must be integers")
    if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("require max_tokens > 0 and 0 <= overlap_tokens < max_tokens")
    if not text.strip():
        return
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer with character offsets is required")
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(ids) != len(offsets) or not ids:
        raise ValueError("nonempty text must yield paired token IDs and offsets")
    previous_start = previous_end = 0
    for start, end in offsets:
        if not 0 <= start < end <= len(text):
            raise ValueError("invalid tokenizer character offset")
        if start < previous_start or end < previous_end:
            raise ValueError("tokenizer offsets must be monotonic")
        previous_start, previous_end = start, end
    source_hash = digest(text)
    stride = max_tokens - overlap_tokens
    for ordinal, start in enumerate(range(0, len(ids), stride)):
        end = min(start + max_tokens, len(ids))
        if start > 0 and end - start < overlap_tokens:
            break
        char_start, char_end = (0, len(text)) if len(ids) <= max_tokens else (
            int(offsets[start][0]), int(offsets[end - 1][1])
        )
        chunk = text[char_start:char_end]
        # The token-window length refers to tokenization of the whole document.
        retokenized = tokenizer.encode(chunk, add_special_tokens=False, truncation=False)
        identity = json.dumps(
            [source_id, document_id, source_hash, char_start, char_end],
            ensure_ascii=False, separators=(",", ":"),
        )
        yield {
            "schema": "preserved-character-span-v1",
            "chunk_id": digest(identity),
            "source_id": source_id,
            "document_id": document_id,
            "document_sha256": source_hash,
            "chunk_ordinal": ordinal,
            "char_start": char_start,
            "char_end": char_end,
            "source_token_start": start,
            "source_token_end": end,
            "source_token_count": end - start,
            "chunk_token_count": len(retokenized),
            "chunk_text": chunk,
            "chunk_sha256": digest(chunk),
            "tokenizer_id": tokenizer_id,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        }
        if end == len(ids):
            break


def convert(input_path: Path, output_path: Path, tokenizer: Any, tokenizer_id: str,
            max_tokens: int = 500, overlap_tokens: int = 50) -> int:
    """Stream local JSONL in source order; refuse to replace an existing output."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    count = 0
    # Publish only a complete file. Hard-link publication is atomic and refuses
    # to overwrite a destination created while conversion was running.
    temporary = None
    try:
        with input_path.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix="." + output_path.name + ".", suffix=".partial", delete=False,
        ) as target:
            temporary = Path(target.name)
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for chunk in chunks(row["text"], tokenizer, source_id=row["source_id"],
                                    document_id=row["document_id"], tokenizer_id=tokenizer_id,
                                    max_tokens=max_tokens, overlap_tokens=overlap_tokens):
                    chunk["input_line"] = line_number
                    target.write(json.dumps(chunk, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, output_path)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), use_fast=True,
                                              local_files_only=True)
    print(convert(args.input, args.output, tokenizer, args.tokenizer_id,
                  args.max_tokens, args.overlap_tokens))


if __name__ == "__main__":
    main()
