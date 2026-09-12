# Preserving original text in token windows

`preserve_text_spans.py` is a prospective research utility. It slices the source
string using fast-tokenizer character offsets instead of decoding token IDs.
This preserves characters represented by unknown tokens and avoids adding
decoder-generated spaces to Chinese and Japanese text.

The current production chunker is unchanged. This utility cannot reconstruct
raw documents from previously decoded chunks. Correct text preservation does
not, by itself, demonstrate better embedding or map quality.

Input is UTF-8 JSONL with explicit `source_id`, `document_id` and `text` strings.
Rows are processed in input order. Document IDs must identify source documents;
the caller is responsible for their provenance. Stable chunk IDs include the
source/document identity, original text hash and character span. Repeated input
documents intentionally produce the same IDs; this utility does not deduplicate.

```bash
python research/preserve_text_spans.py documents.jsonl chunks.jsonl \
  --tokenizer-path /path/to/pinned/local/tokenizer \
  --tokenizer-id bert-base-multilingual-cased@COMMIT
```

The tokenizer loads from local files only. Record its actual file hashes in the
dataset build manifest; the supplied identifier is a provenance label, not a
verification of those files. The output records Python character offsets
(end-exclusive, not UTF-8 byte offsets), original-document token windows,
retokenized chunk lengths, original/chunk hashes and input line numbers.

Defaults are 500 original-document tokens and 50-token overlap. Short documents
retain their entire original string. Long windows retain the characters from
their first token start to their last token end; leading/trailing whitespace
outside that span is not included. Internal whitespace is preserved. Tail
windows shorter than the overlap are skipped. Stop after reaching the final
token, avoiding the legacy loop's possible redundant overlap-only window.
Chunk counts therefore need not equal the legacy output.

A token window can begin or end inside a word. Its retokenized length can differ
from the original-document token count, so consumers must not infer a strict
downstream model token limit from `source_token_count`. This tool does not call
an embedding model or truncate source documents to a model context length.

Output publication is atomic on a local filesystem supporting hard links. An
existing destination is never replaced, including one created concurrently.
Malformed input fails without publishing a partial output; temporary files are
removed on ordinary exceptions. An abrupt process kill can leave a hidden
`.partial` file, which is not a completed dataset.

Run the Unicode, overlap, provenance, CLI and failure/publication tests against
a pinned local mBERT tokenizer:

```bash
python research/test_preserve_text_spans.py /path/to/pinned/local/tokenizer -v
```

The fixtures are self-authored. The first boundary test assumed English
`hello` was one mBERT token; it is two. That fixture was replaced with a known
single-token Chinese character, with no implementation or acceptance-threshold
change. The experimental record is Card046 in latent-labs.
