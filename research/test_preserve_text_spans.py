"""Run with a local mBERT tokenizer; fixtures are self-authored, not corpus text."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoTokenizer
from preserve_text_spans import chunks, convert

TOKENIZER = sys.argv.pop(1)
TOK = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, local_files_only=True)


def emit(text, **kw):
    return list(chunks(text, TOK, source_id="fixture", document_id="document",
                       tokenizer_id="bert-base-multilingual-cased@3f076fdb", **kw))


class Preservation(unittest.TestCase):
    def test_short_source_exact(self):
        text = "  中文：東京、café！ 🧬\n\t"
        self.assertEqual(emit(text)[0]["chunk_text"], text)
        self.assertEqual(emit(" \n\t"), [])

    def test_original_characters_survive_unknown_and_cjk_decode(self):
        text = "中文研究：東京の地図、café 🧬。\n" * 120
        rows = emit(text)
        self.assertGreater(len(rows), 2)
        self.assertTrue(any("🧬" in row["chunk_text"] for row in rows))
        self.assertTrue(all("[UNK]" not in row["chunk_text"] for row in rows))
        self.assertEqual(rows, emit(text))
        for row in rows:
            self.assertEqual(row["chunk_text"], text[row["char_start"]:row["char_end"]])
        for left, right in zip(rows, rows[1:]):
            self.assertEqual(left["source_token_end"] - right["source_token_start"], 50)
            self.assertEqual(text[right["char_start"]:left["char_end"]],
                             left["chunk_text"][right["char_start"] - left["char_start"]:])
        old = TOK.decode(TOK.encode(text, add_special_tokens=False)[:500])
        self.assertIn("[UNK]", old)
        self.assertNotEqual(old, rows[0]["chunk_text"])
        self.assertIn("中文", rows[0]["chunk_text"])
        self.assertIn("中 文", old)

    def test_span_identity_and_tail(self):
        text = "中" * 950
        rows = emit(text)
        self.assertEqual([r["source_token_count"] for r in rows], [500, 500])
        self.assertEqual(len({r["chunk_id"] for r in rows}), 2)
        other = list(chunks(text, TOK, source_id="fixture", document_id="different",
                            tokenizer_id="same"))
        self.assertNotEqual(rows[0]["chunk_id"], other[0]["chunk_id"])
        self.assertEqual([r["source_token_count"] for r in emit("中" * 501)], [500, 51])
        self.assertEqual([r["source_token_count"] for r in emit("中" * 521, overlap_tokens=0)], [500, 21])

    def test_bad_offsets_rejected(self):
        class Broken:
            is_fast = True
            def __call__(self, *args, **kw):
                return {"input_ids": [1], "offset_mapping": [(0, 100)]}
        with self.assertRaisesRegex(ValueError, "offset"):
            list(chunks("short", Broken(), source_id="s", document_id="d", tokenizer_id="t"))

    def test_bad_window_rejected(self):
        for settings in ({"max_tokens": 0}, {"overlap_tokens": 500}, {"max_tokens": True}):
            with self.assertRaises((ValueError, TypeError)):
                emit("hello", **settings)

    def test_jsonl_order_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target, cli = root / "in.jsonl", root / "out.jsonl", root / "cli.jsonl"
            source.write_text("\n".join(json.dumps({"source_id": "s", "document_id": str(i),
                                                      "text": t}, ensure_ascii=False)
                                        for i, t in enumerate(["中文 🧬", "東京\n", " " ])) + "\n")
            self.assertEqual(convert(source, target, TOK, "test"), 2)
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual([r["document_id"] for r in rows], ["0", "1"])
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                convert(source, target, TOK, "test")
            self.assertEqual(target.read_bytes(), original)
            subprocess.run([sys.executable, str(Path(__file__).with_name("preserve_text_spans.py")),
                            str(source), str(cli), "--tokenizer-path", TOKENIZER,
                            "--tokenizer-id", "test"], check=True, capture_output=True)
            self.assertEqual(target.read_bytes(), cli.read_bytes())

    def test_failed_input_never_publishes_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "in", root / "out"
            source.write_text('{"source_id":"s","document_id":"d","text":"hello"}\nINVALID\n')
            with self.assertRaises(json.JSONDecodeError):
                convert(source, target, TOK, "test")
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_publication_race_preserves_other_output(self):
        import os
        link = os.link
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / "in", root / "out"
            source.write_text('{"source_id":"s","document_id":"d","text":"hello"}\n')
            def competitor(src, dst):
                Path(dst).write_text("other producer")
                return link(src, dst)
            with patch("preserve_text_spans.os.link", side_effect=competitor):
                with self.assertRaises(FileExistsError):
                    convert(source, target, TOK, "test")
            self.assertEqual(target.read_text(), "other producer")
            self.assertEqual(list(root.glob("*.partial")), [])


if __name__ == "__main__":
    unittest.main()
