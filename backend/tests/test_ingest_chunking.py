import unittest

from rag.chain import extract_answer_and_sources, normalize_answer_text
from rag.ingest import split_text_into_chunks


class IngestChunkingTests(unittest.TestCase):
    def test_split_text_into_chunks_respects_limit(self):
        text = (
            "This is a long document paragraph intended to be split into smaller pieces. "
            * 30
        )

        chunks = split_text_into_chunks(text, chunk_size=120, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_normalize_answer_text_formats_responses_cleanly(self):
        cleaned = normalize_answer_text(
            "Sure! Here is the answer:\nThe feature uses chunking and retrieval.\n\nSources: sample.pdf"
        )

        self.assertEqual(cleaned, "The feature uses chunking and retrieval.")

    def test_extract_answer_and_sources_parses_sources_from_model_output(self):
        answer, sources = extract_answer_and_sources(
            "Answer: The feature uses chunking and retrieval.\nSources: sample.pdf, docs.md"
        )

        self.assertEqual(answer, "The feature uses chunking and retrieval.")
        self.assertEqual(sources, ["sample.pdf", "docs.md"])


if __name__ == "__main__":
    unittest.main()
