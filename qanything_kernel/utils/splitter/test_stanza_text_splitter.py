import unittest
from unittest.mock import patch
import sys
sys.path.append(".")

from stanza_text_splitter import StanzaTextSplitter, normalize_whitespace, detect_chinese_script, is_meaningful_sentence

class TestStanzaTextSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = StanzaTextSplitter()

    def test_normalize_whitespace(self):
        test_cases = [
            ("This is a test.", "This is a test."),
            ("This  is  a  test.", "This is a test."),
            ("This\nis\na\ntest.", "This\nis\na\ntest."),
            ("This \n\r is \n\r a \r test.", "This\nis\na\ntest."),
            (" This is a test. ", "This is a test."),
        ]
        for input_text, expected_output in test_cases:
            self.assertEqual(normalize_whitespace(input_text), expected_output)

    def test_detect_chinese_script(self):
        test_cases = [
            ("这是一个简体中文测试。", "zh-hans"),
            ("這是一個繁體中文測試。", "zh-hant"),
            ("這是。简体", "zh-hans"),
        ]
        for input_text, expected_output in test_cases:
            self.assertEqual(detect_chinese_script(input_text), expected_output)

    def test_is_meaningful_sentence(self):
        test_cases = [
            ("This is a meaningful sentence.", "en", True),
            ('"..2.,,,', "en", False),
            ("这是一个有意义的句子。", "zh", True),
            ("：子。。。.", "zh", False),
        ]
        for input_text, lang, expected_output in test_cases:
            self.assertEqual(is_meaningful_sentence(input_text, lang), expected_output)

    def test_split_text(self):
        test_cases = [
            ("This is a test. This is another test.", ["This is a test.", "This is another test."]),
            ("这是一个测试。这是另一个测试。", ["这是一个测试。", "这是另一个测试。"]),
        ]
        for input_text, expected_output in test_cases:
            self.assertEqual(self.splitter.split_text(input_text), expected_output)

    def test_split_text_pdf(self):
        splitter = StanzaTextSplitter(pdf=True)
        input_text = "This\n\n\nis\na\ntest.\n\n\nThis\nis\nanother\ntest."
        expected_output = ["This is a test.", "This is another test."]
        self.assertEqual(splitter.split_text(input_text), expected_output)

if __name__ == "__main__":
    unittest.main()