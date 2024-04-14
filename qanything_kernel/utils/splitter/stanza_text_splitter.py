from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
import stanza
import torch
from ftlangdetect import detect
from opencc import OpenCC
from difflib import SequenceMatcher
import random

def normalize_whitespace(text):
    # 将连续的非换行符的空字符转为一个空格
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    # 将包含换行符的连续空字符转换为一个换行符
    text = re.sub(r'\s*(\r\n|\r|\n)\s*', r'\n', text)
    # 移除首尾的空字符
    text = text.strip()
    return text

def detect_chinese_script(text, sample_size=4096):
    # 创建转换器
    s2t = OpenCC('s2t')
    t2s = OpenCC('t2s')

    # 如果文本长度大于设定的样本大小，则进行随机采样
    if len(text) > sample_size:
        sample_indexes = random.sample(range(len(text)), sample_size)
        sampled_text = ''.join([text[i] for i in sorted(sample_indexes)])
    else:
        sampled_text = text

    # 对采样后的文本进行简繁转换
    simplified_text = t2s.convert(sampled_text)
    traditional_text = s2t.convert(sampled_text)

    # 计算转换后与原文本的相似度
    simplified_ratio = SequenceMatcher(None, sampled_text, simplified_text).ratio()
    traditional_ratio = SequenceMatcher(None, sampled_text, traditional_text).ratio()

    # 根据相似度判断文本是简体还是繁体
    if traditional_ratio < simplified_ratio:
        return "zh-hans"
    else:
        return "zh-hant"

def is_meaningful_sentence(sentence, lang):
    if lang == 'en':
        # 对于英语,移除句子中的标点符号和空字符(包括换行符)
        sentence_clean = re.sub(r'[^\w]', '', sentence)
        return len(sentence_clean) > 1
    else:
        # 对于非英语语言,移除句子中的空字符(包括换行符)和常见符号
        symbols = r'[（）：，。、；·•�-‐—―‖…‰※℃℉№○〇〔〕〖〗《》「」『』【】〖〗〘〙〚〛〜・�"\'\(\)\[\]\{\}\<\>/,\.;:!?@#$%\^&\*\+-=\_\|\\\`\~]'
        sentence_clean = re.sub(r'[\s{}]+'.format(re.escape(symbols)), '', sentence)
        return len(sentence_clean) > 1

class StanzaTextSplitter(CharacterTextSplitter):
    pipelines = {}

    def __init__(self, pdf: bool = False, sentence_size: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    @classmethod
    def get_pipeline(cls, lang):
        if lang not in cls.pipelines:
            stanza.download(lang)
            cls.pipelines[lang] = stanza.Pipeline(lang, processors='tokenize', use_gpu=torch.cuda.is_available())
        return cls.pipelines[lang]

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        detected_lang = detect(text.replace('\n', ' ').replace('\r', ' '))['lang']
        if detected_lang == 'zh':
            detected_lang = detect_chinese_script(text)

        nlp = self.get_pipeline(detected_lang)
        doc = nlp(text)

        ls = []
        for sentence in doc.sentences:
            s = normalize_whitespace(sentence.text)
            if is_meaningful_sentence(s, detected_lang):
                ls.append(s)
        return ls
