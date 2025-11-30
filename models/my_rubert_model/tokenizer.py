import re
import json
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab=None, unk_token="[UNK]", pad_token="[PAD]"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        if vocab is None:
            self.vocab = {pad_token:0, unk_token:1}
        else:
            self.vocab = vocab
        self.inv_vocab = {v:k for k,v in self.vocab.items()}

    def build_vocab(self, texts, max_size=10000, min_freq=1):
        counter = Counter()
        for t in texts:
            tokens = self.tokenize(t)
            counter.update(tokens)
        most_common = [w for w,f in counter.most_common(max_size) if f>=min_freq]
        for idx, word in enumerate(most_common, start=len(self.vocab)):
            self.vocab[word] = idx
        self.inv_vocab = {v:k for k,v in self.vocab.items()}

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zа-яё0-9]+", " ", text)
        tokens = text.strip().split()
        return tokens

    def encode(self, text, max_len=50):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]
        if len(ids) < max_len:
            ids += [self.vocab[self.pad_token]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
