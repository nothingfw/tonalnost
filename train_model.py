import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ======================================
# Конфигурация
# ======================================
MAX_LEN = 150
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/simple_model.pth"
TOKENIZER_PATH = "models/tokenizer.json"

print(f"Используем устройство: {DEVICE}")

# ======================================
# Нормализация
# ======================================
def normalize_text(text):
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"ё", "е", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"([!?…]){2,}", r"\1", text)
    return text.strip()


# ======================================
# Простой токенайзер
# ======================================
class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.UNK = "[UNK]"
        self.PAD = "[PAD]"

        if vocab is None:
            self.word2id = {self.PAD: 0, self.UNK: 1}
        else:
            self.word2id = vocab

    def build_vocab(self, texts, min_freq=2):
        freq = {}
        for t in texts:
            for word in t.split():
                freq[word] = freq.get(word, 0) + 1

        for word, cnt in freq.items():
            if cnt >= min_freq:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)

    def encode(self, text, max_len=128):
        tokens = text.split()
        ids = [self.word2id.get(t, self.word2id[self.UNK])
               for t in tokens]

        if len(ids) < max_len:
            ids += [self.word2id[self.PAD]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return SimpleTokenizer(vocab)


# ======================================
# Dataset
# ======================================
class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.texts[idx], MAX_LEN)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ======================================
# Модель
# ======================================
class SimpleSentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_labels=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = self.dropout(h[-1])
        return self.fc(h)


# ======================================
# Загрузка данных
# ======================================
df = pd.read_csv("train.csv")  # Требует столбцы text,label
df["text"] = df["text"].astype(str).apply(normalize_text)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# ======================================
# Токенайзер
# ======================================
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(train_texts)
tokenizer.save(TOKENIZER_PATH)
print(f"Словарь сохранён в {TOKENIZER_PATH}")

# ======================================
# Dataset / DataLoader
# ======================================
train_ds = ReviewsDataset(train_texts, train_labels, tokenizer)
val_ds = ReviewsDataset(val_texts, val_labels, tokenizer)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ======================================
# Модель
# ======================================
model = SimpleSentimentModel(len(tokenizer.word2id)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0
patience = 2
patience_counter = 0


# ======================================
# Обучение
# ======================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dl)

    # ---- Валидация ----
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}, Val Macro-F1: {macro_f1:.4f}")

    # ---- Ранняя остановка ----
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), MODEL_PATH)
        print("🔥 Лучшая модель сохранена!")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⛔ Ранняя остановка — качество больше не растёт.")
            break

print("Обучение завершено.")
print(f"Лучшая модель сохранена в {MODEL_PATH}")
