import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv("data/zends_customer_queries_.csv")

df["intent_code"] = df["intent"].astype("category").cat.codes

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["intent_code"], test_size=0.2
)

train_enc = tokenizer(list(X_train), truncation=True, padding=True)

class Dataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k:v[idx] for k,v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_ds = Dataset(train_enc, list(y_train))

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
)

args = TrainingArguments(
    output_dir="models/intent_model",
    per_device_train_batch_size=8,
    num_train_epochs=2
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()

model.save_pretrained("models/intent_model")
tokenizer.save_pretrained("models/intent_model")

print("MODEL TRAINED")
