import os
from datetime import datetime

import torch
import torch.optim as optim
from absl import app, flags
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("num_epochs", 3, "Number of epochs")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate")
flags.DEFINE_integer("num_workers", 4, "Number of workers for dataloader")


def train(model, train_dataloader, optimizer, num_epochs):
    for epoch in range(1, 1 + num_epochs):
        model.train()
        for batch in tqdm(
            train_dataloader, desc=f"Training epoch {epoch}", leave=False
        ):
            optimizer.zero_grad()
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()


@torch.no_grad()
def test(model, val_dataloader):
    model.eval()
    total = 0
    correct = 0
    for batch in tqdm(val_dataloader, desc=f"Testing", leave=False):
        for key in batch:
            batch[key] = batch[key].cuda()
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        correct += (batch["labels"] == predictions).sum()
        total += len(batch["labels"])
    return correct / total


def main(argv):
    os.environ["HF_DATASETS_CACHE"] = "/cmlscratch/pkattaki/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "bert-base-cased"
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mnli_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    mnli_classifier.cuda()

    def preprocess(examples):
        texts = (examples["premise"], examples["hypothesis"])

        result = tokenizer(
            *texts, padding="max_length", max_length=128, truncation=True
        )
        result["labels"] = examples["label"]
        result["genre_c"] = examples["genre"]

        return result

    dataset = load_dataset("multi_nli")
    dataset = dataset.map(
        preprocess,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=dataset["train"].column_names,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in mnli_classifier.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in mnli_classifier.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=FLAGS.learning_rate)

    src_genres = ["telephone", "fiction"]
    trgt_genres = ["government", "travel"]

    src_train = dataset["train"].filter(lambda x: x["genre_c"] in src_genres)
    src_val = dataset["validation_matched"].filter(lambda x: x["genre_c"] in src_genres)

    src_train = src_train.map(lambda x: x, batched=True, remove_columns=["genre_c"])
    src_val = src_val.map(lambda x: x, batched=True, remove_columns=["genre_c"])

    trgt_train = dataset["train"].filter(lambda x: x["genre_c"] in trgt_genres)
    trgt_val = dataset["validation_matched"].filter(
        lambda x: x["genre_c"] in trgt_genres
    )

    trgt_train = trgt_train.map(lambda x: x, batched=True, remove_columns=["genre_c"])
    trgt_val = trgt_val.map(lambda x: x, batched=True, remove_columns=["genre_c"])

    train_dataloader = DataLoader(
        src_train,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    src_val_data_loader = DataLoader(
        src_val,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    trgt_val_data_loader = DataLoader(
        trgt_val,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    print("Training baseline model on source domain")
    now = datetime.now()
    train(mnli_classifier, train_dataloader, optimizer, FLAGS.num_epochs)
    print(f"Training time: {datetime.now() - now}")
    accuracy = test(mnli_classifier, src_val_data_loader)
    print(f"Source domain accuracy: {100 * accuracy:.2f}%")
    accuracy = test(mnli_classifier, trgt_val_data_loader)
    print(f"Val domain accuracy: {100 * accuracy:.2f}%")

    torch.save(
        {
            "model_state_dict": mnli_classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "ckpts/mnli_classifier_baseline.pt",
    )

    trgt_train_small = trgt_train.train_test_split(test_size=0.1, seed=84818)["test"]
    train_dataloader = DataLoader(
        trgt_train_small,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    print("Finetuning on target domain")
    now = datetime.now()
    train(mnli_classifier, train_dataloader, optimizer, FLAGS.num_epochs)
    print(f"Training time: {datetime.now() - now}")
    accuracy = test(mnli_classifier, src_val_data_loader)
    print(f"Source domain accuracy: {100 * accuracy:.2f}%")
    accuracy = test(mnli_classifier, trgt_val_data_loader)
    print(f"Target domain accuracy: {100 * accuracy:.2f}%")

    torch.save(
        {
            "model_state_dict": mnli_classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "ckpts/mnli_classifier_finetuned.pt",
    )


if __name__ == "__main__":
    app.run(main)
