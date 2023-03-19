import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags
from baselines import test
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, default_data_collator, get_scheduler)

FLAGS = flags.FLAGS


def train(model, train_dataloader, optimizer, scheduler, num_epochs):
    for epoch in range(1, 1 + num_epochs):
        model.train()
        for batch in tqdm(
            train_dataloader, desc=f"Training epoch {epoch}", leave=False
        ):
            optimizer.zero_grad()
            importance_weights = batch.pop("importance_weights")
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            logits = outputs.logits
            losses = F.cross_entropy(logits, batch["labels"], reduction="none")
            loss = torch.inner(losses, importance_weights.cuda())
            loss.backward()

            optimizer.step()
            scheduler.step()


def main(argv):
    os.environ["HF_DATASETS_CACHE"] = "/cmlscratch/pkattaki/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "bert-base-cased"

    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    domain_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    ckpt = torch.load("ckpts/domain_classifier.pt")
    domain_classifier.cuda()
    domain_classifier.load_state_dict(ckpt["model_state_dict"])
    domain_classifier.eval()

    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mnli_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    ckpt = torch.load("ckpts/mnli_classifier_baseline.pt")
    mnli_classifier.cuda()
    mnli_classifier.load_state_dict(ckpt["model_state_dict"])

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

    train_dataset = src_train

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    importance_weights = []
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Computing importance weights"):
            del batch["labels"]
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = domain_classifier(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
            probs = probs / (1 - probs)
            importance_weights.extend(probs)
    train_dataset = train_dataset.add_column("importance_weights", importance_weights)

    train_dataloader = DataLoader(
        train_dataset,
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

    num_training_steps = len(train_dataloader) * FLAGS.num_epochs
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print("Training baseline model on source domain")
    now = datetime.now()
    train(mnli_classifier, train_dataloader, optimizer, scheduler, FLAGS.num_epochs)
    print(f"Training time: {datetime.now() - now}")
    accuracy = test(mnli_classifier, src_val_data_loader)
    print(f"Source domain accuracy: {100 * accuracy:.2f}%")
    accuracy = test(mnli_classifier, trgt_val_data_loader)
    print(f"Target domain accuracy: {100 * accuracy:.2f}%")

    torch.save(
        {
            "model_state_dict": mnli_classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "ckpts/mnli_classifier_reweighted.pt",
    )


if __name__ == "__main__":
    app.run(main)
