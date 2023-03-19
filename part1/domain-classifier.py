import os
from datetime import datetime

import torch
import torch.optim as optim
from absl import app, flags
from baselines import test, train
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

FLAGS = flags.FLAGS


def main(argv):
    os.environ["HF_DATASETS_CACHE"] = "/cmlscratch/pkattaki/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "bert-base-cased"
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    domain_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    domain_classifier.cuda()

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
                for n, p in domain_classifier.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in domain_classifier.named_parameters()
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
    trgt_train = trgt_train.train_test_split(test_size=0.1, seed=84818)["test"]
    trgt_val = trgt_val.map(lambda x: x, batched=True, remove_columns=["genre_c"])

    temp_src_train = src_train.remove_columns("labels")
    temp_src_train = temp_src_train.add_column("labels", [0] * len(temp_src_train))
    temp_trgt_train = trgt_train.remove_columns("labels")
    temp_trgt_train = temp_trgt_train.add_column("labels", [1] * len(temp_trgt_train))
    train_dataset = concatenate_datasets([temp_src_train, temp_trgt_train])

    temp_src_val = src_val.remove_columns("labels")
    temp_src_val = temp_src_val.add_column("labels", [0] * len(temp_src_val))
    temp_trgt_val = trgt_val.remove_columns("labels")
    temp_trgt_val = temp_trgt_val.add_column("labels", [1] * len(temp_trgt_val))
    val_dataset = concatenate_datasets([temp_src_val, temp_trgt_val])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )
    val_data_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    print("Training baseline model on source domain")
    now = datetime.now()
    train(domain_classifier, train_dataloader, optimizer, FLAGS.num_epochs)
    print(f"Training time: {datetime.now() - now}")
    accuracy = test(domain_classifier, val_data_loader)
    print(f"Validation accuracy: {100 * accuracy:.2f}%")

    torch.save(
        {
            "model_state_dict": domain_classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "ckpts/domain_classifier.pt",
    )


if __name__ == "__main__":
    app.run(main)
