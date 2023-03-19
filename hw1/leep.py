import os

import torch
from absl import app, flags
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)
from tqdm import tqdm
FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt", None, "Checkpoint to load")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("num_workers", 4, "Number of workers for dataloader")

def leep(model, num_source_labels, num_target_labels, target_dataloader):
    p_yz = torch.zeros(num_target_labels, num_source_labels)
    num_examples = 0
    with torch.no_grad():
        for batch in tqdm(target_dataloader):
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu()
            for idx, l_y in enumerate(batch["labels"]):
                p_yz[l_y] += probs[idx]
                num_examples += 1
    p_yz = p_yz / num_examples
    p_y_z = p_yz / p_yz.sum(dim=0, keepdim=True)
    leep_score = 0
    with torch.no_grad():
        for batch in tqdm(target_dataloader):
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu()
            leep_score += torch.log((probs * p_y_z[batch["labels"].cpu()]).sum(dim=1)).sum()
    leep_score = leep_score / num_examples
    return leep_score


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
    ckpt = torch.load(FLAGS.ckpt)
    mnli_classifier.load_state_dict(ckpt["model_state_dict"])
    mnli_classifier.eval()
    for p in mnli_classifier.parameters():
        p.requires_grad = False

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

    trgt_genres = ["government", "travel"]

    trgt_train = dataset["train"].filter(lambda x: x["genre_c"] in trgt_genres)

    trgt_train = trgt_train.map(lambda x: x, batched=True, remove_columns=["genre_c"])
    trgt_train = trgt_train.train_test_split(test_size=0.1, seed=84818)["test"]

    trgt_data_loader = DataLoader(
        trgt_train,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    leep_score = leep(mnli_classifier, 3, 3, trgt_data_loader)
    print(f"LEEP score: {leep_score:.2f}")


if __name__ == "__main__":
    app.run(main)
