import os
from datetime import datetime

import evaluate
import numpy as np
import torch
import torch.optim as optim
from absl import app, flags
from accelerate import Accelerator
from baselines import test, train
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, default_data_collator)
from utils import postprocess_qa_predictions

FLAGS = flags.FLAGS

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def train_qa(model, train_dataloader, optimizer, num_epochs):
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
def test_qa(model, squad_val_examples, squad_val, val_dataloader, answer_column_name):
    model.eval()
    accelerator = Accelerator()

    metric = evaluate.load("squad_v2")

    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(
                accelerator.gather_for_metrics(start_logits).cpu().numpy()
            )
            all_end_logits.append(
                accelerator.gather_for_metrics(end_logits).cpu().numpy()
            )

    max_len = max(
        [x.shape[1] for x in all_start_logits]
    )  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(
        all_start_logits, squad_val, max_len
    )
    end_logits_concat = create_and_fill_np_array(all_end_logits, squad_val, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    n_best_size = 20
    version_2_with_negative = True
    max_answer_length = 30
    null_score_diff_threshold = 0.0
    output_dir = "outputs"

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=version_2_with_negative,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
            output_dir=output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(squad_val_examples, squad_val, outputs_numpy)
    eval_metric = metric.compute(
        predictions=prediction.predictions, references=prediction.label_ids
    )
    return eval_metric
    


def main(argv):
    os.environ["HF_DATASETS_CACHE"] = "/cmlscratch/pkattaki/datasets"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "bert-base-cased"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mrc_model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    mnli_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )
    mrc_model.bert = mnli_classifier.bert
    mnli_classifier.cuda()
    mrc_model.cuda()

    def preprocess_mnli(examples):
        texts = (examples["premise"], examples["hypothesis"])

        result = tokenizer(
            *texts, padding="max_length", max_length=128, truncation=True
        )
        result["labels"] = examples["label"]
        result["genre_c"] = examples["genre"]

        return result

    dataset = load_dataset("multi_nli")
    dataset = dataset.map(
        preprocess_mnli,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=dataset["train"].column_names,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in mrc_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in mrc_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=FLAGS.learning_rate)

    squad_train = load_dataset("squad_v2", split="train")
    squad_val_examples = load_dataset("squad_v2", split="validation")

    column_names = squad_train.column_names
    pad_on_right = True
    max_seq_length = 384
    doc_stride = 128
    pad_to_max_length = True

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Training preprocessing
    def preprocess_squad_train(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def preprocess_squad_validation(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    squad_train = squad_train.map(
        preprocess_squad_train, batched=True, remove_columns=column_names
    )
    squad_val = squad_val_examples.map(
        preprocess_squad_validation, batched=True, remove_columns=column_names
    )

    train_dataloader = DataLoader(
        squad_train,
        shuffle=True,
        batch_size=FLAGS.batch_size,
        collate_fn=default_data_collator,
    )
    val_dataloader = DataLoader(
        squad_val, batch_size=FLAGS.batch_size, collate_fn=default_data_collator
    )

    train_qa(mrc_model, train_dataloader, optimizer, 1)
    # accuracy = test_qa(mrc_model, squad_val_examples, squad_val, val_dataloader, answer_column_name)
    # print(f"Accuracy on MRC val: {accuracy}")

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
    trgt_train_small = trgt_train.train_test_split(test_size=0.1, seed=84818)["test"]
    
    train_dataloader = DataLoader(
        trgt_train_small,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    val_data_loader = DataLoader(
        trgt_val,
        collate_fn=default_data_collator,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
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

    print("Finetuning on target domain")
    now = datetime.now()
    train(mnli_classifier, train_dataloader, optimizer, FLAGS.num_epochs)
    print(f"Training time: {datetime.now() - now}")
    accuracy = test(mnli_classifier, val_data_loader)
    print(f"Val domain accuracy: {100 * accuracy:.2f}%")


if __name__ == "__main__":
    app.run(main)
