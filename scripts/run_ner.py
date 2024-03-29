#!/usr/bin/env python
# coding: utf-8

# Added by Jinja template to the final script -----

import sys

if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    config_file = sys.argv[1]
else:
    raise Exception("Must provide a .yaml file as the only parameter")
    
# -----



# # tfBERT: Finetuned BERT model for NER of transcription factors and target genes

# The code in this file finetunes a BERT model for Named Entity Recognition. Jupyter cells with the _remove_cell_ tag will be ignored when using the custom Jinja template included in this project for `nbconvert`, used to convert the Python notebook to a Python script file.

# ## Import packages

# The impots Jupyter cell has a tag of _imports_, which is important for the Jinja template as it will add the `argparse` package below such cell code.

# In[ ]:


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
import random
import torch
from datasets import ClassLabel, load_dataset, Sequence
from pyhere import here  # type: ignore

import transformers
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint



# ## Setup variables and create logger

# In[ ]:



# Define torch device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



# In[ ]:



# Set seeds
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if device == torch.device("mps"):
    torch.mps.manual_seed(seed)
elif device == torch.device("cuda"):
    torch.cuda.manual_seed_all(seed)



# In[ ]:



# Create logger
logger = logging.getLogger(__name__)



# ## Custom classes

# In[ ]:



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str
    cache_dir: Optional[str] = field(default=None)
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    text_column_name: str
    label_column_name: str
    max_seq_length: int
    
    # split_text_into_tokens: bool # will use .split() for text and labels column # TO REMOVE

    dataset_name: Optional[str] = field(default=None)
    load_script: Optional[str] = field(default=None)

    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)

    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    label_all_tokens: bool = field(default=False)

    return_entity_level_metrics: bool = field(default=False)

    metric: str = field(default="seqeval")
    



# ## Custom methods

# In[ ]:



# Tokenize all texts and align the labels with them
def tokenize_and_align_labels(examples):
    
    tokenized_inputs = tokenizer(
        examples[data_args.text_column_name],
        truncation = True,
        max_length = data_args.max_seq_length,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples[data_args.label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index = i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the data_args.label_all_tokens flag
            else:
                if data_args.label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



# In[ ]:



def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list



# In[ ]:



def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 2)

    # Remove  ignored idex (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }



# ## Main

# ### Load arguments

# In[ ]:



parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_yaml_file(config_file)



# ### Setup logger

# In[ ]:



logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M/%S",
    handlers = [logging.StreamHandler(sys.stdout)]
)

if training_args.should_log:
    # The default of training_args.log_level is passive (defaults to warning), so we set log level at info here to have that default
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()



# In[ ]:



logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
)

logger.info(f"Training/evaluation parameters {training_args}")



# ### Loading the dataset

# First, we load the dataset either form Hugging Face or local files.

# In[ ]:



logger.info("### Loading the dataset ###")

# If dataset name from Hugging Face is provided
if data_args.dataset_name is not None:
    raw_datasets = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir)

# Otherwise, import from local files
else:
    data_files = {}
    if data_args.train_file:
        data_files["train"] = data_args.train_file
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file:
        data_files["test"] = data_args.test_file

    if data_args.load_script == "csv":
        raw_datasets = load_dataset("csv", data_files=data_files, sep="\t")
    elif data_args.load_script == "arrow":
        raw_datasets = load_dataset("arrow", data_files=data_files)

logger.info(f"Loaded dataset(s):\n{raw_datasets}")



# Create a label list for the dataset.

# In[ ]:



column_names = raw_datasets["train"].column_names # type: ignore
features = raw_datasets["train"].features # type: ignore



# In[ ]:



labels_are_int = isinstance(features[data_args.label_column_name].feature, ClassLabel)
if labels_are_int:
    label_list = features[data_args.label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list: [str] = get_label_list(raw_datasets["train"][data_args.label_column_name]) # type: ignore
    label_to_id = {l: i for i, l in enumerate(label_list)}

num_labels = len(label_list)

logger.info(f"Number of labels in dataset: {num_labels}")



# ### Load pretrained model and tokenizer

# Initialize the model and move it to the device (e.g. GPU).

# In[ ]:



logger.info("### Loading pretrained model and tokenizer ###")

# Load model with a dropout probability for the classifier head
model = AutoModelForTokenClassification.from_pretrained(
    model_args.model_name,
    num_labels = num_labels,
    classifier_dropout = 0.1,
    cache_dir=model_args.cache_dir
)

model.to(device)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()



# Initialize the tokenizer.

# In[ ]:



tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)



# In[ ]:



# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = dict(enumerate(label_list))



# Create an alternative to `label_list` were "B-" tags are replaced by their corresponding "I-" tags.

# In[ ]:



# Map that sends B-Xxx label to its I-Xxx counterpart
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and (label.replace("B-", "I-") in label_list):
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)



# ### Preprocessing the dataset

# Tokenize all texts and align labels and tokens.

# In[ ]:



logger.info("### Pre-processing the dataset ###")

# Tokenize train dataset
if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Running tokenizer on train dataset"
    )

# Tokenize validation dataset
if training_args.do_eval:
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requieres a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Runing tokenizer on validation dataset"
    )

# TODO: Do the same for do_predict



# Initialize the data collator and metrics to be computed.

# In[ ]:



# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True, pad_to_multiple_of=None)

# Metrics
metric = evaluate.load(data_args.metric, cache_dir=model_args.cache_dir)



# ### Training

# Initialize the Hugging Face `Trainer`.

# In[ ]:



# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=6)]
)



# If the argument `do_train` is set to `True` in the user's config file, then training is performed. First, we look for training checkpoints in the output directory. If there are none, then training starts anew.

# In[ ]:



if training_args.do_train:
    logger.info("### Training ###")
    last_checkpoint: Optional[str] = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Obtain metrics
    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()



# ### Evaluation

# If the argument `do_eval` is set to `True` in the user's config file, then evaluation is performed.

# In[ ]:



if training_args.do_eval:
    logger.info("### Evaluation ###")
    metrics = trainer.evaluate()


    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



# In[ ]:



# TODO: Save metrics and do the same for do_predict


