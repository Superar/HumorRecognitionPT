import pickle
from argparse import Namespace
from logging import getLogger
from pathlib import Path

import datasets
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

logger = getLogger('HumorRecognitionPT')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_parser(subparsers):
    transformer_parser = subparsers.add_parser('transformer')
    transformer_subparser = transformer_parser.add_subparsers()

    # fine-tune
    parser_finetune = transformer_subparser.add_parser('fine-tune')
    parser_finetune.add_argument('--input', '-i',
                                 help='Training data in JSON format.',
                                 required=True, type=Path)
    parser_finetune.add_argument('--output', '-o',
                                 help='Directory path to save the model.',
                                 required=True, type=Path)
    parser_finetune.add_argument('--model', '-m',
                                 help='HuggingFace model to be used.',
                                 required=False, type=str,
                                 default='neuralmind/bert-base-portuguese-cased')
    parser_finetune.set_defaults(command=train)

    # test
    parser_test = transformer_subparser.add_parser('test')
    parser_test.add_argument('--input', '-i',
                             help='Test data in JSON format.',
                             required=True, type=Path)
    parser_test.add_argument('--model', '-m',
                             help='Model directory path.',
                             required=False, type=str,
                             default='neuralmind/bert-base-portuguese-cased')
    parser_test.add_argument('--output', '-o',
                             help='Path to the file to save the predictions in JSON format.',
                             required=False, type=Path,
                             default=None)
    parser_test.set_defaults(command=test)


def train(args: Namespace):
    logger.info(f'Reading corpus: {args.input}')
    corpus = pd.read_json(args.input)
    logger.debug(f'Corpus\n\n{corpus}')

    logger.info(f'Preparing data')
    labels = corpus['Label'].unique()
    num_labels = len(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {label2id[l]: l for l in labels}
    corpus['Label'] = corpus['Label'].map(label2id)
    logger.debug(f'Mapped labels\n\n{corpus["Label"]}')

    data = datasets.Dataset.from_pandas(corpus[['Text', 'Label']])
    data = data.rename_column('Label', 'label')
    logger.debug(f'Data\n\n{data}')

    # Tokenization
    logger.info(f'Loading tokenizer from model: {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(example):
        return tokenizer(example['Text'], truncation=True)

    tokenized_data = data.map(tokenize_function, batched=True)
    logger.debug(f'Batch created\n\n{tokenized_data}')

    # Fine-tuning
    logger.info(f'Creating classification model from model: {args.model}')
    training_args = TrainingArguments(args.output)
    model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                               num_labels=num_labels,
                                                               label2id=label2id,
                                                               id2label=id2label)
    logger.info(f'Model created')
    logger.info(f'Fine-tuning model')
    trainer = Trainer(model,
                      training_args,
                      train_dataset=tokenized_data,
                      data_collator=data_collator,
                      tokenizer=tokenizer)
    trainer.train()


def test(args: Namespace):
    logger.info(f'Loading file {args.input}')
    corpus = pd.read_json(args.input)
    logger.debug(f'Corpus\n\n{corpus}')

    logger.info(f'Loading tokenizer from model: {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info(f'Loading model: {args.model}')
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    text_classification = pipeline('text-classification',
                                   model=model,
                                   tokenizer=tokenizer,
                                   device=device.index)

    logger.info('Making predictions')
    predictions = text_classification(corpus['Text'].to_list())
    results = pd.DataFrame(predictions, index=corpus.index)
    results = results.drop(columns='score')
    results = results.rename(columns={'label': 'Prediction'})
    results['Label'] = corpus['Label']
    logger.debug(f'Predictions\n\n{results}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving predictions to {args.output}')
        results.to_json(args.output, force_ascii=False, indent=4)
        logger.info('Predictions saved')
