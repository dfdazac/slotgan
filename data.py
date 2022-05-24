from typing import Dict, Literal, List
import os.path as osp
import random

import tokenizers
from datasets import load_dataset, load_from_disk, Dataset
import torch
from torch.utils.data import DataLoader
import nltk
from nltk import sent_tokenize
from transformers import AutoTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

DATA_PATH = './data/'
DATASET_TEST = 'test'
DATASET_WIKIDATA5M = 'wikidata5m'
DATASET_CONLL = 'conll'
DATASETS = {DATASET_TEST, DATASET_WIKIDATA5M, DATASET_CONLL}
DATASETS_LITERAL = Literal[DATASET_TEST, DATASET_WIKIDATA5M, DATASET_CONLL]


class TransformersCollator:
    """A class to instantiate a tokenizer with a predefined maximum sequence
    length, implementing a collate_fn.

    Args:
        base_model (str): the base model used to instantiate a tokenizer
            (e.g. 'bert-base-cased')
        max_length (int): maximum length of loaded sentences. Longer sentences
            will be truncated.
        shift_names (bool): if True, sequences within a batch will be shifted
            randomly with padding added before and after, e.g.
            "[PAD] Luke [PAD] [PAD] [PAD]" could be shifted randomly to
            "[PAD] [PAD] [PAD] Luke [PAD]".
    """
    def __init__(self, base_model: str, max_length: int, shift_names: bool,
                 randomize_case: bool = False):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.shift_names = shift_names
        self.randomize_case = randomize_case

    @staticmethod
    def _randomize_case(examples: List[str]):
        sentences = []
        for sample in examples:
            if random.random() < 0.2:
                sample = sample.upper()
            sentences.append(sample)
        return sentences

    def collate_text_fn(self, examples: list):
        """Collate sentences by tokenizing and converting to integer IDs.

        Returns:
            input_ids (torch.Tensor): token int IDs
            attention_mask (torch.Tensor): mask containing 0.0 for padding
            to be ignored, and 1.0 otherwise.
        """
        sentences = [sample['text'] for sample in examples]
        if self.randomize_case:
            sentences = self._randomize_case(sentences)

        encoding = self.tokenizer(sentences, max_length=self.max_length,
                                  truncation=True, padding=True,
                                  return_tensors='pt',
                                  return_token_type_ids=False,
                                  add_special_tokens=True)

        return encoding.data['input_ids'], encoding.data['attention_mask']

    def collate_names_fn(self, examples: list):
        """Collate names by tokenizing and converting to integer IDs.
        If shift_names was set to True, names will be shifted randomly within
        the batch.

        Returns:
            input_ids (torch.Tensor): token int IDs
            attention_mask (torch.Tensor): mask containing 0.0 for padding
            to be ignored, and 1.0 otherwise.
        """
        names = [sample['text'] for sample in examples]
        if self.randomize_case:
            names = self._randomize_case(names)

        batch_size = len(names)

        encoding = self.tokenizer(names, max_length=self.max_length,
                                  truncation=True, padding=True,
                                  return_tensors='pt',
                                  return_token_type_ids=False,
                                  add_special_tokens=False)

        token_ids, mask = encoding['input_ids'], encoding['attention_mask']

        if self.shift_names:
            lengths = mask.sum(dim=-1)
            max_length = lengths.max()
    
            indices = torch.arange(max_length).expand(batch_size, -1)
            shifts = torch.floor(torch.rand(batch_size) * (max_length - lengths + 1))
            indices = (indices + shifts.long().unsqueeze(-1)) % max_length
    
            shifted_token_ids = torch.zeros_like(token_ids)
            shifted_token_ids.scatter_(1, indices, token_ids)
            mask = (shifted_token_ids > 0).float()
            token_ids = shifted_token_ids

        return token_ids, mask


def make_infinite_iterator(data_loader: DataLoader):
    """A generator that re-initializes a DataLoader once it has been
    traversed"""
    while True:
        for data in data_loader:
            yield data


def get_wikidata5m_dataset_pairs(sample: bool = False) -> tuple:
    """Get a a pair of PyTorch Datasets for sentences and entity names,
    respectively, extracted from the Wikidata5M dataset.

    Args:
        sample (bool): If False, look for the files
            entity2textlong.txt with entity descriptions, and entity2names.txt
            listing entity names and aliases. Otherwise, use a test sample.

    Returns:
        text_dataset (Dataset)
        names_dataset (Dataset)
    """
    def extract_wikidata5m_sentences(examples: Dict[str, list]) -> Dict:
        """Given all entries from the Wikidata5M descriptions dataset,
        create a new dataset where an instance is a sentence."""
        sentences = []
        for data in examples['text']:
            text = data[data.find('\t') + 1:].strip()
            sentences.extend(sent_tokenize(text))

        return {'text': sentences}

    def extract_wikidata5m_names(examples: Dict[str, list]) -> Dict:
        """Given all entries from the Wikidata5M names dataset,
        create a new dataset where an instance is a name."""
        names = []
        for data in examples['text']:
            text = data[data.find('\t') + 1:].strip()
            names.extend(text.split('\t'))

        return {'text': names}

    path = osp.join(DATA_PATH, 'Wikidata5M')
    if sample:
        descriptions_file = 'sample.txt'
        names_file = 'samplenames.txt'
    else:
        descriptions_file = 'entity2textlong.txt'
        names_file = 'entity2names-v2.txt'

    descriptions_path = osp.join(path, descriptions_file)
    names_path = osp.join(path, names_file)
    processed_text = osp.join(path, f'processed-{descriptions_file[:-4]}')
    processed_names = osp.join(path, f'processed-{names_file[:-4]}')

    try:
        text_dataset = load_from_disk(processed_text)
        print(f'Loading cached data from {processed_text}')
    except FileNotFoundError:
        text_dataset = load_dataset('text',
                                    data_files=descriptions_path,
                                    split='train')
        text_dataset = text_dataset.map(extract_wikidata5m_sentences,
                                        remove_columns=text_dataset.column_names,
                                        batched=True)
        text_dataset.save_to_disk(processed_text)

    try:
        names_dataset = load_from_disk(processed_names)
        print(f'Loading cached data from {processed_names}')
    except FileNotFoundError:
        names_dataset = load_dataset('text',
                                     data_files=names_path,
                                     split='train')
        names_dataset = names_dataset.map(extract_wikidata5m_names,
                                          remove_columns=names_dataset.column_names,
                                          batched=True)
        names_dataset.save_to_disk(processed_names)

    return text_dataset, names_dataset


def get_conll_dataset_pairs(split: str = 'train') -> tuple:
    """Get a a pair of PyTorch Datasets for sentences and entity names,
    respectively, extracted from the CoNLL 2003 NER dataset.

    Returns:
        text_dataset (Dataset)
        names_dataset (Dataset)
    """
    def extract_sentences_and_entities(examples: Dict[str, list]) -> Dict:
        """Extract mentions of entities from the CoNLL 2003 dataset using
        the provided NER tags."""
        sentences = []
        names = []

        for tokens, tag_ids in zip(examples['tokens'], examples['ner_tags']):
            sentences.append(' '.join(tokens))

            entities = []
            for i, tag_id in enumerate(tag_ids):
                tag = ner_tags[tag_id]
                if tag.startswith('B'):
                    entities.append(tokens[i])
                elif tag.startswith('I'):
                    entities[-1] = entities[-1] + f' {tokens[i]}'

            names.append(entities)

        return {'text': sentences, 'names': names}

    unique_names = set()
    def filter_duplicate_entities(examples: Dict[str, list]) -> Dict:
        """Update a set of unique names with names in a set of example."""
        unique_batch_names = set()
        batch_names = []
        for names_list in examples['names']:
            for name in names_list:
                if name not in unique_batch_names:
                    batch_names.append(name)
                    unique_batch_names.add(name)

        unique_names.update(unique_batch_names)

        return {'text': batch_names}

    dataset = load_dataset('conll2003', split=split)
    ner_tags = dataset.features['ner_tags'].feature.names
    sentences_and_names = dataset.map(extract_sentences_and_entities,
                                      batched=True,
                                      remove_columns=dataset.column_names)

    names_dataset = sentences_and_names.map(filter_duplicate_entities,
                                            batched=True,
                                            remove_columns=sentences_and_names.column_names)
    text_dataset = sentences_and_names.remove_columns('names')

    return text_dataset, names_dataset


def get_dataset_pairs(dataset: str) -> tuple:
    """Load a dataset from a predefined set.

    Args:
        dataset (str): name of the dataset

    Returns:
        text_dataset (Dataset)
        names_dataset (Dataset)
    """
    if dataset in {DATASET_TEST, DATASET_WIKIDATA5M}:
        sample = dataset == DATASET_TEST
        text_dataset, names_dataset = get_wikidata5m_dataset_pairs(sample)
    elif dataset == DATASET_CONLL:
        text_dataset, names_dataset = get_conll_dataset_pairs()
    else:
        raise ValueError(f'Unknown dataset {dataset}.'
                         f' Choose one of {DATASETS}.')

    print(f'Loaded {dataset}, containing {len(text_dataset):,} sentences'
          f' and {len(names_dataset):,} names.')

    return text_dataset, names_dataset


TAG_O = 'O'
TAG_B_PREFIX = 'B-'
TAG_I_PREFIX = 'I-'
TAG_MISC_SUFFIX = 'MISC'
TAG_B_MISC = TAG_B_PREFIX + TAG_MISC_SUFFIX
TAG_I_MISC = TAG_I_PREFIX + TAG_MISC_SUFFIX


def get_conll2003_data(tokenizer: tokenizers.Tokenizer, split: str) -> Dataset:
    """Load the CoNLL dataset for NER evaluation using a specific tokenizer.
    This is necessary since tokenizing might split a word,
    but we want to evaluate model outputs for the original word.
    In this case, we assign a label of -100 to extra wordpieces.

    For example, while originally we might have

    Label    PER     O          O     LOC
    Tokens   Sara    travels    to    Zandvoort

    After tokenizing, we obtain
    Label    PER     O          O      LOC  -100     -100    -100
    Tokens   Sara    travels    to     Z    ##and    ##vo    ##ort

    Args:
        tokenizer (tokenizers.Tokenizer): used to split words and convert tokens
            to ids
        split (str): one of 'training', 'valid', 'test'

    Returns:
        dataset (datasets.Dataset)
    """
    def discard_types(example: Dict) -> Dict:
        """For each example, add a new column where all NER tags of the form
        B-* and I-* are replaced by B-MISC and I-MISC, respectively."""
        example_tag_ids = example['ner_tags']
        entity_tags = []
        for tag_id in example_tag_ids:
            tag = tags_list[tag_id]
            if tag.startswith(TAG_B_PREFIX) or tag.startswith(TAG_I_PREFIX):
                entity_tags.append(tag_to_id[tag[:2] + TAG_MISC_SUFFIX])
            else:
                entity_tags.append(tag_id)

        example['ner_tags'] = entity_tags
        return example

    def tokenize_and_align_labels(examples: Dict[str, list]) -> Dict:
        """Add a new column containing integer token IDs for the tokens
        in the example, and a column for labels for each token.
        Labels for internal wordpieces (e.g. starting with ###) are given
        the special value -100 to be ignored in downstream tasks."""
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True,
                                     is_split_into_words=True,
                                     return_attention_mask=False,
                                     return_token_type_ids=False)
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens ([CLS], [SEP]) have a word id that is None
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first wordpiece of each word only
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # Ignore other wordpieces within a word
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = load_dataset('conll2003', split=split)
    tags_list = dataset.features['ner_tags'].feature.names
    tag_to_id = {tag: i for i, tag in enumerate(tags_list)}

    dataset = dataset.map(discard_types)
    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    dataset.features['labels'].feature = dataset.features['ner_tags'].feature

    features = dataset.features.keys()
    kept_features = {'input_ids', 'labels'}
    removed_columns = [f for f in features if f not in kept_features]
    dataset = dataset.remove_columns(removed_columns)

    return dataset
