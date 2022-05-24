from pprint import pprint
from collections import defaultdict
from argparse import ArgumentParser

from datasets import load_dataset, load_metric
from tqdm import tqdm
import numpy as np

from data import get_conll_dataset_pairs
from utils.span import get_overlap_metrics

TAG_O_ID = 0
TAG_B_ID = 1
TAG_I_ID = 2
ID_TO_TAG = {0: 'O', 1: 'B-ENT', 2: 'I-ENT'}

parser = ArgumentParser()
parser.add_argument('--case_insensitive', '-i', action='store_true')
args = parser.parse_args()
case_insensitive = args.case_insensitive

# Get a set with all entities in the training set of ConLL
_, names_dataset = get_conll_dataset_pairs('train')
names_set = set()
max_name_length = -1
for sample in names_dataset:
    sample_name = sample['text']
    if case_insensitive:
        sample_name = sample_name.lower()
    names_set.add(sample_name)
    if len(sample_name) > max_name_length:
        max_name_length = len(sample_name)

# Iterate over sentences in the CoNLL test set
dataset = load_dataset('conll2003', split='test')
tags_list = dataset.features['ner_tags'].feature.names
metric = load_metric('seqeval')

accumulated_overlap_metrics = defaultdict(float)
mention_count = 0

for sample in tqdm(dataset, desc='Annotating via string matching'):
    words = sample['tokens']
    sentence_length = len(words)
    assigned_tag_ids = np.zeros(sentence_length)

    # Iterate over all possible spans
    for span_length in range(min(sentence_length, max_name_length), 0, -1):
        for span_start in range(sentence_length - span_length + 1):
            span_end = span_start + span_length
            span = words[span_start:span_end]
            candidate = ' '.join(span)
            if case_insensitive:
                candidate = candidate.lower()

            # Check first that the span doesn't contain a tag already
            span_tags = assigned_tag_ids[span_start:span_end]
            if span_tags.sum() > 0:
                continue

            # Assign tags via string matching
            if candidate in names_set:
                assigned_tag_ids[span_start] = TAG_B_ID
                assigned_tag_ids[span_start + 1:span_end] = TAG_I_ID

    # Drop span types
    assigned_tags = [ID_TO_TAG[i] for i in assigned_tag_ids]
    gold_tags = []
    for tag_id in sample['ner_tags']:
        gold_tag = tags_list[tag_id]
        if gold_tag.startswith('O'):
            gold_tags.append(gold_tag)
        else:
            gold_tags.append(f'{gold_tag[:2]}ENT')

    metric.add(prediction=assigned_tags, reference=gold_tags)
    
    overlap_metrics = get_overlap_metrics(assigned_tags, gold_tags)
    num_mentions = 0
    for name, values in overlap_metrics.items():
        accumulated_overlap_metrics[name] += sum(values)
        num_mentions = len(values)
    mention_count += num_mentions

pprint(metric.compute())
for name, value in accumulated_overlap_metrics.items():
    print(f'{name}: {value / mention_count:.3f} over {mention_count} mentions')
