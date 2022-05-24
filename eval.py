from argparse import ArgumentParser
from collections import defaultdict
import os.path as osp
from pprint import pprint

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import transformers
import datasets
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

import data
from models import TokenEmbedding, EntityExtractor, ConvolutionalDiscriminator
from utils.span import get_overlap_metrics


class CoNLLEvaluator:
    def __init__(self, split, tokenizer, batch_size, device, plot=False,
                 instance_eval=False, instance_eval_file=None):
        self.tokenizer = tokenizer

        self.dataset =  data.get_conll2003_data(tokenizer, split)

        self.tags_list = self.dataset.features['labels'].feature.names
        self.tags_array = np.array(self.tags_list)
        self.tag_idx_O = self.tags_list.index(data.TAG_O)
        self.tag_idx_B = self.tags_list.index(data.TAG_B_MISC)
        self.tag_idx_I = self.tags_list.index(data.TAG_I_MISC)

        collator = transformers.DataCollatorForTokenClassification(tokenizer)
        self.loader = DataLoader(self.dataset, batch_size=batch_size,
                                 collate_fn=collator)
        self.plot = plot
        self.device = device
        self.metrics = datasets.load_metric('seqeval')

        if self.plot:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(11, 11))

        self.instance_eval = instance_eval
        if self.instance_eval and instance_eval_file is None:
                raise ValueError('If instance_eval is True,'
                                 ' instance_eval_file must be provided.')
        self.instance_eval_file = instance_eval_file

    def _plot_example(self, hard_mask, input_ids, true_tags):
        slot_sample = hard_mask.cpu().numpy().T
        self.ax.imshow(slot_sample)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        labeled_tokens = []
        for token, tag_id in zip(tokens, true_tags):
            tag = ''
            if tag_id == 0:
                tag = ' [O]'
            elif tag_id == 7:
                tag = ' [B]'
            elif tag_id == 8:
                tag = ' [I]'

            labeled_tokens.append(token + tag)

        self.ax.set_yticks(range(slot_sample.shape[0]))
        self.ax.set_yticklabels(labeled_tokens)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        input('>')
        plt.cla()

    @torch.no_grad()
    def evaluate(self, embedding, generator):
        pred_binary = []
        true_binary = []

        for batch in tqdm(self.loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            padding_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].tolist()

            soft_mask = generator(embedding(input_ids), padding_mask)[-1]
            assignments = soft_mask.argmax(dim=1, keepdim=True)
            hard_mask = torch.zeros_like(soft_mask)  # (B, S, L)
            hard_mask.scatter_(1, assignments, 1.0)

            assignments = assignments.squeeze(1).tolist()
            predictions = []
            references = []

            for batch_index, item in enumerate(assignments):
                length = padding_mask[batch_index].sum().item()

                slot_predictions = item[:length]
                true_tag_ids = labels[batch_index][:length]

                if self.plot:
                    self._plot_example(hard_mask[batch_index, :, :length],
                                       input_ids[batch_index, :length],
                                       true_tag_ids)

                pred_tags = []
                true_tags = []
                prev_slot_id = -1
                for slot_id, true_tag_id in zip(slot_predictions,
                                                true_tag_ids):
                    if true_tag_id == -100:
                        continue

                    true_tags.append(self.tags_list[true_tag_id])

                    if true_tag_id == 0:
                        true_binary.append(0)
                    else:
                        true_binary.append(1)

                    if slot_id == 0:
                        pred_tags.append(data.TAG_O)
                        pred_binary.append(0)
                    elif slot_id != prev_slot_id:
                        pred_tags.append(data.TAG_B_MISC)
                        pred_binary.append(1)
                    else:
                        pred_tags.append(data.TAG_I_MISC)
                        pred_binary.append(1)

                    prev_slot_id = slot_id

                predictions.append(pred_tags)
                references.append(true_tags)

            self.metrics.add_batch(predictions=predictions, references=references)

        results = self.metrics.compute()
        detection_results = classification_report(true_binary, pred_binary,
                                                  output_dict=True)

        results_dict = {'ner_precision': results['overall_precision'],
                        'ner_recall': results['overall_recall'],
                        'ner_f1': results['overall_f1'],
                        'ner_accuracy': results['overall_accuracy']}
        for symbol in ('1', '0'):
            for metric in ('precision', 'recall', 'f1-score'):
                key = f'binary_{metric}_{symbol}'
                results_dict[key] = detection_results[symbol][metric]

        return results_dict

    @torch.no_grad()
    def evaluate_marginal(self, embedding, generator, discriminator=None, threshold=None):
        results_dict = defaultdict(list)

        accumulated_overlap_metrics = defaultdict(float)
        mention_count = 0

        for batch in tqdm(self.loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            padding_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            ignored_labels_mask = labels == -100
            labels[ignored_labels_mask] = self.tag_idx_O

            samples, _, soft_mask = generator(embedding(input_ids), padding_mask)
            # (B, S, L, D), _, (B, S, L)

            if threshold is not None:
                batch_size, num_slots, length, dim = samples.shape
                samples = samples.reshape(batch_size * num_slots, length, dim)
                scores = discriminator(samples)
                # all_scores.extend(scores.squeeze(-1).tolist())
                # (B * S, 1)
                scores = scores.reshape(batch_size, num_slots)
                mask_scores = torch.where(scores > threshold, 1.0, 0.0)
                soft_mask[:, 1:] = soft_mask[:, 1:] * mask_scores.unsqueeze(-1)

            is_entity = (soft_mask.argmax(dim=1) > 0).int()
            # (B, L)

            start_entity = torch.empty_like(is_entity, dtype=torch.bool)
            start_entity[:, 1:] = (is_entity[:, 1:] - is_entity[:, :-1]) == 1
            start_entity[:, 0] = is_entity[:, 0] == 1
            # (B, L)

            pred_idx = torch.empty_like(is_entity)
            pred_idx[is_entity == 0] = self.tag_idx_O
            pred_idx[is_entity == 1] = self.tag_idx_I
            pred_idx[start_entity] = self.tag_idx_B

            pred_labels = self.tags_array[pred_idx.cpu().numpy()]
            true_labels = self.tags_array[labels.numpy()]

            predictions = []
            references = []

            padding_mask = padding_mask.cpu()

            for i in range(pred_labels.shape[0]):
                eval_idx = (padding_mask[i] * ~ignored_labels_mask[i]).bool().numpy()
                sample_prediction = pred_labels[i][eval_idx].tolist()
                sample_reference = true_labels[i][eval_idx].tolist()

                predictions.append(sample_prediction)
                references.append(sample_reference)

                overlap_metrics_dict = get_overlap_metrics(sample_prediction,
                                                         sample_reference)
                span_count = 0
                for metric_name, values in overlap_metrics_dict.items():
                    accumulated_overlap_metrics[metric_name] += sum(values)
                    span_count = len(values)
                mention_count += span_count

                if self.instance_eval:
                    instance_results = self.metrics.compute(predictions=[sample_prediction],
                                                            references=[sample_reference])
                    results_dict['sentence'].append(self.tokenizer.decode(input_ids[i], skip_special_tokens=True))
                    results_dict['labels'].append(sample_reference)
                    results_dict['predictions'].append(sample_prediction)
                    results_dict['precision'].append(instance_results['overall_precision'])
                    results_dict['recall'].append(instance_results['overall_recall'])
                    results_dict['f1'].append(instance_results['overall_f1'])
                    results_dict['accuracy'].append(instance_results['overall_accuracy'])

                    for overlap_metric, values in overlap_metrics_dict:
                        results_dict[overlap_metric].append(np.mean(values))

            if not self.instance_eval:
                self.metrics.add_batch(predictions=predictions,
                                       references=references)

        if self.instance_eval:
            pd.DataFrame.from_dict(results_dict).to_pickle(self.instance_eval_file)
            results = None
        else:
            results = self.metrics.compute()
            for overlap_metric, accumulated_value in accumulated_overlap_metrics.items():
                results[overlap_metric] = accumulated_value / mention_count

        return results


def from_commandline():
    parser = ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('--split', choices=['train', 'validation', 'test'],
                        default='validation')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--instance_eval', action='store_true')
    parser.add_argument('--threshold_analysis', action='store_true')
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = 'bert-base-cased'
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    embedding = TokenEmbedding(base_model).to(device)
    generator = EntityExtractor(base_model, num_slots=10, num_iters=3,
                                stretch=True)
    discriminator = ConvolutionalDiscriminator(generator.hidden_size)
    checkpoint = torch.load(args.checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)
    discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator.to(device)

    checkpoint_name = osp.splitext(osp.basename(args.checkpoint_path))[0]

    instance_eval_file = f'{checkpoint_name}-instance-results.pkl'
    evaluator = CoNLLEvaluator(args.split, tokenizer, args.batch_size, device,
                               args.plot, args.instance_eval,
                               instance_eval_file)

    if args.threshold_analysis:
        threshold_results_dict = defaultdict(list)
        for threshold in np.linspace(-2.0, 0.5, num=50):
            results = evaluator.evaluate_marginal(embedding, generator,
                                                  discriminator,
                                                  threshold)

            threshold_results_dict['threshold'].append(threshold)
            threshold_results_dict['precision'].append(results["overall_precision"])
            threshold_results_dict['recall'].append(results["overall_recall"])
            threshold_results_dict['f1'].append(results["overall_f1"])

        results_file = f'{checkpoint_name}-threshold-results.pkl'
        pd.DataFrame.from_dict(threshold_results_dict).to_pickle(results_file)
    else:
        results = evaluator.evaluate_marginal(embedding, generator, discriminator,
                                              args.threshold)
        pprint(results)


if __name__ == '__main__':
    from_commandline()
