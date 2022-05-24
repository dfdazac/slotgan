from typing import List, Dict
from collections import defaultdict

OVERLAP_PRECISION = 'overlap_precision'
OVERLAP_RECALL = 'overlap_recall'
OVERLAP_F1 = 'overlap_f1'
OVERLAP_METRICS = [OVERLAP_PRECISION, OVERLAP_RECALL, OVERLAP_F1]


class Span:
    def __init__(self, start: int):
        self.start = start
        self.end = None
        self.positions = {start}

    def add(self, position: int):
        self.positions.add(position)

    def set_end(self, end: int):
        self.end = end
        self.add(end)

    def overlaps_with(self, other) -> bool:
        if self.end is None or other.end is None:
            raise ValueError('Span is not closed, overlap cannot be computed')

        return self.start <= other.end and self.end >= other.start

    def jaccard_similarity(self, other: 'Span') -> float:
        intersection = self.positions & other.positions
        union = self.positions | other.positions
        return len(intersection) / len(union)

    def squad_overlap(self, reference: 'Span') -> Dict[str, float]:
        intersection = self.positions & reference.positions
        precision = len(intersection) / len(self)
        recall = len(intersection) / len(reference)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {OVERLAP_PRECISION: precision,
                OVERLAP_RECALL: recall,
                OVERLAP_F1: f1}

    def __repr__(self):
        return f'Span({self.positions.__repr__()})'

    def __len__(self):
        return len(self.positions)


def extract_spans(sequence: List[str]) -> List[Span]:
    """Given a sequence of IOB tags, find all mention spans.

    Examples:

    [O, B-MISC, O] returns [Span({1})]
    [O, B-MISC, I-MISC] returns [Span({1, 2})]
    [O, B-MISC, B-MISC] returns [Span({1}), Span({2})
    [O, B-MISC, O, B-MISC, I-MISC, O] returns [Span({1}), Span({2, 3})]

    Args:
        sequence: a list of strings, each corresponding to a tag in the IOB
            scheme

    Returns
        A list of Span objects, one for each span found in the input sequence.
    """
    all_spans = []
    span_start = -1
    for i, tag in enumerate(sequence):
        if tag.startswith('B'):
            if span_start > -1:
                all_spans[-1].set_end(i - 1)
                span_start = span_end = -1

            span_start = i
            all_spans.append(Span(span_start))
        elif tag.startswith('I'):
            if span_start == -1:
                # Sometimes tokenization causes spans like [B-MISC, I-MISC]
                # to lose e.g. the first token when keeping only indices of
                # interest, resulting in the span [I-MISC]. We assume then
                # that this is a starting span token.
                span_start = i
                all_spans.append(Span(span_start))
            else:
                all_spans[-1].add(i)
        elif tag.startswith('O') and span_start > -1:
            all_spans[-1].set_end(i - 1)
            span_start = span_end = -1

    if span_start > -1:
        all_spans[-1].set_end(i)

    return all_spans


def get_span_overlap_metrics(predicted_spans: List[Span],
                             reference_spans: List[Span]) -> Dict[str, List[float]]:
    """Given a predicted and reference pair of lists containing Spans, compute
    the Jaccard similarity between each mention in the reference list and
    the corresponding overlapping mention in the predicted list.
    If there is no overlapping mention in the prediction, similarity is zero.
    Mentions in the prediction that do not overlap with any reference mention
    are ignored.

    Args:
        predicted_spans
        reference_spans

    Returns
        List of floats containing the Jaccard similarities for reference
        mentions
    """
    similarities = defaultdict(list)
    for gold in reference_spans:
        span_found = False
        for pred in predicted_spans:
            if gold.overlaps_with(pred):
                span_metrics = gold.squad_overlap(pred)
                for metric in OVERLAP_METRICS:
                    similarities[metric].append(span_metrics[metric])
                span_found = True
                break
        if not span_found:
            for metric in OVERLAP_METRICS:
                similarities[metric].append(0.0)

    return similarities


def get_overlap_metrics(predictions: List[str],
                        references: List[str]) -> Dict[str, List[float]]:
    """Given two sequences of IOB tags, compute the mention-level Jaccard
    similarity.

    Args:
        predictions
        references
    """
    predicted_spans = extract_spans(predictions)
    reference_spans = extract_spans(references)
    metric_to_values = get_span_overlap_metrics(predicted_spans, reference_spans)

    return metric_to_values
