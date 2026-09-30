"""
First-Order Autoregressive Language Model
Implementation for AI Laboratory: Bayesian Networks and Autoregressive Language Models
"""

import random
from collections import defaultdict, Counter


class FirstOrderLanguageModel:
    """
    A First-Order Autoregressive Markov Language Model.
    Models P(X_t | X_{t-1}) using explicit conditional probability tables (CPTs)
    learned from transition frequencies in training corpus.
    """

    def __init__(self, start_token="<START>", end_token="<END>"):
        self.start_token = start_token
        self.end_token = end_token
        self.counts = defaultdict(Counter)
        self.probabilities = defaultdict(dict)
        self.vocabulary = set()

    def train(self, tokenized_sentences):
        """
        Calculates token transitions and builds the conditional probability distribution table.
        """
        self.counts.clear()
        self.probabilities.clear()
        self.vocabulary.clear()

        for sentence in tokenized_sentences:
            for w1, w2 in zip(sentence[:-1], sentence[1:]):
                self.counts[w1][w2] += 1
                self.vocabulary.add(w1)
                self.vocabulary.add(w2)

        # Compute P(X_t | X_{t-1}) = C(w_{t-1}, w_t) / sum_k C(w_{t-1}, w_k)
        for w1, next_counts in self.counts.items():
            total_transitions = sum(next_counts.values())
            for w2, count in next_counts.items():
                self.probabilities[w1][w2] = count / total_transitions

    def get_distribution(self, prev_token):
        """
        Returns conditional distribution P(X_t | X_{t-1} = prev_token).
        """
        return self.probabilities.get(prev_token, {})

    def predict_next_greedy(self, prev_token):
        """
        Mode A: Greedy generation -> selects argmax_w P(w | prev_token).
        """
        dist = self.get_distribution(prev_token)
        if not dist:
            return self.end_token
        return max(dist.items(), key=lambda x: x[1])[0]

    def sample_next(self, prev_token):
        """
        Mode B: Probabilistic sampling -> samples w ~ P(w | prev_token).
        """
        dist = self.get_distribution(prev_token)
        if not dist:
            return self.end_token
        tokens, probs = zip(*dist.items())
        return random.choices(tokens, weights=probs, k=1)[0]

    def generate_sentence(self, mode="sampling", max_length=30):
        """
        Generates a sequence token-by-token until <END> or max_length is reached.
        """
        sentence = [self.start_token]
        while sentence[-1] != self.end_token and len(sentence) < max_length:
            curr = sentence[-1]
            if mode == "greedy":
                nxt = self.predict_next_greedy(curr)
            else:
                nxt = self.sample_next(curr)
            sentence.append(nxt)
        return sentence

    def test_normalization(self):
        """
        Verifies the probability invariant: sum_v P(v | w) = 1.0 for all w.
        """
        normalization_results = {}
        for w, dist in self.probabilities.items():
            total_prob = sum(dist.values())
            normalization_results[w] = total_prob
        return normalization_results
