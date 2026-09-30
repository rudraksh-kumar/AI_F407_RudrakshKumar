"""
Second-Order Autoregressive Language Model
Implementation for AI Laboratory: Bayesian Networks and Autoregressive Language Models
"""

import random
from collections import defaultdict, Counter


class SecondOrderLanguageModel:
    """
    A Second-Order Autoregressive Markov Language Model.
    Models P(X_t | X_{t-2}, X_{t-1}) using explicit conditional probability tables (CPTs)
    learned from trigram transition frequencies in training corpus.
    """

    def __init__(self, start_token="<START>", end_token="<END>"):
        self.start_token = start_token
        self.end_token = end_token
        self.counts = defaultdict(Counter)
        self.probabilities = defaultdict(dict)
        self.vocabulary = set()

    def train(self, tokenized_sentences):
        """
        Calculates trigram transitions and builds P(X_t | X_{t-2}, X_{t-1}).
        """
        self.counts.clear()
        self.probabilities.clear()
        self.vocabulary.clear()

        for sentence in tokenized_sentences:
            for i in range(len(sentence) - 2):
                w1, w2, w3 = sentence[i], sentence[i + 1], sentence[i + 2]
                context = (w1, w2)
                self.counts[context][w3] += 1
                self.vocabulary.update([w1, w2, w3])

        # Compute P(X_t | X_{t-2}, X_{t-1}) = C(w_{t-2}, w_{t-1}, w_t) / sum_k C(w_{t-2}, w_{t-1}, w_k)
        for context, next_counts in self.counts.items():
            total_transitions = sum(next_counts.values())
            for w3, count in next_counts.items():
                self.probabilities[context][w3] = count / total_transitions

    def get_distribution(self, context_tuple):
        """
        Returns conditional distribution P(X_t | X_{t-2}=w_{t-2}, X_{t-1}=w_{t-1}).
        """
        return self.probabilities.get(context_tuple, {})

    def predict_next_greedy(self, context_tuple):
        """
        Mode A: Greedy generation -> selects argmax_w P(w | context_tuple).
        """
        dist = self.get_distribution(context_tuple)
        if not dist:
            return self.end_token
        return max(dist.items(), key=lambda x: x[1])[0]

    def sample_next(self, context_tuple):
        """
        Mode B: Probabilistic sampling -> samples w ~ P(w | context_tuple).
        """
        dist = self.get_distribution(context_tuple)
        if not dist:
            return self.end_token
        tokens, probs = zip(*dist.items())
        return random.choices(tokens, weights=probs, k=1)[0]

    def generate_sentence(self, mode="sampling", initial_word="the", max_length=30):
        """
        Generates a sequence starting from (<START>, initial_word) token-by-token.
        """
        sentence = [self.start_token, initial_word]
        while sentence[-1] != self.end_token and len(sentence) < max_length:
            context = (sentence[-2], sentence[-1])
            if mode == "greedy":
                nxt = self.predict_next_greedy(context)
            else:
                nxt = self.sample_next(context)
            sentence.append(nxt)
        return sentence

    def test_normalization(self):
        """
        Verifies the probability invariant: sum_v P(v | w_{t-2}, w_{t-1}) = 1.0.
        """
        normalization_results = {}
        for context, dist in self.probabilities.items():
            total_prob = sum(dist.values())
            normalization_results[context] = total_prob
        return normalization_results
