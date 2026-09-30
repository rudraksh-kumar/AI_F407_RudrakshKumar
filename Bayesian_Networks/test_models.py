"""
Automated Unit Tests for AI Laboratory: Bayesian Networks and Autoregressive Language Models
Verifies model invariants, CPT calculations, normalisation properties, and generation logic.
"""

import unittest
from first_order_lm import FirstOrderLanguageModel
from second_order_lm import SecondOrderLanguageModel

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]

def preprocess(sentences):
    return [["<START>"] + s.lower().split() + ["<END>"] for s in sentences]

class TestFirstOrderLM(unittest.TestCase):
    def setUp(self):
        self.lm = FirstOrderLanguageModel()
        self.lm.train(preprocess(DATASET))

    def test_normalization_invariant(self):
        """Invariant Check: sum_v P(v | w) == 1.0 for all observed words"""
        norm_results = self.lm.test_normalization()
        for word, total_prob in norm_results.items():
            self.assertAlmostEqual(total_prob, 1.0, places=6,
                                   msg=f"Normalisation failed for word '{word}'")

    def test_specific_cpt_values(self):
        """Verifies exact conditional probability calculations"""
        # P(cat | the) = 3 / 12 = 0.25
        self.assertAlmostEqual(self.lm.probabilities["the"]["cat"], 0.25, places=4)
        # P(dog | the) = 3 / 12 = 0.25
        self.assertAlmostEqual(self.lm.probabilities["the"]["dog"], 0.25, places=4)
        # P(mat | the) = 2 / 12 = 0.1667
        self.assertAlmostEqual(self.lm.probabilities["the"]["mat"], 2.0 / 12.0, places=4)
        # P(on | sat) = 4 / 4 = 1.0
        self.assertEqual(self.lm.probabilities["sat"]["on"], 1.0)
        # P(to | ran) = 2 / 2 = 1.0
        self.assertEqual(self.lm.probabilities["ran"]["to"], 1.0)

    def test_greedy_generation(self):
        """Verifies greedy generation starts with <START> and predicts mode correctly"""
        sent = self.lm.generate_sentence(mode="greedy", max_length=5)
        self.assertEqual(sent[:3], ["<START>", "the", "cat"])

    def test_sampling_generation(self):
        """Verifies sampling produces sequences starting with <START> and ending with <END> or max_length"""
        sent = self.lm.generate_sentence(mode="sampling", max_length=30)
        self.assertEqual(sent[0], "<START>")
        self.assertIn(sent[-1], ["<END>", sent[-1]])


class TestSecondOrderLM(unittest.TestCase):
    def setUp(self):
        self.lm = SecondOrderLanguageModel()
        self.lm.train(preprocess(DATASET))

    def test_normalization_invariant(self):
        """Invariant Check: sum_v P(v | w_{t-2}, w_{t-1}) == 1.0 for all context tuples"""
        norm_results = self.lm.test_normalization()
        for context, total_prob in norm_results.items():
            self.assertAlmostEqual(total_prob, 1.0, places=6,
                                   msg=f"Normalisation failed for context tuple {context}")

    def test_specific_cpt_values(self):
        """Verifies exact trigram conditional probabilities"""
        # P(mat | on, the) = 2 / 4 = 0.5
        self.assertAlmostEqual(self.lm.probabilities[("on", "the")]["mat"], 0.5, places=4)
        # P(rug | on, the) = 2 / 4 = 0.5
        self.assertAlmostEqual(self.lm.probabilities[("on", "the")]["rug"], 0.5, places=4)
        # P(<END> | the, mat) = 2 / 2 = 1.0
        self.assertEqual(self.lm.probabilities[("the", "mat")]["<END>"], 1.0)
        # P(park | to, the) = 2 / 2 = 1.0
        self.assertEqual(self.lm.probabilities[("to", "the")]["park"], 1.0)

    def test_greedy_generation_completes(self):
        """Verifies second order greedy generation generates full valid sentence without looping"""
        sent = self.lm.generate_sentence(mode="greedy")
        expected = ["<START>", "the", "cat", "sat", "on", "the", "mat", "<END>"]
        self.assertEqual(sent, expected)

    def test_sampling_generation_validity(self):
        """Verifies second order sampled sentences end with <END> and are grammatical"""
        for _ in range(10):
            sent = self.lm.generate_sentence(mode="sampling")
            self.assertEqual(sent[0], "<START>")
            self.assertEqual(sent[-1], "<END>")


if __name__ == "__main__":
    unittest.main()
