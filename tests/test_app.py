import unittest

from app import model


class SentimentModelTests(unittest.TestCase):
    def test_positive_prediction(self):
        prediction = model.predict("Fantastic quality and amazing service")

        self.assertEqual(prediction["sentiment"], "positive")
        self.assertGreaterEqual(prediction["confidence"], 0)
        self.assertLessEqual(prediction["confidence"], 1)

    def test_negative_prediction(self):
        prediction = model.predict("Terrible quality and rude service")

        self.assertEqual(prediction["sentiment"], "negative")


if __name__ == "__main__":
    unittest.main()
