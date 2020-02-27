import unittest
import Debug.sequential_learning as sq


class ImportTest(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(sq.greet(), "hello, world")


if __name__ == '__main__':
    unittest.main()
