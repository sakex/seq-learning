import unittest
import Debug.sequential_learning as sl
import numpy as np


class ImportTest(unittest.TestCase):
    def eigenspace_consistence(self):
        nvar = 2
        n = 200
        r = 1
        data = np.random.random(n, nvar)
        print(type(data))
        #data = np.apply_along_axis(data)
        self.assert_()


if __name__ == '__main__':
    unittest.main()
