import unittest

from sirpy.utils.lambdaUtils import null_lambda, add_functions, difference_functions


class LambdaUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.add = lambda t, y, _: t + y
        self.sub = lambda t, y, _: t - y

    def test_add_functions(self):
        self.assertEqual(2, add_functions(self.add, self.sub)(1, 100, None))

    def test_difference_functions(self):
        self.assertEqual(0, difference_functions(self.add, self.add)(1, 100, None))

    def test_null_lambda(self):
        self.assertEqual(0, null_lambda(1, 100, None))

if __name__ == '__main__':
    unittest.main()
