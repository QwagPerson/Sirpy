import unittest

from sirpy.utils.attrDict import AttrDict


class attrDictTests(unittest.TestCase):
    def setUp(self) -> None:
        sample_dict = {
            "a": 1,
            "b": 2,
            "c": 3
        }
        self.attr_dict = AttrDict(**sample_dict)

    def test_get_set(self):
        # test get
        self.assertEqual(2, self.attr_dict["b"])
        self.assertNotEqual(2, self.attr_dict["a"])
        # test set
        self.attr_dict["a"] = 100
        self.assertEqual(100, self.attr_dict["a"])

    def test_len_contains(self):
        self.assertEqual(3, len(self.attr_dict))
        self.assertTrue("a" in self.attr_dict)
        self.assertFalse("d" in self.attr_dict)

    def test_iter(self):
        for key in self.attr_dict:
            self.assertTrue(key in self.attr_dict)

    def test_del(self):
        del self.attr_dict["a"]
        self.assertFalse("a" in self.attr_dict)
        self.assertEqual(2, len(self.attr_dict))
    

if __name__ == '__main__':
    unittest.main()
