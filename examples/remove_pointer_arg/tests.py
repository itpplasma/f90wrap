import unittest

from pywrapper import m_test

class TestReturnArray(unittest.TestCase):

    def not_ignored(self):
        _ = m_test.not_to_be_ignored()

    def ignored_1(self):
        with self.assertRaises(AttributeError):
            _ = m_test.to_be_ignored_1()

    def ignored_2(self):
        with self.assertRaises(AttributeError):
            _ = m_test.to_be_ignored_2()

if __name__ == '__main__':
    unittest.main()