import unittest

from nltk.stem.porter import PorterStemmer

from feature_extraction import text_stem, tokenize


class FeatureExtractionTestCase(unittest.TestCase):
    def test_test_stem(self):
        """
        Unittest for function test_stem.
        """
        # test data
        s = "Python is very interesting"
        l = s.split()
        stem_result = ['Python', 'is', 'veri', 'interest']
        # f_result
        stemmer = PorterStemmer()
        f_result = text_stem(l, stemmer)
        # stem_result == f_result ?
        self.assertEqual(stem_result, f_result, msg="\nThe stem result\
                          should be \n{},\n rather than \n{}".format(
                stem_result, f_result))

    def test_tokenize(self):
        """
        Unittest for function tokenize.
        """
        # test data
        s = "Python is very interesting."  # end with a period
        result = ['Python', 'is', 'veri', 'interest']
        # tokenize_result
        tokenize_result = tokenize(s)
        # tokenize_result == result ?
        self.assertEqual(result, tokenize_result, msg="\nThe stem result\
                          should be \n{},\n rather than \n{}".format(
                result, tokenize_result))
