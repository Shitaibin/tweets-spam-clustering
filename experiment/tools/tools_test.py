import unittest
from unittest import TestCase
from tools import remove_hashtag, get_hashtag
from tools import split_record, text_stem, tokenize
from tools import cos_similarity
from tools import merge_tweets, get_m_tags, sort_tag_counts
from tools import get_cluster_tag, get_merge_rules
from tools import timestamp_to_datehour
from nltk.stem.porter import PorterStemmer
from pandas import DataFrame


class ToolsTestCase(TestCase):

    """
    Unittest for tools.py
    """
    ##################################################
    # Unittest for preprocessing data

    def test_get_hashtag(self):
        """
        Unittest for function get_hashtag hashtag.
        """
        self.assertEqual("#ohi", get_hashtag("abcd #ohi"))
        self.assertEqual("", get_hashtag("abcd ohi ojk"))

    def test_remove_hashtag(self):
        """
        Unittest for function remove hashtag.
        """
        tweets = ["abcd #ohi",
                  "abcd #ohi ojk"]
        result = ["abcd ",
                  "abcd  ojk"]
        self.assertEqual(result, remove_hashtag(tweets))

    ##################################################
    # Unittest for Visulize fucntions
    def test_split_record(self):
        """
        Unittest for function split_record.
        """
        # test data
        sid = '123456'
        stamp = '151413'
        degree = '1'
        content = "this's the content.#hashtag"
        url = "http://jkfslkdf.ckd"
        tweet = sid + ' | ' + stamp + ' | ' +\
            degree + ' | ' + content +\
            ' |*|' + url
        tweet_tuple = (sid, stamp, degree, content, url)
        # test result
        ret = split_record(tweet)
        # data and result should be equal
        self.assertEqual(tweet_tuple, ret, msg="Split result should be \
                         {}, rather than{}".format(tweet_tuple, ret))

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

    def test_cos_similarity(self):
        """
        Unittest for function CosSimilirity.

        The input parameters is normalize vector/list. So, cos_similarity
        just caculate the dot product. In this test, we just need
        to test dot product is enough.
        """
        self.assertEqual(8, cos_similarity([1, 2], [2, 3]), msg="The cosine\
                          similarity of (1,2) and (2,3) should be \
                         {}".format(8))
        self.assertEqual(0, cos_similarity([0, 0], [2, 3]), msg="The cosine\
                          similarity of (0,0) and (2,3) should be \
                         {}".format(0))
        self.assertEqual(4, cos_similarity([-1, 2], [2, 3]), msg="The cosine\
                          similarity of (-1,2) and (2,3) should be \
                         {}".format(4))

    def test_merge_tweets(self):
        """
        Unittest for function merge_tweets.
        """
        # test data
        labels_tweets = DataFrame()
        labels = [0, 0, 1, 1]
        tweets = ["This is the first tweets.",
                  "Here is the second tweets.",
                  "Aha, that's the third tweets",
                  "Finally, this is the last one."]
        labels_tweets['label'] = labels
        labels_tweets['tweet'] = tweets
        result = {}
        result[0] = tweets[0] + " " + tweets[1]
        result[1] = tweets[2] + " " + tweets[3]
        # merge_tweets result
        merged_result = merge_tweets(labels_tweets)
        # result === merge_tweets?
        self.assertEqual(result, merged_result, msg="merge tweets\
                          result should be {},\nrather than {}\
                         ".format(result, merged_result))

    ##################################################
    # Unittest for merge cluster fucntions
    def test_get_m_tags(self):
        """
        Unittest for function get_m_tags.
        """
        # test data
        text = "Wow this is amazing, Python is amazing, python is\
             very easy to learn and to use. Do you think so?"
        result = [('python', 2), ('amazing', 2), ('think', 1),
                  ('use', 1), ('learn', 1), ('easy', 1), ('wow', 1)]
        result = sort_tag_counts(result)
        leng = len(result)
        # normal test: 1 <= m <= len(tags of text)
        for m in range(1, leng + 1):
            test_result = get_m_tags(text, m)
            self.assertEqual(result[:m], test_result)

        # off normal lower: m <= 0
        for m in range(-3, 1):
            self.assertRaises(AssertionError, lambda: get_m_tags(text, 0))

        # off normal upper: m > len(tags of text)
        for m in range(leng + 1, leng + 5):
            test_result = get_m_tags(text, m)
            self.assertEqual(result, test_result)

        # text is empty string
        for text in [None, "", "   "]:
            self.assertRaises(AssertionError, lambda: get_m_tags(text, 1))

    def test_get_merge_rules(self):
        """
        Unittest for function get_merge_rules.
        """
        ###########################################
        # Normal test
        # test data
        labels_tags = [(0, 'icloud'),
                       (1, 'python'),
                       (2, 'icloud'),
                       (3, 'icloud'),
                       (4, 'clothing'),
                       (5, 'clothing')]

        result = {0: 0, 1: 1, 2: 0, 3: 0, 4: 4, 5: 4}

        # test result
        test_result = get_merge_rules(labels_tags)

        # unit test
        self.assertEqual(result, test_result)

        ###########################################
        # Boundary test
        # labels_tags is None, or empty, or tag is empty..
        self.assertRaises(TypeError, lambda: get_merge_rules(None))
        self.assertEqual({}, get_merge_rules([]))
        self.assertRaises(TypeError, lambda: get_merge_rules([0, None]))
        self.assertRaises(TypeError, lambda: get_merge_rules([0, '']))

    def test_sort_tag_counts(self):
        """
        Unittest for function sort_tag_counts.
        """
        # test data
        l = [('python', 2), ('amazing', 2), ('think', 1),
             ('use', 1), ('learn', 1)]
        result = [('amazing', 2), ('python', 2), ('learn', 1),
                  ('think', 1), ('use', 1)]
        # test result
        test_result = sort_tag_counts(l)
        # result == test_result ?
        self.assertEqual(result, test_result)

    def test_get_cluster_tag(self):
        """
        Unittest for function get_cluster_tag.
        """
        # not empty string
        text = "Wow this is amazing, Python is amazing, python is\
             very easy to learn and to use. Do you think so?"
        self.assertEqual("amazing", get_cluster_tag(text))

        # empty string
        pass

    def test_timestamp_to_datehour(self):
        """
        Unittest for function timestamp_to_datehour.
        """
        timestamp = '1307000691'
        self.assertEqual("2011060215", timestamp_to_datehour(timestamp))

if __name__ == "__main__":
    unittest.main()
