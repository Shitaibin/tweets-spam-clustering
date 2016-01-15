import unittest

from pandas import DataFrame

from merge_cluster import get_cluster_tag, get_merge_rules
from merge_cluster import merge_tweets, get_m_tags, sort_tag_counts


class MergeClusterTestCase(unittest.TestCase):
    """
    Unittest for tools.py
    """

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


if __name__ == "__main__":
    unittest.main()
