import unittest

from handledata import remove_hashtag, get_hashtag
from handledata import split_record


class DataTestCase(unittest.TestCase):
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
        tweet = sid + ' | ' + stamp + ' | ' + \
                degree + ' | ' + content + \
                ' |*|' + url
        tweet_tuple = (sid, stamp, degree, content, url)
        # test result
        ret = split_record(tweet)
        # data and result should be equal
        self.assertEqual(tweet_tuple, ret, msg="Split result should be \
                         {}, rather than{}".format(tweet_tuple, ret))
