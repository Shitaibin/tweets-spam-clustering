# Author: James Shi
# License: BSD 3 clause

# handle data
# including load data...

from __future__ import print_function

import logging

import numpy as np

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('tweetspamtools')


#################################################
# Load data
def load_data(fname):
    """
    Load data form specific file.

    Return: an array. each term feature take a column.
    """
    tid_list = []
    timestamp_list = []
    degree_list = []
    url_list = []
    content_list = []

    try:
        with open(fname) as f:
            for line in f:
                sid, stamp, degree, content, url = split_record(line)
                tid_list.append(sid)
                timestamp_list.append(stamp)
                degree_list.append(degree)
                content_list.append(content)
                url_list.append(url)
    except Exception as e:
        # TODO: if fname not valid, f is valid, so fix it
        if f: f.close()
        logger.exception(e)
    ary = np.asarray([tid_list, timestamp_list,
                      degree_list, content_list, url_list])
    ary = ary.T
    return ary


def remove_hashtag(tweets):
    """
    If this tweet has hashtag, remove it.

    :tweets: a list of tweets.
    :return: a list of tweets.
    """
    res = []
    for tweet in tweets:
        hashtag = get_hashtag(tweet)
        new_tweet = tweet.replace(hashtag, "")
        res.append(new_tweet)
    return res


def get_hashtag(tweet):
    """
    Get hashtag of a tweet.

    :tweet: string.
    :return: hashtag or empty string.
    """
    hashtag = ""
    idx_of_tag = tweet.rfind('#')
    if idx_of_tag is not -1:
        idx_of_tag_end = tweet.find(' ', idx_of_tag)
        if idx_of_tag_end is -1:  # hashtag is the last word
            hashtag = tweet[idx_of_tag:]
        else:
            hashtag = tweet[idx_of_tag:idx_of_tag_end]

    return hashtag


def split_record(tweet):
    """
    A tweet record is formated like below:

    sender_id | timestamp | degree | tweet content | * | url

    The url may be missed.
    Whatever, get them.
    """

    ret = tweet.split('|')

    # remove_blank_space = lambda s: s.strip()

    def _remove_blank_space(s):
        return s.strip()

    ret = map(_remove_blank_space, ret)

    s_id = ret[0]
    s_stamp = ret[1]
    n_degree = ret[2]
    s_content = ret[3]
    s_url = ret[-1] if len(ret) == 6 else ''
    return s_id, s_stamp, n_degree, s_content, s_url
