# Author: James Shi
# License: BSD 3 clause


# TODO: decompose the left compose

from __future__ import print_function

import logging
import time

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('tweetspamtools')


def timestamp_to_datehour(timestamp):
    """
    Transform timestamp to date and hour.

    :timestamp: string
    :return: string
    """
    t = time.localtime(float(timestamp))
    datehour = time.strftime("%Y%m%d%H", t)
    return datehour
