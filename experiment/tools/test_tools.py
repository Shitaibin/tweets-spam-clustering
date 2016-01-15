import unittest
from unittest import TestCase

from tools import timestamp_to_datehour


class ToolsTestCase(TestCase):
    """
    Unittest for tools.py
    """

    def test_timestamp_to_datehour(self):
        """
        Unittest for function timestamp_to_datehour.
        """
        timestamp = '1307000691'
        self.assertEqual("2011060215", timestamp_to_datehour(timestamp))


if __name__ == "__main__":
    unittest.main()
