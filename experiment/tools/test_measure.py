from unittest import TestCase

from measure import cos_similarity


class MeasureTestCase(TestCase):
    """
    Unittest for tools.py
    """

    ##################################################
    # Unittest for Visulize fucntions




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
