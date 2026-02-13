from unittest import TestCase
from biomodelml.matrices import build_matrix
from Bio.Seq import Seq
from numpy.testing import assert_equal


def test_should_detect_palindrome():
    # ACCTAGGT is a reverse-complement palindrome
    # Both R (seq vs seq) and G (seq vs complement) should be symmetric
    sequence = Seq("ACCTAGGT")
    matrix = build_matrix(sequence, sequence, 20, "N")
    assert_equal(matrix[:,:, 0], matrix[:,:, 0].T)  # R channel symmetric
    assert_equal(matrix[:,:, 1], matrix[:,:, 1].T)  # G channel symmetric

    
def test_should_detect_direct_repeat():
    # TTACGTTACG contains a direct repeat (TTACG appears twice)
    # R channel (seq vs seq) should be symmetric
    sequence = Seq("TTACGTTACG")
    matrix = build_matrix(sequence, sequence, 20, "N")
    assert_equal(matrix[:,:, 0], matrix[:,:, 0].T)

        
def test_should_detect_repeat():
    sequence = Seq("TTACGGCATT")
    matrix = build_matrix(sequence, sequence, 20, "N")
    assert_equal(matrix[:,:, 0], matrix[:,:, 0].T)
    assert_equal(matrix[:,:, 1], matrix[:,:, 1].T)
    assert_equal(matrix[:,:, 2], matrix[:,:, 2].T)


class TestBuildSimpleMatrix(TestCase):
    def setUp(self):
        self.sequence = Seq("ACAT")

    def test_should_build_red_layer(self):
        layer = [
            [
                20, 0, 20, 0
            ],
            [
                0, 20, 0, 0
            ],
            [
                20, 0, 20, 0
            ],
            [
                0, 0, 0, 20
            ]
        ]
        matrix = build_matrix(self.sequence, self.sequence, 20, "N")
        assert_equal(matrix[:,:, 0], layer)

    def test_should_build_green_layer(self):
        # Green channel: seq vs complement (ACAT vs TGTA)
        layer = [
            [
                0, 0, 0, 20
            ],
            [
                0, 0, 0, 0
            ], 
            [
                0, 0, 0, 20
            ],
            [
                20, 0, 20, 0
            ]
        ]

        matrix = build_matrix(self.sequence, self.sequence, 20, "N")
        assert_equal(matrix[:,:, 1], layer)

    def test_should_build_blue_layer(self):
        # Blue channel: positions with no match in R or G
        layer = [
            [
                0, 20, 0, 0
            ], 
            [
                20, 0, 20, 20
            ], 
            [
                0, 20, 0, 0
            ], 
            [
                0, 20, 0, 0
            ]
        ]

        matrix = build_matrix(self.sequence, self.sequence, 20, "N")
        assert_equal(matrix[:,:, 2], layer) 