from biomodelml.variants.sw import SmithWatermanVariant
from biotite.sequence import align, Sequence


class NeedlemanWunschVariant(SmithWatermanVariant):

    name = "Global with Needleman-Wunsch"

    def call_alg(self, seq: Sequence, seq_compare: Sequence) -> align.Alignment:
        matrix = self._get_substitution_matrix()
        max_score = matrix.score_matrix().max()*max(len(seq), len(seq_compare))
        return 1 - align.align_optimal(
            seq, seq_compare, matrix, local=False)[0].score / max_score