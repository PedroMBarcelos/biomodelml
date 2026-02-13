from biomodelml.variants.variant import Variant
from biomodelml.structs import DistanceStruct
from biotite import sequence
from biotite.application import clustalo
from biotite.sequence import align


class ControlVariant(Variant):

    name = "Control with Clustal Omega"

    def __init__(self, fasta_file: str, sequence_type: str):
        super().__init__(fasta_file, sequence_type)
        if sequence_type == self.nucleotide_type:
            mapper = lambda x: sequence.NucleotideSequence(x, True)
        else:
            mapper = sequence.ProteinSequence
        self._sequences = list(map(mapper, self._sequences))

    def build_matrix(self) -> DistanceStruct:
        alignment = clustalo.ClustalOmegaApp.align(self._sequences)

        distances = 1 - align.get_pairwise_sequence_identity(
            alignment, mode="shortest"
        )
        return DistanceStruct(
            names=self._names, matrix=distances, align=alignment)
