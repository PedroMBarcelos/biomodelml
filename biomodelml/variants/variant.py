from Bio import SeqIO
from biomodelml.structs import DistanceStruct
from typing import List


class Variant:
    name = "variant"
    protein_type = "P"
    nucleotide_type = "N"
    def __init__(self, fasta_file: str = None, sequence_type: str = None):
        if not fasta_file:
            return
        if sequence_type not in [self.protein_type, self.nucleotide_type]:
            raise IOError(
                f"Sequence must be a protein or nucleotide and is: {sequence_type}")
        seqs = SeqIO.parse(fasta_file, "fasta")
        self._names = []
        self._sequences = []
        self._fasta_file = fasta_file
        self._sequence_type = sequence_type
        for s in seqs:
            self._names.append(s.description)
            self._sequences.append(s.seq)

    @classmethod
    def from_name_list(cls, name_list = List[str], **kwargs):
        obj = cls(**kwargs)
        obj._names = name_list
        return obj

    def build_matrix(self) -> DistanceStruct:
        raise NotImplementedError()