from Bio import SeqIO
from Bio.Seq import Seq
from biotite.sequence import NucleotideSequence, ProteinSequence, AlphabetError
from biomodelml.structs import SeqTypeStruct


NUCLEOTIDE_SYMBOLS = tuple(NucleotideSequence.alphabet_unamb.get_symbols()) + ("U",)
ALL_NUCLEOTIDE_SYMBOLS = NUCLEOTIDE_SYMBOLS + tuple(NucleotideSequence.alphabet_amb.get_symbols())
PROTEIN_SYMBOLS = tuple(ProteinSequence.alphabet.get_symbols())
SEQ_TYPES = SeqTypeStruct(N=ALL_NUCLEOTIDE_SYMBOLS, P=PROTEIN_SYMBOLS)


def convert_and_remove_unrelated_sequences(seq_path: str, seq_type):
    if seq_type not in SEQ_TYPES.__dataclass_fields__:
        raise IOError("Only sequence type N or P accepted")
    with open(seq_path) as handle:
        sequences = SeqIO.parse(handle, "fasta")
        sanitized_seqs = []
        for s in sequences:
            cleaned_seq = str(s.seq).upper().replace("-", "").replace(".", "")
            if not cleaned_seq:
                print(f"Sequence {s.description} removed")
                continue

            s.seq = Seq(cleaned_seq)
            alphabet = set(s.seq)
            if seq_type == "N":
                if alphabet.issubset(SEQ_TYPES.N):
                    sanitized_seqs.append(s)
                else:
                    print(f"Sequence {s.description} removed")
                continue

            # Protein mode accepts either valid protein symbols directly,
            # or nucleotide symbols that can be translated into proteins.
            if alphabet.issubset(SEQ_TYPES.P):
                sanitized_seqs.append(s)
                continue

            if alphabet.issubset(NUCLEOTIDE_SYMBOLS):
                try:
                    NucleotideSequence(s.seq, False)
                    translated = s.translate(stop_symbol="")
                    translated.description = s.description
                    translated.id = s.id
                    print(f"Sequence {translated.description} translated")
                    sanitized_seqs.append(translated)
                except AlphabetError:
                    print(f"Error on sequence {s.description} and it's removed")
                continue

            print(f"Sequence {s.description} removed")

    print(f"writing {len(sanitized_seqs)} sequences")
    SeqIO.write(sanitized_seqs, f"{seq_path}.{seq_type}.sanitized", "fasta")

# Alias for backwards compatibility with tests
remove_unrelated_sequences = convert_and_remove_unrelated_sequences
