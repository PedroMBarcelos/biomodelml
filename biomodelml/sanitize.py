from Bio import SeqIO
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
            s.seq = s.seq.upper()
            alphabet = set(s.seq)
            if alphabet.issubset(getattr(SEQ_TYPES, seq_type)):
                t = s
                if seq_type == "P" and alphabet.issubset(NUCLEOTIDE_SYMBOLS):
                    try:
                        NucleotideSequence(s.seq, False)
                        t = s.translate(stop_symbol="")
                        t.description = s.description
                        t.id = s.id
                        print(f"Sequence {t.description} translated")
                    except AlphabetError:
                        print(f"Error on sequence {s.description} and it's removed")
                        continue
                
                sanitized_seqs.append(t)
            else:
                print(f"Sequence {s.description} removed")

    print(f"writing {len(sanitized_seqs)} sequences")
    SeqIO.write(sanitized_seqs, f"{seq_path}.{seq_type}.sanitized", "fasta")

# Alias for backwards compatibility with tests
remove_unrelated_sequences = convert_and_remove_unrelated_sequences
