import pytest
from biomodelml.sanitize import remove_unrelated_sequences


def test_should_raise_error_when_wrong_field(tmp_path):
    fasta_file = tmp_path / "fasta.fasta"
    with pytest.raises(IOError) as err:
        remove_unrelated_sequences(str(fasta_file), "D")

def test_should_pass_dna_sequences(tmp_path):
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".N.sanitized"
    test_string = ">testing\nATGCTGA\n"
    with open(fasta_file, "w") as f:
        f.write(test_string)
    remove_unrelated_sequences(str(fasta_file), "N")
    with open(result_file) as f:
        assert f.read() == test_string

def test_should_pass_ptn_sequences(tmp_path):
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".P.sanitized"
    test_string = ">testing\nAYWRSQL\n"
    with open(fasta_file, "w") as f:
        f.write(test_string)
    remove_unrelated_sequences(str(fasta_file), "P")
    with open(result_file) as f:
        assert f.read() == test_string

def test_should_remove_dna_sequences_with_unknown(tmp_path):
    # J is not a valid nucleotide symbol (not even ambiguous)
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".N.sanitized"
    with open(fasta_file, "w") as f:
        f.write(">testing\nATGCTJA\n")
    remove_unrelated_sequences(str(fasta_file), "N")
    with open(result_file) as f:
        assert f.read() == ""

def test_should_remove_ptn_sequences_with_unknown(tmp_path):
    # J is not a valid protein symbol
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".P.sanitized"
    with open(fasta_file, "w") as f:
        f.write(">testing\nAJWRSQL\n")
    remove_unrelated_sequences(str(fasta_file), "P")
    with open(result_file) as f:
        assert f.read() == ""

def test_should_remove_dna_sequences_with_doubt(tmp_path):
    # J is not a valid nucleotide symbol
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".N.sanitized"
    with open(fasta_file, "w") as f:
        f.write(">testing\nATGCTJA\n")
    remove_unrelated_sequences(str(fasta_file), "N")
    with open(result_file) as f:
        assert f.read() == ""

def test_should_remove_ptn_sequences_with_doubt(tmp_path):
    # J is not a valid protein symbol
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".P.sanitized"
    with open(fasta_file, "w") as f:
        f.write(">testing\nAJWRSQL\n")
    remove_unrelated_sequences(str(fasta_file), "P")
    with open(result_file) as f:
        assert f.read() == ""

def test_should_only_remove_dna_sequences_not_supported(tmp_path):
    # Should keep valid sequence and remove invalid one (with J)
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".N.sanitized"
    test_string = ">testing\nATGCTGA\n"
    with open(fasta_file, "w") as f:
        f.write(test_string + ">testing2\nATGCTJA\n")
    remove_unrelated_sequences(str(fasta_file), "N")
    with open(result_file) as f:
        assert f.read() == test_string

def test_should_only_remove_ptn_sequences_with_not_supported(tmp_path):
    # Should keep valid sequence and remove invalid one (with J)
    fasta_file = tmp_path / "fasta.fasta"
    result_file = str(fasta_file) + ".P.sanitized"
    test_string = ">testing\nAYWRSQL\n"
    with open(fasta_file, "w") as f:
        f.write(test_string + ">testing2\nAJWRSQL\n")
    remove_unrelated_sequences(str(fasta_file), "P")
    with open(result_file) as f:
        assert f.read() == test_string