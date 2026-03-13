"""Shared pytest fixtures for BioModelML Phase 1 tests."""
import pytest
from pathlib import Path
from Bio.Seq import Seq
from Bio import SeqIO


@pytest.fixture
def tiny_dna_seq():
    """Minimal DNA sequence for quick unit tests."""
    return Seq("ATGC")


@pytest.fixture
def tiny_protein_seq():
    """Minimal protein sequence for quick unit tests."""
    return Seq("MWVLKSG")


@pytest.fixture
def small_dna_seq():
    """Small DNA sequence for testing with realistic patterns."""
    return Seq("ATGCATGCTAGC")


@pytest.fixture
def example_fasta_path():
    """Path to example FASTA file."""
    return Path(__file__).parent.parent / "data" / "example_sequences.fasta"


@pytest.fixture
def myoglobin_fasta_path():
    """Path to myoglobin ortholog FASTA file (15 sequences)."""
    return Path(__file__).parent.parent / "data" / "orthologs_myoglobin.fasta"


@pytest.fixture
def test_sequences(example_fasta_path):
    """Load example sequences from FASTA file."""
    if example_fasta_path.exists():
        return list(SeqIO.parse(example_fasta_path, "fasta"))
    return []


@pytest.fixture
def matrix_output_dir(tmp_path):
    """Temporary directory for image generation with auto-cleanup."""
    output_dir = tmp_path / "matrices"
    output_dir.mkdir(exist_ok=True)
    return output_dir
