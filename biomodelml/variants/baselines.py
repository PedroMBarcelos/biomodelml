import math
import shutil
import subprocess
import tempfile
from typing import Iterable, Tuple

import numpy
from Bio import AlignIO
from Bio.Align import PairwiseAligner
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from biomodelml.structs import DistanceStruct
from biomodelml.variants.control import ControlVariant
from biomodelml.variants.variant import Variant


_NUC = set("ACGT")


def _iter_valid_nuc_sites(seq_a: str, seq_b: str) -> Iterable[Tuple[str, str]]:
    for a, b in zip(seq_a, seq_b):
        a = a.upper()
        b = b.upper()
        if a in _NUC and b in _NUC:
            yield a, b


def _iter_comparable_sites(seq_a: str, seq_b: str) -> Iterable[Tuple[str, str]]:
    for a, b in zip(seq_a, seq_b):
        a = a.upper()
        b = b.upper()
        if a != '-' and b != '-':
            yield a, b


def _p_distance_from_aligned(seq_a: str, seq_b: str) -> float:
    mismatches = 0
    valid_sites = 0
    for a, b in _iter_comparable_sites(seq_a, seq_b):
        valid_sites += 1
        if a != b:
            mismatches += 1
    if valid_sites == 0:
        return 0.0
    return mismatches / valid_sites


def _gtr_q_matrix(rates, freqs):
    # Rate order follows common convention: AC, AG, AT, CG, CT, GT.
    ac, ag, at, cg, ct, gt = rates
    pi_a, pi_c, pi_g, pi_t = freqs
    pi = numpy.array([pi_a, pi_c, pi_g, pi_t], dtype=numpy.float64)

    r = numpy.array(
        [
            [0.0, ac, ag, at],
            [ac, 0.0, cg, ct],
            [ag, cg, 0.0, gt],
            [at, ct, gt, 0.0],
        ],
        dtype=numpy.float64,
    )
    q = numpy.zeros((4, 4), dtype=numpy.float64)
    for i in range(4):
        for j in range(4):
            if i != j:
                q[i, j] = r[i, j] * pi[j]
        q[i, i] = -numpy.sum(q[i, :])

    # Scale so expected substitution rate is 1.
    expected_rate = -numpy.sum(pi * numpy.diag(q))
    if expected_rate <= 0:
        raise ValueError("Invalid GTR parameters: expected rate must be positive")
    return q / expected_rate


def _gtr_mle_distance(seq_a: str, seq_b: str, rates, freqs, upper_bound: float = 5.0) -> float:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    sites = [(mapping[a], mapping[b]) for a, b in _iter_valid_nuc_sites(seq_a, seq_b)]
    if not sites:
        return 0.0

    q = _gtr_q_matrix(rates, freqs)
    idx_i = numpy.array([a for a, _ in sites], dtype=numpy.int32)
    idx_j = numpy.array([b for _, b in sites], dtype=numpy.int32)

    def neg_log_likelihood(distance: float) -> float:
        p = expm(q * max(distance, 1e-12))
        probs = p[idx_i, idx_j]
        probs = numpy.clip(probs, 1e-12, 1.0)
        return -numpy.sum(numpy.log(probs))

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(1e-8, upper_bound),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


class PDistanceVariant(ControlVariant):
    name = "p-distance"

    def build_matrix(self) -> DistanceStruct:
        gapped = self._build_alignment_strings()
        n = len(self._sequences)
        distances = numpy.zeros((n, n), dtype=numpy.float64)
        for i in range(n):
            for j in range(i, n):
                if gapped is not None:
                    d = _p_distance_from_aligned(gapped[i], gapped[j])
                else:
                    s1, s2 = self._pairwise_align(str(self._sequences[i]), str(self._sequences[j]))
                    d = _p_distance_from_aligned(s1, s2)
                distances[i, j] = d
                distances[j, i] = d
        return DistanceStruct(names=self._names, matrix=distances)

    def _build_alignment_strings(self):
        try:
            alignment = super().build_matrix().align
            return [str(seq) for seq in alignment.get_gapped_sequences()]
        except Exception:
            return None

    @staticmethod
    def _pairwise_align(seq_a: str, seq_b: str):
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -1.0

        alignment = aligner.align(seq_a, seq_b)[0]
        a_blocks, b_blocks = alignment.aligned

        i = 0
        j = 0
        out_a = []
        out_b = []

        for (a0, a1), (b0, b1) in zip(a_blocks, b_blocks):
            while i < a0 and j < b0:
                out_a.append(seq_a[i])
                out_b.append(seq_b[j])
                i += 1
                j += 1
            while i < a0:
                out_a.append(seq_a[i])
                out_b.append('-')
                i += 1
            while j < b0:
                out_a.append('-')
                out_b.append(seq_b[j])
                j += 1

            out_a.append(seq_a[a0:a1])
            out_b.append(seq_b[b0:b1])
            i = a1
            j = b1

        while i < len(seq_a) and j < len(seq_b):
            out_a.append(seq_a[i])
            out_b.append(seq_b[j])
            i += 1
            j += 1
        while i < len(seq_a):
            out_a.append(seq_a[i])
            out_b.append('-')
            i += 1
        while j < len(seq_b):
            out_a.append('-')
            out_b.append(seq_b[j])
            j += 1

        return ''.join(out_a), ''.join(out_b)


class JukesCantorVariant(PDistanceVariant):
    name = "Jukes-Cantor JC69"

    def build_matrix(self) -> DistanceStruct:
        if self._sequence_type != self.nucleotide_type:
            raise ValueError("JC69 baseline is only defined for nucleotide sequences (seq_type='N')")
        gapped = self._build_alignment_strings()
        n = len(self._sequences)
        distances = numpy.zeros((n, n), dtype=numpy.float64)
        for i in range(n):
            for j in range(i, n):
                if gapped is not None:
                    p = _p_distance_from_aligned(gapped[i], gapped[j])
                else:
                    s1, s2 = self._pairwise_align(str(self._sequences[i]), str(self._sequences[j]))
                    p = _p_distance_from_aligned(s1, s2)
                if p >= 0.75:
                    d = 10.0
                else:
                    d = -(3.0 / 4.0) * math.log(1.0 - (4.0 / 3.0) * p)
                distances[i, j] = d
                distances[j, i] = d
        return DistanceStruct(names=self._names, matrix=distances)


class GTRMLEVariant(PDistanceVariant):
    name = "GTR distance (pairwise MLE)"

    def __init__(
        self,
        fasta_file: str,
        sequence_type: str,
        rates=(1.0, 2.0, 1.0, 1.0, 2.0, 1.0),
        freqs=(0.25, 0.25, 0.25, 0.25),
    ):
        super().__init__(fasta_file, sequence_type)
        self._rates = tuple(float(x) for x in rates)
        self._freqs = tuple(float(x) for x in freqs)

    def build_matrix(self) -> DistanceStruct:
        if self._sequence_type != self.nucleotide_type:
            raise ValueError("GTR MLE baseline is only defined for nucleotide sequences (seq_type='N')")
        gapped = self._build_alignment_strings()
        n = len(self._sequences)
        distances = numpy.zeros((n, n), dtype=numpy.float64)
        for i in range(n):
            for j in range(i, n):
                if gapped is not None:
                    s1, s2 = gapped[i], gapped[j]
                else:
                    s1, s2 = self._pairwise_align(str(self._sequences[i]), str(self._sequences[j]))
                d_ij = _gtr_mle_distance(s1, s2, self._rates, self._freqs)
                d_ji = _gtr_mle_distance(s2, s1, self._rates, self._freqs)
                d = 0.5 * (d_ij + d_ji)
                distances[i, j] = d
                distances[j, i] = d
        return DistanceStruct(names=self._names, matrix=distances)


class MuscleIdentityVariant(Variant):
    name = "MUSCLE + p-distance"

    def build_matrix(self) -> DistanceStruct:
        muscle_bin = shutil.which("muscle")
        if muscle_bin is None:
            raise RuntimeError("MUSCLE baseline requested but 'muscle' binary was not found in PATH")

        with tempfile.TemporaryDirectory(prefix="biomodelml_muscle_") as tmp_dir:
            input_fasta = f"{tmp_dir}/input.fasta"
            output_fasta = f"{tmp_dir}/aligned.fasta"

            with open(input_fasta, "w") as f_out:
                for name, seq in zip(self._names, self._sequences):
                    f_out.write(f">{name}\n{str(seq)}\n")

            # Support both MUSCLE v5 and legacy CLI flags.
            cmd_v5 = [muscle_bin, "-align", input_fasta, "-output", output_fasta]
            cmd_legacy = [muscle_bin, "-in", input_fasta, "-out", output_fasta]
            run = subprocess.run(cmd_v5, capture_output=True, text=True)
            if run.returncode != 0:
                run = subprocess.run(cmd_legacy, capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(f"MUSCLE execution failed: {run.stderr.strip()}")

            msa = AlignIO.read(output_fasta, "fasta")
            gapped = [str(rec.seq) for rec in msa]

        n = len(gapped)
        distances = numpy.zeros((n, n), dtype=numpy.float64)
        for i in range(n):
            for j in range(i, n):
                d = _p_distance_from_aligned(gapped[i], gapped[j])
                distances[i, j] = d
                distances[j, i] = d

        return DistanceStruct(names=self._names, matrix=distances)