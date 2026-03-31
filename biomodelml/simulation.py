import pyvolve
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


DEFAULT_GTR_RATES = (1.0, 2.0, 1.0, 1.0, 2.0, 1.0)
DEFAULT_BASE_FREQS = (0.25, 0.25, 0.25, 0.25)

# Amino acid alphabet and properties for protein evolution
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AVILMFYWP")
CHARGED = set("DEKR")
POLAR = set("STNQC")
AROMATIC = set("FYW")


class ProteinEvolutionGenerator:
    """
    Generates synthetic protein evolution data with selection pressure.

    This generator implements a sophisticated evolution model with:
    - Selection pressure based on amino acid properties
    - Insertions and deletions (indels)
    - Poisson-distributed mutation counts
    - Realistic evolutionary dynamics
    """

    def __init__(self, seq_len=500, mutation_rate=0.002, indel_rate=0.0005,
                 selection_strength=0.3):
        """
        Initialize the protein evolution generator.

        Args:
            seq_len (int): Length of the random parent sequences.
            mutation_rate (float): Probability of substitution per site per branch unit.
            indel_rate (float): Probability of insertion/deletion per site per branch unit.
            selection_strength (float): Strength of selection against unfavorable changes (0-1).
        """
        self.seq_len = seq_len
        self.mutation_rate = mutation_rate
        self.indel_rate = indel_rate
        self.selection_strength = selection_strength

    def _generate_random_sequence(self):
        """Generates a single random protein sequence."""
        return "".join(random.choice(AMINO_ACIDS) for _ in range(self.seq_len))

    def _calculate_fitness_penalty(self, old_aa, new_aa):
        """
        Calculate fitness penalty for an amino acid substitution.

        Higher penalty for drastic changes (e.g., hydrophobic to charged).

        Args:
            old_aa (str): Original amino acid
            new_aa (str): New amino acid

        Returns:
            float: Penalty score (0.0 = no penalty, 1.0 = maximum penalty)
        """
        # Same amino acid - no penalty
        if old_aa == new_aa:
            return 0.0

        # Check if properties change
        old_props = {
            'hydrophobic': old_aa in HYDROPHOBIC,
            'charged': old_aa in CHARGED,
            'polar': old_aa in POLAR,
            'aromatic': old_aa in AROMATIC
        }

        new_props = {
            'hydrophobic': new_aa in HYDROPHOBIC,
            'charged': new_aa in CHARGED,
            'polar': new_aa in POLAR,
            'aromatic': new_aa in AROMATIC
        }

        # Count different properties
        differences = sum(old_props[k] != new_props[k] for k in old_props)

        # Heavy penalty for hydrophobic <-> charged changes
        if (old_props['hydrophobic'] and new_props['charged']) or \
           (old_props['charged'] and new_props['hydrophobic']):
            return 1.0

        # Moderate penalties for other changes
        return differences * 0.3

    def _evolve_sequence(self, parent_seq, distance):
        """
        Evolve a sequence along a branch with given evolutionary distance.

        Args:
            parent_seq (str): Starting sequence
            distance (float): Evolutionary distance (branch length)

        Returns:
            str: Evolved sequence
        """
        seq = list(parent_seq)

        # Point mutations with Poisson distribution
        expected_mutations = int(len(seq) * self.mutation_rate * distance)
        # Add Gaussian noise approximation of Poisson
        num_mutations = max(0, int(random.gauss(expected_mutations, expected_mutations ** 0.5 + 0.1)))

        for _ in range(num_mutations):
            if not seq:
                break

            pos = random.randint(0, len(seq) - 1)
            old_aa = seq[pos]
            new_aa = random.choice(AMINO_ACIDS)

            # Apply selection
            penalty = self._calculate_fitness_penalty(old_aa, new_aa)
            if random.random() > penalty * self.selection_strength:
                seq[pos] = new_aa

        # Indels with Poisson distribution
        expected_indels = int(len(seq) * self.indel_rate * distance)
        num_indels = max(0, int(random.gauss(expected_indels, (expected_indels + 1) ** 0.5)))

        for _ in range(num_indels):
            if not seq:
                break

            if random.random() < 0.5 and len(seq) > 10:  # Deletion
                pos = random.randint(0, len(seq) - 1)
                del seq[pos]
            else:  # Insertion
                pos = random.randint(0, len(seq))
                new_aa = random.choice(AMINO_ACIDS)
                seq.insert(pos, new_aa)

        return ''.join(seq)

    def generate_evolution_pair(self):
        """
        Simulates a single evolution event to generate a parent-mutated pair.

        Returns:
            tuple: A tuple containing:
                - parent_seq (SeqRecord): The original sequence.
                - mutated_seq (SeqRecord): The evolved sequence.
                - distance (float): The evolutionary distance used for the simulation.
        """
        # 1. Generate a random parent sequence
        parent_seq_str = self._generate_random_sequence()
        parent_seq_record = SeqRecord(Seq(parent_seq_str), id="parent")

        # 2. Define a random evolutionary distance
        distance = random.uniform(0.01, 1.5)

        # 3. Evolve the sequence
        mutated_seq_str = self._evolve_sequence(parent_seq_str, distance)
        mutated_seq_record = SeqRecord(Seq(mutated_seq_str), id="mutated")

        return parent_seq_record, mutated_seq_record, distance

    def generate_batch(self, num_pairs):
        """
        Generates a batch of evolution pairs.

        Args:
            num_pairs (int): The number of pairs to generate.

        Yields:
            tuple: A tuple for each generated pair (parent_seq, mutated_seq, distance).
        """
        for _ in range(num_pairs):
            yield self.generate_evolution_pair()


class SyntheticEvolutionGenerator:
    """
    Generates synthetic evolutionary data using Pyvolve.
    This class simulates sequence evolution to create pairs of parent-mutated
    sequences along with their evolutionary distance. This data is crucial for
    training the Siamese network when empirical data is scarce.
    """
    def __init__(
        self,
        alphabet='ACGT',
        seq_len=500,
        model_name='gtr',
        kappa=4.0,
        gtr_rates=DEFAULT_GTR_RATES,
        base_freqs=DEFAULT_BASE_FREQS,
        gamma_alpha=0.5,
        gamma_categories=4,
        p_invariant=0.1,
        indel_rate=0.0005,
        indel_size=1,
        max_len=None
    ):
        """
        Initializes the generator.
        Args:
            alphabet (str): The sequence alphabet ('ACGT' for nucleotides).
            seq_len (int): The length of the random parent sequences.
        """
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.model_name = model_name.lower()
        self.kappa = float(kappa)
        self.gtr_rates = tuple(gtr_rates)
        self.base_freqs = tuple(base_freqs)
        self.gamma_alpha = float(gamma_alpha)
        self.gamma_categories = int(gamma_categories)
        self.p_invariant = float(p_invariant)
        self.indel_rate = float(indel_rate)
        self.indel_size = max(1, int(indel_size))
        self.max_len = int(max_len) if max_len is not None else None
        self._pyvolve_indel_supported = False
        self.model = self._get_model()

    def _get_model(self):
        """Returns a nucleotide substitution model for Pyvolve."""
        if self.model_name == 'hky85':
            params = {
                "kappa": self.kappa,
                "state_freqs": list(self.base_freqs),
                "alpha": self.gamma_alpha,
                "num_categories": self.gamma_categories,
                "pinv": self.p_invariant,
            }
            return pyvolve.Model("nucleotide", params)

        if self.model_name == 'gtr':
            params = {
                "mu": {
                    "AC": self.gtr_rates[0],
                    "AG": self.gtr_rates[1],
                    "AT": self.gtr_rates[2],
                    "CG": self.gtr_rates[3],
                    "CT": self.gtr_rates[4],
                    "GT": self.gtr_rates[5],
                },
                "state_freqs": list(self.base_freqs),
                "alpha": self.gamma_alpha,
                "num_categories": self.gamma_categories,
                "pinv": self.p_invariant,
            }
            return pyvolve.Model("nucleotide", params)

        raise ValueError(f"Unsupported model_name: {self.model_name}. Use 'gtr' or 'hky85'.")

    def _build_partition(self):
        """
        Build a Pyvolve partition, attempting indel-aware parameters first.

        Some Pyvolve versions expose different indel option names. We attempt
        a best-effort indel setup and gracefully fall back to substitution-only.
        """
        base_kwargs = {"models": self.model, "size": self.seq_len}
        indel_kwargs = {
            "indel_rate": self.indel_rate,
            "indel_size": self.indel_size,
        }

        if self.indel_rate <= 0:
            self._pyvolve_indel_supported = False
            return pyvolve.Partition(**base_kwargs)

        try:
            self._pyvolve_indel_supported = True
            return pyvolve.Partition(**base_kwargs, **indel_kwargs)
        except TypeError:
            self._pyvolve_indel_supported = False
            print("Warning: This Pyvolve build does not expose indel Partition options; continuing without Pyvolve-managed indels.")
            return pyvolve.Partition(**base_kwargs)

    def _apply_fallback_indels(self, seq):
        """Apply lightweight indels if Pyvolve-level indels are unavailable."""
        if self.indel_rate <= 0:
            return seq

        seq_list = list(seq)
        expected_indels = int(max(0, len(seq_list) * self.indel_rate))
        n_events = max(0, int(random.gauss(expected_indels, (expected_indels + 1) ** 0.5)))

        for _ in range(n_events):
            if not seq_list:
                break
            is_deletion = random.random() < 0.5 and len(seq_list) > 10
            if is_deletion:
                del_size = min(self.indel_size, len(seq_list) - 1)
                if del_size <= 0:
                    continue
                pos = random.randint(0, len(seq_list) - del_size)
                del seq_list[pos:pos + del_size]
            else:
                ins_size = self.indel_size
                pos = random.randint(0, len(seq_list))
                ins = [random.choice(self.alphabet) for _ in range(ins_size)]
                seq_list[pos:pos] = ins

        return ''.join(seq_list)

    def _clamp_sequence_len(self, seq):
        """Clamp sequence length so downstream matrix generation remains valid."""
        if self.max_len is None:
            return seq
        return seq[:self.max_len]

    def _generate_random_sequence(self):
        """Generates a single random DNA sequence."""
        return "".join(random.choice(self.alphabet) for _ in range(self.seq_len))

    def generate_evolution_pair(self):
        """
        Simulates a single evolution event to generate a parent-mutated pair.
        Returns:
            tuple: A tuple containing:
                - parent_seq (SeqRecord): The original sequence.
                - mutated_seq (SeqRecord): The evolved sequence.
                - distance (float): The branch length (evolutionary distance)
                                    used for the simulation.
        """
        # 1. Generate a random parent sequence
        parent_seq_str = self._generate_random_sequence()
        parent_seq_record = SeqRecord(Seq(parent_seq_str), id="parent")

        # 2. Define a simple tree with a random branch length
        distance = random.uniform(0.01, 1.5) # Evolutionary distance
        tree_str = f"(parent:{distance},mutated:0.0);"
        tree = pyvolve.read_tree(tree=tree_str)

        # 3. Set up the partition
        partition = self._build_partition()

        # 4. Evolve the sequence
        evolver = pyvolve.Evolver(
            tree=tree,
            partitions=partition,
            custom_seqs={"parent": parent_seq_str}
        )
        evolver(seqfile=None, ratefile=None, infofile=None)

        # 5. Extract the mutated sequence
        # The evolved sequences are stored in evolver.get_sequences()
        # We need to find the one that is not the parent.
        mutated_seq_str = evolver.get_sequences(anc=False)['mutated']
        if self.indel_rate > 0 and not self._pyvolve_indel_supported:
            mutated_seq_str = self._apply_fallback_indels(mutated_seq_str)
        mutated_seq_str = self._clamp_sequence_len(mutated_seq_str)
        parent_seq_str = self._clamp_sequence_len(parent_seq_str)
        parent_seq_record = SeqRecord(Seq(parent_seq_str), id="parent")
        mutated_seq_record = SeqRecord(Seq(mutated_seq_str), id="mutated")

        return parent_seq_record, mutated_seq_record, distance

    def generate_batch(self, num_pairs):
        """
        Generates a batch of evolution pairs.
        Args:
            num_pairs (int): The number of pairs to generate.
        Yields:
            tuple: A tuple for each generated pair (parent_seq, mutated_seq, distance).
        """
        for _ in range(num_pairs):
            yield self.generate_evolution_pair()


def get_generator(seq_type, **kwargs):
    """
    Factory function to get the appropriate sequence generator.

    Args:
        seq_type (str): 'N' for nucleotides, 'P' for proteins
        **kwargs: Additional arguments passed to the generator

    Returns:
        Generator instance (SyntheticEvolutionGenerator or ProteinEvolutionGenerator)

    Raises:
        ValueError: If seq_type is not 'N' or 'P'
    """
    if seq_type == 'N':
        # Nucleotide generator using Pyvolve
        return SyntheticEvolutionGenerator(**kwargs)
    elif seq_type == 'P':
        # Protein generator with selection pressure
        return ProteinEvolutionGenerator(**kwargs)
    else:
        raise ValueError(f"Invalid seq_type: {seq_type}. Must be 'N' (nucleotide) or 'P' (protein)")


if __name__ == '__main__':
    # Example of how to use the generator
    generator = SyntheticEvolutionGenerator(seq_len=100)
    print("Generating 5 synthetic evolution pairs...")
    for i, (parent, mutated, dist) in enumerate(generator.generate_batch(5)):
        print(f"--- Pair {i+1} (Distance: {dist:.4f}) ---")
        print(f"Parent:  {parent.seq[:60]}...")
        print(f"Mutated: {mutated.seq[:60]}...")
        print("-" * (20 + len(str(i+1)) + len(f"{dist:.4f}")))
