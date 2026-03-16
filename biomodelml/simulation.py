import pyvolve
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

class SyntheticEvolutionGenerator:
    """
    Generates synthetic evolutionary data using Pyvolve.
    This class simulates sequence evolution to create pairs of parent-mutated
    sequences along with their evolutionary distance. This data is crucial for
    training the Siamese network when empirical data is scarce.
    """
    def __init__(self, alphabet='ACGT', seq_len=500):
        """
        Initializes the generator.
        Args:
            alphabet (str): The sequence alphabet ('ACGT' for nucleotides).
            seq_len (int): The length of the random parent sequences.
        """
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.model = self._get_model()

    def _get_model(self):
        """Returns a nucleotide substitution model for Pyvolve."""
        # Using the HKY85 model as a reasonable default for nucleotide evolution
        return pyvolve.Model("nucleotide", {"kappa": 4.0})

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
        partition = pyvolve.Partition(models=self.model, size=self.seq_len)

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

if __name__ == '__main__':
    # Example of how to use the generator
    generator = SyntheticEvolutionGenerator(seq_len=100)
    print("Generating 5 synthetic evolution pairs...")
    for i, (parent, mutated, dist) in enumerate(generator.generate_batch(5)):
        print(f"--- Pair {i+1} (Distance: {dist:.4f}) ---")
        print(f"Parent:  {parent.seq[:60]}...")
        print(f"Mutated: {mutated.seq[:60]}...")
        print("-" * (20 + len(str(i+1)) + len(f"{dist:.4f}")))
