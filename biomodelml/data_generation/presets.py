"""
Preset configurations for AliSim sequence generation.

Defines standard presets (training, benchmark, noise) with sensible defaults
for different use cases, while allowing parameter override for experimentation.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class AliSimConfig:
    """Configuration for AliSim sequence generation.
    
    Attributes:
        alignment_length: Sequence length in base pairs/amino acids
        evolution_rate: Rate of evolution (e.g., 'fast', 'moderate', 'slow')
        model: Substitution model (e.g., 'GTR', 'HKY', 'K80')
        tree_style: Tree shape ('balanced', 'random', 'power-law')
        num_sequences: Number of sequences per alignment
        description: Human-readable description of the preset
    """
    alignment_length: int
    evolution_rate: str
    model: str
    tree_style: str
    num_sequences: int = 20
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AliSimConfig":
        """Create from dictionary."""
        return cls(**data)


# Preset definitions for different use cases
PRESETS: Dict[str, AliSimConfig] = {
    "training": AliSimConfig(
        alignment_length=500,
        evolution_rate="moderate",
        model="GTR",
        tree_style="balanced",
        num_sequences=20,
        description="Balanced synthetic data optimized for deep learning training. "
                    "Moderate evolutionary rate and balanced trees reduce training instability.",
    ),
    "benchmark": AliSimConfig(
        alignment_length=1000,
        evolution_rate="mixed",
        model="HKY",
        tree_style="random",
        num_sequences=20,
        description="Realistic biological diversity. Longer alignments with mixed evolution rates "
                    "simulate natural sequence variation for benchmarking algorithms.",
    ),
    "noise": AliSimConfig(
        alignment_length=300,
        evolution_rate="high",
        model="GTR",
        tree_style="power-law",
        num_sequences=20,
        description="High evolutionary variation for robustness testing. Short alignments with "
                    "aggressive evolution rates test model stability under noise.",
    ),
}


def get_preset(preset_name: str) -> AliSimConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset ('training', 'benchmark', 'noise')
        
    Returns:
        AliSimConfig for the requested preset
        
    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    return PRESETS[preset_name]


def validate_preset_overrides(
    base_config: AliSimConfig,
    alignment_length: Optional[int] = None,
    evolution_rate: Optional[str] = None,
    model: Optional[str] = None,
    tree_style: Optional[str] = None,
    num_sequences: Optional[int] = None,
) -> AliSimConfig:
    """
    Create a new config from base preset with optional overrides.
    
    Args:
        base_config: Base AliSimConfig to override
        alignment_length: Override alignment length (optional)
        evolution_rate: Override evolution rate (optional)
        model: Override substitution model (optional)
        tree_style: Override tree style (optional)
        num_sequences: Override number of sequences (optional)
        
    Returns:
        New AliSimConfig with overrides applied
    """
    return AliSimConfig(
        alignment_length=alignment_length or base_config.alignment_length,
        evolution_rate=evolution_rate or base_config.evolution_rate,
        model=model or base_config.model,
        tree_style=tree_style or base_config.tree_style,
        num_sequences=num_sequences or base_config.num_sequences,
        description=base_config.description,
    )


# Model selections for nucleotides vs proteins
NUCLEOTIDE_MODELS = [
    "JC",
    "K80",
    "HKY",
    "TN93",
    "GTR",
    "GTR+Gamma",
]

PROTEIN_MODELS = [
    "Blosum62",
    "Dayhoff",
    "JTT",
    "LG",
    "WAG",
    "Q.mammal",
    "Q.plant",
    "Q.yeast",
]

TREE_STYLES = [
    "balanced",
    "random",
    "power-law",
]

EVOLUTION_RATES = [
    "fast",
    "moderate",
    "slow",
    "mixed",
    "high",
]
