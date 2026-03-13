import os
import numpy
import itertools
import cv2
from Bio.Seq import Seq
from biotite.sequence.align import SubstitutionMatrix
from biotite.sequence import NucleotideSequence, ProteinSequence
from biomodelml.submatrix import SNEATH, PROTSUB

NUCLEOTIDES = NucleotideSequence.alphabet_unamb.get_symbols()
HYDROPHILIC = "STYNQDE"
HYDROPHOBIC = "GAVCPLIMWFKRH"
ALL_PROTEINS = ProteinSequence.alphabet.get_symbols()
PROTEINS = HYDROPHILIC+HYDROPHOBIC

def _weight_ptns(seq1: Seq, seq2: Seq, rows: numpy.ndarray, max_window: int):

    rgb_dict = {}
    total_aa_combines = itertools.combinations_with_replacement(ALL_PROTEINS, 2)
    substmatrix = SubstitutionMatrix.dict_from_str(PROTSUB)
    min_subst = min(substmatrix.values())
    max_subst = max(substmatrix.values()) + abs(min_subst)
    sneath = SubstitutionMatrix.dict_from_str(SNEATH)
    min_sneath = min(sneath.values())
    max_sneath = max(sneath.values())-min_sneath
    for aa1, aa2 in total_aa_combines:

        color_by_subst = min(max_window, max(0, round((substmatrix[aa1, aa2]+abs(min_subst))*max_window/max_subst)))
        color_by_sneath = min(max_window, max(0, round((sneath.get((aa1, aa2), min_sneath)-min_sneath)*max_window/(max_sneath))))
        combinations = max_window if aa1 == aa2 else 0

        if ((aa1 in PROTEINS) and (aa2 in PROTEINS)):
            rgb_dict[(aa1, aa2)] = rgb_dict[(aa2, aa1)] = (color_by_subst,
                                                            combinations,
                                                            color_by_sneath
                                                            )

        else:
            rgb_dict[(aa1, aa2)] = rgb_dict[(aa2, aa1)] = (color_by_subst,
                                                            0,
                                                            0,
                                                            )
    for line, letter1 in enumerate(seq2):
        for col, letter2 in enumerate(seq1):
            rows[line, col, :] = rgb_dict[letter1, letter2]

    return rows


def _weight_seqs(seq1: Seq, seq2: Seq, rows: numpy.ndarray, max_window: int):
    indexes = dict()
    for line, letter in enumerate(seq2):
        if (letter in NUCLEOTIDES):
            if (letter not in indexes):
                indexes[letter] = numpy.where(
                    numpy.array(list(seq1)) == seq2[line])[0]
            idx = indexes[letter]
            rows[line, idx] = max_window
    return rows


def build_matrix(seq1: Seq, seq2: Seq, max_window: int, seq_type: str):
    """
    Primeira sequência é a coluna e segunda é a linha.
    Retorna a soma de todas as janelas em 2 dimensões, na normal e na reversa.

    :rtype: numpy array (len segunda, len primeira, 2)
    """
    len2 = len(seq2)
    len1 = len(seq1)
    rows = numpy.zeros((len2, len1, 3), numpy.uint8)
    if seq_type == "N":
        seq1 = str(seq1)
        seq2_complement = str(seq2.complement())
        seq2 = str(seq2)

        #  red
        rows[:, :, 0] = _weight_seqs(seq1, seq2, rows[:, :, 0], max_window)
        #  green
        rows[:, :, 1] = _weight_seqs(
            seq1, seq2_complement, rows[:, :, 1], max_window)
        #  blue
        all_lines, all_columns = numpy.where(
            (rows[:, :, 0] == 0) & (rows[:, :, 1] == 0))
        rows[all_lines, all_columns, 2] = max_window
    elif seq_type == "P":

        rows = _weight_ptns(seq1, seq2, rows, max_window)

    return rows


def _save_grayscale(
    matrix: numpy.ndarray,
    output_path: str,
    filename: str
):
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(
        os.path.join(output_path, filename), matrix
    )


def _produce_grayscale_groups(
        matrix: numpy.ndarray,
        output_path: str,
        filename: str):

    groups = {
        "gray_max": numpy.max,
        "gray_mean": numpy.mean
    }

    for name, group in groups.items():
        _save_grayscale(group(matrix, axis=2), os.path.join(
            output_path, name), filename)


def _produce_grayscale_by_channel(
        channel_name: str,
        matrix: numpy.ndarray,
        output_path: str,
        filename: str):

    channels = {
        "gray_r": 0,
        "gray_g": 1,
        "gray_b": 2
    }

    _save_grayscale(matrix[:, :, channels[channel_name]],
                    os.path.join(output_path, channel_name), filename)


def _produce_by_channel(
    channel_name: str,
    matrix: numpy.ndarray,
    output_path: str,
    filename: str
):
    channels_not = {
        "red": (1, 2),
        "green": (0, 2),
        "blue": (0, 1),
        "red_blue": (1,),
        "red_green": (2,),
        "green_blue": (0,)
    }
    new_matrix = matrix.copy()
    os.makedirs(os.path.join(output_path, channel_name), exist_ok=True)
    for channel in channels_not.get(channel_name, []):
        new_matrix[:, :, channel] = 0
    
    # Convert RGB to BGR for OpenCV
    bgr_matrix = cv2.cvtColor(new_matrix, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        os.path.join(output_path, channel_name, filename),
        bgr_matrix
    )


def _produce_channel_images(generate_variations=False, **kwargs):
    if len(kwargs["matrix"].shape) == 3:
        _produce_by_channel("full", **kwargs)
        if generate_variations:
            _produce_by_channel("red", **kwargs)
            _produce_by_channel("green", **kwargs)
            _produce_by_channel("blue", **kwargs)
            _produce_by_channel("red_blue", **kwargs)
            _produce_by_channel("red_green", **kwargs)
            _produce_by_channel("green_blue", **kwargs)
            _produce_grayscale_by_channel("gray_r", **kwargs)
            _produce_grayscale_by_channel("gray_g", **kwargs)
            _produce_grayscale_by_channel("gray_b", **kwargs)
            _produce_grayscale_groups(**kwargs)
    else:
        kwargs["output_path"] = os.path.join(kwargs["output_path"], "full")
        _save_grayscale(**kwargs)


def save_image_by_matrices(
        name1: str, name2: str, seq1: Seq, seq2: Seq,
        max_window: int, output_path: str, seq_type: str,
        generate_variations: bool = False):
    """
    Generate and save sequence comparison matrix as images.
    
    Args:
        name1, name2: Sequence identifiers
        seq1, seq2: Biological sequences
        max_window: Maximum pixel value
        output_path: Root directory for image storage
        seq_type: 'N' for nucleotides, 'P' for proteins
        generate_variations: If True, generate all 11 channel variations.
                            If False (default), generate only full/ RGB image.
    """
    matrix = build_matrix(seq1, seq2, max_window, seq_type)
    filename = f"{name1}x{name2}.png" if name1 != name2 else f"{name1}.png"
    _produce_channel_images(
        generate_variations=generate_variations,
        matrix=matrix, output_path=output_path, filename=filename)


def extract_channel(image_array: numpy.ndarray, channel: str) -> numpy.ndarray:
    """
    Extract specific channel(s) from RGB image array in memory.
    No disk I/O required - use this instead of loading from channel subdirectories.
    
    Args:
        image_array: RGB numpy array (H, W, 3) with dtype uint8
        channel: Channel type to extract:
                 'red', 'green', 'blue' - single color channel (others zeroed)
                 'red_blue', 'red_green', 'green_blue' - channel combinations
                 'gray_r', 'gray_g', 'gray_b' - grayscale of individual channels
                 'gray_max' - grayscale using max(R, G, B)
                 'gray_mean' - grayscale using mean(R, G, B)
                 'full' - returns original array unchanged
    
    Returns:
        numpy.ndarray: Extracted channel(s) as RGB (for color) or grayscale array
    
    Examples:
        >>> img = build_matrix(seq1, seq2, 255, "N")
        >>> red_only = extract_channel(img, 'red')  # Shape: (H, W, 3), red channel only
        >>> gray = extract_channel(img, 'gray_max')  # Shape: (H, W), grayscale
    """
    if channel == 'full':
        return image_array.copy()
    
    # Single color channels (RGB with specific channel, others zeroed)
    if channel == 'red':
        result = image_array.copy()
        result[:, :, 1:3] = 0
        return result
    elif channel == 'green':
        result = image_array.copy()
        result[:, :, [0, 2]] = 0
        return result
    elif channel == 'blue':
        result = image_array.copy()
        result[:, :, 0:2] = 0
        return result
    
    # Channel combinations
    elif channel == 'red_blue':
        result = image_array.copy()
        result[:, :, 1] = 0  # Zero green
        return result
    elif channel == 'red_green':
        result = image_array.copy()
        result[:, :, 2] = 0  # Zero blue
        return result
    elif channel == 'green_blue':
        result = image_array.copy()
        result[:, :, 0] = 0  # Zero red
        return result
    
    # Grayscale conversions (return 2D arrays)
    elif channel == 'gray_r':
        return image_array[:, :, 0]
    elif channel == 'gray_g':
        return image_array[:, :, 1]
    elif channel == 'gray_b':
        return image_array[:, :, 2]
    elif channel == 'gray_max':
        return numpy.max(image_array, axis=2)
    elif channel == 'gray_mean':
        return numpy.mean(image_array, axis=2).astype(numpy.uint8)
    
    else:
        raise ValueError(
            f"Unknown channel type: {channel}. "
            f"Valid options: 'red', 'green', 'blue', 'red_blue', 'red_green', "
            f"'green_blue', 'gray_r', 'gray_g', 'gray_b', 'gray_max', 'gray_mean', 'full'"
        )
