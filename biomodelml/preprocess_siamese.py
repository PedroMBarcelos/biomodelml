import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from biomodelml.data_generation.training_dataset import TrainingDataset
from biomodelml.variants.siamese_sliding_window import SiameseSlidingWindowVariant
from biomodelml.cli.train_siamese import _collect_pair_examples

def is_window_informative(window: np.ndarray, threshold: float = 0.85) -> bool:
    if np.std(window) < 1e-3:
        return False
    zero_ratio = np.mean(window <= 1e-2)
    return zero_ratio < threshold

def main():
    parser = argparse.ArgumentParser(description="Pré-processamento Otimizado Siamese Multi-Core")
    parser.add_argument("dataset_root", help="Diretório images_output2/")
    parser.add_argument("sequence_type", choices=["N", "P"], help="Tipo de sequência (N ou P)")
    parser.add_argument("--window-size", type=int, default=200, help="Tamanho da janela")
    parser.add_argument("--stride", type=int, default=None, help="Passo da janela")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k factor")
    parser.add_argument("--sparsity-threshold", type=float, default=0.85, help="Filtro de descarte")
    parser.add_argument("--output-dir", default="preprocessed_data", help="Onde salvar")
    parser.add_argument("--shard-size", type=int, default=2000, help="Tamanho do bloco ampliado")
    parser.add_argument("--workers", type=int, default=12, help="Número de núcleos de CPU")
    
    args = parser.parse_args()
    
    dataset = TrainingDataset(args.dataset_root, lazy_load=True)
    pairs = list(_collect_pair_examples(dataset))
    
    if not pairs:
        print("Nenhum par encontrado.")
        return

    variant = SiameseSlidingWindowVariant(
        image_folder=args.dataset_root,
        window_size=args.window_size,
        stride=args.stride,
        top_k=args.top_k,
        sequence_type=args.sequence_type
    )
    
    out_path = Path(args.output_dir) / f"window_{args.window_size}_{args.sequence_type}"
    out_path.mkdir(parents=True, exist_ok=True)
    
    label_scale = max(pair.label for pair in pairs) if pairs else 1.0
    if label_scale <= 0: label_scale = 1.0

    # --- DICIONÁRIO DE CACHE DE IMAGENS ---
    # Evita abrir o mesmo arquivo do disco repetidamente
    image_cache = {}
    print("Pré-carregando e processando geometria das imagens exclusivas na RAM...")
    
    # Coleta todas as imagens únicas necessárias
    unique_indices = set([p.left_index for p in pairs] + [p.right_index for p in pairs])
    for idx in tqdm(unique_indices, desc="Filtrando Janelas"):
        sample = dataset[idx]
        specs = variant._window_specs(sample.image_array)
        valid_windows = []
        for top, left, bottom, right in specs:
            win = sample.image_array[top:bottom, left:right]
            padded_win = variant._pad_window(win)
            if is_window_informative(padded_win, args.sparsity_threshold):
                valid_windows.append(padded_win)
        image_cache[idx] = valid_windows

    print("\nIniciando extração e cruzamento na GPU (Processamento em Paralelo)...")
    
    shard_x, shard_y = [], []
    shard_count = 0

    def process_pair(pair):
        windows1 = image_cache[pair.left_index]
        windows2 = image_cache[pair.right_index]
        if not windows1 or not windows2:
            return None
        
        # Cruzamento via multiplicação de tensores na GPU
        distances, _ = variant._window_distance_matrix_from_windows(windows1, windows2)
        return distances.astype(np.float32), np.float32(pair.label / label_scale)

    # Executa o processamento dos pares distribuindo pelas threads da CPU para alimentar a GPU continuamente
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_pair, pairs), total=len(pairs), desc="Processando Pares"))

    print("\nEmpacotando e salvando Shards no disco...")
    for res in results:
        if res is None: continue
        matrix, label = res
        shard_x.append(matrix)
        shard_y.append(label)
        
        if len(shard_x) == args.shard_size:
            max_height = max(m.shape[0] for m in shard_x)
            max_width = max(m.shape[1] for m in shard_x)
            x_batch = np.zeros((len(shard_x), max_height, max_width, 1), dtype=np.float32)
            for m_idx, m in enumerate(shard_x):
                h, w = m.shape[:2]
                x_batch[m_idx, :h, :w, 0] = m
            y_batch = np.array(shard_y, dtype=np.float32).reshape(-1, 1)
            
            np.save(out_path / f"x_shard_{shard_count}.npy", x_batch)
            np.save(out_path / f"y_shard_{shard_count}.npy", y_batch)
            shard_x, shard_y = [], []
            shard_count += 1

    # Trata o último bloco
    if shard_x:
        max_height = max(m.shape[0] for m in shard_x)
        max_width = max(m.shape[1] for m in shard_x)
        x_batch = np.zeros((len(shard_x), max_height, max_width, 1), dtype=np.float32)
        for m_idx, m in enumerate(shard_x):
            h, w = m.shape[:2]
            x_batch[m_idx, :h, :w, 0] = m
        y_batch = np.array(shard_y, dtype=np.float32).reshape(-1, 1)
        np.save(out_path / f"x_shard_{shard_count}.npy", x_batch)
        np.save(out_path / f"y_shard_{shard_count}.npy", y_batch)
        shard_count += 1

    with open(out_path / "metadata.json", "w") as f:
        json.dump({"label_scale": float(label_scale), "total_shards": shard_count, "window_size": args.window_size, "sequence_type": args.sequence_type}, f, indent=2)
        
    print(f"\n[FIM] Sucesso! {shard_count} shards armazenados.")

if __name__ == "__main__":
    main()