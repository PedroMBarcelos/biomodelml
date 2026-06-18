import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from biomodelml.variants.siamese_sliding_window import SiameseSlidingWindowVariant
from biomodelml.data_generation.training_dataset import TrainingDataset
from biomodelml.cli.train_siamese import _collect_pair_examples, _build_variant

def is_window_informative(window: np.ndarray, threshold: float = 0.85) -> bool:
    """
    Retorna Falso se a janela for quase toda em branco/preta (fundo sem dados biológicos).
    Ajuste o threshold para controlar a agressividade do descarte.
    """
    # Se o desvio padrão for muito baixo, a janela é homogênea (vazia)
    if np.std(window) < 1e-3:
        return False
    
    # Se a porcentagem de pixels de fundo (zeros ou perto disso) for maior que o threshold, descarta
    zero_ratio = np.mean(window <= 1e-2)
    return zero_ratio < threshold

def main():
    parser = argparse.ArgumentParser(description="Pré-processamento Otimizado Siamese")
    parser.add_argument("dataset_root", help="Diretório images_output2/")
    parser.add_argument("sequence_type", choices=["N", "P"], help="Tipo de sequência (N ou P)")
    parser.add_argument("--window-size", type=int, default=200, help="Tamanho da janela")
    parser.add_argument("--stride", type=int, default=None, help="Passo da janela")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k factor")
    parser.add_argument("--sparsity-threshold", type=float, default=0.85, help="Filtro de descarte de janelas vazias")
    parser.add_argument("--output-dir", default="preprocessed_data", help="Onde salvar os shards")
    parser.add_argument("--shard-size", type=int, default=500, help="Tamanho do bloco")
    
    args = parser.parse_args()

    args.feature_input_shape = (224, 224, 3)
    args.dropout = 0.1
    args.learning_rate = 1e-3
    args.batch_size = 2
    args.epochs = 10

    dataset = TrainingDataset(args.dataset_root, lazy_load=True)
    pairs = list(_collect_pair_examples(dataset))
    
    if not pairs:
        print("Nenhum par encontrado.")
        return

    args = parser.parse_args()
    
    # Substitua a chamada do CLI antiga por esta instanciação direta:
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
        
    shard_x = []
    shard_y = []
    shard_count = 0
    
    print(f"Iniciando fatiamento inteligente (Sparsity Threshold: {args.sparsity_threshold})...")
    
    for idx, pair in enumerate(tqdm(pairs)):
        left_sample = dataset[pair.left_index]
        right_sample = dataset[pair.right_index]
        
        # --- ENGENHARIA DE RECORTE INTELIGENTE COM FILTRO ---
        # Extrai posições válidas baseadas no seu método _window_specs original
        specs1 = variant._window_specs(left_sample.image_array)
        specs2 = variant._window_specs(right_sample.image_array)
        
        # Filtra as janelas da imagem 1 tirando as vazias
        windows1 = []
        for top, left, bottom, right in specs1:
            win = left_sample.image_array[top:bottom, left:right]
            padded_win = variant._pad_window(win)
            if is_window_informative(padded_win, args.sparsity_threshold):
                windows1.append(padded_win)
                
        # Filtra as janelas da imagem 2
        windows2 = []
        for top, left, bottom, right in specs2:
            win = right_sample.image_array[top:bottom, left:right]
            padded_win = variant._pad_window(win)
            if is_window_informative(padded_win, args.sparsity_threshold):
                windows2.append(padded_win)
        
        # Se alguma das proteínas ficou sem janelas válidas após o filtro, ignora o par
        if not windows1 or not windows2:
            continue
            
        # Calcula as características e a matriz de distância cruzada exata
        distances, _ = variant._window_distance_matrix_from_windows(windows1, windows2)
        matrix = distances.astype(np.float32)
        label = np.float32(pair.label / label_scale)
        
        shard_x.append(matrix)
        shard_y.append(label)
        
        # Salvamento em blocos compactados
        if len(shard_x) == args.shard_size or idx == len(pairs) - 1:
            if not shard_x: continue
            
            # Força o preenchimento estático para que o lote inteiro caiba na assinatura do Keras
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

    with open(out_path / "metadata.json", "w") as f:
        json.dump({
            "label_scale": float(label_scale),
            "total_shards": shard_count,
            "window_size": args.window_size,
            "sequence_type": args.sequence_type
        }, f, indent=2)
        
    print(f"\nDataset reduzido e salvo com sucesso em {shard_count} shards!")

if __name__ == "__main__":
    main()