import torch
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import os
import faiss
from typing import List, Tuple, Dict, Any
import csv
from time import time


ROOT_PATH = os.getcwd()
CHUNKS_PATH = os.path.join(ROOT_PATH, "chunks")

ORACC_NORM_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_norm_embeddings.csv")
ORACC_NORM_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_norm_meta.csv")

ORACC_LEMMA_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_lemma_embeddings.csv")
ORACC_LEMMA_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_lemma_meta.csv")

ORACC_FORMS_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_embeddings.csv")
ORACC_FORMS_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_meta.csv")

ORACC_NORM_POS_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_norm_pos_embeddings.csv")
ORACC_NORM_POS_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_norm_pos_meta.csv")

ORACC_LEMMA_POS_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_lemma_pos_embeddings.csv")
ORACC_LEMMA_POS_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_lemma_pos_meta.csv")

ORACC_FORMS_POS_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_embeddings.csv")
ORACC_FORMS_POS_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_meta.csv")

ORACC_FORMS_NORMALISED_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_normalised_embeddings.csv")
ORACC_FORMS_NORMALISED_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_normalised_meta.csv")

ORACC_FORMS_POS_NORMALISED_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_normalised_embeddings.csv")
ORACC_FORMS_POS_NORMALISED_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_normalised_meta.csv")

ORACC_SIGNS_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_signs_embeddings.csv")
ORACC_SIGNS_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_signs_meta.csv")

ORACC_SIGNS_NORMALISED_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_signs_normalised_embeddings.csv")
ORACC_SIGNS_NORMALISED_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_signs_normalised_meta.csv")

ORACC_SIGNSGDL_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_signs_gdl_embeddings.csv")
ORACC_SIGNSGDL_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_signs_gdl_meta.csv")

EMBS_NORM_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_normalised_e5.npy")
EMBS_NORM_POS_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_normalised_pos_e5.npy")
EMBS_LEMMA_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_lemma_e5.npy")
EMBS_LEMMA_POS_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_lemma_pos_e5.npy")
EMBS_FORMS_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_forms_e5.npy")
EMBS_FORMS_NORMALISED_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_forms_normalised_e5.npy")
EMBS_FORMS_POS_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_forms_pos_e5.npy")
EMBS_FORMS_POS_NORMALISED_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_forms_pos_normalised_e5.npy")
EMBS_SIGNS_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_signs_e5.npy")
EMBS_SIGNS_NORMALISED_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_signs_normalised_e5.npy")
EMBS_SIGNSGDL_PATH_E5 = os.path.join(CHUNKS_PATH,"embeddings_signs_gdl_e5.npy")

EMBS_NORM_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_normalised_miniLM.npy")
EMBS_NORM_POS_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_normalised_pos_miniLM.npy")
EMBS_LEMMA_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_lemma_miniLM.npy")
EMBS_LEMMA_POS_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_lemma_pos_miniLM.npy")
EMBS_FORMS_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_forms_miniLM.npy")
EMBS_FORMS_POS_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_forms_pos_miniLM.npy")
EMBS_FORMS_NORMALISED_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_forms_normalised_miniLM.npy")
EMBS_FORMS_POS_NORMALISED_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_forms_pos_normalised_miniLM.npy")
EMBS_SIGNS_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_signs_miniLM.npy")
EMBS_SIGNS_NORMALISED_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_signs_normalised_miniLM.npy")
EMBS_SIGNSGDL_PATH_MiniLM = os.path.join(CHUNKS_PATH,"embeddings_signs_gdl_miniLM.npy")

IDS_NORM_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_norm.csv")
IDS_NORM_POS_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_norm_pos.csv")
IDS_LEMMA_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_lemma.csv")
IDS_LEMMA_POS_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_lemma_pos.csv")
IDS_FORMS_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_forms.csv")
IDS_FORMS_POS_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_forms_pos.csv")
IDS_FORMS_NORMALISED_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_forms_normalised.csv")
IDS_FORMS_POS_NORMALISED_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_forms_pos_normalised.csv")
IDS_SIGNS_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_signs.csv")
IDS_SIGNS_NORMALISED_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_signs_normalised.csv")
IDS_SIGNSGDL_PATH = os.path.join(CHUNKS_PATH,"chunk_ids_signs_gdl.csv")

FAISS_NORM_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_norm_e5.faiss")
FAISS_NORM_POS_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_norm_pos_e5.faiss")
FAISS_LEMMA_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_lemma_e5.faiss")
FAISS_LEMMA_POS_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_lemma_pos_e5.faiss")
FAISS_FORMS_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_forms_e5.faiss")
FAISS_FORMS_POS_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_forms_pos_e5.faiss")
FAISS_FORMS_NORMALISED_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_forms_normalised_e5.faiss")
FAISS_FORMS_POS_NORMALISED_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_forms_pos_normalised_e5.faiss")
FAISS_SIGNS_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_signs_e5.faiss")
FAISS_SIGNS_NORMALISED_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_signs_normalised_e5.faiss")
FAISS_SIGNSGDL_PATH_E5 = os.path.join(CHUNKS_PATH, "oracc_signs_gdl_e5.faiss")

FAISS_NORM_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_norm_miniLM.faiss")
FAISS_NORM_POS_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_norm_pos_miniLM.faiss")
FAISS_LEMMA_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_lemma_miniLM.faiss")
FAISS_LEMMA_POS_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_lemma_pos_miniLM.faiss")
FAISS_FORMS_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_forms_miniLM.faiss")
FAISS_FORMS_POS_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_forms_pos_miniLM.faiss")
FAISS_FORMS_NORMALISED_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_forms_normalised_miniLM.faiss")
FAISS_FORMS_POS_NORMALISED_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_forms_pos_normalised_miniLM.faiss")
FAISS_SIGNS_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_signs_miniLM.faiss")
FAISS_SIGNS_NORMALISED_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_signs_normalised_miniLM.faiss")
FAISS_SIGNSGDL_PATH_MiniLM = os.path.join(CHUNKS_PATH, "oracc_signs_gdl_miniLM.faiss")


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def process_chunks(input_csv_path:str, output_embeddings_path:str, output_ids_path:str, model_name:str, device:str='cuda', batch_size:int=256):
    start_ = time()

    df = pd.read_csv(input_csv_path)
    ids = df['chunk_id'].astype(str).tolist()
    texts = df['text'].astype(str).tolist()

    model = SentenceTransformer(model_name, device=device)
    all_vecs = []

    total = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=total):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype('float32')
        
        all_vecs.append(vecs)

    E = np.vstack(all_vecs)
    np.save(output_embeddings_path, E)
    pd.Series(ids, name='chunk_id').to_csv(output_ids_path, index=False)

    end_ = time()
    print("Processing time:", end_ - start_)

    print("Saved to:", output_embeddings_path, E.shape, " / ", output_ids_path, len(ids))

def process_chunks_POS(input_csv_path:str, output_embeddings_path:str, output_ids_path:str,
                   model_name:str, device:str='cuda', batch_size:int=2048,
                   text_prefix:str='passage: '):
    
    start_ = time()

    df = pd.read_csv(input_csv_path)
    ids = df["chunk_id"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    if text_prefix:
        texts = [f"{text_prefix}{t}" for t in texts]

    model = SentenceTransformer(model_name, device=device)

    model.max_seq_length = 96       # (zkus 96; když chceš víc přesnosti, dej 128)
    if device == "cuda":
        model = model.to(torch.float16)
        torch.set_float32_matmul_precision("high")

    all_vecs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=total):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False
                           ).astype("float32")
        all_vecs.append(vecs)
    E = np.vstack(all_vecs)
    np.save(output_embeddings_path, E)
    pd.Series(ids, name="chunk_id").to_csv(output_ids_path, index=False)

    end_ = time()
    print("Processing time:", end_ - start_)

    print("Uloženo:", output_embeddings_path, E.shape, " / ", output_ids_path, len(ids))

def select_paths(mode='normalised', model='e5'):
    if mode == 'normalised':
        if model == 'e5':
            return (ORACC_NORM_embed_csv_PATH, EMBS_NORM_PATH_E5, IDS_NORM_PATH, FAISS_NORM_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_NORM_embed_csv_PATH, EMBS_NORM_PATH_MiniLM, IDS_NORM_PATH, FAISS_NORM_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'normalised_POS':
        if model == 'e5':
            return (ORACC_NORM_POS_embed_csv_PATH, EMBS_NORM_POS_PATH_E5, IDS_NORM_POS_PATH, FAISS_NORM_POS_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_NORM_POS_embed_csv_PATH, EMBS_NORM_POS_PATH_MiniLM, IDS_NORM_POS_PATH, FAISS_NORM_POS_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'lemma':
        if model == 'e5':
            return (ORACC_LEMMA_embed_csv_PATH, EMBS_LEMMA_PATH_E5, IDS_LEMMA_PATH, FAISS_LEMMA_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_LEMMA_embed_csv_PATH, EMBS_LEMMA_PATH_MiniLM, IDS_LEMMA_PATH, FAISS_LEMMA_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'lemma_POS':
        if model == 'e5':
            return (ORACC_LEMMA_POS_embed_csv_PATH, EMBS_LEMMA_POS_PATH_E5, IDS_LEMMA_POS_PATH, FAISS_LEMMA_POS_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_LEMMA_POS_embed_csv_PATH, EMBS_LEMMA_POS_PATH_MiniLM, IDS_LEMMA_POS_PATH, FAISS_LEMMA_POS_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'forms':
        if model == 'e5':
            return (ORACC_FORMS_embed_csv_PATH, EMBS_FORMS_PATH_E5, IDS_FORMS_PATH, FAISS_FORMS_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_FORMS_embed_csv_PATH, EMBS_FORMS_PATH_MiniLM, IDS_FORMS_PATH, FAISS_FORMS_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'forms_POS':
        if model == 'e5':
            return (ORACC_FORMS_POS_embed_csv_PATH, EMBS_FORMS_POS_PATH_E5, IDS_FORMS_POS_PATH, FAISS_FORMS_POS_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_FORMS_POS_embed_csv_PATH, EMBS_FORMS_POS_PATH_MiniLM, IDS_FORMS_POS_PATH, FAISS_FORMS_POS_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'forms_normalised':
        if model == 'e5':
            return (ORACC_FORMS_NORMALISED_embed_csv_PATH, EMBS_FORMS_NORMALISED_PATH_E5, IDS_FORMS_NORMALISED_PATH, FAISS_FORMS_NORMALISED_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_FORMS_NORMALISED_embed_csv_PATH, EMBS_FORMS_NORMALISED_PATH_MiniLM, IDS_FORMS_NORMALISED_PATH, FAISS_FORMS_NORMALISED_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'forms_POS_normalised':
        if model == 'e5':
            return (ORACC_FORMS_POS_NORMALISED_embed_csv_PATH, EMBS_FORMS_POS_NORMALISED_PATH_E5, IDS_FORMS_POS_NORMALISED_PATH, FAISS_FORMS_POS_NORMALISED_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_FORMS_POS_NORMALISED_embed_csv_PATH, EMBS_FORMS_POS_NORMALISED_PATH_MiniLM, IDS_FORMS_POS_NORMALISED_PATH, FAISS_FORMS_POS_NORMALISED_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'signs':
        if model == 'e5':
            return (ORACC_SIGNS_embed_csv_PATH, EMBS_SIGNS_PATH_E5, IDS_SIGNS_PATH, FAISS_SIGNS_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_SIGNS_embed_csv_PATH, EMBS_SIGNS_PATH_MiniLM, IDS_SIGNS_PATH, FAISS_SIGNS_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'signs_normalised':
        if model == 'e5':
            return (ORACC_SIGNS_NORMALISED_embed_csv_PATH, EMBS_SIGNS_NORMALISED_PATH_E5, IDS_SIGNS_NORMALISED_PATH, FAISS_SIGNS_NORMALISED_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_SIGNS_NORMALISED_embed_csv_PATH, EMBS_SIGNS_NORMALISED_PATH_MiniLM, IDS_SIGNS_NORMALISED_PATH, FAISS_SIGNS_NORMALISED_PATH_MiniLM, 'all-MiniLM-L6-v2')
    elif mode == 'signs_gdl':
        if model == 'e5':
            return (ORACC_SIGNSGDL_embed_csv_PATH, EMBS_SIGNSGDL_PATH_E5, IDS_SIGNSGDL_PATH, FAISS_SIGNSGDL_PATH_E5, 'intfloat/e5-base-v2')
        elif model == 'MiniLM':
            return (ORACC_SIGNSGDL_embed_csv_PATH, EMBS_SIGNSGDL_PATH_MiniLM, IDS_SIGNSGDL_PATH, FAISS_SIGNSGDL_PATH_MiniLM, 'all-MiniLM-L6-v2')
    else:
        raise ValueError("Unknown mode: " + mode)

if __name__ == "__main__":
    for mode in ['forms', 'forms_POS', 'signs', 'signs_gdl', 'forms_POS_normalised', 'signs_normalised', 'forms_normalised']: # ['forms_POS', 'normalised_POS', 'lemma_POS', 'forms', 'lemma', 'normalised', 'signs', 'signs_gdl']
        for model in ['e5', 'MiniLM']:
            paths = select_paths(mode=mode, model=model)

            print("Processing:", mode, model)

            if model == 'e5':
                process_chunks_POS(input_csv_path=paths[0], output_embeddings_path=paths[1], output_ids_path=paths[2], model_name=paths[4])

            elif model == 'MiniLM':
                process_chunks(input_csv_path=paths[0], output_embeddings_path=paths[1], output_ids_path=paths[2], model_name=paths[4])
            
            else:
                print("Unknown model:", model)