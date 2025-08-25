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
from backend.intertextulity_package import OraccCorpus, select_paths


ROOT_PATH = os.getcwd()
CORPUS_PATH = os.path.join(ROOT_PATH, 'CORPUS')
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


""" Chunking functions. ----------------------------------------------------- """


def make_windows(seq: List[str], window: int, stride: int, drop_last: bool=False) -> List[Tuple[int,int,List[str]]]:
    """
    Creates a list of strideping windows from the input sequence.

    :param seq: Input sequence (list of tokens/characters)
    :param window: Size of the window
    :param stride: Stride (step size) for moving the window
    :param drop_last: Whether to drop the last window if it's smaller than the specified size
    :return: List of tuples (start_idx, end_idx, subseq)
    """
    n = len(seq)
    out = []
    if n == 0 or window <= 0 or stride <= 0:
        return out

    i = 0
    while i < n:
        j = i + window
        if j > n:
            if drop_last:
                break
            j = n
        out.append((i, j, seq[i:j]))
        if j == n:
            break
        i += stride
    return out


def chunkORACCtext(input_orrac_corpus: OraccCorpus, oracc_text_ID:str, mode: str='normalised', chunk_size: int=10, stride: int=5, drop_last: bool=False, unknown_policy: str='compress', skip_all_unknown: bool=True) -> List[Dict[str, Any]]:
    """ Parsing ORACC to chunks. 

    :param input_orrac_corpus: The ORACC corpus object
    :param oracc_text_ID: The ID of the ORACC text to process
    :param mode: The mode for text retrieval (e.g., 'normalised')
    :param chunk_size: The size of each chunk
    :param stride: The stride between chunks
    :param drop_last: Whether to drop the last chunk if it's smaller than chunk_size
    :param unknown_policy: Whether to drop unknown words (select 'compress'|'keep')
    :param skip_all_unknown: Whether to skip chunks that are entirely unknown
    :return: A tuple containing a list of chunks as tuples (start_idx, end_idx, subseq)
    """

    oraccText = input_orrac_corpus.get_data_by_id(oracc_text_ID, mode=mode)

    windows = make_windows(oraccText, window=chunk_size, stride=stride, drop_last=drop_last)

    out = []
    for s, e, subseq_raw in windows:
        # DISPLAY data
        text_display = ' '.join(list(subseq_raw))

        # EMBEDDING data (UKNOWN handling)
        if unknown_policy == 'compress':
            emb_tokens = [t for t in subseq_raw if t != '∎']
        elif unknown_policy == 'keep':
            emb_tokens = list(subseq_raw)
        else:
            raise ValueError("unknown_policy must be 'compress'|'keep'")

        text_embed = ' '.join(emb_tokens).strip()
        if skip_all_unknown and text_embed == '' or text_embed == '∎':
            continue

        out.append({
            'start': s,
            'end': e,
            'text_display': text_display,
            'text_embed': text_embed,
        })
    return out


def export_corpus_to_csv(corpus: OraccCorpus, out_embed_csv: str, out_meta_csv: str, mode: str, chunk_size: int = 10, stride: int = 5, drop_last: bool = False, unknown_policy: str='compress', skip_all_unknown: bool=True):

    if mode=='normalised':
        unit_tag='n'
    elif mode=='forms':
        unit_tag='f'
    elif mode=='forms_normalised':
        unit_tag='fn'
    elif mode=='lemma':
        unit_tag='l'
    elif mode=='forms_pos':
        unit_tag='fp'
    elif mode=='forms_pos_normalised':
        unit_tag='fpn'
    elif mode=='lemma_pos':
        unit_tag='lp'
    elif mode=='normalised_pos':
        unit_tag='np'
    elif mode=='signs':
        unit_tag='s'
    elif mode=='signs_normalised':
        unit_tag='sn'
    elif mode=='signs_gdl':
        unit_tag='sg'
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'normalised', 'forms', 'lemma', 'normalised_pos', 'forms_pos', 'lemma_pos', 'signs', 'signs_gdl', 'forms_normalised', 'forms_POS_normalised', 'signs_normalised'.")

    with open(out_embed_csv, "w", newline="", encoding="utf-8") as fe, \
         open(out_meta_csv,  "w", newline="", encoding="utf-8") as fm:
        we = csv.writer(fe)
        wm = csv.writer(fm)
        we.writerow(["chunk_id", "text"])
        wm.writerow(["chunk_id", "text_id", "start", "end", "text_display"])

    total_embed = 0
    total_meta = 0

    # Streaming text chunks
    for textID in tqdm(corpus.texts, desc='Processing ORACC texts'):
        recs = chunkORACCtext(
            input_orrac_corpus=corpus,
            oracc_text_ID=textID,
            mode=mode,
            chunk_size=chunk_size,
            stride=stride,
            drop_last=drop_last,
            unknown_policy=unknown_policy, 
            skip_all_unknown=skip_all_unknown
        )

        with open(out_embed_csv, 'a', newline='', encoding='utf-8') as fe, \
             open(out_meta_csv,  'a', newline='', encoding='utf-8') as fm:
            we = csv.writer(fe)
            wm = csv.writer(fm)

            for r in recs:
                chunk_id = f"{textID}:{unit_tag}:{r['start']}-{r['end']}"

                # meta zapisujeme vždy (aby šlo v UI projít vše, i když embed není)
                wm.writerow([chunk_id, textID, r['start'], r['end'], r['text_display']])
                total_meta += 1

                # do embedding CSV jen neprázdné texty (tvoje logika už ∎ vyhodila)
                if r['text_embed']:
                    we.writerow([chunk_id, r['text_embed']])
                    total_embed += 1

    print(f'Saved → {out_embed_csv}: {total_embed} rows')
    print(f'Saved → {out_meta_csv}:  {total_meta} rows')


""" Embedding functions. ---------------------------------------------------- """

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def _resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def process_chunks(input_csv_path:str, output_embeddings_path:str, output_ids_path:str, model_name:str, device:str='auto', batch_size:int | None = None):
    start_ = time()

    device = _resolve_device(device)

    if batch_size is None:
        batch_size = 1024 if device == 'cuda' else 128

    df = pd.read_csv(input_csv_path)
    ids = df['chunk_id'].astype(str).tolist()
    texts = df['text'].astype(str).tolist()

    model = SentenceTransformer(model_name, device=device)
    if device == 'cuda':
        model = model.to(torch.float16)
        torch.set_float32_matmul_precision('high')
    all_vecs = []

    total = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=total):
        batch = texts[i:i+batch_size]
        with torch.inference_mode():
            vecs = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype('float32')
        
        all_vecs.append(vecs)

    E = np.vstack(all_vecs)
    np.save(output_embeddings_path, E)
    pd.Series(ids, name='chunk_id').to_csv(output_ids_path, index=False)

    end_ = time()
    print("Processing time:", end_ - start_)

    print("Saved to:", output_embeddings_path, E.shape, " / ", output_ids_path, len(ids))

def process_chunks_POS(input_csv_path:str, output_embeddings_path:str, output_ids_path:str,
                   model_name:str, device:str='auto', batch_size:int | None = None,
                   text_prefix:str='passage: '):
    
    start_ = time()

    device = _resolve_device(device)

    if batch_size is None:
        batch_size = 2048 if device == 'cuda' else 128

    df = pd.read_csv(input_csv_path)
    ids = df['chunk_id'].astype(str).tolist()
    texts = df['text'].astype(str).tolist()

    if text_prefix:
        texts = [f"{text_prefix}{t}" for t in texts]

    model = SentenceTransformer(model_name, device=device)

    model.max_seq_length = 96
    if device == 'cuda':
        model = model.to(torch.float16)
        torch.set_float32_matmul_precision('high')

    all_vecs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=total):
        batch = texts[i:i+batch_size]
        with torch.inference_mode():
            vecs = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype('float32')
        all_vecs.append(vecs)
    E = np.vstack(all_vecs)
    np.save(output_embeddings_path, E)
    pd.Series(ids, name='chunk_id').to_csv(output_ids_path, index=False)

    end_ = time()
    print('Processing time:', end_ - start_)

    print('Saved:', output_embeddings_path, E.shape, '/' , output_ids_path, len(ids))
    

""" Creating FAISS index ---------------------------------------------------- """


def make_FAISS(input_embeddings_path:str, output_faiss_path:str, nlist:int=1024):

    MIN_NLIST, MAX_NLIST = 256, 32768
    TRAIN_MULTIPLIER = 128

    print('\tLoading embeddings...')
    E = np.load(input_embeddings_path).astype('float32')
    N, d = E.shape
    print(f'\tVectors: {N}, dim: {d}')

    nlist = int(4 * (N ** 0.5))
    nlist = max(MIN_NLIST, min(nlist, MAX_NLIST))
    print(f'\tnlist = {nlist}')

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    train_size = min(N, nlist * TRAIN_MULTIPLIER)
    print(f'\tTrain size: {train_size}')
    rng = np.random.default_rng(42)
    train_idx = rng.choice(N, size=train_size, replace=False)
    train_vecs = E[train_idx]

    print('\tTraining IVF...')
    index.train(train_vecs)
    assert index.is_trained

    print('\tAdding vectors to index...')
    index.add(E)   # lze i po dávkách, ale add() si data zkopíruje

    faiss.write_index(index, output_faiss_path)
    print(f'DONE. Index saved: {output_faiss_path}  |  ntotal={index.ntotal}')


if __name__ == "__main__":

    oracc_corpus = oracc_corpus = OraccCorpus(projects_path=CORPUS_PATH, files_prefix='prnd_no_comp')
    
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_NORM_embed_csv_PATH, out_meta_csv=ORACC_NORM_meta_csv_PATH, mode='normalised')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_LEMMA_embed_csv_PATH, out_meta_csv=ORACC_LEMMA_meta_csv_PATH, mode='lemma')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_FORMS_embed_csv_PATH, out_meta_csv=ORACC_FORMS_meta_csv_PATH, mode='forms')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_FORMS_NORMALISED_embed_csv_PATH, out_meta_csv=ORACC_FORMS_NORMALISED_meta_csv_PATH, mode='forms_normalised')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_NORM_POS_embed_csv_PATH, out_meta_csv=ORACC_NORM_POS_meta_csv_PATH, mode='normalised_pos')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_LEMMA_POS_embed_csv_PATH, out_meta_csv=ORACC_LEMMA_POS_meta_csv_PATH, mode='lemma_pos')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_FORMS_POS_embed_csv_PATH, out_meta_csv=ORACC_FORMS_POS_meta_csv_PATH, mode='forms_pos')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_FORMS_POS_NORMALISED_embed_csv_PATH, out_meta_csv=ORACC_FORMS_POS_NORMALISED_meta_csv_PATH, mode='forms_pos_normalised')
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_SIGNS_embed_csv_PATH, out_meta_csv=ORACC_SIGNS_meta_csv_PATH, mode='signs', chunk_size=25, stride=10)
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_SIGNS_NORMALISED_embed_csv_PATH, out_meta_csv=ORACC_SIGNS_NORMALISED_meta_csv_PATH, mode='signs_normalised', chunk_size=25, stride=10)
    export_corpus_to_csv(corpus=oracc_corpus, out_embed_csv=ORACC_SIGNSGDL_embed_csv_PATH, out_meta_csv=ORACC_SIGNSGDL_meta_csv_PATH, mode='signs_gdl', chunk_size=25, stride=10)
    
    print('Chunking was finished.')

    print('')

    print('Embedding chunks... this make take a lot of time (hours) ...')
    
    for mode in ['normalised', 'normalised_POS', 'lemma', 'lemma_POS', 'forms', 'forms_normalised', 'forms_POS', 'forms_POS_normalised', 'signs', 'signs_normalised', 'signs_gdl']:
        for model in ['e5', 'MiniLM']:
            paths = select_paths(mode=mode, model=model)

            print("Processing:", mode, model)

            if model == 'e5':
                process_chunks_POS(input_csv_path=paths[0], output_embeddings_path=paths[1], output_ids_path=paths[2], model_name=paths[4])

            elif model == 'MiniLM':
                process_chunks(input_csv_path=paths[0], output_embeddings_path=paths[1], output_ids_path=paths[2], model_name=paths[4])
            
            else:
                print("Unknown model:", model)

    print("All chunks were embedded.")

    print('')

    print('Making the FAISS index...')

    for mode in ['normalised', 'normalised_pos', 'lemma', 'lemma_pos', 'forms', 'forms_pos', 'forms_normalised', 'forms_pos_normalised', 'signs', 'signs_normalised', 'signs_gdl']:
        for model in ['vect_e5', 'vect_MiniLM']:
            paths = select_paths(mode=mode, model=model)

            make_FAISS(input_embeddings_path=paths[1], output_faiss_path=paths[3])