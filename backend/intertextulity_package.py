""" This script serves to provide direct intertextuality analysis for the ORACC corpus. """
import os
import joblib
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any, Set, Iterable
from rapidfuzz.distance import Levenshtein
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
import sys
import gc

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict, Any
import re
from rapidfuzz import fuzz
import torch
import time


__version__ = 'BETA_0.0.1'
__author__ = 'František Válek'
__license_software__ = 'CC0 1.0 Universal'
__license_oracc_data__ = 'CC BY-SA 3.0' # see http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html; for individual datasets further authors are relevant (links are included for reference to dataset)

""" TODO List
- Edit descriptions (such as model name in rendering)
- Add detailed description on functionality within the app
- Add proper functions documentation
- Add case-insensivity!!! (Annunitum != annunitum and that is a problem!!)
- Add rendering of results with broader context
- Add support for signs interchangeability (normalisation of signs; on level of signs, on level of "normalised" mode) - this is probably already secured by Signs-GDL mode (but not in forms!!)
"""

""" DEFINING PATHS ------------------------------------------------------ """

ROOT_PATH = os.getcwd()
PROJECTS_DATA_PATH = os.path.join(ROOT_PATH, 'projectsdata')
CORPUS_PATH = os.path.join(ROOT_PATH, 'CORPUS')

""" Loading and preprocessing corpus functions. ------------------------ """ # To download and update the ORACC dataset, use functions in another python file. # TODO

# NOTE: in case of POS variants, we want to ignore named entities/Proper Nouns. (see https://oracc.museum.upenn.edu/doc/help/languages/propernouns/index.html)
PN_POSs = ['AN', 'CN', 'DN', 'EN', 'FN', 'GN', 'LN', 'MN', 'ON', 'PN', 'QN', 'RN', 'SN', 'TN', 'WN', 'YN']

def load_json_corpus(json_corpus_name:str, load_path=CORPUS_PATH) -> dict:
    return joblib.load(os.path.join(load_path, f'{json_corpus_name}.joblib'))

def parsejson(text_json:dict):
    text_forms = []
    text_lemma = []
    text_normalised = []
    
    text_signs = []
    text_signs_gdl = []

    text_forms_pos = []
    text_lemma_pos = []
    text_normalised_pos = []

    def extract_from_node(obj):
        if isinstance(obj, dict):
            if obj.get("node") == "l" and isinstance(obj.get("f"), dict):
                f = obj["f"]

                pos  = f.get("pos") or f.get("epos")
                form = f.get("form")
                lemma = f.get("cf")
                norm = f.get("norm") or f.get("norm0")

                # TODO: check this (sys.intern) --> but it seems that it mostly slows the process down and there is virtually no RAM drop...
                # form = sys.intern(f.get("form") or "")
                # lemma = sys.intern(f.get("cf") or "")
                # norm = sys.intern(f.get("norm") or f.get("norm0") or "")

                text_forms.append(form)
                text_lemma.append(lemma)
                text_normalised.append(norm)

                if pos in PN_POSs:
                    text_forms_pos.append(f"PN_{pos}")
                    text_lemma_pos.append(f"PN_{pos}")
                    text_normalised_pos.append(f"PN_{pos}")
                # if pos in PN_POSs:
                #     text_forms_pos.append(sys.intern(f"PN_{pos}"))
                #     text_lemma_pos.append(sys.intern(f"PN_{pos}"))
                #     text_normalised_pos.append(sys.intern(f"PN_{pos}"))
                else:
                    text_forms_pos.append(form)
                    text_lemma_pos.append(lemma)
                    text_normalised_pos.append(norm)

                for g in f.get("gdl", []):
                    if isinstance(g, dict):
                        if "v" in g:
                            text_signs.append(g["v"])
                        if "gdl_sign" in g:
                            text_signs_gdl.append(g["gdl_sign"])
                        for sub in g.get("seq", []):
                            if "v" in sub:
                                text_signs.append(sub["v"])
                            if "gdl_sign" in sub:
                                text_signs_gdl.append(sub["gdl_sign"])

                # for g in f.get("gdl", []):
                #     if isinstance(g, dict):
                #         if "v" in g:
                #             text_signs.append(sys.intern(g["v"]))
                #         if "gdl_sign" in g:
                #             text_signs_gdl.append(sys.intern(g["gdl_sign"]))
                #         for sub in g.get("seq", []):
                #             if "v" in sub:
                #                 text_signs.append(sys.intern(sub["v"]))
                #             if "gdl_sign" in sub:
                #                 text_signs_gdl.append(sys.intern(sub["gdl_sign"]))

            for value in obj.values():
                extract_from_node(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_from_node(item)

    def change_unknowns(input_list:list):
        unknowns = [None, 'x', 'X']
        return ["■" if item in unknowns else item for item in input_list]

    extract_from_node(text_json)

    text_forms = change_unknowns(text_forms)
    text_lemma = change_unknowns(text_lemma)
    text_normalised = change_unknowns(text_normalised)
    text_signs = change_unknowns(text_signs)
    text_signs_gdl = change_unknowns(text_signs_gdl)
    text_forms_pos = change_unknowns(text_forms_pos)
    text_lemma_pos = change_unknowns(text_lemma_pos)
    text_normalised_pos = change_unknowns(text_normalised_pos)

    return {'text_forms': text_forms, 'text_lemma': text_lemma, 'text_normalised': text_normalised, 'text_signs': text_signs, 'text_signs_gdl': text_signs_gdl, 'text_forms_pos': text_forms_pos, 'text_lemma_pos': text_lemma_pos, 'text_normalised_pos': text_normalised_pos}


# NOTE: backup (delete, if the new version works better)
# def normalize_signs(input_: str) -> str:
#     """ Normalises signs in the text (e.g., ša = ša₂ = ša₃)"""
#     for num in '₁₂₃₄₅₆₇₈₉₀':
#         while num in input_:
#             input_ = input_.replace(num, '')

#     return input_


# def normalize_signs_list(input_: list) -> list:
#     """ Normalises signs in the text (e.g., ša = ša₂ = ša₃)"""
#     return [normalize_signs(item) for item in input_]

SUB_NUM = str.maketrans('', '', '₀₁₂₃₄₅₆₇₈₉')

def normalize_signs(s: str) -> str:
    return s.translate(SUB_NUM)

def normalize_signs_list(lst: list) -> list:
    return [sys.intern(normalize_signs(x) if x is not None else "■") for x in lst]

class OraccProjectCorpus:
    def __init__(self, json_corpus):
        self.corpus = json_corpus
        self.texts =  [text_id for text_id in json_corpus]
        self.texts_data = [json_corpus[text_id] for text_id in json_corpus]
        self.size = len(json_corpus)
        
        analysed_corpus, texts_with_errors, empty_texts = self.AnalyseCorpus()
        
        self.Lemma = analysed_corpus['lemma']
        self.Forms = analysed_corpus['forms']
        self.Normalised = analysed_corpus['normalised']
        self.Signs = analysed_corpus['signs']
        self.SignsGDL = analysed_corpus['signs_gdl']
        
        self.LemmaPOS = analysed_corpus.get('lemma_pos')
        self.FormsPOS = analysed_corpus.get('forms_pos')
        self.NormalisedPOS = analysed_corpus.get('normalised_pos')
        
        self.SignsNormalised = analysed_corpus.get('signs_normalised')
        self.FormsNormalised = analysed_corpus.get('forms_normalised')
        self.FormsPOSNormalised = analysed_corpus.get('forms_pos_normalised')

        self.texts_with_errors = texts_with_errors
        self.empty_texts = empty_texts
    
    def AnalyseCorpus(self) -> dict: 
        texts_with_errors = []
        empty_texts = []
        
        corpus_data = {}
        
        full_corpus_forms = []
        full_corpus_lemma = []
        full_corpus_normalised = []
        full_corpus_signs = []
        full_corpus_signs_gdl = []
        full_corpus_forms_pos = []
        full_corpus_lemma_pos = []
        full_corpus_normalised_pos = []
        full_corpus_signs_normalised = []
        full_corpus_forms_normalised = []
        full_corpus_forms_pos_normalised = []

        # print('\tAnalyzing texts in the corpus.', self.size, 'texts to be processed.')
        
        for text_id in self.texts:
            
            try:
                text_analysed = parsejson(self.corpus[text_id])
            except:
                # TODO: find out the problems with these texts!
                #print('ERROR with a text:', text_id)
                texts_with_errors.append(text_id)
                text_analysed = {'text_forms': [], 'text_lemma': [], 'text_normalised': [], 'text_signs': [], 'text_signs_gdl': [], 'text_forms_pos': [], 'text_lemma_pos': [], 'text_normalised_pos': [], 'signs_normalised': [], 'forms_normalised': [], 'forms_pos_normalised': []}

            corpus_data[text_id] = text_analysed
            
            full_corpus_forms.append(text_analysed['text_forms'])
            full_corpus_lemma.append(text_analysed['text_lemma'])
            full_corpus_normalised.append(text_analysed['text_normalised'])
            full_corpus_signs.append(text_analysed['text_signs'])
            full_corpus_signs_gdl.append(text_analysed['text_signs_gdl'])
            full_corpus_forms_pos.append(text_analysed['text_forms_pos'])
            full_corpus_lemma_pos.append(text_analysed['text_lemma_pos'])
            full_corpus_normalised_pos.append(text_analysed['text_normalised_pos'])

            full_corpus_signs_normalised.append(normalize_signs_list(text_analysed['text_signs']))
            full_corpus_forms_normalised.append(normalize_signs_list(text_analysed['text_forms']))
            full_corpus_forms_pos_normalised.append(normalize_signs_list(text_analysed['text_forms_pos']))

            if text_analysed == {'text_forms': [], 'text_lemma': [], 'text_normalised': [], 'text_signs': [], 'text_signs_gdl': [], 'text_forms_pos': [], 'text_lemma_pos': [], 'text_normalised_pos': [], 'signs_normalised': [], 'forms_normalised': [], 'forms_pos_normalised': []}:
                empty_texts.append(text_id)

        return {'corpus_data': corpus_data, 'forms': full_corpus_forms, 'lemma': full_corpus_lemma, 'normalised': full_corpus_normalised, 'signs': full_corpus_signs, 'signs_gdl': full_corpus_signs_gdl, 'forms_pos': full_corpus_forms_pos, 'lemma_pos': full_corpus_lemma_pos, 'normalised_pos': full_corpus_normalised_pos, 'signs_normalised': full_corpus_signs_normalised, 'forms_normalised': full_corpus_forms_normalised, 'forms_pos_normalised': full_corpus_forms_pos_normalised}, texts_with_errors, empty_texts


class OraccCorpus():
    def __init__(self, projects_path:str, files_prefix:str='prnd_') -> None: # def __init__(self, input_projects:dict) -> None:
        # self.projects = input_projects # not needed, RAM saving

        all_texts = []
        
        lemma_corpus = []
        forms_corpus = []
        normalised_corpus = []
        signs_corpus = []
        signs_gdl_corpus = []
        forms_pos_corpus = []
        lemma_pos_corpus = []
        normalised_pos_corpus = []
        signs_normalised_corpus = []
        forms_normalised_corpus = []
        forms_pos_normalised_corpus = []

        texts_with_errors = []
        empty_texts = []
    
        # for project_name, project_data in tqdm(input_projects.items(), desc='Processing projects'):
        # for project_name in tqdm(list(input_projects.keys()), desc='Processing projects...'):

        files_to_process = [x for x in os.listdir(projects_path) if x.startswith(files_prefix) and x.endswith('.joblib')]


        for project_file in tqdm(files_to_process, desc='Processing project files...'):
            # project_data = input_projects.pop(project_name)  # saving RAM (but still working with full dataset --> not good)

            if project_file.startswith(files_prefix) and project_file.endswith('.joblib'):
                project_data = load_json_corpus(project_file[:-7], load_path=projects_path)
                OPC_project = OraccProjectCorpus(json_corpus=project_data)

                del OPC_project.corpus  # saving RAM

                for text in OPC_project.Lemma:
                    lemma_corpus.append(text)
                
                for text in OPC_project.Forms:
                    forms_corpus.append(text)
                    
                for text in OPC_project.Normalised:
                    normalised_corpus.append(text)

                for text in OPC_project.Signs:
                    signs_corpus.append(text)
                
                for text in OPC_project.SignsGDL:
                    signs_gdl_corpus.append(text)

                for text_id in OPC_project.texts_with_errors:
                    texts_with_errors.append(text_id)

                for text_id in OPC_project.empty_texts:
                    empty_texts.append(text_id)

                for text in OPC_project.FormsPOS:
                    forms_pos_corpus.append(text)

                for text in OPC_project.LemmaPOS:
                    lemma_pos_corpus.append(text)

                for text in OPC_project.NormalisedPOS:
                    normalised_pos_corpus.append(text)

                for text_id in OPC_project.texts:
                    all_texts.append(text_id)

                for text in OPC_project.SignsNormalised:
                    signs_normalised_corpus.append(text)

                for text in OPC_project.FormsNormalised:
                    forms_normalised_corpus.append(text)

                for text in OPC_project.FormsPOSNormalised:
                    forms_pos_normalised_corpus.append(text)

                del project_data, OPC_project # saving RAM
                gc.collect() # saving RAM
            
            else:
                continue

        print('Corpus size:', len(lemma_corpus), 'texts.')
        print('Texts with errors:', len(texts_with_errors), 'texts.')
        
        for text_id in texts_with_errors:
            print('\t', text_id)

        print('Empty texts:', len(empty_texts), 'texts.')

        self.texts = all_texts
        self.lemma_corpus = lemma_corpus
        self.forms_corpus = forms_corpus
        self.normalised_corpus = normalised_corpus
        
        self.signs_corpus = signs_corpus
        self.signs_gdl_corpus = signs_gdl_corpus
        
        self.forms_pos_corpus = forms_pos_corpus
        self.lemma_pos_corpus = lemma_pos_corpus
        self.normalised_pos_corpus = normalised_pos_corpus

        self.signs_normalised_corpus = signs_normalised_corpus
        self.forms_normalised_corpus = forms_normalised_corpus
        self.forms_pos_normalised_corpus = forms_pos_normalised_corpus

        self.texts_with_errors = texts_with_errors
        self.empty_texts = empty_texts


    def get_data_by_id(self, text_id, mode='forms', print_=False) -> list:
        """ Print text data for debugging purposes. """
        try:
            txt_idx = self.texts.index(text_id)
        except ValueError:
            print(f'Text ID {text_id} not found in the corpus.')
            return []

        if mode == 'forms':
            if print_:
                print(f'Forms: {self.forms_corpus[txt_idx]}')
            return self.forms_corpus[txt_idx]
        elif mode == 'lemma':
            if print_:
                print(f'Lemmas: {self.lemma_corpus[txt_idx]}')
            return self.lemma_corpus[txt_idx]
        elif mode == 'normalised':
            if print_:
                print(f'Normalised: {self.normalised_corpus[txt_idx]}')
            return self.normalised_corpus[txt_idx]
        elif mode == 'signs':
            if print_:
                print(f'Signs: {self.signs_corpus[txt_idx]}')
            return self.signs_corpus[txt_idx]
        elif mode == 'signs_gdl':
            if print_:
                print(f'Signs GDL: {self.signs_gdl_corpus[txt_idx]}')
            return self.signs_gdl_corpus[txt_idx]
        elif mode == 'forms_pos':
            if print_:
                print(f'Forms POS: {self.forms_pos_corpus[txt_idx]}')
            return self.forms_pos_corpus[txt_idx]
        elif mode == 'lemma_pos':
            if print_:
                print(f'Lemmas POS: {self.lemma_pos_corpus[txt_idx]}')
            return self.lemma_pos_corpus[txt_idx]
        elif mode == 'normalised_pos':
            if print_:
                print(f'Normalised POS: {self.normalised_pos_corpus[txt_idx]}')
            return self.normalised_pos_corpus[txt_idx]
        elif mode == 'signs_normalised':
            if print_:
                print(f'Signs Normalised: {self.signs_normalised_corpus[txt_idx]}')
            return self.signs_normalised_corpus[txt_idx]
        elif mode == 'forms_normalised':
            if print_:
                print(f'Forms Normalised: {self.forms_normalised_corpus[txt_idx]}')
            return self.forms_normalised_corpus[txt_idx]
        elif mode == 'forms_pos_normalised':
            if print_:
                print(f'Forms POS Normalised: {self.forms_pos_normalised_corpus[txt_idx]}')
            return self.forms_pos_normalised_corpus[txt_idx]
        else:
            if print_:
                print(f'Mode print set wrong! Use "forms", "lemma", "normalised", "signs", "signs_gdl", "forms_pos", "lemma_pos", "normalised_pos", "signs_normalised", "forms_normalised", or "forms_pos_normalised".')
            return []
        

""" Intertextuality functions ------------------------------------------- """

def token_edit_distance_inner(query: List[str], target: List[str], max_total_ed: Optional[int] = None, unknown_token: str = '■'):
    m, n = len(query), len(target)
    if m == 0 or m > n:
        return None

    best_sum: Optional[int] = None
    best_hits: List[Tuple[int, int, List[str]]] = []

    for i in range(0, n - m + 1):
        s = 0
        for j in range(m):
            if query[j] == unknown_token:
                d = 0
            else:
                d = Levenshtein.distance(query[j], target[i + j])
            
            s += d
            
            # DŮLEŽITÉ: přeruš jen když je PRŮBĚŽNÁ suma HORŠÍ, ne shodná
            if best_sum is not None and s > best_sum:
                break
        else:
            if best_sum is None or s < best_sum:
                best_sum = s
                best_hits = [(s, i, target[i:i+m])]
            elif s == best_sum:
                best_hits.append((s, i, target[i:i+m]))

    if best_sum is None:
        return None
    if max_total_ed is not None and best_sum > max_total_ed:
        return None

    return best_hits

def token_edit_distance(query: List[str], target: List[str], max_total_ed: Optional[int] = None, unknown_token: str = '■') -> Optional[int]:
    """ This function serves to search for hits, considering edit distance on the level of full tokens. (e.g., 'inuma ilu ibnu awilutam' // 'inuma blabla ibnu awilutam' has edit distance 1) """
    m, n = len(query), len(target)
    if m == 0 or n == 0:
        return None

    # DP a backtrack
    dp = [[0]*(n+1) for _ in range(m+1)]
    bt = [[0]*(n+1) for _ in range(m+1)]  # 0=diag, 1=up(delete), 2=left(insert)

    for j in range(n+1):
        dp[0][j] = 0                 # substring může začít kdekoliv
    for i in range(1, m+1):
        dp[i][0] = i                 # smazání i tokenů z query
        bt[i][0] = 1

    for i in range(1, m+1):
        ai = query[i-1]
        for j in range(1, n+1):
            bj = target[j-1]
            cost = 0 if (ai == bj or ai == unknown_token) else 1
            del_q = dp[i-1][j] + 1        # delete ai
            ins_q = dp[i][j-1] + 1        # insert bj
            sub  = dp[i-1][j-1] + cost    # match/replace (UNKNOWN matchuje cokoliv za 0)

            if sub <= del_q and sub <= ins_q:
                dp[i][j] = sub; bt[i][j] = 0
            elif del_q <= ins_q:
                dp[i][j] = del_q; bt[i][j] = 1
            else:
                dp[i][j] = ins_q; bt[i][j] = 2

    best = min(dp[m][1:])  # nejlepší vzdálenost přes všechna možná zakončení
    if max_total_ed is not None and best > max_total_ed:
        return None

    hits: List[Tuple[int, int, int, List[str]]] = []
    for j in range(1, n+1):
        if dp[m][j] != best:
            continue
        # backtrack z (m, j) na začátek substringu (řádek i==0)
        i, jj = m, j
        while i > 0:
            move = bt[i][jj]
            if move == 0:      # diag
                i -= 1; jj -= 1
            elif move == 1:    # up (delete v query)
                i -= 1
            else:              # left (insert v targetu)
                jj -= 1
        start, end = jj, j
        hits.append((best, start, end, target[start:end]))

    return hits if hits else None


Doc = List[str]

@dataclass
class SimpleIndex:
    postings: Dict[str, List[int]]     # token -> [doc_id]
    doc_unique: List[Set[str]]         # interní doc_id -> unikátní tokeny
    df: Dict[str, int]                 # token -> DF
    N: int                             # počet dokumentů
    ids2ext: List[Any]                 # interní doc_id -> původní ID
    ext2ids: Dict[Any, int]            # původní ID -> interní doc_id

def build_inverted_index(docs: Iterable[Doc], external_ids: Optional[Iterable[Any]] = None, stop: Optional[Set[str]] = ('■')) -> SimpleIndex:
    """
    Vstup: `docs` je iterovatelný přes dokumenty, každý dokument je list tokenů (List[str]).
    Výstup: invertovaný index + základní statistiky.
    """
    stop = stop or set()
    postings_sets: Dict[str, Set[int]] = defaultdict(set)
    doc_unique: List[Set[str]] = []
    ext_ids: List[Any] = []
    ext2int: Dict[Any, int] = {}

    ext_iter = iter(external_ids) if external_ids is not None else None

    for internal_id, tokens in enumerate(docs):
        u = {t for t in tokens if t not in stop}
        doc_unique.append(u)
        for t in u:
            postings_sets[t].add(internal_id)

        if ext_iter is not None:
            ext_id = next(ext_iter)  # vyhodí StopIteration, pokud délky nesedí
        else:
            ext_id = internal_id     # fallback: původní ID = interní index
        ext_ids.append(ext_id)
        ext2int[ext_id] = internal_id

    # finalize
    postings = {t: sorted(ids) for t, ids in postings_sets.items()}
    df = {t: len(ids) for t, ids in postings.items()}
    N = len(doc_unique)

    return SimpleIndex(postings=postings, doc_unique=doc_unique, df=df, N=N,
                       ids2ext=ext_ids, ext2ids=ext2int)


def set_correct_benchmark(query: list, max_total_ed: int, mode: str) -> float:
    """ Set the correct benchmark for selection of possible documents based on the query based on its length and the max_total_ed. """
    benchmark = 1.0
    if max_total_ed > 0:
        if mode == 'edit_distance_inner':
            sorted_q = sorted(query, key=len)
            num_of_tokens = 0
            tokens_len = 0
            for token in sorted_q:
                num_of_tokens += 1
                tokens_len += len(token)
                if tokens_len > max_total_ed:
                    break
            
            print((num_of_tokens / len(query)))
            benchmark = ceil((num_of_tokens / len(query)) * 10) / 10

        elif mode == 'edit_distance_tokens':
            print((max_total_ed / len(query)))
            benchmark = ceil((max_total_ed / len(query)) * 10) / 10

    benchmark = 1-benchmark

    if benchmark == 0.0:
        benchmark = 0.1

    return benchmark


def select_documents_for_single_token(index: SimpleIndex, term: str) -> List[int]:
    """ Selects documents containing the given token. """
    return index.postings.get(term, [])

def select_documents_for_tokens(index: SimpleIndex, terms: List[str], benchmark: float = 0.8, stop: Optional[Set[str]] = ('■')) -> Set[int]:
    """
    Selects documents containing at least the benchmark proportion of (unique) tokens from the query.
    benchmark=0.8 => must match at least 80% of query tokens.
    """
    qset = set(terms)
    if not qset:
        return set()

    if stop:
        qset -= stop

    counts = defaultdict(int)  # doc_id -> kolik dotazových tokenů se našlo
    for t in qset:
        for doc_id in index.postings.get(t, ()):
            counts[doc_id] += 1

    required = ceil(benchmark * len(qset))  # integer práh

    if required == len(qset) and benchmark != 1.0:
        required = len(qset) - 1
        # NOTE: this is needed for short queries!!

    return {doc_id for doc_id, c in counts.items() if c >= required}


def load_data_by_mode(mode:str, oracc_corpus: OraccCorpus):
    """ Loads relevant data from the ORACC corpus according to the specified mode. """
    if mode == 'normalised':
        normalised_inverted_index = build_inverted_index(oracc_corpus.normalised_corpus, oracc_corpus.texts)
        normalised_stops = set(['■', 'ina', 'ana', 'u', 'ša'])
        return normalised_inverted_index, normalised_stops
    elif mode == 'normalised_pos':
        normalised_pos_inverted_index = build_inverted_index(oracc_corpus.normalised_pos_corpus, oracc_corpus.texts)
        normalised_pos_stops = set(['■', 'ina', 'ana', 'u', 'ša'])
        return normalised_pos_inverted_index, normalised_pos_stops
    elif mode == 'lemma':
        lemma_inverted_index = build_inverted_index(oracc_corpus.lemma_corpus, oracc_corpus.texts)
        lemma_stops = set(['■', 'ina', 'ana', 'u', 'ša', 'i-na', 'a-na'])
        return lemma_inverted_index, lemma_stops
    elif mode == 'lemma_pos':
        lemma_pos_inverted_index = build_inverted_index(oracc_corpus.lemma_pos_corpus, oracc_corpus.texts)
        lemma_pos_stops = set(['■', 'ina', 'ana', 'u', 'ša', 'i-na', 'a-na'])
        return lemma_pos_inverted_index, lemma_pos_stops
    elif mode == 'forms':
        forms_inverted_index = build_inverted_index(oracc_corpus.forms_corpus, oracc_corpus.texts)
        forms_stops = set(['■', 'ina', 'ana', 'u', 'ša', 'i-na', 'a-na'])
        return forms_inverted_index, forms_stops
    elif mode == 'forms_pos':
        forms_pos_inverted_index = build_inverted_index(oracc_corpus.forms_pos_corpus, oracc_corpus.texts)
        forms_pos_stops = set(['■', 'ina', 'ana', 'u', 'ša', 'i-na', 'a-na'])
        return forms_pos_inverted_index, forms_pos_stops
    elif mode == 'forms_normalised':
        forms_inverted_index = build_inverted_index(oracc_corpus.forms_normalised_corpus, oracc_corpus.texts)
        forms_stops = set(['■', 'ina', 'ana', 'u', 'ša', 'i-na', 'a-na'])
        return forms_inverted_index, forms_stops
    elif mode == 'forms_pos':
        forms_pos_inverted_index = build_inverted_index(oracc_corpus.forms_pos_normalised_corpus, oracc_corpus.texts)
        forms_pos_stops = set(['■', 'ina', 'ana', 'u', 'ša', 'i-na', 'a-na'])
        return forms_pos_inverted_index, forms_pos_stops
    elif mode == 'signs':
        signs_inverted_index = build_inverted_index(oracc_corpus.signs_corpus, oracc_corpus.texts)
        signs_stops = set(['■'])
        return signs_inverted_index, signs_stops
    elif mode == 'signs_normalised':
        signs_inverted_index = build_inverted_index(oracc_corpus.signs_normalised_corpus, oracc_corpus.texts)
        signs_stops = set(['■'])
        return signs_inverted_index, signs_stops
    elif mode == 'signs_gdl':
        signs_gdl_inverted_index = build_inverted_index(oracc_corpus.signs_gdl_corpus, oracc_corpus.texts)
        signs_gdl_stops = set(['■'])
        return signs_gdl_inverted_index, signs_gdl_stops
    else:
        raise ValueError(f"Unknown mode: {mode}")

def search_for_query_in_target_dataset(mode: str, processing:str, query: List[str], ORACCtarget_dataset: OraccCorpus, max_total_ed: int = 5, target_inverted_idx=None, stop=None, ignore_texts=None) -> pd.DataFrame:
    """
    Searches for a query in the target dataset and looks for intertextuality within it. 
    1) Returns a set of document IDs that match the query, 
    2) Applies edit-distance functions on it.

    :param mode: The mode of the search ('normalised', 'lemma', 'forms', 'normalised_pos', 'lemma_pos', 'forms_pos', 'signs', 'signs_gdl')
    :param processing: The processing type indicating on which level the edit distance is applied ('inner', 'token')
    :param query: The query terms to search for
    :param ORACCtarget_dataset: The target dataset to search within (must be OraccCorpus class)
    :return: pandas dataframe of results
    """

    benchmark = set_correct_benchmark(query=query, max_total_ed=max_total_ed, mode=processing)

    if not target_inverted_idx or not stop:
        target_inverted_idx, stop = load_data_by_mode(mode, ORACCtarget_dataset)
    
    selected_documents = select_documents_for_tokens(target_inverted_idx, query, stop=stop, benchmark=benchmark)

    print(len(selected_documents), 'documents selected for query:', query, 'in', mode, 'mode.')

    hits = {}

    print('Max total edit distance allowed:', max_total_ed)

    res_id = 0

    for doc_id in selected_documents:
        if ignore_texts and doc_id in ignore_texts:
            continue
        else:            
            ORACC_doc_id = target_inverted_idx.ids2ext[doc_id]
            target_data = ORACCtarget_dataset.get_data_by_id(ORACC_doc_id, mode=mode)
            
            if processing == 'edit_distance_inner':
                hits_inner = token_edit_distance_inner(query, target_data, max_total_ed=max_total_ed)
                if hits_inner:
                    for hit in hits_inner:
                        hits[res_id] = {'ORACC_doc_id': ORACC_doc_id, 'matched_sequence': ' '.join(hit[2]), 'edit_distance': hit[0]}
                        res_id += 1

            elif processing == 'edit_distance_tokens':
                hits_full_tokens = token_edit_distance(query, target_data, max_total_ed=max_total_ed)
                if hits_full_tokens:
                    for hit in hits_full_tokens:
                        hits[res_id] = {'ORACC_doc_id': ORACC_doc_id, 'matched_sequence': ' '.join(hit[3]), 'edit_distance': hit[0]}
                        res_id += 1

            else:
                print(f"Unknown processing type: {processing}. Use 'edit_distance_inner' or 'edit_distance_tokens'.")


    out_df = pd.DataFrame.from_dict(hits, orient='index')
    out_df.index.name = 'result_id'
    out_df.reset_index(inplace=True)

    # --- řazení výsledků primárně dle edit distance ---
    if not out_df.empty:
        # pojisti, že je to numerické
        out_df['edit_distance'] = pd.to_numeric(out_df['edit_distance'], errors='coerce')
        # primární klíč: edit_distance; sekundárně ORACC_doc_id a result_id pro stabilitu
        out_df.sort_values(
            ['edit_distance', 'ORACC_doc_id', 'result_id'],
            ascending=[True, True, True],
            inplace=True,
            kind='mergesort'
        )
        out_df.reset_index(drop=True, inplace=True)

    return out_df

    # out_df = pd.DataFrame.from_dict(hits, orient='index')
    # out_df.index.name = 'result_id'
    # out_df.reset_index(inplace=True)

    # return out_df


def skip_empty_query(query: List[str], stop: Set[str], min_tokens: int=2) -> bool:
    """
    Check if a query is empty or contains only stop words (or if it is long enough).
    """
    query_tokens = set(query) - stop
    return [(len(query_tokens) < min_tokens), len(query_tokens)]


def parse_query_text(query:List[str], window_size:int=5, stride:int=3) -> List[str]:
    """
    Parse the input query (list of strings) into strided queries of a specified size. If the window size is shorter than the query, the full query is returned. If the last window overlaps the query, the last window is calculated from the back.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError('window_size and stride must be positive integers.')
    n = len(query)
    if n == 0:
        return []
    if n <= window_size:
        return [query[:]] # short query --> one window

    # standard starts by stride
    starts = list(range(0, n - window_size + 1, stride))

    # adding the last window if not present (standard window over the limit)
    last_start = n - window_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return [query[s:s + window_size] for s in starts]


def get_core_project(text_id: str) -> str:
    parts = text_id.split('/')
    return '/'.join(parts[:len(parts)-1])


def find_intertextualities_of_text(oracc_corpus:OraccCorpus, text_id:str, window_size:int=5, stride:int=3, mode:str='normalised', processing: str='edit_distance_inner', ignore_itself=True, ignore_core_project=False, edit_distance_tolerance=5, if_min_tokens_lower_tolerance_to=0, min_tokens: int=2) -> pd.DataFrame:
    """
    Add description
    """
    # NOTE: deadline is set --> this may cause problems, check if that happens!!
    deadline = (time.monotonic() + 60)

    queries = parse_query_text(oracc_corpus.get_data_by_id(text_id, mode=mode), window_size=window_size, stride=stride)
    print(f'Input text has been parsed to {len(queries)} queries.')

    target_inverted_idx, stop = load_data_by_mode(mode, oracc_corpus)

    ignore_texts = []
    if ignore_itself:
        ignore_texts = [text_id]

    if ignore_core_project:
        print(f'Ignoring texts from the same core project as {text_id}.')
        core_project = get_core_project(text_id)
        if core_project:
            for t_id in oracc_corpus.texts:
                if get_core_project(t_id) == core_project:
                    ignore_texts.append(t_id)

    ignore_texts = set(ignore_texts)  # make it a set for faster lookups

    first_ = True
    for query in queries:
        # Killing long runs
        if deadline and time.monotonic() >= deadline:
            return 'timeout'

        # Skipping empty queries and lowering tolerance with limit queries
        if skip_empty_query(query=query, stop=stop, min_tokens=min_tokens)[0]:
            continue

        elif skip_empty_query(query=query, stop=stop, min_tokens=min_tokens)[1] == min_tokens:
            ed_tolerance = if_min_tokens_lower_tolerance_to

        else:
            ed_tolerance = edit_distance_tolerance

        q_hits = search_for_query_in_target_dataset(mode=mode, processing=processing, query=query, ORACCtarget_dataset=oracc_corpus, max_total_ed=ed_tolerance, target_inverted_idx=target_inverted_idx, stop=stop, ignore_texts=ignore_texts)

        q_hits['query'] = ' '.join(query)

        if first_:
            hits = q_hits
            first_ = False

        else:
            hits = pd.concat([hits, q_hits], ignore_index=True).drop_duplicates()

    # filtering ignore texts
    if ignore_texts:
        hits = hits[~hits['ORACC_doc_id'].isin(ignore_texts)]

    return hits


def render_results_to_html(results:pd.DataFrame, query:list, mode:str, processing:str, max_total_ed:int) -> str:
    """ This function renders results to HTML format to be included in the Analysis section. """

    html = f'<p><b>Query:</b> {" ".join(query)}</p>'
    html += f'<p><b>Mode:</b> {mode}</p>'
    html += f'<p><b>Processing:</b> {processing}</p>'
    html += f'<p><b>Max total edit distance allowed:</b> {max_total_ed}</p>'
    html += f'<p><b>Number of matches found:</b> {len(results)}</p>'

    if len(results) > 100:
        html += f'<p>Too many results to display. Showing first 100 results (download the rest as CSV or XLSX):</p>'
        results = results.head(100)

    html += f'<p>--------------</p>'

    for index, row in results.iterrows():
        html += f'<h4>Text ID: {row["ORACC_doc_id"]} (<a href="http://oracc.org/{row["ORACC_doc_id"]}" target="_blank" rel="noopener noreferrer">see ORACC</a>)</h4>'
        html += f'<p><b>Match:</b> {row["matched_sequence"]} (edit distance: {row["edit_distance"]})</p>'
        html += f'<p>--------------</p>'
    return html


def render_results_to_html_text_id(results: pd.DataFrame, text_id:str, mode:str, processing:str, ignore_self:bool, ignore_core_project:bool) -> str:
    # TODO:
    html = f'<p><b>Text ID:</b> {text_id} (<a href="http://oracc.org/{text_id}" target="_blank" rel="noopener noreferrer">ORACC</a>)</p>'
    html += f'<p><b>Mode:</b> {mode}</p>'
    html += f'<p><b>Processing:</b> {processing}</p>'
    html += f'<p><b>Ignore self:</b> {ignore_self}</p>'
    html += f'<p><b>Ignore core project:</b> {ignore_core_project}</p>'
    html += f'<p><b>Number of matches found:</b> {len(results)}</p>'

    if len(results) > 100:
        html += f'<p>Too many results to display. Showing first 50 results (download the rest as CSV or XLSX):</p>'
        results = results.head(50)

    html += f'<p>--------------</p>'

    for index, row in results.iterrows():
        html += f'<h4>Text ID: {row["ORACC_doc_id"]} (<a href="http://oracc.org/{row["ORACC_doc_id"]}" target="_blank" rel="noopener noreferrer">see ORACC</a>)</h4>'
        html += f'<p><b>Query:</b> {row["query"]}</p>'
        html += f'<p><b>Match:</b> {row["matched_sequence"]} (edit distance: {row["edit_distance"]})</p>'
        html += f'<p>--------------</p>'
    return html


""" Functions for vector comparison. ----------------------------------------------- """
# NOTE: embedings and FAISS indexes are prepared in a different python file (currently jupyter process_ORACC_copurs.ipynb  TODO: make it into a proper python script that can be easily run)

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

ORACC_LEMMA_pos_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_lemma_pos_embeddings.csv")
ORACC_LEMMA_pos_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_lemma_pos_meta.csv")

ORACC_FORMS_pos_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_embeddings.csv")
ORACC_FORMS_pos_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_meta.csv")

ORACC_FORMS_NORMALISED_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_normalised_embeddings.csv")
ORACC_FORMS_NORMALISED_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_normalised_meta.csv")

ORACC_FORMS_pos_NORMALISED_embed_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_normalised_embeddings.csv")
ORACC_FORMS_pos_NORMALISED_meta_csv_PATH = os.path.join(CHUNKS_PATH, "oracc_forms_pos_normalised_meta.csv")

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

def select_paths(mode:str, model:str):
    if mode == 'normalised':
        if model == 'vect_e5':
            return (ORACC_NORM_embed_csv_PATH, EMBS_NORM_PATH_E5, IDS_NORM_PATH, FAISS_NORM_PATH_E5, 'intfloat/e5-base-v2', ORACC_NORM_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_NORM_embed_csv_PATH, EMBS_NORM_PATH_MiniLM, IDS_NORM_PATH, FAISS_NORM_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_NORM_meta_csv_PATH)
    elif mode == 'normalised_pos':
        if model == 'vect_e5':
            return (ORACC_NORM_POS_embed_csv_PATH, EMBS_NORM_POS_PATH_E5, IDS_NORM_POS_PATH, FAISS_NORM_POS_PATH_E5, 'intfloat/e5-base-v2', ORACC_NORM_POS_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_NORM_POS_embed_csv_PATH, EMBS_NORM_POS_PATH_MiniLM, IDS_NORM_POS_PATH, FAISS_NORM_POS_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_NORM_POS_meta_csv_PATH)
    elif mode == 'lemma':
        if model == 'vect_e5':
            return (ORACC_LEMMA_embed_csv_PATH, EMBS_LEMMA_PATH_E5, IDS_LEMMA_PATH, FAISS_LEMMA_PATH_E5, 'intfloat/e5-base-v2', ORACC_LEMMA_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_LEMMA_embed_csv_PATH, EMBS_LEMMA_PATH_MiniLM, IDS_LEMMA_PATH, FAISS_LEMMA_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_LEMMA_meta_csv_PATH)
    elif mode == 'lemma_pos':
        if model == 'vect_e5':
            return (ORACC_LEMMA_pos_embed_csv_PATH, EMBS_LEMMA_POS_PATH_E5, IDS_LEMMA_POS_PATH, FAISS_LEMMA_POS_PATH_E5, 'intfloat/e5-base-v2', ORACC_LEMMA_pos_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_LEMMA_pos_embed_csv_PATH, EMBS_LEMMA_POS_PATH_MiniLM, IDS_LEMMA_POS_PATH, FAISS_LEMMA_POS_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_LEMMA_pos_meta_csv_PATH)
    elif mode == 'forms':
        if model == 'vect_e5':
            return (ORACC_FORMS_embed_csv_PATH, EMBS_FORMS_PATH_E5, IDS_FORMS_PATH, FAISS_FORMS_PATH_E5, 'intfloat/e5-base-v2', ORACC_FORMS_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_FORMS_embed_csv_PATH, EMBS_FORMS_PATH_MiniLM, IDS_FORMS_PATH, FAISS_FORMS_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_FORMS_meta_csv_PATH)
    elif mode == 'forms_pos':
        if model == 'vect_e5':
            return (ORACC_FORMS_pos_embed_csv_PATH, EMBS_FORMS_POS_PATH_E5, IDS_FORMS_POS_PATH, FAISS_FORMS_POS_PATH_E5, 'intfloat/e5-base-v2', ORACC_FORMS_pos_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_FORMS_pos_embed_csv_PATH, EMBS_FORMS_POS_PATH_MiniLM, IDS_FORMS_POS_PATH, FAISS_FORMS_POS_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_FORMS_pos_meta_csv_PATH)
    elif mode == 'forms_normalised':
        if model == 'vect_e5':
            return (ORACC_FORMS_NORMALISED_embed_csv_PATH, EMBS_FORMS_NORMALISED_PATH_E5, IDS_FORMS_NORMALISED_PATH, FAISS_FORMS_NORMALISED_PATH_E5, 'intfloat/e5-base-v2', ORACC_FORMS_NORMALISED_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_FORMS_NORMALISED_embed_csv_PATH, EMBS_FORMS_NORMALISED_PATH_MiniLM, IDS_FORMS_NORMALISED_PATH, FAISS_FORMS_NORMALISED_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_FORMS_NORMALISED_meta_csv_PATH)
    elif mode == 'forms_pos_normalised':
        if model == 'vect_e5':
            return (ORACC_FORMS_pos_NORMALISED_embed_csv_PATH, EMBS_FORMS_POS_NORMALISED_PATH_E5, IDS_FORMS_POS_NORMALISED_PATH, FAISS_FORMS_POS_NORMALISED_PATH_E5, 'intfloat/e5-base-v2', ORACC_FORMS_pos_NORMALISED_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_FORMS_pos_NORMALISED_embed_csv_PATH, EMBS_FORMS_POS_NORMALISED_PATH_MiniLM, IDS_FORMS_POS_NORMALISED_PATH, FAISS_FORMS_POS_NORMALISED_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_FORMS_pos_NORMALISED_meta_csv_PATH)
    elif mode == 'signs':
        if model == 'vect_e5':
            return (ORACC_SIGNS_embed_csv_PATH, EMBS_SIGNS_PATH_E5, IDS_SIGNS_PATH, FAISS_SIGNS_PATH_E5, 'intfloat/e5-base-v2', ORACC_SIGNS_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_SIGNS_embed_csv_PATH, EMBS_SIGNS_PATH_MiniLM, IDS_SIGNS_PATH, FAISS_SIGNS_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_SIGNS_meta_csv_PATH)
    elif mode == 'signs_normalised':
        if model == 'vect_e5':
            return (ORACC_SIGNS_NORMALISED_embed_csv_PATH, EMBS_SIGNS_NORMALISED_PATH_E5, IDS_SIGNS_NORMALISED_PATH, FAISS_SIGNS_NORMALISED_PATH_E5, 'intfloat/e5-base-v2', ORACC_SIGNS_NORMALISED_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_SIGNS_NORMALISED_embed_csv_PATH, EMBS_SIGNS_NORMALISED_PATH_MiniLM, IDS_SIGNS_NORMALISED_PATH, FAISS_SIGNS_NORMALISED_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_SIGNS_NORMALISED_meta_csv_PATH)
    elif mode == 'signs_gdl':
        if model == 'vect_e5':
            return (ORACC_SIGNSGDL_embed_csv_PATH, EMBS_SIGNSGDL_PATH_E5, IDS_SIGNSGDL_PATH, FAISS_SIGNSGDL_PATH_E5, 'intfloat/e5-base-v2', ORACC_SIGNSGDL_meta_csv_PATH)
        elif model == 'vect_MiniLM':
            return (ORACC_SIGNSGDL_embed_csv_PATH, EMBS_SIGNSGDL_PATH_MiniLM, IDS_SIGNSGDL_PATH, FAISS_SIGNSGDL_PATH_MiniLM, 'all-MiniLM-L6-v2', ORACC_SIGNSGDL_meta_csv_PATH)
    else:
        raise ValueError("Unknown mode: " + mode)
    

def query_to_pos(query: str) -> str:
    out = []
    for tok in query.split():
        # logogramy celé VELKÉ necháme být
        if tok.isupper():
            out.append(tok)
            continue
        # heuristika na jména: aspoň jeden segment začíná velkým písmenem
        segs = tok.split("-")
        if any(seg and seg[0].isalpha() and seg[0].isupper() for seg in segs):
            out.append("PN_RN")
        else:
            out.append(tok)  # zachovat diakritiku i case
    return " ".join(out)

""" e5 model ===== """

_POS_BASE = {'an', 'cn', 'dn', 'en', 'fn', 'gn', 'ln', 'mn', 'on', 'pn', 'qn', 'rn', 'sn', 'tn', 'wn', 'yn'}

def _weighted_overlap_POS(q: str, t: str, tag_w: float = 0.3) -> float:
    def is_tag(tok): return all(p in _POS_BASE for p in tok.split("_"))
    def toks(s): 
        ts = re.findall(r"\w+", str(s), flags=re.UNICODE)
        return [t.lower() for t in ts if not t.isdigit()]
    q_toks, t_toks = set(toks(q)), set(toks(t))
    def w(tok): return (tag_w if is_tag(tok) else 1.0)
    denom = sum(w(t) for t in q_toks) or 1.0
    num   = sum(w(t) for t in q_toks if t in t_toks)
    return num / denom


def search_query_e5(query: str, faiss_idx_path:str, ids_path:str, meta_csv_path:str, topk: int = 50, nprobe: int = 256, cand: int = 2000) -> pd.DataFrame:
    # Loading E5 model
    e5 = SentenceTransformer('intfloat/e5-base-v2', device="cuda")
    e5.max_seq_length = 96
    e5 = e5.to(torch.float16)
    torch.set_float32_matmul_precision("high")

    # --- načti POS index/ids/meta postavené na E5 embeddingách ---
    index_pos_e5 = faiss.read_index(faiss_idx_path)
    ids_pos = pd.read_csv(ids_path)["chunk_id"].astype(str).tolist()
    meta_pos = pd.read_csv(meta_csv_path).astype({"chunk_id":"string"})
    
    
    # 1) převede dotaz do POS tvaru, zachová diakritiku/case u obsahu
    q_pos = query_to_pos(query)             # např. "PN_RN rīm tuqumtim"
    # 2) E5: dotaz musí mít prefix "query: "
    q_emb = e5.encode([f"query: {q_pos}"], normalize_embeddings=True).astype("float32")
    # 3) FAISS
    index_pos_e5.nprobe = nprobe
    D, I = index_pos_e5.search(q_emb, max(topk, cand))
    hit_ids = [ids_pos[i] for i in I[0]]
    embed_s = [float(s) for s in D[0]]

    df = pd.DataFrame({"chunk_id": hit_ids, "embed_score": embed_s}).merge(meta_pos, on="chunk_id", how="left")
    
    # malý strukturální signál z POS řetězce (slabá váha)
    q_pos_lower = q_pos.lower()
    df["lex"] = df["text_display"].astype(str).str.lower().apply(lambda t: _weighted_overlap_POS(q_pos_lower, t, tag_w=0.3))
    df["score"] = 0.9*df["embed_score"] + 0.1*df["lex"]

    df = df.drop_duplicates("chunk_id").sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df)+1))
    # df["text_display"] = df["text_display"].astype(str).str.slice(0, 180)
    return df[["rank","score","embed_score","lex","chunk_id","text_id","start","end","text_display"]]


""" MiniLM model ===== """


def _lex_sim(query: str, text: str) -> float:
    q = str(query).lower()
    t = str(text).lower().replace("∎", " ")
    return max(fuzz.partial_ratio(q, t), fuzz.token_set_ratio(q, t)) / 100.0

def search_query_MiniLM(query: str, faiss_idx_path:str, ids_path:str, meta_path:str, topk: int = 50, nprobe: int = 128, cand: int = 1000) -> pd.DataFrame:
    
    mini = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
    torch.set_float32_matmul_precision("high")

    index_norm = faiss.read_index(faiss_idx_path)
    ids_norm   = pd.read_csv(ids_path)["chunk_id"].astype(str).tolist()
    meta_norm  = pd.read_csv(meta_path).astype({"chunk_id":"string"})
    meta_norm["text_display"] = meta_norm["text_display"].astype(str)

    # 1) FAISS kandidáti
    index_norm.nprobe = nprobe
    q_emb = mini.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index_norm.search(q_emb, max(topk, cand))

    # 2) Poskládat výsledky + meta
    hit_ids  = [ids_norm[i] for i in I[0]]
    embed_s  = [float(s) for s in D[0]]
    df = pd.DataFrame({"chunk_id": hit_ids, "embed_score": embed_s}).drop_duplicates("chunk_id")
    df = df.merge(meta_norm, on="chunk_id", how="left")

    # 3) Lehký lexikální rerank
    df["lex"] = df["text_display"].apply(lambda t: _lex_sim(query, t))
    df["score"] = 0.7 * df["embed_score"] + 0.3 * df["lex"]

    # 4) Řazení a výstup
    df = df.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df)+1))
    # df["text_display"] = df["text_display"].str.slice(0, 180)
    return df[["rank","score","embed_score","lex","chunk_id","text_id","start","end","text_display"]]


def search_vectors(query:str, mode:str, model:str) -> pd.DataFrame:
    """
    Searches for vectors in the ORACC corpus using the specified mode and model.
    Returns a DataFrame with the results.
    """
    paths = select_paths(mode=mode, model=model)

    model_name = paths[4]
    faiss_idx_path = paths[3]
    ids_path = paths[2]
    meta_csv_path = paths[5]

    if model == 'vect_e5':
        search_results = search_query_e5(query, faiss_idx_path, ids_path, meta_csv_path)

    elif model == 'vect_MiniLM':
        search_results = search_query_MiniLM(query, faiss_idx_path, ids_path, meta_csv_path)

    else:
        print(f"Unknown model: {model}. Use 'e5' or 'MiniLM'.")
        search_results = pd.DataFrame()

    return search_results


def render_vector_results_to_html(results: pd.DataFrame, query: str, mode: str, processing: str) -> str:
    """
    Renders the search results to HTML format.
    """
    html = f'<p><b>Query:</b> {query}</p>'
    html += f'<p><b>Mode:</b> {mode}</p>'
    html += f'<p><b>Processing:</b> {processing}</p>'

    html += f'<p>Showing first 15 matches sorted by vector proximity and lexical similarity. You can download first 50 matches in CSV or XLSX.</p>'
    results = results.head(15)

    html += f'<p>--------------</p>'

    for _, row in results.iterrows():
        html += f'<p><b>Text ID:</b> {row["text_id"]} (<a href="http://oracc.org/{row["text_id"]}" target="_blank" rel="noopener noreferrer">ORACC</a>)</p>'
        html += f'<p>Rank: {row["rank"]} | Score: {row["score"]:.4f} (embed {row["embed_score"]:.4f}; lex {row["lex"]:.4f})</p>'
        html += f'<p><b>Matched text:</b> {row["text_display"]}</p>'
        html += f'<p>--------------</p>'
    
    return html

if __name__ == "__main__":
    print("This is a package for intertextuality detection.")    
    