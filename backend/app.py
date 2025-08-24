from intertextulity_package import load_json_corpus, OraccCorpus, find_intertextualities_of_text, search_for_query_in_target_dataset, render_results_to_html, render_vector_results_to_html, search_vectors, normalize_signs, render_results_to_html_text_id

from monitor import start_resource_logger
import atexit

from flask import Flask, request, jsonify, send_from_directory, Response, redirect
from flask_cors import CORS
import os
import logging
import uuid
import pandas as pd
from io import BytesIO, StringIO
from cachelib import SimpleCache
import json
from time import time
import math


__version__ = 'BETA_0.0.1'
__author__ = 'František Válek'
__license_software__ = 'CC0 1.0 Universal'
__license_oracc_data__ = 'CC BY-SA 3.0' # see http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html; for individual datasets further authors are relevant (links are included for reference to dataset)

""" TODO List
- Add references to the texts and authors to ORACC (links only?)
- Add proper functions documentation
- Implement text normalization
- Add support for signs interchangeability (normalisation of signs; on level of signs, on level of "normalised" mode)
"""

ROOT_PATH = os.getcwd()
PROJECTS_DATA_PATH = os.path.join(ROOT_PATH, 'projectsdata')
CORPUS_PATH = os.path.join(ROOT_PATH, 'CORPUS')

logging.basicConfig(level=logging.DEBUG)

""" Loading ORACC corpus ---------------------------------------------- """

# NOTE: start logging performance
# stop_logger = start_resource_logger(interval=3.0)

print("Loading ORACC corpus... please, wait, it may take a while.")
start_ = time()
oracc_corpus = OraccCorpus(projects_path=CORPUS_PATH, files_prefix='prnd_no_comp')  # loading pruned corpus
end_ = time()
print(f"ORACC corpus loaded in {end_ - start_:.2f} seconds.")

# atexit.register(lambda: stop_logger.set())


""" DEFINING FLASK APP ------------------------------------------------ """

app = Flask(__name__, static_folder='../frontend')
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})

cache = SimpleCache()

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


""" Working with corpus ---------------------------------- """

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
            benchmark = math.ceil((num_of_tokens / len(query)) * 10) / 10

        elif mode == 'edit_distance_tokens':
            print((max_total_ed / len(query)))
            benchmark = math.ceil((max_total_ed / len(query)) * 10) / 10

    benchmark = 1-benchmark

    if benchmark == 0.0:
        benchmark = 0.1

    return benchmark


@app.route('/get_text_example/<text_id>/<mode>/<normalise_signs>', methods=['POST'])
def get_text_example(text_id, mode, normalise_signs):
    """ Retrieve the text example from the corpus using the text_id. """
    text_id = text_id.strip()

    if normalise_signs == 'false':
        normalise_signs = False
    elif normalise_signs == 'true':
        normalise_signs = True

    while '-' in text_id:
        text_id = text_id.replace('-', '/')

    if normalise_signs and mode in ['signs', 'forms', 'forms_pos']:
        mode = f'{mode}_normalised'

    text_example = oracc_corpus.get_data_by_id(text_id, mode)[:15]
    print(f"Text example for {text_id} in mode {mode}: {text_example}")
    

    if text_example:
        return jsonify({'text_example': ' '.join(text_example), 'mode': mode}), 200
    else:
        return jsonify({'error': 'Text example not found (possibly a wrong ID or not selected mode)'}), 404
    

@app.route('/get_text_example_full/<text_id>/<mode>/<normalise_signs>', methods=['POST'])
def get_text_example_full(text_id, mode, normalise_signs):
    """ Retrieve the full text example from the corpus using the text_id. """
    text_id = text_id.strip()

    if normalise_signs == 'false':
        normalise_signs = False
    elif normalise_signs == 'true':
        normalise_signs = True

    while '-' in text_id:
        text_id = text_id.replace('-', '/')

    if normalise_signs and mode in ['signs', 'forms', 'forms_pos']:
        mode = f'{mode}_normalised'

    text_example = oracc_corpus.get_data_by_id(text_id, mode)
    print(f"Text example for {text_id} in mode {mode}: {text_example}")
    

    if text_example:
        return jsonify({'text_example': ' '.join(text_example), 'mode': mode}), 200
    else:
        return jsonify({'error': 'Text example not found (possibly a wrong ID or not selected mode)'}), 404


""" ANALYSIS SECTION -------------------------------------------------- """


@app.route('/analyse_input/<mode>/<processing>/<max_total_ed>/<query>/<normalise_signs>', methods=['POST'])
def analyse_input(mode, processing, max_total_ed, query, normalise_signs):
    """ Call the appropriate function based on the mode and processing type. """
    max_total_ed = int(max_total_ed)

    if normalise_signs == 'false':
        normalise_signs = False
    elif normalise_signs == 'true':
        normalise_signs = True

    # NOTE: this may not work as intended (are some signs with '-' or '.' in the corpus?) TODO: check this!!
    if mode in ['signs', 'signs_gdl']:
        while '-' in query:
            query = query.replace('-', ' ')
        while '.' in query:
            query = query.replace('.', ' ')

    if normalise_signs and mode in ['signs', 'forms', 'forms_pos']:
        mode = f'{mode}_normalised'
        query = normalize_signs(query)

    input_id = str(uuid.uuid4())

    if processing in ['vect_e5', 'vect_MiniLM']:
        results = search_vectors(mode=mode, model=processing, query=query)

        print(results.head(10))
        print(f'Analysis results ({len(results)}):')

        results = results.head(100) # limit 100 results for export

        results_html = render_vector_results_to_html(results=results, query=query, mode=mode, processing=processing)

        # TODO: implement export to CSV and XLSX (here, the JSON will probably not work as it is a pd df!!!! --> the best may be to make all the results to pd dfs and save csv/xls files to cache right away!)
        cache.set(f'results_{input_id}', results, timeout=3600)
        logging.debug(f"Processed data saved to cache with key: results_{input_id}")

    elif processing in ['edit_distance_inner', 'edit_distance_tokens']:
        query = query.split()

        benchmark = set_correct_benchmark(query, max_total_ed, mode=processing)
        print(f'Benchmark was document selection was set to: {benchmark}')

        results = search_for_query_in_target_dataset(mode=mode, processing=processing, query=query, ORACCtarget_dataset=oracc_corpus, max_total_ed=max_total_ed, benchmark=benchmark)

        print(f'Analysis results ({len(results)}):')
        print(results.head(10))

        results_html = render_results_to_html(results=results, query=query, mode=mode, processing=processing, max_total_ed=max_total_ed)

        # TODO: implement export to CSV and XLSX
        cache.set(f'results_{input_id}', results, timeout=3600)
        logging.debug(f"Processed data saved to cache with key: results_{input_id}")

    return jsonify({'input_id': input_id, 'data_for_download':  f'results_{input_id}', 'results_html': results_html})


# TODO: analyse whole text by ID
@app.route('/analyse_text_by_id/<text_id>/<mode>/<processing>/<max_total_ed>/<normalise_signs>/<window_len>/<stride>/<ignore_self>/<ignore_core_project>', methods=['POST'])
def analyse_text_by_id(text_id, mode, processing, max_total_ed, normalise_signs, window_len, stride, ignore_self, ignore_core_project):

    # Call the appropriate function based on the mode and processing type
    max_total_ed = int(max_total_ed)
    window_len = int(window_len)
    stride = int(stride)

    if normalise_signs == 'false':
        normalise_signs = False
    elif normalise_signs == 'true':
        normalise_signs = True

    if ignore_self == 'false':
        ignore_self = False
    elif ignore_self == 'true':
        ignore_self = True

    if ignore_core_project == 'false':
        ignore_core_project = False
    elif ignore_core_project == 'true':
        ignore_core_project = True

    while '-' in text_id:
        text_id = text_id.replace('-', '/')

    input_id = str(uuid.uuid4())

    if processing in ['vect_e5', 'vect_MiniLM']:
        results_html = '<p>vector search is not yet prepared for the whole documents</p>'
        # results = search_vectors(mode=mode, model=processing, query=text_id)

        # print(results.head(10))
        # print(f'Analysis results ({len(results)}):')

        # results = results.head(100) # limit 100 results for export

        # results_html = render_vector_results_to_html(results=results, query=text_id, mode=mode, processing=processing)

        # # TODO: implement export to CSV and XLSX (here, the JSON will probably not work as it is a pd df!!!! --> the best may be to make all the results to pd dfs and save csv/xls files to cache right away!)
        # cache.set(f'results_{input_id}', results, timeout=3600)
        # logging.debug(f"Processed data saved to cache with key: results_{input_id}")

    elif processing in ['edit_distance_inner', 'edit_distance_tokens']:
        results = find_intertextualities_of_text(oracc_corpus, text_id, window_size=window_len, stride=stride, mode=mode, benchmark=0.8, ignore_itself=ignore_self, ignore_core_project=ignore_core_project, edit_distance_tolerance=max_total_ed, if_min_tokens_lower_tolerance_to=0)

        if results == 'timeout':
            logging.warning(f"Analysis timed out for text ID: {text_id}")
            results_html = '<p>Analysis timed out. Probably, there were too many queries and possible hits to analyse. Timelimit is set to 20 seconds. For larger queries, douwnload the app from the <a href="https://github.com/valekfrantisek/ORACC-JSON" target="blank">GitHub page</a> of the project and run the analysis on your own device.</p>'

            return jsonify({'input_id': input_id, 'data_for_download':  f'results_{input_id}', 'results_html': results_html})

        print(f'Analysis results ({len(results)}):')
        print(results.head(10))

        results_html = render_results_to_html_text_id(results=results, text_id=text_id, mode=mode, processing=processing, ignore_self=ignore_self, ignore_core_project=ignore_core_project)

        cache.set(f'results_{input_id}', results, timeout=3600)
        logging.debug(f"Processed data saved to cache with key: results_{input_id}")

    return jsonify({'input_id': input_id, 'data_for_download':  f'results_{input_id}', 'results_html': results_html})


""" DOWNLOAD FUNCTIONS ------------------------------------------------ """


@app.route('/download_csv/<filename>', methods=['GET'])
def download_csv(filename):
    df = cache.get(filename)
    if df is None:
        return jsonify({'error': 'File not found or expired'}), 404
    
    output = df.to_csv(index=False)
    
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}.csv'}
    )
    
    
@app.route('/download_xlsx/<filename>', methods=['GET'])
def download_xlsx(filename):
    df = cache.get(filename)
    if df is None:
        return jsonify({'error': 'File not found or expired'}), 404

    df.index.name = 'res_id'
    df.reset_index(inplace=True)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    
    return Response(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename={filename}.xlsx'}
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)