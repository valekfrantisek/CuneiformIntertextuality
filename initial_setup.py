""" This python file includes script that runs the initial setup for running intertextuality detection on your own device. """
import os
import requests
import pandas as pd
import time
import zipfile
import json
import joblib


__version__ = 'BETA_0.0.1'
__author__ = 'František Válek'
__license_software__ = 'CC0 1.0 Universal'
__license_oracc_data__ = 'CC BY-SA 3.0' # see http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html; for individual datasets further authors are relevant (links are included for reference to dataset)

""" Creating structure for downloads and corpuses. """
ROOT_PATH = os.getcwd()
ZIP_PATH = os.path.join(ROOT_PATH, 'jsonzip')
PROJECTS_DATA_PATH = os.path.join(ROOT_PATH, 'projectsdata')
PROJECTS_METADATA_PATH = os.path.join(ROOT_PATH, 'projectsmetadata')
CORPUS_PATH = os.path.join(ROOT_PATH, 'CORPUS')
CHUNKS_PATH = os.path.join(ROOT_PATH, 'chunks')
FRONTEND_PATH = os.path.join(ROOT_PATH, 'frontend')

CSV_PROJECTS_DF = os.path.join(PROJECTS_METADATA_PATH, 'projects.csv')
LIST_OF_PROJECTS = os.path.join(PROJECTS_METADATA_PATH, 'projects.txt')


def create_structure():
    os.makedirs(ZIP_PATH, exist_ok=True)
    os.makedirs(PROJECTS_DATA_PATH, exist_ok=True)
    os.makedirs(CORPUS_PATH, exist_ok=True)
    os.makedirs(CHUNKS_PATH, exist_ok=True)
    os.makedirs(PROJECTS_METADATA_PATH, exist_ok=True)


""" DOWNLOADING FUNCTIONS """

def get_existing_projects():
    projects_url = 'https://oracc.museum.upenn.edu/projectlist.html'
    response = requests.get(projects_url, verify=False)

    lines_in_html = response.text.split('\n')

    projects_dict = {}
    run_shortcuts = []

    idx=0
    for line in lines_in_html:
        if 'href="./' in line:
            line_parts = line.split('href="./')
            line_parts_2 = line_parts[1].split('">')
            project_shortcut = line_parts_2[0]
            project_shortcut = project_shortcut.replace('/', '-')
            if project_shortcut in run_shortcuts:
                continue
            else:
                line_parts_3 = line_parts_2[1].split('</a>')
                project_name = line_parts_3[0]
                projects_dict[idx] = {'name': project_name, 'shortcut': project_shortcut, 'project_json_link': f'https://oracc.museum.upenn.edu/json/{project_shortcut}.zip'}
                run_shortcuts.append(project_shortcut)
                idx += 1
            
    print('\tUp-to-date list of projects has been created.')

    # Extracting projects to csv file projects.csv
    projects_df = pd.DataFrame.from_dict(projects_dict)
    projects_df = projects_df.transpose()
    projects_df.to_csv(CSV_PROJECTS_DF)
    print(f'\tFile projects.csv has been saved to {CSV_PROJECTS_DF}.')

    # Extracting list of projects to txt file projects.txt
    with open(LIST_OF_PROJECTS, 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(run_shortcuts))
    print(f'\tFile projects.txt has been saved to {LIST_OF_PROJECTS}.')


def download_jsons():
    with open(LIST_OF_PROJECTS, 'r') as f:
        projects = f.read().split('\n')
    
    CHUNK = 16 * 1024
    
    for project in projects:
        url = 'http://build-oracc.museum.upenn.edu/json/' + project + '.zip'
        file = 'jsonzip/' + project + '.zip'
        r = requests.get(url, verify=False)
        if r.status_code == 200:
            print('\tDownloading ' + url + ' saving as ' + file)
            with open(file, 'wb') as f:
                for c in r.iter_content(chunk_size=CHUNK):
                    f.write(c)
            print('\tWaiting 3 seconds in order not to overload the ORACC server')
            time.sleep(3)
        else:
            print('\t' + url + ' does not exist.')


def extract_and_delete_zip():
    zipped_projects = os.listdir(ZIP_PATH)
    for z_file in zipped_projects:
        if z_file[-4:] == '.zip':
            with zipfile.ZipFile(os.path.join(ZIP_PATH, z_file), 'r') as zip_ref:
                zip_ref.extractall(PROJECTS_DATA_PATH)

            os.remove(os.path.join(ZIP_PATH, z_file))
    
            print(f'\tFile {z_file} has been extracted to folder projectsdata and deleted.')


""" Preparing data for analysis and frontend. """


def find_corpusjson_folders_projects(start_path):
    data = {}

    key = 0
    
    for root, dirs, files in os.walk(start_path):
        if "corpusjson" in dirs:
            relative_path = os.path.relpath(os.path.join(root, "corpusjson"), start_path)

            relative_path_metadata = os.path.relpath(os.path.join(root, "metadata.json"), start_path)

            data[key] = {
                "corpusjson_paths": relative_path.replace("\\", "/"),
                "metadata_path": relative_path_metadata.replace("\\", "/")
            }
            

            dirs.remove("corpusjson")  # Prevent recursion to the corpusjson folder
            key += 1
    
    return data


def list_project_texts(project_name:str):
    texts_with_errors = []
    
    # Find corpusjson folders in the project:
    data = find_corpusjson_folders_projects(os.path.join(PROJECTS_DATA_PATH, project_name))
                
    project_data = {}
    
    for key in data:
        full_path = os.path.join(PROJECTS_DATA_PATH, project_name, data[key]['corpusjson_paths'])
        metadata_path = os.path.join(PROJECTS_DATA_PATH, project_name, data[key]['metadata_path'])

        files_in_folder = os.listdir(full_path)
        text_ids = []

        if len(files_in_folder) > 0:
            print(f'\t\tFound {len(files_in_folder)} files in {project_name}/{full_path[:-11]} project')
            
            for json_file_name in files_in_folder:
                text_ids.append(json_file_name[:-5])  # Remove .json extension

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            full_project_name = metadata['config']['name']

        pd_key = f'{project_name}/{data[key]['corpusjson_paths'][:-11]}'
        if pd_key.endswith('/'):
            pd_key = pd_key[:-1]

        project_data[pd_key] = {'full_project_name': full_project_name, 'text_id': text_ids}

    return project_data


def extract_info():
    all_project_data = {}

    for project_name in os.listdir(PROJECTS_DATA_PATH):
        project_data = list_project_texts(project_name)

        for key, value in project_data.items():
            all_project_data[key] = value

    return all_project_data


def create_texts_map_for_frontend():
    data_info = extract_info()

    json_data = '{\n"projects": [\n'

    for project, info in data_info.items():
        json_data += '\t{ "value": "'+project+'", "label": "'+info['full_project_name']+'" },\n'

    # NOTE: deleting the last comma
    json_data = json_data[:-2]
    
    json_data += '\n\t],\n"textsByProject": {'
    for project, info in data_info.items():
        json_data += f'\n\t"{project}": ['

        i=0
        for text_id in info['text_id']:
            if i == len(info['text_id']) - 1:
                json_data += f'\n\t\t{{"value": "{project}/{text_id}"}}'
            else:
                json_data += f'\n\t\t{{"value": "{project}/{text_id}"}},'
            i += 1

        json_data += '\n\t],'

    # NOTE: deleting the last comma
    json_data = json_data[:-1]

    json_data += '\n\t}\n}'

    with open(os.path.join(FRONTEND_PATH, 'texts_map.json'), 'w', encoding='utf-8') as f:
        f.write(json_data)

    print('\tFrontend data structure created successfully.')


def find_corpusjson_folders(start_path):
    corpusjson_paths = []
    
    for root, dirs, files in os.walk(start_path):
        if "corpusjson" in dirs:
            relative_path = os.path.relpath(os.path.join(root, "corpusjson"), start_path)
            corpusjson_paths.append(relative_path.replace("\\", "/"))
            dirs.remove("corpusjson")  # Prevent recursion to the corpusjson folder
    
    return corpusjson_paths


def extract_jsons_from_project(project_name:str):
    texts_with_errors = []
    
    # Find corpusjson folders in the project:
    corpusjson_folders = find_corpusjson_folders(os.path.join(PROJECTS_DATA_PATH, project_name))
                
    project_jsons = {}
    
    for corpusjson_folder in corpusjson_folders:
        full_path = os.path.join(PROJECTS_DATA_PATH, project_name, corpusjson_folder)
        text_id_prefix = f'{project_name}/{corpusjson_folder[:-11]}'
        files_in_folder = os.listdir(full_path)
        if len(files_in_folder) > 0:
            print(f'Found {len(files_in_folder)} files in {project_name}/{corpusjson_folder[:-11]} project')
            
            for json_file_name in files_in_folder:
                with open(os.path.join(full_path, json_file_name), 'r', encoding='utf-8') as json_file:
                    text_id = f'{text_id_prefix}/{json_file_name[:-5]}'.replace('//', '/') # in case there are no subprojects, there are double slashes --> remove them
                    #print(text_id)
                    try:
                        json_data = json.load(json_file)
                        project_jsons[text_id] = json_data
                    except:
                        texts_with_errors.append(text_id)
                        
    return project_jsons, texts_with_errors


def save_json_corpus(json_corpus:dict, save_name:str, save_path=CORPUS_PATH, compression=None):
    """ Save the ORACC corpus to a joblib file. """
    if compression:
        joblib.dump(json_corpus, os.path.join(save_path, f'{save_name}.joblib'), compress=compression)
    else:
        joblib.dump(json_corpus, os.path.join(save_path, f'{save_name}.joblib'))


DROP_KEYS = {
    "type","implicit","id","ref","lang","role","gg","gdl_type","name",
    "ngram","delim","inst","break","ho","hc","value","subtype","label","sig"
}


def prune_obj(x, drop=DROP_KEYS):
    """ Recursive deleting of keys from dicts and lists. """
    if isinstance(x, dict):
        return {k: prune_obj(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, list):
        return [prune_obj(v, drop) for v in x]
    return x


def save_individual_project_jsons(prefix:str='prnd_no_comp_', prune=True, compression=0):
    """ Saving individual joblib files for each project - possibly better for RAM. """

    projects_texts_with_errors = {}

    for project_name in os.listdir(PROJECTS_DATA_PATH):
        project_jsons, texts_with_errors = extract_jsons_from_project(project_name)
        projects_texts_with_errors[project_name] = texts_with_errors

        if prune:
            project_jsons = prune_obj(project_jsons)

        save_json_corpus(project_jsons, f'{prefix}{project_name}', compression=compression)

    save_json_corpus(projects_texts_with_errors, f'{prefix}texts_with_errors_for_individual_files', compression=compression)

if __name__ == '__main__':
    # print('Running initial setup...')
    
    # print('Creating folder structure...')
    # create_structure()

    # print('Getting existing projects...')
    # get_existing_projects()

    # print('Downloading json zip files...')
    # download_jsons()

    # print('Extracting and deleting zip files...')
    # extract_and_delete_zip()

    # print('Creating frontend data structure...')
    # create_texts_map_for_frontend()

    print('Creating pruned JSONs for individual projects...')
    save_individual_project_jsons()

    print('Initial setting and preparation of corpus is done. Now, you can perform intertextuality search that is based on string comparison (with edit distance).')

    print('')

    print('If you want to perform search with vector comparison, use the "chunk_et_embed.py" script.')
