""" This python file includes script that runs the initial setup for running intertextuality detection on your own device. """
import os
import requests
import pandas as pd
import time
import zipfile


__version__ = 'BETA_0.0.1'
__author__ = 'František Válek'
__license_software__ = 'CC0 1.0 Universal'
__license_oracc_data__ = 'CC BY-SA 3.0' # see http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html; for individual datasets further authors are relevant (links are included for reference to dataset)

""" Creating structure for downloads and corpuses. """
ROOT_PATH = os.getcwd()
ZIP_PATH = os.path.join(os.getcwd(), 'jsonzip')
EXTRACT_PATH = os.path.join(os.getcwd(), 'projectsdata')
PROJECTS_METADATA_PATH = os.path.join('projectsmetadata')
CORPUS_PATH = os.path.join(ROOT_PATH, 'CORPUS')
CHUNKS_PATH = os.path.join(ROOT_PATH, 'chunks')

CSV_PROJECTS_DF = os.path.join(PROJECTS_METADATA_PATH, 'projects.csv')
LIST_OF_PROJECTS = os.path.join(PROJECTS_METADATA_PATH, 'projects.txt')


def create_structure():
    os.makedirs(ZIP_PATH, exist_ok=True)
    os.makedirs(EXTRACT_PATH, exist_ok=True)
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
                zip_ref.extractall(EXTRACT_PATH)
    
            os.remove(os.path.join(ZIP_PATH, z_file))
    
            print(f'\tFile {z_file} has been extracted to folder projectsdata and deleted.')


""" Preparing data for analysis """

if __name__ == '__main__':
    print('Running initial setup...')
    
    print('Creating folder structure...')
    create_structure()

    print('Getting existing projects...')
    get_existing_projects()

    print('Downloading json zip files...')
    download_jsons()

    print('Extracting and deleting zip files...')
    extract_and_delete_zip()

    