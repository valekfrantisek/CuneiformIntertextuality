# ORACC Intertextuality
This project includes scripts that allow to run intertextuality queries over the [Open Richly Anotated Cuneiform Corpus(ORACC)](https://oracc.museum.upenn.edu/) dataset.

The description is yet to be finised.
1) Functions to download ORACC JSON data are in [download_ORACC-JSON.ipynb](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/download_ORACC-JSON.ipynb)
2) Functions to preprocess the corpus are in [DEVEL_process_ORACC_corpus.ipynb](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/DEVEL_process_ORACC_corpus.ipynb). This is a development notebook, therefor a bit of a mess and it will be polished in the future.
3) Functionalities for intertextality detection are in [intertextulity_package.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/intertextulity_package.py) in the [backend directory](https://github.com/valekfrantisek/CuneiformIntertextuality/tree/main/backend). However, these require preprocessed datased (not uploaded to GitHub due to its size).
4) The functionalities will be hopefully soon made available in an online app, with user-friendly UI.

## Structure

## Modes

## How to use
### Requirements
Python packages
- Install requirements from the [requirements.txt](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/requirements.txt).
- Install troch using this command (for GPU): pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
- Alternatively, install torch for CPU (functionality to embed models with CPU may need editions in the script - this is a work in progress): pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

### Before running the web-based application
The data in the repository do not include the ORACC corpus and trained models for vector search. To prepare all the necessary files, you should run the script [initial_setup.py](). This does not exist yet, it is a work in progress.

## License
The script may be used freely, under the license [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

The project here on GitHub does not include ORACC data. If you download them using this script, they are usually licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed), although some projects may include further specifications. See [ORACC licensing portal](http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html).

### ORACC-JSON by Niek Veldhuis
Some parts of the script was based on [ORACC-JSON project](https://github.com/niekveldhuis/ORACC-JSON) created by Niek Veldhuis. Namely, the download functions for ORACC corpus, and some parts of the JSON parser.

## Contact
František Válek (frantisek.valek@upce.cz), [University of Pardubice](https://www.upce.cz/), [Faculty of Philosophy and Arts](https://ff.upce.cz/), [Department of Philosophy and Religion](https://kfr.upce.cz/en)
