# ORACC Intertextuality
This project includes scripts that allow to run intertextuality queries over the [Open Richly Anotated Cuneiform Corpus(ORACC)](https://oracc.museum.upenn.edu/) dataset.

## How to use
### Requirements
Python packages
- Install requirements from the [requirements.txt](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/requirements.txt).
- Install troch using this command (for GPU - recommended): pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
- Alternatively, if you do not have GPU install torch for CPU: pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

### Before running the web-based application
The data in the repository do not include the ORACC corpus and trained models for vector search. To prepare all the necessary files, you should run the script [initial_setup.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/initial_setup.py). Running this script then allows you to perform the intertextuality detection based on string comparisons.

To perform vectorised searches, you need to run [chunk_et_embed.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/chunk_et_embed.py). Running this script requieres cuda and a lot of time (depending on your device). It is prepared to run on GPU with extensive use of RAM. If you run it on CPU and have RAM below  16 GB, try to do some fine tuning (esp. lowering batch_size parametrs)

### Runing the application
Functionalities for intertextality detection are in [intertextulity_package.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/intertextulity_package.py) in the [backend directory](https://github.com/valekfrantisek/CuneiformIntertextuality/tree/main/backend). These functions can be run via a web-like application that communicates with the backend through [flask app](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/app.py). You can run the frontend locally [HTML](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/frontend/index.html), e.g., using VS Code [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer). Then, run the [flask app](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/app.py) and once the corpus is loaded, you may start your queries.

## Intertextuality Modes
### Simple Approach (string comparison)
description TBD

### "Semantic" Approach (vector-based comparison)
description TBD

## License
The script may be used freely, under the license [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

The project here on GitHub does not include ORACC data. If you download them using this script, they are usually licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed), although some projects may include further specifications. See [ORACC licensing portal](http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html).

### ORACC-JSON by Niek Veldhuis
Some parts of the script was based on [ORACC-JSON project](https://github.com/niekveldhuis/ORACC-JSON) created by Niek Veldhuis. Namely, the download functions for ORACC corpus, and some parts of the JSON parser.

## Contact
František Válek (frantisek.valek@upce.cz), [University of Pardubice](https://www.upce.cz/), [Faculty of Philosophy and Arts](https://ff.upce.cz/), [Department of Philosophy and Religion](https://kfr.upce.cz/en)
