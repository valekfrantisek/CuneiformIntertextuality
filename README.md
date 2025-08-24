# ORACC Intertextuality
This project includes scripts that allow to run intertextuality queries over the [Open Richly Anotated Cuneiform Corpus(ORACC)](https://oracc.museum.upenn.edu/) dataset.

The description is yet to be finised.
1) Functions to download ORACC JSON data are in [download_ORACC-JSON.ipynb](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/download_ORACC-JSON.ipynb)
2) Functions to preprocess the corpus are in [process_ORACC_corpus.ipynb](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/process_ORACC_corpus.ipynb). This is a development notebook, therefor a bit of a mess and it will be polished in the future. Some functionalities, namely vectorization are then moved to [embed.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/embed.py), because it runs slow and python runs faster than jupyter.
3) Functionalities for intertextality detection are in [intertextulity_package.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/intertextulity_package.py) in the [backend directory](https://github.com/valekfrantisek/CuneiformIntertextuality/tree/main/backend). However, these require preprocessed datased (not uploaded to GitHub due to its size).
4) The functionalities will be hopefully soon made available in an online app, with user-friendly UI.

In addition, the requirements are not correct and it returns errors at this point --> Either install manually, or wait until I solve this bug.

## Structure

## Modes

## How to use

## License
The script may be used freely, under the license [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

The project here on GitHub does not include ORACC data. If you download them using this script, they are usually licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed), although some projects may include further specifications. See [ORACC licensing portal](http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html).

### ORACC-JSON by Niek Veldhuis
Some parts of the script was based on [ORACC-JSON project](https://github.com/niekveldhuis/ORACC-JSON) created by Niek Veldhuis. Namely, the download functions for ORACC corpus, and some parts of the JSON parser.

## Contact
František Válek (frantisek.valek@upce.cz), [University of Pardubice](https://www.upce.cz/), [Faculty of Philosophy and Arts](https://ff.upce.cz/), [Department of Philosophy and Religion](https://kfr.upce.cz/en)
