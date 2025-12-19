# ORACC Intertextuality
This project includes scripts that allow to run intertextuality queries over the [Open Richly Anotated Cuneiform Corpus(ORACC)](https://oracc.museum.upenn.edu/) dataset.

## How to use
The Cuneiform Intertextuality detection is intended to run as an accessible web application. So far, it is not being run anywhere, but hopefully this will happen soon.

## How to use - locally
### Requirements
Python packages
- Install requirements from the [requirements.txt](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/requirements.txt).
- Install troch using this command (for GPU - recommended): pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
- Alternatively, if you do not have GPU install torch for CPU: pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

### Before running the web-based application
The data in the repository do not include the ORACC corpus and trained models for vector search. To prepare all the necessary files, you should run the script [initial_setup.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/initial_setup.py). Running this script then allows you to perform the intertextuality detection based on string comparisons.

To perform vectorised searches, you need to run [chunk_et_embed.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/chunk_et_embed.py). Running this script requieres cuda and a lot of time (depending on your device). It is prepared to run on GPU with extensive use of RAM. If you run it on CPU and have RAM below  16 GB, try to do some fine tuning (esp. lowering batch_size parametrs)

Alternatively, you can download the [preprocessed data](https:/digitalhumanities.upce.cz/downloads/chunks.zip) and extract them to directory /chunks.

### Runing the application
Functionalities for intertextality detection are in [intertextulity_package.py](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/intertextulity_package.py) in the [backend directory](https://github.com/valekfrantisek/CuneiformIntertextuality/tree/main/backend). These functions can be run via a web-like application that communicates with the backend through [flask app](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/app.py). You can run the frontend locally [HTML](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/frontend/index.html), e.g., using VS Code [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer). Then, run the [flask app](https://github.com/valekfrantisek/CuneiformIntertextuality/blob/main/backend/app.py) and once the corpus is loaded, you may start your queries.

## Intertextuality Modes
### Simple Approach (string comparison)
The simple approach is based on string comparisons. Reflecting the peculiarities of the cuneiform writing system, the system can work with several modes. Below, you can see examples of how the same cuneiform text can look like represented in different modes:
1) **Normalised**: *Zimrī-Lîm rīm tuqumtim lunaʾʾid*
2) **Lemma**: *Zimrī-Lîm rīmu tuqumtu nâdu*
3) **Forms**: *zi-im-ri-lim ri-im tu-qu₂-um-tim lu-na-i-id*
4) **Normalised with POS (part of speech) tagging for NE (named entities)**: *PN_RN rīm tuqumtim lunaʾʾid*
5) **Lemma with POS tagging for NE**: *PN_RN rīmu tuqumtu nâdu*
6) **Forms with POS tagging for NE**: *PN_RN ri-im tu-qu₂-um-tim lu-na-i-id*
7) **Signs**: *zi im ri lim ri im tu qu₂ um tim lu na i id*
8) **Signs (ground forms)**: *ZI IM RI IGI RI IM TU KU UM DIM LU NA I A₂*

On the level of intertextuality processing you can then select edit distance tolerance, either on the level of individual tokens (i.e., words or signs) or characters within them. For example, edit distance on the level of tokens between *Zimrī-Lîm rīm tuqumtim* and *Šamšī-Addu rīm tuqumtim* is 1, but the edit distance on the level of characters within tokens is 7.

Each of these setting is suitable for different purposes and the usesrs are invited to experiment.

### "Semantic" Approach (vector-based comparison)
The "semantic" (or better: vector-based) approach is an attempt to focus on intertextuality beyond the level of direct intertextualities. This option is for now available only whn the app is run locally. Using two types of vector embeding (infloat/e5-base-v2 and all-MiniLM-L6-v2) returns interesting results, however, proper improvement is needed. Those who are interested in this comparison are invited to explore the script and experiment with it. More detailed description of the current state is TBD.

## Work in Progress
I am working on creating more functionalities for this script (e.g., more extensive working with named enitites, fine-tuning the vector-based search, working with metadata of texts such as date, provenance, and languages, to enable network analyses...)

## License
The script may be used freely, under the license [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

The project here on GitHub does not include ORACC data. If you download them using this script, they are usually licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed), although some projects may include further specifications. See [ORACC licensing portal](http://oracc.ub.uni-muenchen.de/doc/about/licensing/index.html).

### ORACC-JSON by Niek Veldhuis
Some parts of the script was based on [ORACC-JSON project](https://github.com/niekveldhuis/ORACC-JSON) created by Niek Veldhuis. Namely, the download functions for ORACC corpus, and some parts of the JSON parser.

## Contact
František Válek (frantisek.valek@upce.cz), [University of Pardubice](https://www.upce.cz/), [Faculty of Philosophy and Arts](https://ff.upce.cz/), [Department of Philosophy and Religion](https://kfr.upce.cz/en)
