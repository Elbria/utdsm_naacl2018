# Cross-topic distributional semantic representations via unsupervised mappings

This is the code for our NAACL2019 submission. The code implements the models described in the two following papers:

* [Topic-based Distributional Semantic Models](https://ieeexplore.ieee.org/document/8334459/), [code](https://github.com/fenchri/mixture-tdsms)
>Fenia Christopoulou, Eleftheria Briakou, Elias Iosif, Alexandros Potamianos

* [Cross-topic distributional semantic representations via unsupervised mappings](https://aclweb.org/anthology/papers/N/N19/N19-1110/)
>Eleftheria Briakou, Nikos Athanasiou, Alexandros Potamianos

## Setup

Create virtual environment and run the main script from the project directory:

* ```$ python3 -m venv utdsm_venv```

* ``bash main.sh``

## Data

The code expects two-versions of _pre-processed_ data:

* sentence-level

* document-level

``Note: Toy corpora are also provided under data/``

## Citation
Please cite the following papers when using this software:

> @inproceedings{briakou-etal-2019-cross,\
>    title = "Cross-Topic Distributional Semantic Representations Via Unsupervised Mappings",\
>    author = "Briakou, Eleftheria  and Athanasiou, Nikos  and Potamianos, Alexandros",\
>    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",\
>    pages = "1052--1061",\
>    year = "2019",\
>    publisher = "Association for Computational Linguistics",\
>}


> @inproceedings{christopoulou2018mixture,\
>  title={Mixture of topic-based distributional semantic and affective models},\
>  author={Christopoulou, Fenia and Briakou, Eleftheria and Iosif, Elias and Potamianos, Alexandros},\
>  booktitle={2018 IEEE 12th International Conference on Semantic Computing (ICSC)},\
>  pages={203--210},\
>  year={2018},\
>  organization={IEEE}\
> }

