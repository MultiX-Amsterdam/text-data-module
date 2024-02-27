#!/bin/sh

# DO NOT put the installation of pytorch in this file
# Users need to check the following website to install pytorch
# https://pytorch.org/get-started/locally/
# The reason is that users can have different hardware settings

pip install --upgrade numpy~=1.26.4
pip install --upgrade scipy~=1.12.0
pip install --upgrade matplotlib~=3.8.2
pip install --upgrade ipython~=8.21.0
pip install --upgrade pandas~=2.2.0
pip install --upgrade scikit-learn~=1.4.0
pip install --upgrade seaborn~=0.13.2
pip install --upgrade tqdm~=4.66.2
pip install --upgrade pyarrow~=15.0.0
pip install --upgrade nltk~=3.8.1
pip install --upgrade gensim~=4.3.2
pip install --upgrade spacy~=3.5.0
pip install --upgrade wordcloud~=1.9.3
