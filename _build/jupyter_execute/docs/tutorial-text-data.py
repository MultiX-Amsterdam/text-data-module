#!/usr/bin/env python
# coding: utf-8

# # Tutorial (Text Data Processing)

# (Last updated: Feb 27, 2024)[^credit]
# 
# [^credit]: Credit: this teaching material is created by [Robert van Straten](https://github.com/robertvanstraten) and revised by [Alejandro Monroy](https://github.com/amonroym99) under the supervision of [Yen-Chia Hsu](https://github.com/yenchiah).

# This tutorial will familiarize you with the data science pipeline of processing text data. We will go through the various steps involved in the Natural Language Processing (NLP) pipeline for topic modelling and topic classification, including tokenization, lemmatization, and obtaining word embeddings. We will also build a neural network using PyTorch for multi-class topic classification using the dataset.
# 
# The AG's News Topic Classification Dataset contains news articles from four different categories, making it a nice source of text data for NLP tasks. We will guide you through the process of understanding the dataset, implementing various NLP techniques, and building a model for classification.

# You can use the following links to jump to the tasks and assignments:
# *   [Task 3: Preprocess Text Data](#t3)
#     *   [Tokenization](#t3-1)
#     *   [Part-of-speech tagging](#t3-2)
#     *   [Stemming / Lemmatization](#t3-3)
#     *   [Stopword Removal](#t3-4)
#     *   [Assignment for Task 3.1: Tokenization and Lemmatization](#a3-1)
#     *   [Assignment for Task 3.2: Word Counts](#a3-2)
#     *   [Assignment for Task 3.3: Stop Words Removal](#a3-3)
#     *   [Another Option: spaCy](#spacy)
# *   [Task 4: Unsupervised Learning - Topic Modelling](#t4)
#     *   [Assignment for Task 4](#a4)
# *   [Task 5: Supervised Learning - Topic Classification](#t5)
#     *   [Compute Word Embeddings](#t5-1)
#     *   [Build the Classifier](#t5-2)
#     *   [Optional Assignment for Task 5](#a5)

# ## Scenario

# The [AG's News Topic Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) is a collection of over 1 million news articles from more than 2000 news sources. The dataset was created by selecting the 4 largest classes from the original corpus, resulting in 120,000 training samples and 7,600 testing samples. The dataset is provided by the academic community for research purposes in data mining, information retrieval, and other non-commercial activities. We will use it to demonstrate various NLP techniques on real data, and in the end, make 2 models with this data. The files train.csv and test.csv contain all the training and testing samples as comma-separated values with 3 columns: class index, title, and description. Download train.csv and test.csv for the following tasks. 

# ## Import Packages

# We put all the packages that are needed for this tutorial below:

# In[1]:


import os
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

from gensim.models import Word2Vec

from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, confusion_matrix

from tqdm.notebook import tqdm

from wordcloud import WordCloud

from xml.sax import saxutils as su

# Add tqdm functions to pandas.
tqdm.pandas()


# ## Task Answers

# The code block below contains answers for the assignments in this tutorial. **Do not check the answers in the next cell before practicing the tasks.**

# In[2]:


def check_answer_df(df_result, df_answer, n=1):
    """
    This function checks if two output dataframes are the same.

    Parameters
    ----------
    df_result : pandas.DataFrame
        The result from the output of a function.
    df_answer: pandas.DataFrame
        The expected output of the function.
    n : int
        The numbering of the test case.
    """
    try:
        assert df_answer.equals(df_result)
        print(f"Test case {n} passed.")
    except Exception:
        print(f"Test case {n} failed.")
        print("Your output is:")
        display(df_result)
        print("Expected output is:")
        display(df_answer)


def check_answer_np(arr_result, arr_answer, n=1):
    """
    This function checks if two output numpy arrays are the same.

    Parameters
    ----------
    arr_result : numpy.ndarray
        The result from the output of a function.
    arr_answer: numpy.ndarray
        The expected output of the function.
    n : int
        The numbering of the test case.
    """
    try:
        assert np.array_equal(arr_result, arr_answer)
        print(f"Test case {n} passed.")
    except Exception:
        print(f"Test case {n} failed.")
        print("Your output is:")
        print(arr_result)
        print("Expected output is:")
        print(arr_answer)


def answer_tokenize_and_lemmatize(df):
    """
    Tokenize and lemmatize the text in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "text" column.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the added "tokens" column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Apply the tokenizer to create the tokens column.
    df["tokens"] = df["text"].progress_apply(word_tokenize)

    # Apply the lemmatizer on every word in the tokens list.
    df["tokens"] = df["tokens"].progress_apply(
        lambda tokens: [lemmatizer.lemmatize(token, wordnet_pos(tag)) for token, tag in nltk.pos_tag(tokens)]
    )
    
    return df


def answer_get_word_counts(df, token_col="tokens"):
    """
    Generate dataframes with the word counts for each class in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "class" and "tokens" columns.
    token_col: str
        Name of the column that stores the tokens.

    Returns
    -------
    pandas.DataFrame:
        There should be three columns in this dataframe.
        The "class" column shows the document class.
        The "tokens" column means tokens in the document class.
        The "count" column means the number of appearances of each token in the class.
        The dataframe should be sorted by the "class" and "count" columns.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # We need to filter out non-words.
    # Notice that the token column contains an array of tokens (not just one token).
    df[token_col] = df[token_col].apply(lambda tokens: [token.lower() for token in tokens if token.isalpha()])

    # Each item in the token column contains an array, which cannot be used directly.
    # Our goal is to count the tokens.
    # Thus, we need to explode the tokens so that every token gets its own row.
    # Then, at the later step, we can group the tokens and count them. 
    df = df.explode(token_col)

    # Option 1:
    # - First, perform the groupby function on class and token.
    # - Then, get the size of how many rows per token (i.e., token counts).
    # - Finally, add the counts as a new column.
    counts = df.groupby(["class", token_col]).size().reset_index(name="count")

    # Option 2 has a similar logic but uses the pivot_table function.
    # counts = counts.pivot_table(index=["class", "tokens"], aggfunc="size").reset_index(name="count")

    # Sort the values on the class and count.
    counts = counts.sort_values(["class", "count"], ascending=[True, False])

    return counts


def answer_remove_stopwords(df):
    """
    Remove stopwords from the tokens.

    Parameters
    ----------
    df : pandas.DataFrame
        There should be three columns in this dataframe.
        The "class" column shows the document class.
        The "tokens" column means tokens in the document class.
        The "count" column means the number of appearances of each token in the class.
        The dataframe should be sorted by the "class" and "count" columns.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the stopwords rows removed.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Using a set for quicker lookups.
    stopwords_set = set(stopwords_list)

    # Filter stopwords from tokens.
    df = df[~df["tokens"].isin(stopwords_set)]

    return df


def answer_get_index_of_top_n_items(array, n=3):
    """
    Given an NumPy array, return the indexes of the top "n" number of items according to their values. 

    Parameters
    ----------
    array : numpy.ndarray
        A 1D NumPy array.
    n : int
        The top "n" number of items that we want.

    Returns
    -------
    numpy.ndarray
        The indexes of the top "n" items.
    """
    return array.argsort()[:-n-1:-1]


# <a name="t3"></a>

# ## Task 3: Preprocess Text Data

# In this task, we will preprocess the text data from the AG News Dataset. First, we need to load the files.

# In[3]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# For performance reasons, we will only use a small subset of this dataset.

# In[4]:


df_train = df_train.groupby("Class Index").head(1000)
df_test = df_test.groupby("Class Index").head(100)


# In[5]:


display(df_train, df_test)


# As you can see, all the classes are distributed evenly in the train and test data.

# In[6]:


display(df_train["Class Index"].value_counts(), df_test["Class Index"].value_counts())


# To make the data more understandable, we will make the classes more understandable by adding a `class` column from the original `Class Index` column, containing the category of the news article. To process both the title and news text together, we will combine the `Title` and `Description` columns into one `text` column. We will deal with just the train data until the point where we need the test data again.

# In[7]:


def reformat_data(df):
    """
    Reformat the Class Index column to a Class column and combine
    the Title and Description columns into a Text column.
    Select only the class_idx, class and text columns afterwards.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.

    Returns
    -------
    pandas.DataFrame
        The reformatted dataframe.
    """
    # Make the class column using a dictionary.
    df = df.rename(columns={"Class Index": "class_idx"})
    classes = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    df["class"] = df["class_idx"].apply(classes.get)

    # Use string concatonation for the Text column and unescape html characters.
    df["text"] = (df["Title"] + " " + df["Description"]).apply(su.unescape)

    # Select only the class_idx, class, and text column.
    df = df[["class_idx", "class", "text"]]
    return df


# In[8]:


df_train_reformat = reformat_data(df_train)
display(df_train_reformat)


# <a name="t3-1"></a>

# ### Tokenization 

# Tokenization is the process of breaking down a text into individual tokens, which are usually words but can also be phrases or sentences. It helps language models to understand and analyze text data by breaking it down into smaller, more manageable pieces. While it may seem like a trivial task, tokenization can be applied in multiple ways and thus be a complex and challenging task influencing NLP applications.
# 
# For example, in languages like English, it is generally straightforward to identify words by using spaces as delimiters. However, there are exceptions, such as contractions like "can't" and hyphenated words like "self-driving". And in Dutch, where multiple nouns can be combined into one bigger noun without any delimiter this can be hard. How would you tokenize "hippopotomonstrosesquippedaliofobie"? In other languages, such as Chinese and Japanese, there are no spaces between words, so identifying word boundaries is much more difficult. 
# 
# To illustrate the use of tokenization, let's consider the following example, which tokenizes a sample text using the `word_tokenize` function from the NLTK package. That function uses a pre-trained tokenization model for English.

# In[9]:


# Sample text.
text = "The quick brown fox jumped over the lazy dog. The cats couldn't wait to sleep all day."

# Tokenize the text.
tokens = word_tokenize(text)

# Print the text and the tokens.
print("Original text:", text)
print("Tokenized text:", tokens)


# <a name="t3-2"></a>

# ### Part-of-speech tagging

# Part-of-speech (POS) tagging is the process of assigning each word in a text corpus with a specific part-of-speech tag based on its context and definition. The tags typically include nouns, verbs, adjectives, adverbs, pronouns, prepositions, conjunctions, interjections, and more. POS tagging can help other NLP tasks disambiguate a token somewhat due to the added context.

# In[10]:


pos_tags = nltk.pos_tag(tokens)
print(pos_tags)


# <a name="t3-3"></a>

# ### Stemming / Lemmatization

# Stemming and lemmatization are two common techniques used in NLP to preprocess and normalize text data. Both techniques involve transforming words into their root form, but they differ in their approach and the level of normalization they provide.
# 
# Stemming is a technique that involves reducing words to their base or stem form by removing any affixes or suffixes. For example, the stem of the word "lazily" would be "lazi". Stemming is a simple and fast technique that can be useful. However, it can also produce inaccurate or incorrect results since it does not consider the context or part of speech of the word.
# 
# Lemmatization, on the other hand, is a more sophisticated technique that involves identifying the base or dictionary form of a word, also known as the lemma. Unlike stemming, lemmatization can consider the part of speech of the word, which can make it more accurate and reliable. With lemmatization, the lemma of the word "lazily" would be "lazy". Lemmatization can be slower and more complex than stemming but provides a higher level of normalization.

# In[11]:


# Initialize the stemmer and lemmatizer.
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def wordnet_pos(nltk_pos):
    """
    Function to map POS tags to wordnet tags for lemmatizer.
    """
    if nltk_pos.startswith("V"):
        return wordnet.VERB
    elif nltk_pos.startswith("J"):
        return wordnet.ADJ
    elif nltk_pos.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# Perform stemming and lemmatization seperately on the tokens.
stemmed_tokens = [stemmer.stem(token) for token in tokens]
lemmatized_tokens = [lemmatizer.lemmatize(token, wordnet_pos(tag))
                     for token, tag in nltk.pos_tag(tokens)]

# Print the results.
print("Stemmed text:", stemmed_tokens)
print("Lemmatized text:", lemmatized_tokens)


# <a name="t3-4"></a>

# ### Stopword Removal

# Stopword removal is a common technique used in NLP to preprocess and clean text data by removing words that are considered to be of little or no value in terms of conveying meaning or information. These words are called "stopwords" and they include common words such as "the", "a", "an", "and", "or", "but", and so on.
# 
# The purpose of stopword removal in NLP is to improve the accuracy and efficiency of text analysis and processing by reducing the noise and complexity of the data. Stopwords are often used to form grammatical structures in a sentence, but they do not carry much meaning or relevance to the main topic or theme of the text. So by removing these words, we can reduce the dimensionality of the text data, improve the performance of machine learning models, and speed up the processing of text data. NLTK has a predefined list of stopwords for English.

# In[12]:


# English stopwords in NLTK.
stopwords_list = stopwords.words('english')
print(stopwords_list)


# <a name="a3-1"></a>

# ### Assignment for Task 3.1: Tokenization and Lemmatization

# The first step is to tokenize and lemmatize the sentences. **Your task (which is your assignment) is to write functions to do the following:**
# - Since we want to use our text to make a model later on, we need to preprocess it. Add a `tokens` column to the `df_train` dataframe with the text tokenized, then lemmatize those tokens. You must use the POS tags when lemmatizing.
#     - Hint: Use the `pandas.Series.apply` function with the imported `nltk.tokenize.word_tokenize` function. Recall that you can use the `pd.Series.apply?` syntax in a code cell for more information.
#     - Hint: use the `nltk.stem.WordNetLemmatizer.lemmatize` function to lemmatize a token. Use the `wordnet_pos` function to obtain the POS tag for the lemmatizer. 
# - Tokenizing and lemmatizing the entire dataset can take a while too. Use `tqdm` and the `pandas.Series.progress_apply` to show progress bars for the operations.

# In[13]:


def tokenize_and_lemmatize(df):
    """
    Tokenize and lemmatize the text in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "text" column.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the added "tokens" column.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################


# Our goal is to have a dataframe that looks like the following. **For simplicity, we only show the top 5 most frequent words.** Your data frame should have more rows.

# In[14]:


# This part of code will take some time to run.
answer_df_with_tokens = answer_tokenize_and_lemmatize(df_train_reformat)
answer_df_with_tokens.groupby("class").head(n=5)


# The code below tests if the output of your function matches the expected output.

# In[15]:


df_with_tokens = tokenize_and_lemmatize(df_train_reformat)
check_answer_df(df_with_tokens, answer_df_with_tokens)


# <a name="a3-2"></a>

# ### Assignment for Task 3.2: Word Counts

# In[16]:


def get_word_counts(df, token_col="tokens"):
    """
    Generate dataframes with the word counts for each class in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "class" and "tokens" columns.
    token_col: str
        Name of the column that stores the tokens.

    Returns
    -------
    pandas.DataFrame:
        There should be three columns in this dataframe.
        The "class" column shows the document class.
        The "tokens" column means tokens in the document class.
        The "count" column means the number of appearances of each token in the class.
        The dataframe should be sorted by the "class" and "count" columns.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################


# Our goal is to have a dictionary of dataframes (one per class) that look like the following. **For simplicity, we only show the top 5 most frequent words.** Your data frame should have more rows.

# In[17]:


answer_word_counts = answer_get_word_counts(answer_df_with_tokens, token_col="tokens")
answer_word_counts.groupby("class").head(n=5)


# The code below tests if the output of your function matches the expected output.

# In[18]:


word_counts = get_word_counts(df_with_tokens, token_col="tokens")
check_answer_df(answer_word_counts, word_counts)


# In the following function, we use the [wordcloud](https://amueller.github.io/word_cloud/auto_examples/simple.html#sphx-glr-auto-examples-simple-py) package to visualize the word counts that you just computed.

# In[19]:


def visualize_word_counts(df, class_col="class", token_col="tokens", count_col="count"):
    """
    Displays word clouds given a word counts dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe with three columns:
        - The "class" column (each document class)
        - The "tokens" column (showing tokens in each document class)
        - The "count" column (showing counts for each token)
    class_col : str
        Name of the class column (if different from "class").
    token_col : str
        Name of the token column (if different from "tokens").
    count_col : str
        Name of the count column (if different from "count").
    """
    # Groupby the class column and loop through all of them
    for name, df_group in df.groupby(class_col):
        # Compute a dictionary with word frequencies
        frequencies = dict(zip(df_group[token_col], df_group[count_col]))
        # Generate word cloud from frequencies
        wordcloud = WordCloud(background_color="white", width=1000, height=500, random_state=42).generate_from_frequencies(frequencies)
        # Display image
        plt.axis("off")
        plt.title("Class: " + name)
        plt.imshow(wordcloud)
        plt.show()


# In[20]:


visualize_word_counts(answer_word_counts)


# <a name="a3-3"></a>

# ### Assignment for Task 3.3: Stop Words Removal

# The stop words make it difficult for us to identify representative words for each class. Let's display the word counts using the data without stop words. But we need to remove the stop words first. **Your task (which is your assignment) is to write functions to do the following:**
# - Remove the stopwords from the tokens column in the dataframe.
#   - Hint: use the `pandas.DataFrame.isin` function.
#   - Hint: use the `stopwords_list` variable to help you check if a token is a stop word.

# In[21]:


def remove_stopwords(df):
    """
    Remove stopwords from the tokens.

    Parameters
    ----------
    df : pandas.DataFrame
        There should be three columns in this dataframe.
        The "class" column shows the document class.
        The "tokens" column means tokens in the document class.
        The "count" column means the number of appearances of each token in the class.
        The dataframe should be sorted by the "class" and "count" columns.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the stopwords rows removed.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################


# Our goal is to have a dictionary of dataframes (one per class) that look like the following. **For simplicity, we only show the top 5 most frequent words.** Your data frame should have more rows.

# In[22]:


answer_word_counts_no_stopword = answer_remove_stopwords(answer_word_counts)
answer_word_counts_no_stopword.groupby("class").head(n=5)


# The code below tests if the output of your function matches the expected output.

# In[23]:


word_counts_no_stopword = remove_stopwords(word_counts)
check_answer_df(word_counts_no_stopword, answer_word_counts_no_stopword)


# In[24]:


visualize_word_counts(answer_word_counts_no_stopword)


# <a name="spacy"></a>

# ### Another Option: spaCy

# spaCy is another library used to perform various NLP tasks like tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and much more. It provides pre-trained models for different languages and domains, which can be used as-is but also can be fine-tuned on a specific task or domain.
# 
# In an object-oriented way, spaCy can be thought of as a collection of classes and objects that work together to perform NLP tasks. Some of the important functions and classes in spaCy include:
# 
# - `nlp`: The core function that provides the main functionality of spaCy. It is used to process text and create a `Doc` object.
# - [`Doc`](https://spacy.io/api/doc): A container for accessing linguistic annotations like tokens, part-of-speech tags, named entities, and dependency parse information. It is created by the `nlp` function and represents a processed document.
# - [`Token`](https://spacy.io/api/token): An object representing a single token in a `Doc` object. It contains information like the token text, part-of-speech tag, lemma, embedding, and much more.
# 
# When a text is processed by spaCy, it is first passed to the `nlp` function, which uses the loaded model to tokenize the text and applies various linguistic annotations like part-of-speech tagging, named entity recognition, and dependency parsing in the background. The resulting annotations are stored in a `Doc` object, which can be accessed and manipulated using various methods and attributes.

# In[25]:


# Load the small English model in spaCy.
# Disable Named Entity Recognition and the parser in the model pipeline since we're not using it.
# Check the following website for the spaCy NLP pipeline:
# - https://spacy.io/usage/processing-pipelines
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Process the text using spaCy.
doc = nlp(text)

# This becomes a spaCy Doc object, which prints nicely as the original string.
print(doc)


# The `Doc` object can be iterated over to access each `Token` object in the document. We can also directly access multiple attributes of the `Token` objects. For example, we can directly access the lemma of the token with `Token.lemma_` and check if a token is a stop word with `Token.is_stop`. To make it easy to see them, we put them in a data frame.

# In[26]:


spacy_doc_attributes = [(token, token.lemma_, token.is_stop) for token in doc]
pd.DataFrame(data=spacy_doc_attributes, columns=["token", "lemma", "is_stopword"])


# The above example only deals with one sentence. Now we need to deal with all the sentences in all the classes. Below is a function to add a column with a `Doc` representation of the `text` column to the dataframe.

# In[27]:


def add_spacy_doc(df):
    """
    Add a column with the spaCy Doc objects.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "text" column.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the added "doc" column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Get the number of CPUs in the machine.
    n_process = max(1, os.cpu_count()-2)

    # Use multiple CPUs to speed up computing.
    df["doc"] = [doc for doc in tqdm(nlp.pipe(df["text"], n_process=n_process), total=df.shape[0])]

    return df


# Now we can add the spaCy tokens using the above function. This step will take some time since it needs to process all the sentences. So we added a progress bar.

# In[28]:


df_with_nltk_tokens_and_spacy_doc = add_spacy_doc(answer_df_with_tokens)
display(df_with_nltk_tokens_and_spacy_doc)


# The following function will add the spacy tokens to our original dataframe.

# In[29]:


def add_spacy_tokens(df):
    """
    Add a column with a list of lemmatized tokens, without stopwords and numbers.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "doc" column.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the "spacy_tokens" column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    df["spacy_tokens"] = df["doc"].apply(
        lambda tokens: [token.lemma_ for token in tokens if token.is_alpha and not token.is_stop]
    )

    return df


# We can run the code below to add the spacy tokens.

# In[30]:


df_with_nltk_tokens_and_spacy_tokens = add_spacy_tokens(df_with_nltk_tokens_and_spacy_doc)
display(df_with_nltk_tokens_and_spacy_tokens)


# Now we can use the function that we wrote before to get the word count from the spacy tokens.

# In[31]:


spacy_word_counts = answer_get_word_counts(df_with_nltk_tokens_and_spacy_tokens, token_col="spacy_tokens")
spacy_word_counts.groupby("class").head(n=5)


# <a name="t4"></a>

# ## Task 4: Unsupervised Learning - Topic Modeling

# Topic modelling is a technique used in NLP that aims to identify the underlying topics or themes in a collection of texts. One way to perform topic modelling is using the probabilistic model Latent Dirichlet Allocation (LDA).
# 
# LDA assumes that each document in a collection is a mixture of different topics, and each topic is a probability distribution over a set of words. The model then infers the underlying topic distribution for each document in the collection and the word distribution for each topic. LDA is trained using an iterative algorithm that maximizes the likelihood of observing the given documents.
# 
# To use LDA, we need to represent the documents as a bag of words, where the order of the words is ignored and only the frequency of each word in the document is considered. This bag-of-words representation allows us to represent each document as a vector of word frequencies, which can be used as input to the LDA algorithm. Computing LDA might take a moment on our dataset size.

# In[32]:


# Convert preprocessed text to bag-of-words representation using CountVectorizer.
vectorizer = CountVectorizer(max_features=1000)


# We will use the `fit_transform` function in the vectorizer. But in this case, we need a string that represents a sentence as the input. So, we can just join all the tokens together into one string. We also reset the index for consistency.

# In[33]:


df_strings = df_with_nltk_tokens_and_spacy_tokens["spacy_tokens"].apply(lambda x: " ".join(x))
df_strings = df_strings.reset_index(drop=True)
df_strings


# Then, we can use the `fit_transform` function to get the bag of words vector.

# In[34]:


X = vectorizer.fit_transform(df_strings.values)


# We convert the original matrix to a data frame to make it easier to see the bag of words. The columns indicate tokens, and the values for each cell indicate the word counts. The number of columns in the data frame matches the `max_features` parameter in the `CountVectorizer`. The number of rows matches the size of the training data.

# In[35]:


pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


# Now we have the bag of words vector. We can use the vector for LDA topic modeling.

# In[36]:


# Define the number of topics to model with LDA.
num_topics = 4

# Fit LDA to the feature matrix. Verbose so we know what iteration we are on.
# The random state is just for producing consistent results.
lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, random_state=42, verbose=True)
f = lda.fit(X)


# Now we can check the topic vectors in the LDA model. Each vector represents the topic in a high dimensional space. The high dimensional space is formed by using the word tokens. So, the vectors can also be viewed as weights that represents the number of importance that a word token was assigned to the topic. In the following code block, we print the shape of the vectors. The row size should match the number of topics that we set before. The column size should match the `max_features` parameter, which means the number of words.

# In[37]:


lda.components_.shape


# <a name="a4"></a>

# ### Assignment for Task 4

# We want to get the weights for each word in each topic and visualize them using word clouds. In the above case, the shape should be `(4, 1000)`, which means we have 4 topics, and each topic is represented by a distribution (i.e., weights) of 1000 words. To make the world cloud visualization simple, we only wants to use the top `n` number of words with the highest weights.
# 
# **Your task (which is your assignment) is to write functions to do the following:**
# - Given a 1D NumPy array, return the indexes of the top `n` number of items according to their values. In other words, we want the indexes that can help us select the highest `n` values. For example, for `n=3` in array `[3,1,2,4,0]`, the function should return `[3,0,2]`, because the highest value is `4` with index `3` in the original array, and so on.
#   - Hint: use the `numpy.argsort` function.
# - Notice that the `numpy.argsort` function gives you the indexes from the array items with the lowest value, which is not what we want. You need to figure out a way to reverse a numpy array and select the top `n` items.

# In[38]:


def get_index_of_top_n_items(array, n):
    """
    Given an NumPy array, return the indexes of the top "n" number of items according to their values. 

    Parameters
    ----------
    array : numpy.ndarray
        A 1D NumPy array.
    n : int
        The top "n" number of items that we want.

    Returns
    -------
    numpy.ndarray
        The indexes of the top "n" items.
    """
    ###################################
    # Fill in your answer here
    return None
    ###################################


# The following code shows the example that we mentioned above.

# In[39]:


A = np.array([3,1,2,4,0])
answer_top_n_for_A = answer_get_index_of_top_n_items(A, n=3)
answer_top_n_for_A


# The code below tests if the output of your function matches the expected output.

# In[40]:


B = lda.components_[0]
top_n_for_topic_0 = get_index_of_top_n_items(B, n=10)
answer_top_n_for_topic_0 = answer_get_index_of_top_n_items(B, n=10)
check_answer_df(top_n_for_topic_0, answer_top_n_for_topic_0)


# We can now use the function that we just implemented in the following function to help us get the weights for the top `n` words for each topic.

# In[41]:


def get_word_weights_for_topics(lda_model, vectorizer, n=100):
    """
    Get weights for words for each topic.
    
    Parameters
    ----------
    lda_model : sklearn.decomposition.LatentDirichletAllocation
        The LDA model.
    vectorizer : sklearn.feature_extraction.text.CountVectorizer
        The count vectorizer.
    n : int
        Number of important words that we want to get.
    
    Returns
    -------
    dict of pandas.DataFrame
        A dictionary with data frames.
    """
    words = vectorizer.get_feature_names_out()
    n = len(words) if n is None else n
    topic_word_weights = {}
    
    for idx, topic_vector in enumerate(lda_model.components_):
        top_features_ind = answer_get_index_of_top_n_items(topic_vector, n=n)
        top_features = [words[i] for i in top_features_ind]
        weights = topic_vector[top_features_ind]
        df = pd.DataFrame(weights, index=top_features, columns=["weight"])
        df = df.sort_values(by="weight", ascending=False)
        topic_word_weights[idx] = df

    return topic_word_weights


# Now we can take a look at the data first. For simplicity, we only print the first 10 important words for each topic.

# In[42]:


topic_word_weights = get_word_weights_for_topics(lda, vectorizer, n=100)
for k, v in topic_word_weights.items():
    print(f"\nTopic #{k}:")
    print(" ".join(v.index[0:10]))
    display(v.iloc[0:10])


# Then, we can use the word weights to create word clouds.

# In[43]:


# Generate a word cloud for each topic.
for topic_idx, words in topic_word_weights.items():
    frequencies = dict(zip(words.index, words["weight"]))
    wordcloud = WordCloud(background_color="white", width=1000, height=500).generate_from_frequencies(frequencies)
    # Display image
    plt.axis("off")
    plt.title(f"Topic {topic_idx}")
    plt.imshow(wordcloud)
    plt.show()


# Compare this with the word cloud visualizations in the pre-processing step previously. Does the LDA topic modeling represent the actural four document classes in the training data? What do you think?
# 
# For this task, we mainly use a qualitative way to evaluate topic modeling by visually inspecting the word clouds. There are also quantiative ways to evaluate the models, but they are not covered in this course. If you are interested in this, check the following resources:
# - [Demonstration of the topic coherence pipeline in Gensim](https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/topic_coherence_tutorial.ipynb)
# - [Performing Model Selection Using Topic Coherence](https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/topic_coherence_model_selection.ipynb)
# - [Benchmark testing of coherence pipeline on Movies dataset](https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/topic_coherence-movies.ipynb)

# <a name="t5"></a>

# ## Task 5: Supervised Learning - Topic Classification

# Topic classification is a task in NLP that involves automatically assigning a given text document to one or more predefined categories or topics. This task is essential for various applications, such as document organization, search engines, sentiment analysis, and more.
# 
# In recent years, deep learning models have shown remarkable performance in various NLP tasks, including topic classification. We will explore a neural network-based approach for topic classification using the PyTorch framework. PyTorch provides an efficient way to build and train neural networks with a high degree of flexibility and ease of use.

# <a name="t5-1"></a>

# ### Compute Word Embeddings

# We will first look at word embeddings, which represent words as vectors in a high-dimensional space. The key idea behind word embeddings is that words with similar meanings tend to appear in similar contexts, and therefore their vector representations should be close together in this high-dimensional space. Word embeddings have been widely used in various NLP tasks such as sentiment analysis, machine translation, and information retrieval.
# 
# There are several techniques to generate word embeddings, but one of the most popular methods is the Word2Vec algorithm, which is based on a neural network architecture. Word2Vec learns embeddings by predicting the probability of a word given its context (continuous bag of words or skip-gram model). The output of the network is a set of word vectors that can be used as embeddings.
# 
# We can train a Word2Vec model ourselves, but keep in mind that later on, it's not nice if we don't have embeddings for certain words in the test set. So let's first apply the familiar preprocessing steps to the test set:

# In[44]:


# Reformat the test set.
df_test_reformat = reformat_data(df_test)

# NLTK preprocessing.
df_test_with_tokens = answer_tokenize_and_lemmatize(df_test_reformat)

# spaCy preprocessing.
df_test_with_nltk_tokens_and_spacy_tokens = add_spacy_tokens(add_spacy_doc(df_test_with_tokens))

display(df_test_with_nltk_tokens_and_spacy_tokens)


# To obtain the complete model, we combine the `tokens` column into one series and call the `Word2Vec` function.

# In[45]:


# Rename the very long variables
df_train_preprocessd = df_with_nltk_tokens_and_spacy_tokens
df_test_preprocessd = df_test_with_nltk_tokens_and_spacy_tokens

# Get all tokens into one series.
tokens_both = pd.concat([df_train_preprocessd["tokens"], df_test_preprocessd["tokens"]])

# Train a Word2Vec model on the NLTK tokens.
w2v_model = Word2Vec(tokens_both.values, vector_size=40, min_count=1)


# To obtain the embeddings, we can use the `Word2Vec.wv[word]` syntax. To get multiple vectors nicely next to each other in a 2D matrix, we can call `numpy.vstack`.

# In[46]:


print(np.vstack([w2v_model.wv[word] for word in ["rain", "cat", "dog"]]))


# The spaCy model we used has a `Tok2Vec` algorithm in its pipeline, so we can directly access the 2D matrix of all word vectors on a document with the `Doc.tensor` attribute. Keep in mind this still contains the embeddings of the stopwords.

# In[47]:


print(doc.tensor)


# To prepare the word embeddings for classification, we will add a `tensor` column to both the dataframes for training and testing. Each cell in the `tensor` column should be a tensor array, representing the word embedding vector for the text in the corresponding row. The tensors need to have the same size for both the training and test sets, so we also need to pad the tensors with smaller sizes by adding zeros at the end.

# In[48]:


def add_padded_tensors(df1, df2):
    """
    Add a tensor column to the dataframes, with every tensor having the same dimensions.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe containing at least the "tokens" or "doc" column.
    df2 : pandas.DataFrame
        The second dataframe containing at least the "tokens" or "doc" column.

    Returns
    -------
    tuple[pandas.DataFrame]
        The dataframes with the added tensor column.
    """
    # Copy the dataframes to avoid editing the originals.
    df1 = df1.copy(deep=True)
    df2 = df2.copy(deep=True)

    # Add tensors (option 1: using the Word2Vec model that we created).
    #for df in [df1, df2]:
    #    df["tensor"] = df["tokens"].apply(
    #        lambda tokens: np.vstack([w2v_model.wv[token] for token in tokens])
    #    )
    
    # Add tensors (option 2: using the spaCy tensors).
    for df in [df1, df2]:
        df["tensor"] = df["doc"].apply(lambda doc: doc.tensor)

    # Determine the largest amount of columns in both the training and test sets.
    largest = max(df1["tensor"].apply(lambda x: x.shape[0]).max(),
                  df2["tensor"].apply(lambda x: x.shape[0]).max())

    # Pad the tensors with zeros so that they have equal size.
    for df in [df1, df2]:
        df["tensor"] = df["tensor"].apply(
            lambda x: np.pad(x, ((0, largest - x.shape[0]), (0, 0)))
        )

    return df1, df2


# In[49]:


df_train_with_tensor, df_test_with_tensor = add_padded_tensors(df_train_preprocessd, df_test_preprocessd)
display(df_test_with_tensor)


# <a name="t5-2"></a>

# ### Build the Classifier 

# Our neural network will take the embedding representation of the document as input and predict the corresponding topic using a softmax output layer. We will evaluate the performance of our model using various metrics such as accuracy, precision, recall, and F1-score.
# 
# The following code demonstrates how to implement a neural network for topic classification in PyTorch. First let's do some more preparations for our inputs, turning them into PyTorch tensors.

# In[50]:


# Transform spaCy tensors into PyTorch tensors.
input_train = torch.from_numpy(np.stack(df_train_with_tensor["tensor"]))
input_test = torch.from_numpy(np.stack(df_test_with_tensor["tensor"]))

# Get the labels, move to 0-indexed instead of 1-indexed.
train_labels = torch.from_numpy(df_train_with_tensor["class_idx"].values) - 1
test_labels = torch.from_numpy(df_test_with_tensor["class_idx"].values) - 1

# One-hot encode labels for training.
train_target = torch.zeros((len(train_labels), 4))
train_target = train_target.scatter_(1, train_labels.unsqueeze(1), 1).unsqueeze(1)


# Then, it is time to define our network. The neural net consists of three fully connected layers (`fc1`, `fc2`, and `fc3`) with ReLU activation (`relu`) in between each layer. We flatten the input tensor using `view` before passing it through the fully connected layers. Finally, we apply the softmax activation function (`softmax`) to the output tensor to obtain the predicted probabilities for each class.

# In[51]:


class TopicClassifier(nn.Module):
    def __init__(self, input_width, input_length, output_size):
        super(TopicClassifier, self).__init__()
        self.input_width = input_width
        self.input_length = input_length
        self.output_size = output_size

        self.fc1 = nn.Linear(input_width * input_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Flatten the input tensor.
        x = x.view(-1, self.input_width * self.input_length)

        # Pass through the fully connected layers with ReLU activation.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        # Apply softmax activation to the output.
        x = self.softmax(x)
        return x


# Now it's time to train our network, this may take a while, but the current loss will be printed after every epoch.
# If you want to run the code faster, you can also put this notebook on Google Colab and use its provided GPU to speed up computing.

# In[52]:


# Define parameters.
n_classes = len(train_labels.unique())
input_size = input_train.shape[1:]
num_epochs = 5
lr = 0.001

# Define model, loss function and optimizer.
model = TopicClassifier(*input_size, output_size=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training loop.
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(zip(input_train, train_target)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# <a name="a5"></a>

# ### Optional Assignment for Task 5

# The following code evaluates the model using a confusion matrix.

# In[53]:


# Evaluate the neural net on the test set.
model.eval()

# Sample from the model.
with torch.no_grad():
    test_outputs = model(input_test)
    # Reuse our previous function to get the label with biggest probability.
    test_pred = np.argmax(test_outputs.detach(), axis=1)

# Set model back to training mode.
model.train()

# Compute the confusion matrix
cm = confusion_matrix(test_labels, test_pred)

# Plot the confusion matrix using seaborn
labels = ["World", "Sports", "Business", "Sci/Tech"]
h = sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=labels, yticklabels=labels)
ax = plt.xlabel("Predicted Labels")
ax = plt.ylabel("True Labels")


# If you do not feel done with text data yet, there is always more to do. In this optional assignment, you can experiment with the number of epochs, learning rate, vector size, optimizer, neural network architecture, regularization, etc. Also, we only use a small subset of this dataset for performance issues. If you have a high-end computer, you can go to the beginning of this tutorial to increase the size of the subset.
# 
# Even during the preprocessing, we could have done some things differently, like making everything lowercase and removing punctuation. Be aware that every choice you make along the way trickles down into your pipeline and can have some effect on your results. Also, take the time to write the code to evaluate the model with more metrics, such as accuracy, precision, recall, and the F1 score.

# In[ ]:




