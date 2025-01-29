import os
import numpy as np
from nltk.corpus import wordnet
from xml.sax import saxutils as su
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from wordcloud import WordCloud

# Add tqdm functions to pandas.
tqdm.pandas()

def check_answer_df(df_result, df_answer, n=1):
    """
    This function checks if two output dataframes are the same.

    Parameters
    ----------
    df_result : pandas.DataFrame
        The result from the output of a function.
    df_answer : pandas.DataFrame
        The expected output of the function.
    n : int
        The numbering of the test case.
    """
    try:
        assert df_answer.equals(df_result)
        print("Test case %d passed." % n)
    except:
        print("Test case %d failed." % n)
        print("")
        print("Your output is:")
        print(df_result)
        print("")
        print("Expected output is:")
        print(df_answer)


def check_answer_np(arr_result, arr_answer, n=1):
    """
    This function checks if two output numpy arrays are the same.

    Parameters
    ----------
    arr_result : numpy.ndarray
        The result from the output of a function.
    arr_answer : numpy.ndarray
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


def wordnet_pos(nltk_pos):
    """
    Function to map POS tags to wordnet tags for lemmatizer.

    Parameters
    ----------
    nltk_pos : str
        The nltk POS tag.

    Returns
    -------
    str
        The wordnet POS tag.
    """
    if nltk_pos.startswith("V"):
        return wordnet.VERB
    elif nltk_pos.startswith("J"):
        return wordnet.ADJ
    elif nltk_pos.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


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


def add_spacy_doc(df, nlp):
    """
    Add a column with the spaCy Doc objects.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "text" column.
    nlp : spaCy language model
        A spaCy language model (https://spacy.io/models/en).

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