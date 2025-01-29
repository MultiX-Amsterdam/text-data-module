import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from util.util import wordnet_pos


def answer_tokenize_and_lemmatize(df):
    """
    This function is the answer of task 3.1.
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
    lemmatizer = WordNetLemmatizer()
    df["tokens"] = df["tokens"].progress_apply(
        lambda tokens: [lemmatizer.lemmatize(token, wordnet_pos(tag)) for token, tag in nltk.pos_tag(tokens)]
    )

    return df


def answer_get_word_counts(df, token_col="tokens"):
    """
    This function is the answer of task 3.2.
    Generate dataframes with the word counts for each class in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the "class" and "tokens" columns.
    token_col : str
        Name of the column that stores the tokens.

    Returns
    -------
    pandas.DataFrame
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
    This function is the answer of task 3.3.
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
    stopwords_list = stopwords.words('english')
    stopwords_set = set(stopwords_list)

    # Filter stopwords from tokens.
    df = df[~df["tokens"].isin(stopwords_set)]

    return df


def answer_get_index_of_top_n_items(array, n=3):
    """
    This function is the answer of task 4.
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