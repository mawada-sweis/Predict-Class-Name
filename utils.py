# Install required third-party libraries
# !pip install nltk

import re

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


# Download necessary nltk data
# nltk.download('stopwords')


stop_words_nltk = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def sampling_data(data) -> pd.DataFrame:
    """Perform data sampling to address class imbalance in a dataset.

    Args:
        data (pd.DataFrame): The dataset to be sampled. 
        It must contain a 'category' column indicating the class labels.

    Returns:
        pd.DataFrame: A DataFrame containing the original data with additional 
        instances of minority class samples appended.
    """
    df_minority = data[data['category'].map(data['category'].value_counts()) == 1]
    return pd.concat([data, df_minority])


def split_create_dfs(X, y, dist=False) -> tuple:
    """Split the data into training and testing sets and create DataFrames for each set.

    Args:
        X (DataFrame): The feature data.
        y (DataFrame): The target data.
        dist (bool, optional): If True, print class distribution information for each set. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements in order:
            - X_train (DataFrame): Features of the training set.
            - X_test (DataFrame): Features of the testing set.
            - y_train (DataFrame): Targets of the training set.
            - y_test (DataFrame): Targets of the testing set.
            - train_data (DataFrame): DataFrame containing content, file_name, and category for the training set.
            - test_data (DataFrame): DataFrame containing content, file_name, and category for the testing set.
    """
    # Split the data into training (60%), dev (20%), and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # combine the content and category again into a DataFrame for each set:
    train_data = pd.DataFrame({'content': X_train.content, 'file_name': X_train.file_name, 'category': y_train})
    test_data = pd.DataFrame({'content': X_test.content, 'file_name': X_test.file_name, 'category': y_test})
    
    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))
    
    if dist == True:
        # Checking the distribution of classes in each set
        print("Training set class distribution:\n", y_train.value_counts(normalize=True))
        print("Testing set class distribution:\n", y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test, train_data, test_data


def display_dist_random_class(category_data, train_category, test_category) -> None:
    """Display the distribution of randomly selected classes in each dataset.

    Args:
        category_data (Series): A series containing the category data for the entire dataset.
        train_category (Series): A series containing the category data for the training set.
        test_category (Series): A series containing the category data for the testing set.
    """
    
    # Select a few classes to inspect. Adjust the number of classes as needed.
    selected_classes = category_data.value_counts().index[:5]

    # Get the distribution of these classes in each set
    train_distribution = train_category.value_counts()[selected_classes]
    test_distribution = test_category.value_counts()[selected_classes]

    # Normalize the distributions to get percentages
    train_distribution_normalized = train_distribution / train_distribution.sum()
    test_distribution_normalized = test_distribution / test_distribution.sum()

    # Combine the distributions into a single DataFrame for display
    distributions = pd.DataFrame({
        'Train': train_distribution_normalized,
        'Test': test_distribution_normalized
    })

    # Display the distributions
    print(distributions)


def reorganize_data(content_data, file_name_data) -> tuple:
    """Reorganize content and file name data.

    Args:
        content_data (Series): The data representing content.
        file_name_data (Series): The data representing file names.

    Returns:
        tuple: A tuple containing the reorganized content data and file name data.
    """
    content_data = content_data.str.replace(r'\s+', ' ', regex=True).str.strip()
    file_name_data = file_name_data.astype(int)
    
    return content_data, file_name_data 


def lowercase_text(text) -> str:
    """Convert a string to lowercase.
    
    Args:
        text (str): The text to be converted.
    
    Returns:
        str: The converted lowercase string.
    """
    return text.lower()


def tokenize_text(row) -> list:
    """Tokenize a string into words.
    
    Args:
        row (str): The text to be tokenized.
    
    Returns:
        list: A list of words from the tokenized text.
    """
    return word_tokenize(row)


def remove_special_chars(words) -> list:
    """Remove special characters from each word in a list of words.
    
    Args:
        words (list): A list of words to process.
    
    Returns:
        list: A list of words with special characters removed.
    """
    pattern = re.compile(r'[^\w.]+|(\'s)+')
    return [re.sub(pattern, '', word) for word in words]


def remove_stopword(content) -> str:
    """Remove stopwords from a list of words.
    
    Args:
        content (list): A list of words from which stopwords are to be removed.
    
    Returns:
        list: A list of words with stopwords removed.
    """
    return [word for word in content if word not in stop_words_nltk]


def lemmatize_sentence(sentence) -> list:
    """Lemmatize each word in a list of words.
    
    Args:
        sentence (list): A list of words to be lemmatized.
    
    Returns:
        list: A list of lemmatized words.
    """
    return [lemmatizer.lemmatize(word) for word in sentence]


def remove_extra_spaces(lst) -> list:
    """Remove extra spaces from each string in a list.
    
    Args:
        lst (list): A list of strings to process.
    
    Returns:
        list: A list of strings with extra spaces removed.
    """
    return [element for element in lst if element.strip()]


def remove_periods(words) -> list:
    """Remove any period element.

    Args:
        words (list): A list of words.

    Returns:
        list: A list of words without period elements.
    """
    return [word for word in words if len(word)!=1 and word!='.']


def create_frequency_table(documents, vocabulary) -> pd.DataFrame:
    """Create the bag of words matrix based on the list of documents.

    Args:
        docs_list (pd.Series): A series of list of words, where each row represent a document, 
        which contain a list of words.
        vocabulary (list): --
        is_list (bool, optional)
    Returns:
        pd.DataFrame: The bag of words table which contain the frequency of each word in each document.
    """
    
    # get all rows(documents) to a list of list instead of series
    documents = documents.tolist()

    # create a dictionary of vocabulary words with its indexes
    word_index = {word: i for i, word in enumerate(vocabulary)}

    # create the bag of words matrix in shape of
    # [number of documents, size of vocabulary]
    vocab_size = len(vocabulary)
    bow_matrix = [[0] * vocab_size for _ in range(len(documents))]

    # calculate the frequency for each word in each document
    for doc_idx, doc in enumerate(documents):
        for word in doc:
            if word in word_index:
                word_idx = word_index[word]
                bow_matrix[doc_idx][word_idx] += 1

    # turn the matrix to df for ease display
    df_bow = pd.DataFrame(bow_matrix, columns=word_index.keys())
    
    return df_bow


def calculate_prior_and_bigdoc(documents, classes) -> (dict, dict):
    # Initialize the count for each category and bigDoc structure
    category_counts = classes.value_counts()
    bigDoc = {}
    
    for document, category in zip(documents, classes):
        # Append document to the correct category in bigDoc
        if category in bigDoc:
            bigDoc[category].extend(document)
        else:
            bigDoc[category] = document
    
    # Calculate the prior probability for each category
    num_documents = len(documents)
    prior = {category: np.log(count / num_documents) for category, count in category_counts.items()}
        
    return prior, bigDoc



def classify_new_document(document_tokens, vocabulary, prior, log_likelihood):
    # Count the frequencies of words in the document using Counter
    document_vector = Counter(document_tokens)
    
    # Filter out words not in the training vocabulary
    document_vector = {word: freq for word, freq in document_vector.items() if word in vocabulary}
    
    class_posteriors = {}
    
    for class_ in prior:
        # Start with the log prior probability
        class_posteriors[class_] = prior[class_]
        
        # Incrementally update the posterior probability for words in the document
        for word, freq in document_vector.items():
            if word in log_likelihood[class_]:
                class_posteriors[class_] += freq * log_likelihood[class_][word]
    
    # Predict the class with the highest posterior probability
    predicted_class = max(class_posteriors, key=class_posteriors.get)
    
    return predicted_class
