import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, TruncatedSVD
from nltk.stem import WordNetLemmatizer
from gensim import models, matutils


# Tokenizer that will also lemmatize
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        tokens = re.findall(r'(\b\w\w+\b)', articles)
        return [self.wnl.lemmatize(t) for t in tokens]

def get_eigenvectors(reducer, id2word):
    eigenvecs = []
    for component, vector in enumerate(reducer.components_):
        top_10 = sorted(
            [(id2word[index], value) for index, value in enumerate(vector)],
            key = lambda x: x[1], reverse = True)[:10]
        descriptive_words = [x[0] for x in top_10]
        print(f'{component}: {descriptive_words}')

def reduce_nlp_data(vectorizer, data, n_components, reducer):

    transformed_data = vectorizer.fit_transform(data)
    id2word = {identifier: word for word, identifier in vectorizer.vocabulary_.items()}

    if reducer == 'lda':
        corpus = matutils.Sparse2Corpus(transformed_data.transpose())
        lda = models.LdaModel(corpus=corpus, num_topics=n_components, minimum_probability=0.03,
                                 id2word=id2word, passes=10, random_state = 42)
        print(lda.print_topics())
        lda_corpus = lda[corpus]
        return lda, matutils.corpus2csc(lda_corpus).toarray().transpose()
    elif reducer == 'svd':
        SVD = TruncatedSVD(n_components, n_iter = 10, random_state = 42)
        svd_data = SVD.fit_transform(transformed_data)
        get_eigenvectors(SVD, id2word)
        return SVD, svd_data
    elif reducer == 'nmf':
        nmf = NMF(n_components, random_state = 42)
        nmf_data = nmf.fit_transform(transformed_data)
        get_eigenvectors(nmf, id2word)
        return nmf, nmf_data

    else:
        return None, None

def combine_scale(nlp_data, numerical_data, standard_scaler):
    # appended values from numerical features to my csc corpus
    combined_data = np.hstack((nlp_data, numerical_data))
    # standard scaled the entire array due to the presence of the categorical features
    scaled_data = standard_scaler.fit_transform(combined_data)
    return standard_scaler, scaled_data

def plot_tsne(tsne, data, labels):
    Xtrain = tsne.fit_transform(data);
    plt.rcParams.update({'figure.figsize': (10, 10)})
    sns.scatterplot(Xtrain[:, 0], Xtrain[:, 1], labels, size = 2, alpha = 'auto', markers = '.');
