import nltk
import pandas as pd
import re

def ohe() ->None:
    """ ONE HOT ENCODING """
    
def bow() -> None:
    """ spam/ham example(cleaning to vectorization) to vectorise the dataset.
    
    STEPS: data cleaning 
    description description description description description description. 
    description description description description description description. 
    description description description description description description. 

    Args:
        a (int): afded.
        b (int): fcsd.

    Returns:
        None

    Raises:
        None
    """
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    def cleaning(df):
        df["cleaned_messages"] = df["messages"].map(lambda  x: re.sub(pattern=r'[^a-zA-Z\s]', 
                                                                        repl='', 
                                                                        string= x).lower().split())
            
        # applying stemming ad stop words
        df['cleaned_messages'] = df['cleaned_messages'].map(stemmer_and_stopwords)

        # print(df["messages"])
        # print(df['cleaned_messages'])

        # save data to better format so that i will not perform above operatons again and again
        df.to_pickle("ml_for_nlp/Data/spamhamdata.pkl")
        # print(f"memory used : {df['cleaned_messages'].memory_usage(deep=True)/(1024*1024)} mb")

    def stemmer_and_stopwords(message:list) -> str:
        for index, word in enumerate(message):
            if word not in set(stopwords.words('english')):
                message[index] = stemmer.lemmatize(word=word, pos='v')
                # print(f"{word} -----> {stemmer.lemmatize(word=word, pos='v')}")
            else:
                message.pop(index)

        final_message = " ".join(message)        
        return final_message

    # check for latest saved check point or else perform the cleaning
    import os
    csv_file = r"ml_for_nlp/Data/spamhamdata.csv"
    pkl_file = os.path.splitext(csv_file)[0] + ".pkl"

    df = None

    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
    else:
        df = pd.read_csv(filepath_or_buffer= r"ml_for_nlp/Data/spamhamdata.csv", 
                        sep='\t',
                        names=['label', 'messages'])
        cleaning(df)
    
    # create bow
    from sklearn.feature_extraction.text import CountVectorizer
    bow_vectorizer = CountVectorizer(max_features=2500,binary=True)
    x = bow_vectorizer.fit_transform(df['cleaned_messages'])

    print(x.shape)
    print(bow_vectorizer.vocabulary_) # type: ignore

def ngram() -> None:
    # first run the BOW so that we bill be having pkl format data 
    # cleaner function can be made global but for every technique cleaning procedure is same
    # so use the existing pickle file

    import os
    pkl_file = r"ml_for_nlp/Data/spamhamdata.pkl"
    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)

    # create bow
    from sklearn.feature_extraction.text import CountVectorizer

    # check ngram (unigrams only(1,1), unigram and bigram(1,2), unigrma and trigram(1,3), 
    # unigram and bigram(2,3), trigram only(3,3))
    ngram_vectorizer = CountVectorizer(max_features=100,binary=True, ngram_range=(3,3))
    x = ngram_vectorizer.fit_transform(df['cleaned_messages'])    # type: ignore

    print(ngram_vectorizer.vocabulary_) # type: ignore

def tfidf() -> None:
    """ RERM FREQUENCY INVERSE DOCUMNET FREQUENCY """
    # first run the BOW so that we bill be having pkl format data 
    # cleaner function can be made global but for every technique cleaning procedure is same
    # so use the existing pickle file

    import os
    pkl_file = r"ml_for_nlp/Data/spamhamdata.pkl"
    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)

    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(2,2))
    x = tfidf_vectorizer.fit_transform(df["cleaned_messages"]) # type: ignore
    # print(x)

    for word,index in tfidf_vectorizer.vocabulary_.items():
        print(word, index)

def word2vec() -> None:
    """ use google's pre trained ML model for creating vectors 
        with 300 fetures for a single word """
    
    from gensim import downloader as api
    from gensim.models import word2vec, keyedvectors

    w2v_model = api.load("word2vec-google-news-300")
    print("Model loaded successfully!")
    
    king_vector = w2v_model['king'] # type: ignore
    queen_vector = w2v_model['queen'] # type: ignore
    
    print(f"\nVector for 'king': \n {king_vector[:10]}")
    print(f"\nVector for 'queen': \n {queen_vector[:10]}")
    print(f"\nsimilar words for king: \n{w2v_model.most_similar('king')}") # type: ignore

    # # Famous analogy,also mentioned in google papers: king - man + woman = queen
    result = w2v_model.most_similar(positive=['king', 'woman'], negative=['man']) # type: ignore
    print(result)

def main() -> None:
    # bow()
    # ngram()
    # tfidf()
    word2vec()

if __name__ == "__main__":
    main()
