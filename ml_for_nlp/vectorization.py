import nltk
import pandas as pd
import re


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
    vectorizer = CountVectorizer(max_features=2500,binary=True)
    x = vectorizer.fit_transform(df['cleaned_messages'])

    print(x.shape)







def main() -> None:
    bow()

if __name__ == "__main__":
    main()
