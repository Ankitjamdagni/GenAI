import nltk
# from nltk.tokenize import sent_tokenize
# nltk.download('punkt_tab')


def tokenization():
    # corpus ~= paragraph
    corpus = r"On September 12, 2024, Tesla Inc. announced a new partnership with Panasonic to build advanced lithium-ion batteries at its Gigafactory in Nevada, USA. The company’s CEO, Elon Musk, stated that the project could create more than 10,000 jobs and reduce production costs by 15%. Meanwhile, experts from Harvard University predicted that the demand for electric vehicles would increase by 25% annually over the next decade. Despite the optimism, several analysts warned about potential supply chain issues and environmental concerns related to lithium mining."
    # print(corpus)

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    ## document ~= sentence (sentence tokenization)
    # from nltk.tokenize import sent_tokenize
    # document = sent_tokenize(text=corpus, language="english")
    # for index,sentences in enumerate(document):
    #     print(f"\nsentence{index+1}: {sentences} ")

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # from nltk.tokenize import word_tokenize
    ## word tokenization
    # words = word_tokenize(corpus)  # word_tokenize is different from wordpunct_tokenize (tokenize punctuations also)
    # for index,words in enumerate(words):
    #     print(f"word{index+1:} {words},")

    ## also
    # for sentences in document:
    #     words = word_tokenize(sentences)
    #     print(f"{words},")

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # from nltk.tokenize import TreebankWordTokenizer
    # tokeinzer = TreebankWordTokenizer()
    # data = tokeinzer.tokenize(text=corpus)
    # print(data)

def stemming():
    # finding root of the word also called lemma
    words = ["running", "runner", "easily", "fairness", "connected", "connections",
             "flies", "crying", "bigger", "happiest", "studies","studying","agreement", "denied"]

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    ## PorterStemmer
    # from nltk.stem import PorterStemmer
    # stemmer = PorterStemmer()
    # for word in words:
    #     print(f"{stemmer.stem(word)} is the stem word for {word}")

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    ## regex stemmer
    # from nltk.stem import RegexpStemmer
    # stemmer = RegexpStemmer(regexp='ing$|s$|er$|able$')
    # for word in words:
    #     print(f"{stemmer.stem(word)} for the word {word}")

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    ## snowball stremmer
    # from nltk.stem import SnowballStemmer
    # stemmer = SnowballStemmer(language="english")
    # for word in words:
    #     print(f"{stemmer.stem(word)} for the word {word}") 

    # also
    # from nltk.stem.snowball import EnglishStemmer
    # stemmer = EnglishStemmer()
    # for word in words:
    #     print(f"{stemmer.stem(word)} for the word {word}")   




def main() -> None:
    # word tokenization + sentence tokenization + ...
    # tokenization()
    # word stemming
    stemming()




if __name__ == "__main__":
    main()
