import nltk
# DOWNLOAD THE RESOURCES AS PER REQUIREMENT
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')

def tokenization() -> None:
    # corpus ~= paragraph
    corpus = """ On September 12, 2024, Tesla Inc. announced a new partnership with Panasonic to build 
                advanced lithium-ion batteries at its Gigafactory in Nevada, USA. 
                The company’s CEO, Elon Musk, stated that the project could create more than 10,000 jobs 
                and reduce production costs by 15%. Meanwhile, experts from Harvard University predicted 
                that the demand for electric vehicles would increase by 25% annually over the next decade. 
                Despite the optimism, several analysts warned about potential supply chain issues and 
                environmental concerns related to lithium mining.  """
    print(corpus)

    # _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # document ~= sentence (sentence tokenization)
    from nltk.tokenize import sent_tokenize
    document = sent_tokenize(text=corpus, language="english")
    for index,sentences in enumerate(document):
        print(f"\nsentence{index+1}: {sentences} ")

    # _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    from nltk.tokenize import word_tokenize
    # word tokenization
    words = word_tokenize(corpus)  # word_tokenize is different from wordpunct_tokenize (tokenize punctuations also)
    for index,words in enumerate(words):
        print(f"word{index+1:} {words},")

    # also
    for sentences in document:
        words = word_tokenize(sentences)
        print(f"{words},")

    # _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    from nltk.tokenize import TreebankWordTokenizer
    tokeinzer = TreebankWordTokenizer()
    data = tokeinzer.tokenize(text=corpus)
    print(data)

def stemming() -> None:
    """ stem word only """
    # finding root of the word also called lemma
    words = ["running", "runner", "easily", "fairness", "connected", "connections",
             "flies", "crying", "bigger", "happiest", "studies","studying","agreement", "denied"]

    # _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # PorterStemmer
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    for word in words:
        print(f"{stemmer.stem(word)} is the stem word for {word}")

    # _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # regex stemmer
    from nltk.stem import RegexpStemmer
    stemmer = RegexpStemmer(regexp='ing$|s$|er$|able$')
    for word in words:
        print(f"{stemmer.stem(word)} for the word {word}")

    # _#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # snowball stremmer (extra advantage: it converts all words into small letters )
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer(language="english")
    for word in words:
        print(f"{stemmer.stem(word)} for the word {word}") 

    # also
    from nltk.stem.snowball import EnglishStemmer
    stemmer = EnglishStemmer()
    for word in words:
        print(f"{stemmer.stem(word)} for the word {word}")   

def lemmatization() -> None:
    """ root words not the stem word
        slower than stemming as it compared from existing corpus but still have various use cases like:
         chatbots, Q/A,  """
    words = ["running", "runner", "easily", "fairness", "connected", "connections",
            "flies", "crying", "bigger", "happiest", "studies","studying","agreement", "denied"]
    # WORDNET LEMMITIZER
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    print("IF PART OF SPEECH IS GIVEN AS NOUN::")
    for word in words:
        print(f"{lemmatizer.lemmatize(word=word, pos='n')} <-------------------- {word}")

    print("IF PART OF SPEECH IS GIVEN AS VERB::")
    for word in words:
        print(f"{lemmatizer.lemmatize(word=word, pos='v')} <-------------------- {word}")

    print("IF PART OF SPEECH IS GIVEN AS ADJECTIVE::")
    for word in words:
        print(f"{lemmatizer.lemmatize(word=word, pos='a')} <-------------------- {word}")

    print("IF PART OF SPEECH IS GIVEN AS ADVERB::")
    for word in words:
        print(f"{lemmatizer.lemmatize(word=word, pos='r')} <-------------------- {word}")

def stop_words() -> None:
    from nltk.corpus import stopwords
    
    # default stop words
    # print(stopwords.words('english'))

    # we can add stopwords to the list also

    paragraph = """ Dear Ankit,	 
 	I extend my heartfelt greetings to all of you on the auspicious occasion of Deepavali, a festival filled with energy and enthusiasm. This is the second Deepavali after the grand construction of the Ram Temple in Ayodhya. Lord Shri Ram teaches us to uphold righteousness and also gives us the courage to fight injustice. We have seen a living example of this a few months ago during Operation Sindoor. During Operation Sindoor, Bharat not only upheld righteousness but also avenged injustice.	 
 	This Deepavali is particularly special because, for the first time, lamps will be lit in many districts across the country, including remote areas. These are the districts where Naxalism and Maoist terrorism have been eradicated from the root. In recent times, we have seen many individuals abandoning the path of violence and joining the mainstream of development, expressing faith in the Constitution of our country. This is a major achievement for the nation.
 	Amid these historic achievements, the country has also embarked on next-generation reforms in recent days. On the first day of Navratri, lower GST rates were implemented. During this "GST Bachat Utsav” (Savings Festival), citizens are saving thousands of crores of rupees.
 	In a world going through multiple crises, Bharat has emerged as a symbol of both stability and sensitivity. We are also on track to become the third-largest economy in the world in the near future.
 	In this journey of a “Viksit” (Developed) and “Aatmanirbhar Bharat” (self-reliant India), our primary responsibility as citizens is to fulfill our duties towards the nation.
 	Let us adopt “Swadeshi” (local products) and proudly say: "This is Swadeshi!" Let us promote the spirit of “Ek Bharat, Shreshtha Bharat”. Let us respect all languages. Let us maintain cleanliness. Let us prioritize our health. Let us reduce the use of oil in our food by 10% and embrace Yoga. All these efforts will rapidly move us towards a “Viksit Bharat”.
 	Deepavali also teaches us that when one lamp lights another, its light doesn't diminish, but it grows further. With the same spirit, let us light lamps of harmony, cooperation and positivity in our society and surroundings this Deepavali.
    Once again, wishing you all a very Happy Deepavali.
 	Yours,
    Narendra Modi """

    # steps
    # take paragrapg -> convert to sentences -> convert into words-> apply stop words -> find stem words by stemming
    from nltk import sent_tokenize
    sentences = sent_tokenize(paragraph)
    # print(sentences)

    for sentence in sentences:
        print(sentences)

    





    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer(language="english")




def main() -> None:
    ## word tokenization + sentence tokenization + ...
    # tokenization()
    
    ## word stemming
    # stemming()
    
    ## Lemmatization
    # lemmatization()
    
    ## stopwords
    stop_words() 



if __name__ == "__main__":
    main()
