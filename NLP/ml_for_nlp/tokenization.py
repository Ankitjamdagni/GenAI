import nltk
# DOWNLOAD THE RESOURCES AS PER REQUIREMENT
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

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

    stem_word_list = []
    clean_word_list = []
    # steps
    # take paragrapg -> convert to sentences -> convert into words-> apply stop words -> find stem words by stemming

    # step 1 : paragraph/corpus to sentences/documents
    from nltk import sent_tokenize
    sentences = sent_tokenize(paragraph)
    # print(len(sentences))

    # from nltk.stem import SnowballStemmer
    # stemmer = SnowballStemmer(language="english")

    from nltk.tokenize import word_tokenize

    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    
    # step 2 : sentences/documnets to words
    for index,sentence in enumerate(sentences):
        # print(f"{index} : {(sentence)}")
        
        # convert sentences to words
        words = word_tokenize(sentence)

        for subindex, word in enumerate(words):
            # stemword = stemmer.stem(word)
            stemword = stemmer.lemmatize(word, pos="v")
            # print(f"{index} {word} --> {stemword}")
            stem_word_list.append(stemword.lower())

            if stemword not in set(stopwords.words("english")):
                clean_word_list.append(stemword)

    refined_paragraph = " ".join(stem_word_list)
    print("_______________________________________")
    print("paragraph :: \n ", paragraph)
    print("_______________________________________")
    print("refined_paragraph :: \n ", refined_paragraph)
    print("_______________________________________")
    cleaned_paragraph = " ".join(clean_word_list)
    print("cleaned_paragraph :: \n ", cleaned_paragraph)

def postag() -> None:
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


    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk import pos_tag

    # step 1: convert to sentences 
    # step 2: convert sentences to words
    # step 2: eliminate stop words in sentences
    # step 3: pass sentence to postag after joining

    sentences = sent_tokenize(paragraph)

    for index, sentence in enumerate(sentences):

        words = word_tokenize(sentence)
        print(f"all words :: \n {words}")
        # exit()
        for subindex, word in enumerate(words):
            if word in set(stopwords.words("english")):
                words.pop(subindex)

        print(f"stopword output :: \n {words}")     
        # exit()
        
        postag = pos_tag(words)
        print(f"after postag :: \n {postag}")
        print("______________________________________")
    
def ner() -> None:
    """ NAMED ENTITY RECOFNITION (NER) """
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

    from nltk import word_tokenize

    words = word_tokenize(paragraph)
    # print(words)

    from nltk import pos_tag

    postags = pos_tag(words)
    # print(postags)

    from nltk import ne_chunk
    ne_chunk(postags).draw()
    print(ne_chunk(postags))


def main() -> None:
    ## word tokenization + sentence tokenization + ...
    # tokenization()
    
    ## word stemming
    # stemming()
    
    ## Lemmatization
    # lemmatization()
    
    ## stopwords
    # stop_words()

    ## PARTS OF SPEECH TAGGING
    # postag()
    
    # NAMED ENTITY RECOFNITION (NER)
    ner()



if __name__ == "__main__":
    main()
