import spacy

nlp = spacy.load("en_core_web_sm")
def Get_NLP_Info(sentence):
    #+-------------------------------------------------------------+
    #|Input: Sentence as a string e.g. "Hello this is a sentence"  |
    #|Process: Uses Spacy to get a dependency parse of the sentence|
    #|Output: Returns Dependcy parse                               |
    #+-------------------------------------------------------------+
     #Load english languages version of spacy
    DependencyParse = nlp(sentence) #Get the NLP information
    return DependencyParse
def Get_POS_Tags(sentence):
    #+-------------------------------------------------------------+
    #|Input: Sentence as a string e.g. "Hello this is a sentence"  |
    #|Process: Uses Spacy to get POS tags of each word             |
    #|Output: Returns list of tuples (word, posTag)                |
    #+-------------------------------------------------------------+

    nlp = spacy.load("en_core_web_sm")#Load english languages version of spacy
    ProcessedText = nlp(sentence) #Get the NLP information
    
    #Get the POS tags
    WordAndPOSTag = [(token.text, token.pos_) for token in ProcessedText]
    return WordAndPOSTag