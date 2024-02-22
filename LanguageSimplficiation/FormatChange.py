import spacy
import Parsers
import string
def Remove_Punc_From_String_List(strList):
    newStrList = []
    for string in strList:
        if any(string.isalnum()) == True:
            newStrList.appned(string)
    return newStrList
def Fix_Capitilisation_String(mystr):
    doc = Parsers.Get_NLP_Info(mystr)
    #only capitalise if position in sentence = 1 or is a entity
    #else upcapitalise
    newStrList = []
    for token in doc:
        if token.ent_type_:
            newStrList.append(token.text)
        elif token.i == 0:
            newStrList.append(token.text.capitalize())
        else: 
            newStrList.append(token.text.lower())
    return ''.join(newStrList)
def Fix_Capitilisation_list(strList):
    return Fix_Capitilisation_String(''.join(strList)).split()
def StrList_To_String(StrList):
    return ' '.join(StrList)
def StrList_From_Spacy(spacyList):
    return [token.text for token in spacyList]
def Flatten(listOfLists):
    return [item for sublist in listOfLists for item in sublist]
def CleanSpacedApostraphies(sentenceString):
    cleaned_string = sentenceString.replace(" '", "'").replace(" .", ".").replace(" !", "!").replace(" ?", "?")  
    return cleaned_string
def ExtractSentences(textStr):
    #convert string of text into list of sentence strings
    doc = Parsers.Get_NLP_Info(textStr.replace("\n", ""))
    sentences = [sentence.text for sentence in doc.sents]
    return sentences
def EnumerateSentenceList(sentenceList):
   return [(sentence, i) for i, sentence in enumerate(sentenceList)]
def ExtractSentenceFrom4Tuple(enumSentenceList):
    return [sentence for sentence, _, _, _ in enumSentenceList]
def CreateEnumerated4Tuple(sentenceList):
    return [(sentence, i, None, None) for i, sentence in enumerate(sentenceList)]
def ExtractSentenceAndIndexFrom4Tuple(weirdTuple):
    return [(sentence, i) for sentence, i, _, _ in weirdTuple]
