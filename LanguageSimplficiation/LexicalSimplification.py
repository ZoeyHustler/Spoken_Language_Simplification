from transformers import pipeline #used for BERT
import Evaluation
import Parsers
import FormatChange
from nltk.corpus import wordnet
"""
from textblob import Word as w
"""
from itertools import product
import sys, os
#from en import parse, tenses, singularize
def Lexicon_Simplify_Main(text):
    #text = list of enumerated Sentences (sentence, ,i)
    methodDict = Lexicon_Simplify_Main_All_Methods(text)
    return methodDict["Overlap_Zipf_20"]
    newText = []
    for sentence,i,syntaxUpdate in text:
        doc = Parsers.Get_NLP_Info(sentence) 
        #simpleLexiconSentencePlusChanged = Lexical_Simplify(doc)
        simpleLexiconSentencePlusChanged = Lexical_Simplify_All_Methods(doc)
        simpleLexiconSentence, lexiconUpdate = simpleLexiconSentencePlusChanged
        cleanSentence = FormatChange.CleanSpacedApostraphies(FormatChange.StrList_To_String(simpleLexiconSentence))
        newText.append((cleanSentence, i, syntaxUpdate, lexiconUpdate))
    return newText
def Lexicon_Simplify_Main_All_Methods(text):
    methodNewTextDictionary = {"Overlap_Zipf_5": [],"Overlap_Zipf_20": [],"Overlap_Zipf_100": [],
                               "Overlap_BertScore_5": [],"Overlap_BertScore_20": [],"Overlap_BertScore_100": [],
                               "Wordnet_Zipf_5": [],
                               "Bert_Zipf_5": [],"Bert_Zipf_20": [],"Bert_Zipf_100": [],
                               "Bert_BertScore_5": []}
    #Create new part -> Backup method (want overlap with Bert as a backup)
    #Do a wordvector similarity as a method to choose the best word -> is there a way to combine this with zipf?????????
    for sentence,i,syntaxUpdate in text:
        doc = Parsers.Get_NLP_Info(sentence)
        functionReturnThing = Lexical_Simplify_All_Methods(doc)
        methodsToWordList, methodsToHasUpdated = functionReturnThing
        for method, newSentence in methodsToWordList.items():
            cleanSentence = FormatChange.CleanSpacedApostraphies(FormatChange.StrList_To_String(newSentence))
            methodNewTextDictionary[method].append((cleanSentence, i, syntaxUpdate, methodsToHasUpdated[method]))
    return methodNewTextDictionary
def Lexical_Simplify_All_Methods(spacySentence):
    #complex words using zipf (ignore entities + punctuation)
    wordsZipf = GetWordListZipf(spacySentence)
    complexWords = GetComplexWords(wordsZipf)
    
    #for complex Words get Bert simplification
    newWordAllMethodsList = {"Overlap_Zipf_5": [],"Overlap_Zipf_20": [],"Overlap_Zipf_100": [],
                             "Overlap_BertScore_5": [],"Overlap_BertScore_20": [],"Overlap_BertScore_100": [],
                             "Wordnet_Zipf_5": [],
                             "Bert_Zipf_5": [],"Bert_Zipf_20": [],"Bert_Zipf_100": [],
                             "Bert_BertScore_5": []}
    #TODO: preset this dictionary with all of the different methods -> so we can add non complex words
    lexiconChangedForMethods = {"Overlap_Zipf_5": False,"Overlap_Zipf_20": False,"Overlap_Zipf_100": False,
                                "Overlap_BertScore_5": False,"Overlap_BertScore_20": False,"Overlap_BertScore_100": False,
                                "Wordnet_Zipf_5": False,
                                "Bert_Zipf_5": False,"Bert_Zipf_20": False,"Bert_Zipf_100": False,
                                "Bert_BertScore_5": False}
    
    LexiconChanged = False
    newWordList = []
    for wordComplexTuple in complexWords:
        token, isComplex = wordComplexTuple
        if isComplex:
            methodToWordDict = WordReplacementStrategyAll(spacySentence, token.i)
            for method, word in methodToWordDict.items():
                if not lexiconChangedForMethods[method] and (word != token.text): lexiconChangedForMethods[method] = True
                newWordAllMethodsList[method].append(word)
            #check if word is same, or not
            
        else:
            #default keep word same if not complex: repeat for all values in dictionary
            for _, wordList in newWordAllMethodsList.items():
                wordList.append(token.text)
    return (newWordAllMethodsList, lexiconChangedForMethods)
def Lexical_Simplify(spacySentence):  
    #complex words using zipf (ignore entities + punctuation)
    wordsZipf = GetWordListZipf(spacySentence)
    complexWords = GetComplexWords(wordsZipf)
    
    #for complex Words get Bert simplification
    newWordList = []
    LexiconChanged = False
    
    for wordComplexTuple in complexWords:
        token, isComplex = wordComplexTuple
        if isComplex:
            newWord = GetWordReplacmentOverlap(spacySentence, token.i)
            #check if word is same, or not
            if not LexiconChanged and (newWord != token.text): LexiconChanged = True
            newWordList.append(newWord)
        else:
            #default keep the word and do not replace
            newWordList.append(token.text)
    return (newWordList, LexiconChanged)
def GetWordListZipf(spacySentence):
    #+----------------------------------------------------------------------------------------+
    #| Input: Sentence as a list of words ["Hello", "this", "is", "a", "sentence"]            |
    #| Process: Labels all words with a complexity score                                      |
    #| Output: List of tuples of (word, complexityScore) [("Hello", 0.4), ("this", 0.1), ...] |
    #+----------------------------------------------------------------------------------------+
    return [(token, Evaluation.Get_Zipf_Value(token.text)) for token in spacySentence]
def GetComplexWords(taggedSentence, cutoff = 5):
    #Complex Word -> Non entity, non punctuation that has a zipf score of < cutoff
    taggedComplexSentence = []
    for (token, zipf) in taggedSentence:
        if token.ent_type_: #Check if it is an entity (name place etc)
            taggedComplexSentence.append((token,False))
            continue
        if token.is_punct: #Check if punctuation
            taggedComplexSentence.append((token,False))
            continue
        if zipf > cutoff: #check the zipf value to determine if complex
            taggedComplexSentence.append((token, False))
        else:
            taggedComplexSentence.append((token, True))
    return taggedComplexSentence    
#--------------------------------------------------
def WordReplacementStrategyAll(spacySentence, indexOfComplexWord):
    #generation methods: Overlap, Wordnet, Bert
    #ranking method: Zipf, BertScore
    #numBertGenerations: 5, 20, 100

    #Get WordNet options
    wordNetList = WordNetReplacementLemmas(spacySentence[indexOfComplexWord].text)
    #print(spacySentence[indexOfComplexWord].text)
    #backup if Wordnet fails -> store original word again
    if wordNetList == []:
        wordNetList = [spacySentence[indexOfComplexWord].text]
    #Get Bert options (word, score) #100 options
    bertListFull = BertReplacmentWordAndScore(FormatChange.StrList_From_Spacy(spacySentence),indexOfComplexWord)
    
    generationMethodsA = ["Overlap", "Wordnet", "Bert"]
    rankingMethodsA = ["Zipf", "BertScore"]
    numBertGenerationsA = [5, 20, 100]
    methodToWordDictionary = {}
    for generationMethod, rankingMethod, numBertGenerations in product(generationMethodsA,rankingMethodsA,numBertGenerationsA):
        #cut out the one's I can't do (WordNet generation + Bert Score), no point doing multiple bert + bertScore as the highest ranked ones are generated first
        if (generationMethod == "Wordnet") and ((rankingMethod == "BertScore") or numBertGenerations != 5):
            continue
        if (generationMethod == "Bert") and (rankingMethod == "BertScore") and (numBertGenerations != 5):
            continue
        #cut out some of the bertlist
        bertList = bertListFull[:numBertGenerations]
        #currently a list of word, score -> turn this into a dictionary and a list
        bertWordToScoreDict = dict(bertList)
        bertWords = [item[0] for item in bertList]
    
        #create generatedWordList
        wordReplacementExists = True
        generationList = []
        if generationMethod == "Overlap":
            overlaps = FindBertWordNetOverlaps(bertWords,wordNetList)
            if len(overlaps) == 0:
                overlaps = [spacySentence[indexOfComplexWord].text]
                wordReplacementExists = False
            generationList = overlaps
        elif generationMethod == "Wordnet":
            generationList = wordNetList
        elif generationMethod == "Bert":
            generationList = bertWords
    
        #substituion ranking
        if rankingMethod == "Zipf":
            bestWord = ChooseBestWordReplacementZipf(generationList)
        elif rankingMethod == "BertScore":
            if wordReplacementExists:
                bestWord = max(generationList, key=lambda word: bertWordToScoreDict[word])
            else:
                bestWord = generationList[0]
        methodName = f"{generationMethod}_{rankingMethod}_{numBertGenerations}"
        methodToWordDictionary[methodName] = bestWord
    return methodToWordDictionary
def WordReplacementStrategyChoose(spacySentence, indexOfComplexWord, generationMethod, rankingMethod, numBertGenerations):
    #generation methods: Overlap, Wordnet, Bert
    #ranking method: Zipf, BertScore
    #numBertGenerations: 5, 20, 100

    #Get WordNet options
    wordNetList = WordNetReplacementLemmas(spacySentence[indexOfComplexWord].text)
    #Get Bert options (word, score) #100 options
    bertListFull = BertReplacmentWordAndScore(FormatChange.StrList_From_Spacy(spacySentence),indexOfComplexWord)
    

    #cut out some of the bertlist
    bertList = bertList[:numBertGenerations]
    #currently a list of word, score -> turn this into a dictionary and a list
    bertWordToScoreDict = dict(bertList)
    bertWords = [item[0] for item in bertList]
    
    #create generatedWordList
    generationList = []
    if generationMethod == "Overlap":
        overlaps = FindBertWordNetOverlaps(bertWords,wordNetList)
        if len(overlaps) == 0:
            overlaps = [spacySentence[indexOfComplexWord].text]
        generationList = overlaps
    elif generationMethod == "Wordnet":
        generationList = bertWords
    elif generationMethod == "Bert":
        generationList = wordNetList
    
    #substituion ranking
    if rankingMethod == "Zipf":
        bestWord = ChooseBestWordReplacementZipf(generationList)
    elif rankingMethod == "BertScore":
        bestWord = max(generationList, key=lambda word: bertWordToScoreDict[word])
    return bestWord
def GetWordReplacmentOverlap(spacySentence, indexOfComplexWord):
    
    bertOptionList = BertReplacementsJustWord(FormatChange.StrList_From_Spacy(spacySentence),indexOfComplexWord)
    # print(bertOptionList)
    wordNetOptions = WordNetReplacementLemmas(spacySentence[indexOfComplexWord].text)
    #new idea: lemmatize the bert options to compare to wordnet -> bert options are most likely correctly conjugated
    overlaps = FindBertWordNetOverlaps(bertOptionList,wordNetOptions)
    if len(overlaps) == 0:
        overlaps = [spacySentence[indexOfComplexWord].text]
    bestOverlap = ChooseBestWordReplacementZipf(overlaps)
    return bestOverlap #USED
def BertReplacementsJustWord(sentence, indexOfComplexWord):
    #+---------------------------------------------------------------------------------+
    #| Input: Sentence as list of words ["Hello", "this", "is", "a", "sentence", "."]  |
    #| Input: Integer index for which word in the sentence is complex                  |
    #| Process: Finds list of possible simplifications for the complex word            |
    #| Output: Returns list of possible words to replace the complex word with         |
    #+---------------------------------------------------------------------------------+
    #mask the sentence (replace index with MASK then add original sentence on the end)
    maskedSentence = FormatChange.StrList_To_String(sentence[:indexOfComplexWord] + ["[MASK]"] + sentence[indexOfComplexWord+1:] + sentence)
    
    #Set up and use BERT API to produce list of possible sentences
    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    PossibleSentences = unmasker(maskedSentence, top_k=20)
    
    wordReplacementChoices = [choice["token_str"] for choice in PossibleSentences]
    return wordReplacementChoices #USED
def WordNetReplacementLemmas(word):
    baseSynonymList = wordnet.synsets(word)
    #can access an expanded list by getting the "lemma names" of each result (gives a 2d list so we flatten this)
    #then get only the unique results from this
    expandedList = list(set(FormatChange.Flatten([synonym._lemma_names for synonym in baseSynonymList])))
    return expandedList #USED
def FindBertWordNetOverlaps(bertList, wordNetList):
    #lemmatize both options
    Bertlemmas = [(token.lemma_, token.text) for token in Parsers.Get_NLP_Info(" ".join(bertList))]
    wordNetLemmas = [(token.lemma_,token.text) for token in Parsers.Get_NLP_Info(" ".join(wordNetList))]
    #overlapLemmaTuples = list(set(Bertlemmas) & set(wordNetLemmas))
    set_list1 = set(tuple[0] for tuple in Bertlemmas)
    set_list2 = set(tuple[0] for tuple in wordNetLemmas)
    result_list = [b for a, b in Bertlemmas if a in set_list2]
    #overlapWords = [item[1] for item in overlapLemmaTuples]
    return result_list #USED
def ChooseBestWordReplacementZipf(wordChoices):
    #get zipf scores
    zipfScores = GetWordListZipf(Parsers.Get_NLP_Info(" ".join(wordChoices)))
    #choose the word with the highest zipf score (more common)
    bestWord = max(zipfScores, key=lambda x: x[1])[0] #sort by the zipf score. Take the minimum -> get the word corresponding to this score
    return bestWord.text #USED
#WORDNET + BERT with BERT
def BertReplacmentWordAndScore(sentence, indexOfComplexWord, numGenerations=100):
    #mask the sentence (replace index with MASK then add original sentence on the end)
    maskedSentence = FormatChange.StrList_To_String(sentence[:indexOfComplexWord] + ["[MASK]"] + sentence[indexOfComplexWord+1:] + sentence)
    
    #Set up and use BERT API to produce list of possible sentences
    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    PossibleSentences = unmasker(maskedSentence, top_k=numGenerations)
    
    wordReplacementChoices = [(choice["token_str"], choice["score"]) for choice in PossibleSentences]
    return wordReplacementChoices



