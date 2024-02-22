from ast import Num
import math
import re
import Parsers
from nltk.corpus import cmudict #cmudict for counting syllables in pronounciation
import numpy
from scipy.spatial import distance #calculating cosine distance
from rouge import Rouge
import time #just for testing purposes
import nltk
import spacy #For gunning Fox Index to find lemmas
from nltk.stem import WordNetLemmatizer #Find lemmatizations of words for Gunning_Fox_Index
import FormatChange
from wordfreq import zipf_frequency
gloveWordVectorsFile = "glove.6B.200d.txt" #for word2VecReading from a file but I've already got this
wordFreqFile = "Freq100000.txt"

def FullEvaluation(oldText, newTextTuple):
    #old text -> list of sentences
    #newText -> list of (sentence, originalIndex, SyntaxUpdated, LexiconUpdated)
    #make oldText a 4,tuple to make aligning easier
    tuple4Old = FormatChange.CreateEnumerated4Tuple(oldText)
    
    oldSentenceListComplexity = 0
    newSentenceListComplexity = 0
    
    newText = FormatChange.ExtractSentenceFrom4Tuple(newTextTuple)
    
    semanticPreservationCalc = Calc_Semantic_Presevation_Text(FormatChange.ExtractSentenceAndIndexFrom4Tuple(tuple4Old),FormatChange.ExtractSentenceAndIndexFrom4Tuple(newTextTuple))
    #1) Get Reading Easy + Gunning Fog oldText
    oldTextComplexityMeasures = EvaluationPerTextSingle(oldText)
    #2) Get Reading Easy + Gunning Fog newText
    newTextComplexityMeasures = EvaluationPerTextSingle(newText)
    assert 1==1
    #3) Get Dependency Information oldText
    oldSentenceListComplexity = EvaluationPerSentenceSingle(FormatChange.ExtractSentenceAndIndexFrom4Tuple(tuple4Old))
    #4) Get Dependency Information newText
    newSentenceListComplexity = EvaluationPerSentenceSingle(FormatChange.ExtractSentenceAndIndexFrom4Tuple(newTextTuple))
    
    return (semanticPreservationCalc,oldTextComplexityMeasures,newTextComplexityMeasures,oldSentenceListComplexity,newSentenceListComplexity)
def EvaluationPerSentenceSingle(textPlusIndex):
    #textPlusIndex is a list of tuples (sentence, originalIndex)
    #Dependency parsing
    PerSentenceEvaluationList = []
    for sentence,i in textPlusIndex:
        PerSentenceEvaluationList.append((Calculate_Dependency_Parse_Evaluations(sentence),i))
    return PerSentenceEvaluationList
def EvaluationPerSentenceComparison(oldtext, newtext):
    SemanticPreservationScores = Calculate_Semantic_Preservation_Evaluations(oldtext, newtext)
    return SemanticPreservationScores
def EvaluationPerTextSingle(text):
    FRES = Calculate_Flesch_Reading_Ease_Score_Full_Text(text)
    GFI = Calculate_Gunning_Fog_Index(text)
    return {"FRES": FRES, "GFI": GFI}

#Complexity Evaluations
#-----------------------------------------------------------------------------------
#Syntax Dependnecy Graphs Evaluations
def Calculate_Dependency_Distances(DependencyParse):
    #+--------------------------------------------------------------+
    #| Input: Dependency Parse of the sentence (dictionary?)        |
    #| Process: Takes each dependency and calculates distances      |
    #| Output: List of dependency distances for each word (list of )|
    #+--------------------------------------------------------------+
    DependencyDistanceList = []
    for token in DependencyParse:
        startIndex = token.i
        endIndex = token.head.i
        Distance = abs(endIndex - startIndex)
        DependencyDistanceList.append(Distance)
    return DependencyDistanceList
def Calculate_Max_Dependency_Depth(DependencyParse):
    #+---------------------------------------------------------------------------------------+
    #| Input: Dependency Tree of a sentence                                                  |
    #| Process: For each token follow the edges to get to the root, find maximum path length |
    #| Output: Maximum Depth of any token from the root                                      |
    #+---------------------------------------------------------------------------------------+
    maxDepth = 0
    def Calculate_Dependency_Depth(token):
        if token.head == token: #At the Root verb of the sentence
            return 0 
        else: #Follow the head and recurse
            return 1+ Calculate_Dependency_Depth(token.head) 
        
    for token in DependencyParse: #For each token Calculate number of dependencies edges to get to the root
        currentTokenDepth = Calculate_Dependency_Depth(token)  
        if currentTokenDepth > maxDepth: 
            maxDepth = currentTokenDepth
    return maxDepth
def Calculate_Dependent_Count_For_Tokens(DependencyParse):
    #+-----------------------------------------------------------------+
    #| Input: Dependency Parse of a sentence                           |
    #| Process: For each token finds the number of dependents it has   |
    #| Process: Dependent = Head of arrow in graph                     |
    #| Output: For each token returns how many dependent it has        |
    #| Output: Returns as dict (token,dependentsCount)                 |
    #+-----------------------------------------------------------------+

    #Store as a dict first: (key,value)=(token, dependents)
    numDependentsDict = {}
    root = None
    for token in DependencyParse:
        if numDependentsDict.get(token.head) is None:
            #Does not exist -> create new entry
            
            numDependentsDict[token.head] = 1
        else:
            #Does exist -> Increment entry
            numDependentsDict[token.head] += 1
            
        if (token.head == token): #find root verb of tree for later
            root = token
    #Subtract 1 from the root verb as it points to itself
    numDependentsDict[root] -= 1
    return numDependentsDict
def Calculate_Max_Number_Dependents(NumberDependentsDict):
    #+-------------------------------------------------------------------+
    #| Input: Dictionary of (tokens:Number Dependents)                   |
    #| Process: Calculates maximum dependents a word has in the sentence |
    #| Output: Maximum number of dependents                              |
    #+-------------------------------------------------------------------+
    return max(NumberDependentsDict.values())
def Calculate_Average_Number_Dependents(NumberDependentsDict):
    valuesList = list(NumberDependentsDict.values())
    return (sum(valuesList) / len(valuesList))
def Calculate_Dependency_Parse_Evaluations(sentence):
    #+------------------------------------------------------------+
    #| Input: Sentence as a string                                |
    #| Process: Evaluation using dependency parse information     |
    #| Output: Returns Dictionary of Dependnecy Evaluation        |
    #| Output: MaxArcLength, AverageArcLength, MaxDependencyDepth |
    #| Output: MaxNumberDependents, AverageNumberDependents       |
    #+------------------------------------------------------------+

    #Parse the sentence
    DependencyParse = Parsers.Get_NLP_Info(sentence)
    
    #Dependency Distance Information
    DependencyDistanceList = Calculate_Dependency_Distances(DependencyParse)
    MaxArcLength = max(DependencyDistanceList)
    AverageArcLength = sum(DependencyDistanceList) / len(DependencyDistanceList) 

    #Dependency Depth Information
    MaxDependencyDepth = Calculate_Max_Dependency_Depth(DependencyParse)
    
    #Number Dependents Informations
    NumberDependentsDict = Calculate_Dependent_Count_For_Tokens(DependencyParse)
    MaxNumberDependents = Calculate_Max_Number_Dependents(NumberDependentsDict)
    AverageNumberDependents = Calculate_Average_Number_Dependents(NumberDependentsDict)

    #Put in dictionary to store
    DependencyInfoDict = {"MaxArcLength": MaxArcLength, "AverageArcLength": AverageArcLength,
                      "MaxDependencyDepth": MaxDependencyDepth,
                      "MaxNumberDependents": MaxNumberDependents, "AverageNumberDependents": AverageNumberDependents
                      }
    return DependencyInfoDict
#-----------------------------------------------------------------------------------
#Mixed Evaluations
def Calculate_Flesch_Reading_Ease_Score_Full_Text(text):
    #+----------------------------------------------------------------------------------+
    #|Input: List of sentences ["Hello this is a sentence.", "This is another sentence"]|
    #|Process: Calculates Number of sentences, Number of Words, Number of Syllables     |
    #|Output: Flesch_Reading_Ease_Score                                                 |
    #+----------------------------------------------------------------------------------+

    #Calculate Number Sentences
    NumSentences = len(text)
    
    #Calculate Number Words
    combinedText = " ".join(text) #Combine all words into one list
    doc = Parsers.Get_NLP_Info(combinedText)
    LowerSplitWords = [token.text.lower() for token in doc]
    NumWords = len(LowerSplitWords)

    #Calculate Number of Syllables
    NumSyllables = 0  
    CMUPronounciationDictionary = cmudict.dict() #To give Number of syllables for a word
    for word in LowerSplitWords:
        if word in CMUPronounciationDictionary:
            Pronounciation = CMUPronounciationDictionary[word][0]
            #Pronounciation splits words into phonemes
            #Syllable in a phoneme is marked by having a digit at the end (stress level)
            #Count the number of phonemes with a digit at the end
            for phoneme in Pronounciation:
                if phoneme[-1].isdigit():
                    NumSyllables += 1

    #Calulate Flesch Reading Ease Score
    WordsPerSentence = NumWords / NumSentences
    SyllablesPerWord = NumSyllables / NumWords
    Flesch_Reading_Ease_Score = 206.835 - (1.015 * WordsPerSentence) - (84.6 * SyllablesPerWord)
    return Flesch_Reading_Ease_Score
def Calculate_Gunning_Fog_Index(text):
    #+----------------------------------------------------------------------------------+
    #|Input: List of sentences ["Hello this is a sentence.", "This is another sentence"]|
    #|Process: Calculates Number of sentences, Number of Words, Number of Complex words |
    #|Output: Gunning Fog Index                                                         |
    #+----------------------------------------------------------------------------------+
    
    #Calculate Number of sentences
    NumSentences = len(text)
    
    #Process the text using Spacy
    combinedText = " ".join(text)
    nlp = spacy.load('en_core_web_sm')
    ProcessedText = nlp(combinedText)
    
    #Get the number of words (don't include punctuation)
    
    NumWords = len([token.text for token in ProcessedText if not token.is_punct])
    
    #Get the numebr of complex words (non proper nouns that are >= 3 syllables)
    NumberComplexWords = 0
    CMUPronounciationDictionary = cmudict.dict()
    for token in ProcessedText:
        #Check if the word is a proper noun -> skip if it is  
        if token.pos_ == "PROPN":
            continue 
       
        #Count Number of Syllables -> Check if number syllables in lemma > 3
        Lemma = token.lemma_ #Take the lemmaziation of the token -> (ignores common suffixes like -ed)     
        if Lemma in CMUPronounciationDictionary:
            Pronounciation = CMUPronounciationDictionary[Lemma][0] #Take the most common pronounciation
            #Pronounciation splits words into phonemes
            #Syllable in a phoneme is marked by having a digit at the end (stress level)
            #Count the number of phonemes with a digit at the end
            
            NumSyllables = 0
            for phoneme in Pronounciation:
                if phoneme[-1].isdigit():
                    NumSyllables += 1
            if NumSyllables >= 3: #Check if the number of syllables is greater than 3 (i.e. complex word)
                NumberComplexWords += 1    
    GunningFoxIndex = 0.4 * ((NumWords / NumSentences) + (100 * (NumberComplexWords / NumWords)))           
    return GunningFoxIndex
#-----------------------------------------------------------------------------------
#Lexical Evaluations
def Build_Frequency_Dictionary():
    frequency_dict = {}
    with open(wordFreqFile, 'r', encoding='utf-8') as file:
        for line in file:
            word, frequency = line.strip().split()
            frequency_dict[word] = int(frequency)
    return frequency_dict
freqDict = Build_Frequency_Dictionary()
def Get_Word_Frequency(word):
    return freqDict.get(word,0) #default value 0 if the word is not found in the dictionary
def Get_Average_Log_Word_Frequency(strList):
    def safe_log(x): #returns log of number but returns 0 if x = 0
        if x == 0:
            return 0
        else:
            return math.log(x,10)
    sumLogFreq = 0
    for word in strList:
        freq = Get_Word_Frequency(word)
        logFreq = safe_log(freq)
        sumLogFreq += logFreq
    return sumLogFreq / len(strList)
def Get_Average_Word_Frequency_Ignore_Entity(strList):
    #Use this because Places names aren't "complex" but will not appear in word lists
    doc = Parsers.Get_NLP_Info(FormatChange.StrList_To_String(strList))
    nonEntityWords = []
    for token in doc:
        if not token.ent_type_:
            nonEntityWords.append(token.text)
    averageLogFreq = Get_Average_Log_Word_Frequency(nonEntityWords)
    return averageLogFreq

def Get_Zipf_Value(word):
    return zipf_frequency(word, 'en', wordlist='large')
#-----------------------------------------------------------------------------------
#Semantic Preservation Evaluations
def Calc_Semantic_Presevation_Text(oldTextWithIndex, newTextWithIndex):
    #combine newTexts with the same index -> so they match the old text
    #Use a dictionary to do this
    groupSentenceDict = {}
    for sentence, index in newTextWithIndex:
        if index in groupSentenceDict:
            groupSentenceDict[index].append(sentence)
        else:
            groupSentenceDict[index] = [sentence]
    reWorkedSentencesNew = [(' '.join(sentence)) for index, sentence in groupSentenceDict.items()]
    removedIndexOld = [sentence for sentence, index in oldTextWithIndex]
    semPrevScoreList = []
    for old, new in zip(removedIndexOld,reWorkedSentencesNew):
        semPrevScores = Calculate_Semantic_Preservation_Evaluations(old,new)
        semPrevScoreList.append(semPrevScores)
    assert 1==1
def Calculate_Semantic_Preservation_Evaluations(oldSentence, newSentence):
    #each sentence is a string
    #turn into list for cosine similarity -> changing this to using a spacy tokenizer instead as they are better
    strList1 = [token.text for token in Parsers.Get_NLP_Info(oldSentence)]
    strList2 = [token.text for token in Parsers.Get_NLP_Info(newSentence)]
    #strlist1 = sentence1.split()
    #strlist2 = sentence2.split()   
    #cosineSimilarity = Calculate_Sentence_Cosine_Similarity(strList1, strList2) #CURRENTLY NOT WORKING
    
    #onlyDifferentSemantic = Calc_Semantic_Preservation_Only_Changed_Words(strList1,strList2) #CURRENTLY NOT WORKING
    #get the rouge scores
    rougeScores = Calculate_Rouge_Scores(newSentence, oldSentence) #need in order testSentence, referenceSentence
    #put all the data into a single dictionary (flatten the rouge dictionary)
    SemanticPreservationScore = {}
    #SemanticPreservationScore["CosineSim"] = cosineSimilarity #CURRENTLY NOT WORKING
    rouge1 = rougeScores[0]["rouge-1"]
    rouge2 = rougeScores[0]["rouge-2"]
    rougel = rougeScores[0]["rouge-l"]
    SemanticPreservationScore["Rouge1Fscore"] = rouge1["f"]
    SemanticPreservationScore["Rouge1Precision"] = rouge1["p"]
    SemanticPreservationScore["Rouge1Recall"] = rouge1["r"]
    
    SemanticPreservationScore["Rouge2Fscore"] = rouge2["f"]
    SemanticPreservationScore["Rouge2Precision"] = rouge2["p"]
    SemanticPreservationScore["Rouge2Recall"] = rouge2["r"]
    
    SemanticPreservationScore["RougeLFscore"] = rougel["f"]
    SemanticPreservationScore["RougeLPrecision"] = rougel["p"]
    SemanticPreservationScore["RougeLRecall"] = rougel["r"]
    
    return SemanticPreservationScore
def Get_Word2Vec_Dictionary(): #This isn't really used now because I've stored the vector in a numpy file
    word2vecDictionary = {}
    with open(gloveWordVectorsFile, "r", encoding="utf-8") as file:
        for wordVector in file:
            
            #WordVector is of format: "word" value1 value2 value3 ...
            wordAndValues = wordVector.split() #gives format ["word", dim1, dim2, dim3, ...]
            word = wordAndValues[0] #get the Word from the vector (first element)
            print(word)
            #create numpy array for the remaining values (the dimensions of the vector)
            vectorisedWord = numpy.array(wordAndValues[1:], dtype='float32')
            
            #store this word and vector in a dictionary
            word2vecDictionary[word] = vectorisedWord
    numpy.save("GloveWord2VecDict.npy", word2vecDictionary)
    return word2vecDictionary
def Calculate_Sentence_Cosine_Similarity(sentence1, sentence2):
    #+----------------------------------------------------------------------+
    #| Input: 2 Sentence lists: ["I", "want", "to", "pet",   "the", "dog"]  |
    #|                          ["I", "want", "to", "touch", "the", "dog"]  |
    #| Process: 1) Get word2vec Dictionary and find each word vector used   |
    #|           2) Averages word vectors to get sentence2Vec               |
    #|           3) Computes cosine similarity                              |
    #| Output: Cosine Similarity between the two sentences  between -1 and 1|
    #+----------------------------------------------------------------------+

    #get the word2vec dictionary
    word2vecDictionary = numpy.load("GloveWord2VecDict.npy", allow_pickle=True).item() #allow_pickle means that it can be loaded in without problems


    #Create 2D numpy array, get word vector for each word in the sentence
    sentence1wordVectorsNPArray = numpy.array([word2vecDictionary[word.lower()] for word in sentence1])
    sentence2wordVectorsNPArray = numpy.array([word2vecDictionary[word.lower()] for word in sentence2])
    
    #Calculate Average vector for each sentence -> get sentenceVector
    sentence1Vector = numpy.mean(sentence1wordVectorsNPArray, axis=0) #axis=0 means [[1,2,3],[4,5,6],[7,8,9]] gives average of [4,5,6]
    sentence2Vector = numpy.mean(sentence2wordVectorsNPArray, axis=0)

    #Normalize the vectors
    sentence1VecNorm = sentence1Vector / numpy.linalg.norm(sentence1Vector) #divide the vector by its size
    sentence2VecNorm = sentence2Vector / numpy.linalg.norm(sentence2Vector) #divide the vector by its size

    #Calculate cosine distance
    cosineDistance = distance.cosine(sentence1VecNorm, sentence2VecNorm)

    #Subtract from 1 to get cosine similarity
    cosineSimilarity = 1-cosineDistance
    return cosineSimilarity
def Calculate_Word_Cosine_Similarity(word1, word2):
    #+---------------------------------------------------------------------+
    #| Input: 2 strings with one word e.g. "touch", "pet"                  |
    #| Process: 1) Get word2vec Dictionary and find each word vector used  |
    #|          2) Compute cosine similarity                               |
    #| Output: Cosine Similarity between the two words between -1 and 1    |
    #+---------------------------------------------------------------------+  
    
    #Get the word2vec dictionary  
    word2vecDictionary = numpy.load("GloveWord2VecDict.npy", allow_pickle=True).item() #THIS WILL NOT WORK!!!!!! I HAVE REMOVED THIS BECAUSE IT IS TOO BIG FILE
    
    #Get the word vectors of the words
    word1Vector = word2vecDictionary[word1]
    word2Vector = word2vecDictionary[word2]

    #Normalize the vectors 
    word1VecNorm = word1Vector / numpy.linalg.norm(word1Vector) #divide the vector by its size
    word2VecNorm = word2Vector / numpy.linalg.norm(word2Vector) #divide the vector by its size
    
    #Calculate cosine distance
    cosineDistance = distance.cosine(word1VecNorm, word2VecNorm)

    #Subtract from 1 to get cosine similarity
    cosineSimilarity = 1-cosineDistance
    return cosineSimilarity
def Calculate_Rouge_Scores(testSentence, referenceSentence):
    #+--------------------------------------------------------------------------------------+
    #| Input: Sentence produced by program and gold standard simplification                 |
    #|        e.g. "I want to pet the dog", "I want to touch the dog"                       |
    #| Process: Uses ROUGE library, calculates F1_score, Precision and Recall for ROUGE1    |
    #| Output: Returns dictionary of ROUGE 1,2,L scores:                                    |
    #|[{                                                                                    |
    #|"rouge-1": {"f": 0.4786324739396596,"p": 0.6363636363636364,"r": 0.3835616438356164}, |
    #|"rouge-2": {"f": 0.2608695605353498,"p": 0.3488372093023256,"r": 0.20833333333333334},|
    #|"rouge-l": {"f": 0.44705881864636676,"p": 0.5277777777777778,"r": 0.3877551020408163} |
    #|}]                                                                                    |
    #+--------------------------------------------------------------------------------------+
    rouge = Rouge()
    return rouge.get_scores(testSentence, referenceSentence)

def Calc_Semantic_Preservation_Only_Changed_Words(oldSentence, newSentence):
    semPrevScoreList = []
    #in each sentence get the list of all tokens and find the ones that are unique to each text
    uniqueNew = [word for word in newSentence if word not in oldSentence]
    uniqueOld = [word for word in oldSentence if word not in newSentence]
    if uniqueNew == [] and uniqueOld == []:
        cosineSim = 1
    else:
        cosineSim = Calculate_Sentence_Cosine_Similarity(uniqueNew, uniqueOld)
    return cosineSim
