import os
import math
from nltk.corpus import wordnet as wn #wordNet for synonyms
from transformers import pipeline #used for BERT
import spacy #dependency parser
from transformers.data.data_collator import numpy_default_data_collator #Not using currently
import pyttsx3
#import modules that I have created to split up my code better
import LexicalSimplification
import SyntacticSimplification
import Evaluation
import SpeechToText
import Parsers
import FormatChange

def Evaluate_Text_Complexity(text):
    #+----------------------------------------------------------------------------+
    #| Input: List of sentences ["This is a sentence", "This is another sentence] |
    #| Process: Uses all evaluation metrics of the text                           |
    #| Output: Result of all evaluation metrics as a dictionary                   |
    #+----------------------------------------------------------------------------+

    #Calculate dependency parsing evaluations
    PerSentenceEvaluationList = []
    for sentence in text:
        PerSentenceEvaluationList.append(Evaluation.Calculate_Dependency_Parse_Evaluations(sentence))
    
    #Calculate Flesch-Reading-Ease Score for the whole text
    Flesch_Reading_Ease_Score = Evaluation.Calculate_Flesch_Reading_Ease_Score_Full_Text(text)

    #Calculate Gunning Fog index for whole text
    Gunning_Fox_Index = Evaluation.Calculate_Gunning_Fog_Index(text)
    return "" #TODO
def TextSimplification(enumeratedText):
    #Input: List of enumerated sentences [("this is sentence 0",0), ("this is sentence 1", 1)]
    #Simplify Syntax then Lexicon (in case lexicon plays up and messes with sentence structure)
    #Going to store a true/False whether there was an update to syntax or lexicon?
    SimpleSyntaxTextEnum = SyntacticSimplification.Syntax_Simplification_Main(enumeratedText)
    SimpleLexiconTextEnum = LexicalSimplification.Lexicon_Simplify_Main(SimpleSyntaxTextEnum)
    return SimpleLexiconTextEnum 
def AudioFileToSentenceList(audioFilePath):
    #1) Do speech to text processing
    jsonText = SpeechToText.Transcribe(audioFilePath)
    #2) Convert JSON to list of sentences
    sentenceList = FormatChange.ExtractSentences(jsonText)
    return sentenceList
def TextToSpeechSave(textStr, savedFileName="UnnamedFile"):
    newFileName = GetUniqueFileName(savedFileName+"_Audio", "Audio_Output_Files", "mp3")
    engine = pyttsx3.init()
    engine.save_to_file(textStr, newFileName)
    engine.runAndWait()
def TextSave(textStr, savedFileName="UnnamedFile"):
    newFileName = GetUniqueFileName(savedFileName+"_Transcription", "Text_Output_Files", "txt")
    with open(newFileName, "w", encoding="utf-8") as file:
        file.write(textStr)
def GetUniqueFileName(filename, direc, extension):
    counter = 1
    while os.path.exists(os.path.join(direc, filename)):
        filename = f"{filename}_{counter}"
        counter += 1
    file_path = os.path.join(direc, f"{filename}.{extension}")
    return file_path
def SpeakSimplifiedText(textStr):
    engine = pyttsx3.init()
    engine.say(textStr)
    engine.runAndWait()
def CreateInputFilePath(inputFileName, isAudio):
    if isAudio: #is mp3 file
        fileNameExt = f"{inputFileName}.mp3"
        file_path = os.path.join("Audio_Input_Files", fileNameExt)
    else: #is txt file
        fileNameExt = f"{inputFileName}.txt"
        file_path = os.path.join("Text_Input_Files", fileNameExt)
    return file_path
def TextFileToSentenceList(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    sentenceList = FormatChange.ExtractSentences(file_contents)
    return sentenceList
def printEval(textEval):
    assert 1==1
    semantic = textEval[0]
    for i, sentence in enumerate(semantic):
        for key,value in sentence.items():
            print(f"sentence {i}, {key}: {value}")
    oldTextComp = textEval[1]
    print(f"old Text, Flesch-Reading-Ease-Score: {oldTextComp['FRES']}")
    print(f"old Text, Gunning Fog Index: {oldTextComp['GFI']}")
    newTextComp = textEval[2]
    print(f"new Text, Flesch-Reading-Ease-Score: {newTextComp['FRES']}")
    print(f"new Text, Gunning Fog Index: {newTextComp['GFI']}")
    oldSentComp = textEval[3]
    for tup in oldSentComp:
        sentence, index = tup
        for key,value in sentence.items():
            print(f"old sentence {index}, {key}: {value}")
    newSentComp = textEval[4]
    for tup in newSentComp:
        sentence, index = tup
        for key,value in sentence.items():
            print(f"new sentence {index}, {key}: {value}")
    assert 1==1  
#ask user if they want audio or text file
isAudioInput = input("Enter 1 for audio, Enter 0 for text: ")
isAudio = (isAudioInput == "1")
#ask user for filename
if isAudio:
    inputfileName = input("Enter audio input file name: ")
else:
    inputfileName = input("Enter text input file name: ")
#ask user for filename to store to

outputFileName = input("Enter output file name: ")

filePath = CreateInputFilePath(inputfileName,isAudio)
if os.path.exists(filePath):
    #check if audio or text
    if isAudio:
        sentenceList = AudioFileToSentenceList(filePath)
    else:        
        sentenceList = TextFileToSentenceList(filePath)
    #Enumerate sentence list (add integer with each sentence)
    enumeratedSentenceList = FormatChange.EnumerateSentenceList(sentenceList)
    simpleTextListEnum = TextSimplification(enumeratedSentenceList)
    simpleTextList = FormatChange.ExtractSentenceFrom4Tuple(simpleTextListEnum)
    #evaluate the text for help purposes
    textEval = Evaluation.FullEvaluation(sentenceList, simpleTextListEnum)
    #need to do something with this evaluation
    simpleText = FormatChange.StrList_To_String(simpleTextList)
    TextToSpeechSave(simpleText, outputFileName)
    TextSave(simpleText, outputFileName)
    #SpeakSimplifiedText(simpleText)
    printEval(textEval)    
    print(simpleText)
else:
    print("Input file does not exist")

    
