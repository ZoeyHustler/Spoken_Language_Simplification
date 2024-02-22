import Parsers
import FormatChange
import spacy
import string
from spacy import displacy


def Syntax_Simplification_Main(text):
    #Input List of enumerated Sentences (sentences + integer)
    #Output List of strings (sentences) where each has been passed through the DEPSYM algorithm
    newTextSemiEnumerated = []
    for sentence,i in text:
        NewSentence1, NewSentence2 = DEPSYM_Simplify(sentence)
        #If there was no simplification -> NewSentence2 is None
        #clean spacy apostraphies reverts errors like "I 'm" to "I'm"
        cleanSentence1 = FormatChange.CleanSpacedApostraphies(FormatChange.StrList_To_String(NewSentence1))
        #newTextSemiEnumerated.append((cleanSentence1,i))
        if len(NewSentence2) > 0:
            #There has been a syntax update
            newTextSemiEnumerated.append((cleanSentence1,i,True))
            cleanSentence2 = FormatChange.CleanSpacedApostraphies(FormatChange.StrList_To_String(NewSentence2))
            newTextSemiEnumerated.append((cleanSentence2,i, True)) #keep same i -> can combine split sentences back later for analysis
        else:
            #There has not been a syntax update
            newTextSemiEnumerated.append((cleanSentence1,i,False))
    return newTextSemiEnumerated

#Split this into Apposite, Relative, Conjoint (conj, advcl, ccomp) clauses (can come back to passive)
def DEPSYM_Simplify(sentence):
    #1) Get NLP information
    spaciedSentence = Parsers.Get_NLP_Info(sentence)
    #2) Get sentence as a list of tokens in case of failure
    wordList  = [token.text for token in spaciedSentence]
    #3) Assume we can only simplify if the root is a verb
    """
    for token in spaciedSentence:
        print(f"{token.text} --> {token.dep_} --> {token.head.text}")
    """
    if Is_Root_Verb(spaciedSentence) is False:
        return (wordList, [])
    
    #-----------------------------------------------------
    #check for different cases
    if Contains_Apposite_Clause(spaciedSentence) and Does_An_Nsubj_Attach_To_Root(spaciedSentence):
        SimplifiedPair = Simplify_Apposite_Clause(spaciedSentence)
    elif Contains_Conjoint_Clause_Conj(spaciedSentence):
        SimplifiedPair = Simplify_Conjoint_Clause_Conj(spaciedSentence)
    elif Contains_Conjoint_Clause_Advcl(spaciedSentence):
        SimplifiedPair = Simplify_Conjoint_Clause_Advcl(spaciedSentence)
    elif Contains_Conjoint_Clause_Ccomp(spaciedSentence):
        SimplifiedPair = Simplify_Conjoint_Clause_Ccomp(spaciedSentence)
    elif Contains_Relative_Clause(spaciedSentence) and Does_An_Nsubj_Attach_To_Root(spaciedSentence):
         SimplifiedPair = Simplify_Relative_Clause(spaciedSentence)
    else: 
        return (wordList, []) #idk how to simplfy otherwise
    return SimplifiedPair

def Simplify_Apposite_Clause(spaciedSentence):
    #Split the sentence into the 3 components
    sub, appos, obj = Split_Apposite_Clause(spaciedSentence)
    #remove punctuation from subject sentence
    sub = Remove_Punctuation_From_Token_List(sub)
    """
    The required auxiliary verb is determined using the tense of the root, and the singularity/plurality
    of the main subject. 
    """
        
    tense = Determine_Root_Tense(spaciedSentence)
    plurality = Determine_Is_Subject_Plural(spaciedSentence)
    person = Determine_Person(spaciedSentence)
    auxVerb = Determine_Auxilliary_Verb(tense, plurality, person)
        
    subObjSentence = Create_First_Sentence(sub, obj)
    subApposSentence = Create_Second_Sentence(sub, appos, auxVerb)
    strListFirst = Pure_Strings_List_From_Mixed_String_Token_List(subObjSentence)
    strListSecond =  Pure_Strings_List_From_Mixed_String_Token_List(subApposSentence)
    return strListFirst, strListSecond
def Simplify_Relative_Clause(spaciedSentence):
    #check if there is an nsubj -> will mess up otherwise

    #check if the relative clause it attached to the subject or the object
    if Is_Relative_Clause_Attached_To_Subject(spaciedSentence):
        #split sentence with relative clause in the subject
        sub, relcl, obj = Split_Relative_Clause_Subject(spaciedSentence)
        sub = Remove_Punctuation_From_Token_List(sub)
        #don't need to add auxillery verb
        subObjSentence = Create_First_Sentence_Relative_Subject(sub, obj)
        subRelclSentence = Create_Second_Sentence_Relative_Subject(sub, relcl)
                
        strListFirst = Pure_Strings_List_From_Mixed_String_Token_List(subObjSentence)
        strListSecond =  Pure_Strings_List_From_Mixed_String_Token_List(subRelclSentence)
        return strListFirst, strListSecond
    else:
        #Bob lives in Delhi, where his brother lives -> Bob lives in Delhi. Delhi is where his brother lives
        #split sentence with relative clause in the object
        sub, relcl, obj = Split_Relative_Clause_Object(spaciedSentence)
        #remove punctuation from the object section
        obj = Remove_Punctuation_From_Token_List(obj)
        tense = Determine_Relative_Clause_Tense(spaciedSentence)
        plurality = Determine_Object_Plurality(spaciedSentence)
        person = Determine_Object_Person(spaciedSentence)
        auxVerb = Determine_Auxilliary_Verb(tense, plurality, person)
            
        subObjSentence = Create_First_Sentence_Relative_Object(sub, obj)
        ObjRelclSentence = Create_Second_Sentence_Relative_Object(obj, relcl, auxVerb)
        #need to clean this up
        strListFirst = Pure_Strings_List_From_Mixed_String_Token_List(subObjSentence)
        strListSecond =  Pure_Strings_List_From_Mixed_String_Token_List(ObjRelclSentence)
        return subObjSentence, ObjRelclSentence
def Simplify_Conjoint_Clause_Conj(spaciedSentence):
    baseClause, cc, secondClause = Split_Conjoint_Clause_Conj(spaciedSentence)
    strListFirst = Pure_Strings_List_From_Mixed_String_Token_List(baseClause)
    strListSecond =  Pure_Strings_List_From_Mixed_String_Token_List(secondClause)
    return baseClause, secondClause
def Simplify_Conjoint_Clause_Advcl(spaciedSentence):
    BaseClause, ConjunctionWord, SecondClause = Split_Conjoint_Clause_Advcl(spaciedSentence)
    newConjunctionWord = Check_Conjunction_Word_Type(ConjunctionWord)
    if newConjunctionWord is None:
        wordList  = [token.text for token in spaciedSentence]
        return (wordList, []) #assume we cannot make a simplification -> return original sentence
    elif newConjunctionWord == "but":
        FirstSentence, SecondSentence = Advcl_Type1_Create_Sentences(BaseClause, newConjunctionWord, SecondClause)
    elif newConjunctionWord == "so" and newConjunctionWord != ConjunctionWord[0].text:
        FirstSentence, SecondSentence = Advcl_Type2_Create_Sentences_Swap_Order(BaseClause, newConjunctionWord, SecondClause)
    elif newConjunctionWord == "so":
        FirstSentence, SecondSentence = Advcl_Type2_Create_Sentences_Keep_Order(BaseClause, newConjunctionWord, SecondClause)
    else:
        #need to pass in the sentence to identify what the tense of the second clause verb is
        FirstSentence, SecondSentence = Advcl_Type3_Create_Sentences(spaciedSentence, BaseClause, newConjunctionWord, SecondClause)
    #sentence is a mix of pure strings and tokens -> simplfy this
    strListFirst = Pure_Strings_List_From_Mixed_String_Token_List(FirstSentence)
    strListSecond = Pure_Strings_List_From_Mixed_String_Token_List(SecondSentence)
    return (strListFirst, strListSecond) #we can make a simplfiication -> return both sentences
def Simplify_Conjoint_Clause_Ccomp(spaciedSentence):
    BaseClause, connectionWord, SecondaryClause = Split_Conjoint_Clause_Ccomp(spaciedSentence)
    FirstSentence, SecondSentence = Ccomp_Create_Sentences(BaseClause, connectionWord, SecondaryClause)
    FirstSentence, SecondSentence = Correct_Sentence_Order(FirstSentence, SecondSentence)
    return (FirstSentence, SecondSentence)

def Is_Root_Verb(spaciedSentence):
    root = [token for token in spaciedSentence if token.head == token][0]
    if "VERB" in root.pos_ or "AUX" in root.pos_:
        return True
    else: 
        return False
#Apposite Clauses
def Contains_Apposite_Clause(spaciedSentence):
    """
    spaciedSentence: nlp information of sentence
    """
    
    #Get the dependencies in the sentence
    #Check if there exists a appos dependency tag in the sentence
    existsApposDependency = False
    for token in spaciedSentence:
        if token.dep_ == "appos": existsApposDependency = True
    return existsApposDependency
def Split_Apposite_Clause(spaciedSentence):
    """
    take in sentence "The jeans, my favourite pair, need to be washed"
    split it into 3 components. "The jeans", "My favourite pair", "need to be washed"
    """
    #get the root of the sentence (where the head=itself)
    root = [token for token in spaciedSentence if token.head == token][0]
    RootSubtree = sorted(list(root.subtree), key=lambda x: x.i)
    #get the nsubj going from the root
    nsubj = [child for child in root.children if child.dep_ == "nsubj"][0]
    NsubjSubtree = sorted(list(nsubj.subtree), key=lambda x: x.i) #get the subtree of the nsubj -> part 1
    #English is an SVO language -> subject always at the start -> root verb in the middle -> object section always after
    appos = [child for child in nsubj.children if child.dep_ == "appos"][0]
    ApposSubtree = sorted(list(appos.subtree), key=lambda x: x.i)
    
    #I've got 3 sections
    #1) The jeans, my favourite pair, need to be washed
    #2) The jeans, my favourite pair,
    #3) my favourite pair
    
    #remove 3) from 2)
    PureSubject = []
    i = 0
    while i < len(NsubjSubtree):
        # Check if a sublist in list1 is equal to list2
        if NsubjSubtree[i:i + len(ApposSubtree)] == ApposSubtree:
            i += len(ApposSubtree)  # Skip the sublist in list1
        else:
            PureSubject.append(NsubjSubtree[i])
            i += 1
    #remove 2) from 1)
    PureObject = []
    i = 0
    while i < len(RootSubtree):
        # Check if a sublist in list1 is equal to list2
        if RootSubtree[i:i + len(NsubjSubtree)] == NsubjSubtree:
            i += len(NsubjSubtree)  # Skip the sublist in list1
        else:
            PureObject.append(RootSubtree[i])
            i += 1
    return (PureSubject, ApposSubtree, PureObject)
def Determine_Root_Tense(spaciedSentence):
    #get the root verb of the sentence
    root = [token for token in spaciedSentence if token.head == token][0]
    verb, tense = (root.text, root.tag_)
    return tense
def Determine_Is_Subject_Plural(spaciedSentence):
    root = [token for token in spaciedSentence if token.head == token][0]
    nsubj = [child for child in root.children if child.dep_ == "nsubj"][0]
    a = nsubj.morph.get("Number","")
    if "NOUN" in nsubj.pos_ and "Plur" in nsubj.morph.get("Number", ""):
        return "Plural"
    elif "NOUN" in nsubj.pos_ and "Sing" in nsubj.morph.get("Number", ""):
        return "Singular"
    else:
        return "Unknown"
def Determine_Person(spaciedSentence):
    #returns if in 1st, 2nd or 3rd person
    root = [token for token in spaciedSentence if token.head == token][0]
    nsubj = [child for child in root.children if child.dep_ == "nsubj"][0]
    PersonType = nsubj.morph.get("Person","")
    if len(PersonType) == 1:
        return int(PersonType[0]) #if it is stated that it is 1st 2nd 3rd person return that
    else:
        return 3 #else assume it is 3rd person
def Determine_Auxilliary_Verb(tense, plurality, person):
    #we want to get the correct form of "to be"
    if plurality == "Plural":
        if tense == "VB" or tense == "VBP" or tense == "VBZ": #present
            return "are"
        elif tense == "VBD": #past tense
            return "were"
        elif tense == "MD": #future tense
            return "will be"
    else: #assume singular even if unknown
        if tense == "VB" or tense == "VBP" or tense == "VBZ": #present
            if person == 1:
                return "am"
            elif person == 2:
                return "are"
            else: 
                return "is"
        elif tense == "VBD": #past tense
            if person == 1:
                return "was"
            elif person == 2:
                return "were"
            else: 
                return "was"
        elif tense == "MD": #future tense
            return "will be"
    return "are" #back up in case it all fails 
def Remove_Punctuation_From_Token_List(wordList):
    return [x for x in wordList if not x.dep_ == "punct"]
def Create_First_Sentence(sub, obj):
    #sub + obj
    newSentence = []
    for token in sub:
        newSentence.append(token.text)
    for token in obj:
        newSentence.append(token.text)
    newSentence.append(".")
    return newSentence
def Create_Second_Sentence(sub, appos, auxVerb):
    #sub + auxVerb + obj
    newSentence = []
    for token in sub:
        newSentence.append(token.text)
    newSentence.append(auxVerb)
    for token in appos:
        newSentence.append(token.text)
    newSentence.append(".")
    return newSentence
#Relative Clauses
def Contains_Relative_Clause(spaciedSentence):
    #check for a relcl dependency in the dependency tree
    existsRelclDependency = False
    for token in spaciedSentence:
        if token.dep_ == "relcl": existsRelclDependency = True
    return existsRelclDependency
def Does_An_Nsubj_Attach_To_Root(spaciedSentence):
    #get the root
    root = [token for token in spaciedSentence if token.head == token][0]
    nsubj = [child for child in root.children if child.dep_ == "nsubj"]
    return len(nsubj) > 0
def Is_Relative_Clause_Attached_To_Subject(spaciedSentence):
    root = [token for token in spaciedSentence if token.head == token][0]
    nsubj = [child for child in root.children if child.dep_ == "nsubj"][0]
    #check if one of the dependents of the child has a relcl dependency. If not the relative clause is in the object
    if len([child for child in nsubj.children if child.dep_ == "relcl"]) > 0:
        return True
    else:
        return False
def Split_Relative_Clause_Subject(spaciedSentence):
    root = [token for token in spaciedSentence if token.head == token][0]
    RootSubtree = sorted(list(root.subtree), key=lambda x: x.i) #need to sort this as dependency parsers with overlaps mess up the ordering
    
    nsubj = [child for child in root.children if child.dep_ == "nsubj"][0]
    NsubjSubtree = sorted(list(nsubj.subtree), key=lambda x: x.i)
    
    relcl = [child for child in nsubj.children if child.dep_ == "relcl"][0]
    RelclSubtree = sorted(list(relcl.subtree), key=lambda x: x.i)

    #1) Root subtree: "The bike, which I have had for ten years, is falling apart."
    #2) NsubjSubtree: "The bike, which I have had for ten years"
    #3) RelclSubtree: "which I have had for ten years"
    #remove 3) from 2)
    PureSubject = []
    i = 0
    while i < len(NsubjSubtree):
        # Check if a sublist in list1 is equal to list2
        if NsubjSubtree[i:i + len(RelclSubtree)] == RelclSubtree:
            i += len(RelclSubtree)  # Skip the sublist in list1
        else:
            PureSubject.append(NsubjSubtree[i])
            i += 1
    #remove 2) from 1)
    PureObject = []
    i = 0
    while i < len(RootSubtree):
        # Check if a sublist in list1 is equal to list2
        if RootSubtree[i:i + len(NsubjSubtree)] == NsubjSubtree:
            i += len(NsubjSubtree)  # Skip the sublist in list1
        else:
            PureObject.append(RootSubtree[i])
            i += 1
    return (PureSubject, RelclSubtree, PureObject)
def Split_Relative_Clause_Object(spaciedSentence):
    #Bob lives in Delhi, (where his brother lives) -> Bob lives in Delhi. Delhi is (where his brother lives)

    root = [token for token in spaciedSentence if token.head == token][0]
    RootSubtree = sorted(list(root.subtree), key=lambda x: x.i)
    
    relcl = [token for token in spaciedSentence if token.dep_=="relcl"][0]
    RelclSubtree = sorted(list(relcl.subtree), key=lambda x: x.i)
    
    mainObject = relcl.head
    MainObjectSubtree = sorted(list(mainObject.subtree), key=lambda x: x.i)
    
    #1) Get "Object-Relcl"
    #2) Sentence 1) Root + Object - Relcl
    #3) Sentence 2) Object + Auxillery Verb + Relcl

    PureObject = []
    i = 0
    while i < len(MainObjectSubtree):
        # Check if a sublist in list1 is equal to list2
        if MainObjectSubtree[i:i + len(RelclSubtree)] == RelclSubtree:
            i += len(RelclSubtree)  # Skip the sublist in list1
        else:
            PureObject.append(MainObjectSubtree[i])
            i += 1
    PureSubject = []
    i = 0
    while i < len(RootSubtree):
        # Check if a sublist in list1 is equal to list2
        if RootSubtree[i:i + len(MainObjectSubtree)] == MainObjectSubtree:
            i += len(MainObjectSubtree)  # Skip the sublist in list1
        else:
            PureSubject.append(RootSubtree[i])
            i += 1
    return (PureSubject, RelclSubtree, PureObject)
def Determine_Relative_Clause_Tense(spaciedSentence):
    relcl = [token for token in spaciedSentence if token.dep_ == "relcl"][0]
    verb, tense = (relcl.text, relcl.tag_)
    return tense
def Determine_Object_Plurality(spaciedSentence):
    relcl = [token for token in spaciedSentence if token.dep_=="relcl"][0]
    mainObject = relcl.head
    if "NOUN" in mainObject.pos_ and "Plur" in mainObject.morph.get("Number", ""):
        return "Plural"
    elif "NOUN" in mainObject.pos_ and "Sing" in mainObject.morph.get("Number", ""):
        return "Singular"
    else:
        return "Unknown"
def Determine_Object_Person(spaciedSentence):
    relcl = [token for token in spaciedSentence if token.dep_=="relcl"][0]
    mainObject = relcl.head
    PersonType = mainObject.morph.get("Person","")
    if len(PersonType) == 1:
        return int(PersonType[0]) #if it is stated that it is 1st 2nd 3rd person return that
    else:
        return 3 #else assume it is 3rd person
def Create_First_Sentence_Relative_Object(sub, obj):
    newSentence = []
    for token in sub:
        newSentence.append(token.text)
    for token in obj:
        newSentence.append(token.text)
    newSentence.append(".")
    return newSentence
def Create_Second_Sentence_Relative_Object(obj, relcl, auxVerb):
    newSentence = []
    for token in obj:
        newSentence.append(token.text)
    newSentence.append(auxVerb)
    for token in relcl:
        newSentence.append(token.text)
    newSentence.append(".")
    return newSentence
def Create_First_Sentence_Relative_Subject(sub, obj):
    newSentence = []
    for token in sub:
        newSentence.append(token.text)
    for token in obj:
        newSentence.append(token.text)
    newSentence.append(".")
    return newSentence
def Create_Second_Sentence_Relative_Subject(sub, relcl):
    newSentence = []
    for token in sub:
        newSentence.append(token.text)
    for token in relcl:
        newSentence.append(token.text)
    newSentence.append(".")
    return newSentence
#conjoint clauses
def Contains_Conjoint_Clause_Conj(spaciedSentence):
    #must contain the conj tag with the head at the root verb (ignores things like "I like apples and oranges")

    root = [token for token in spaciedSentence if token.head == token][0]
    #check if one of the roots childen has the conj tag
    conj = [child for child in root.children if child.dep_ == "conj"]
    cc = [child for child in root.children if child.dep_ == "cc"]
    if len(conj) > 0 and len(cc) > 0:
        return True
    return False
def Split_Conjoint_Clause_Conj(spaciedSentence):
    #split into 3 parts: Base clause, additional clause, conjunction word
    root = [token for token in spaciedSentence if token.head == token][0]
    RootSubtree = sorted(list(root.subtree), key=lambda x: x.i)
    
    conj = [child for child in root.children if child.dep_ == "conj"][0]
    ConjSubtree = sorted(list(conj.subtree), key=lambda x: x.i)
    
    cc = [child for child in root.children if child.dep_ == "cc"][0]
    CcSubtree = sorted(list(cc.subtree), key=lambda x: x.i)
    
    baseClausePlusCC = []
    i = 0
    while i < len(RootSubtree):
        # Check if a sublist in list1 is equal to list2
        if RootSubtree[i:i + len(ConjSubtree)] == ConjSubtree:
            i += len(ConjSubtree)  # Skip the sublist in list1
        else:
            baseClausePlusCC.append(RootSubtree[i])
            i += 1
    baseClause = []
    i = 0
    while i < len(baseClausePlusCC):
        # Check if a sublist in list1 is equal to list2
        if baseClausePlusCC[i:i + len(CcSubtree)] == CcSubtree:
            i += len(CcSubtree)  # Skip the sublist in list1
        else:
            baseClause.append(baseClausePlusCC[i])
            i += 1
    return baseClause, CcSubtree, ConjSubtree
#advcl
def Contains_Conjoint_Clause_Advcl(spaciedSentence):
    root = [token for token in spaciedSentence if token.head == token][0]
    #check if one of the roots childen has the conj tag
    advcl = [child for child in root.children if child.dep_ == "advcl"]
    if len(advcl) > 0:
        return True
    return False
def Split_Conjoint_Clause_Advcl(spaciedSentence):
    #split into base clause + extra clause (find root of second clause from end of advcl tag)
    #there might also be a "mark" -> separate this
    root = [token for token in spaciedSentence if token.head == token][0]
    RootSubtree = sorted(list(root.subtree), key=lambda x: x.i)
    advcl = [child for child in root.children if child.dep_ == "advcl"][0]
    AdvclSubtree = sorted(list(advcl.subtree), key=lambda x: x.i)
    
    #check if there is a "mark" tag attached to second clause root verb
    mark = [child for child in advcl.children if child.dep_ == "mark"]
    if len(mark)>0:
        #there is a mark -> separate this
        MarkSubtree = sorted(list(mark[0].subtree), key=lambda x: x.i)
        #remove advcl from root
        PureRootClause = []
        i = 0
        while i < len(RootSubtree):
            # Check if a sublist in list1 is equal to list2
            if RootSubtree[i:i + len(AdvclSubtree)] == AdvclSubtree:
                i += len(AdvclSubtree)  # Skip the sublist in list1
            else:
                PureRootClause.append(RootSubtree[i])
                i += 1
        #remove mark from advcl subtree
        PureSecondClause = []
        i = 0
        while i < len(AdvclSubtree):
            # Check if a sublist in list1 is equal to list2
            if AdvclSubtree[i:i + len(MarkSubtree)] == MarkSubtree:
                i += len(MarkSubtree)  # Skip the sublist in list1
            else:
                PureSecondClause.append(AdvclSubtree[i])
                i += 1 
    else:
        PureRootClause = []
        i = 0
        while i < len(RootSubtree):
            # Check if a sublist in list1 is equal to list2
            if RootSubtree[i:i + len(AdvclSubtree)] == AdvclSubtree:
                i += len(AdvclSubtree)  # Skip the sublist in list1
            else:
                PureRootClause.append(RootSubtree[i])
                i += 1
        #We check for one other conjunction word "so" -> if this does not exist then we assume no conjunction word -> idk how to simplify
        #"So" will be an advmod of the secondary clause at the beginning of the second clause
        advmodSo = [child for child in root.children if advcl.dep_ == "advmod"]
        if len(advmodSo) > 0:
            #check if the word is so
            if advmodSo[0].text == "so":
                MarkSubtree = advmodSo[0]
                PureSecondClause = []
                i = 0
                while i < len(AdvclSubtree):
                    # Check if a sublist in list1 is equal to list2
                    if AdvclSubtree[i:i + len(MarkSubtree)] == MarkSubtree:
                        i += len(MarkSubtree)  # Skip the sublist in list1
                    else:
                        PureSecondClause.append(AdvclSubtree[i])
                        i += 1
            else:
                PureSecondClause = AdvclSubtree
        else:
            PureSecondClause = AdvclSubtree
        
        
    return PureRootClause, MarkSubtree, PureSecondClause
def Check_Conjunction_Word_Type(ConjunctionWord):
    newConjunctionWord = None
    if ConjunctionWord is not None:
        if ConjunctionWord[0].text in ["although", "whereas", "however"]:
            #type 1 -> replace with but
            newConjunctionWord = "but"
        elif ConjunctionWord[0].text in ["because", "as"]:
            #type 2 -> replace with so -> swap sentence order later
            newConjunctionWord = "so"
        elif ConjunctionWord[0].text in ["before", "after", "once", "since", "when"]:
            #type 3 -> keep same word -> add "this + auxVerb before"
            newConjunctionWord = ConjunctionWord[0].text
        else: 
            #else we don't know 
            newConjunctionWord = None
    return newConjunctionWord
    assert 1==1
def Advcl_Type1_Create_Sentences(BaseClause, newConjunctionWord, SecondClause):
    #although words
    #replaced with but
    FirstSentence = BaseClause
    SecondSentence = newConjunctionWord + SecondClause 
    return (FirstSentence,SecondSentence)
def Advcl_Type2_Create_Sentences_Keep_Order(BaseClause, newConjunctionWord, SecondClause):
    FirstSentence = BaseClause
    SecondSentence = newConjunctionWord + SecondClause
    return (FirstSentence, SecondSentence)
def Advcl_Type2_Create_Sentences_Swap_Order(BaseClause, newConjunctionWord, SecondClause):
    #sometimes sentences can become ambiguous from swapping order
    #I like apples because he hates them
    #He hates them. So i like apples. -> does not make sense
    FirstSentence = SecondClause
    SecondSentence =  [newConjunctionWord] + BaseClause
    return (FirstSentence, SecondSentence)
def Advcl_Type3_Create_Sentences(spaciedSentence, BaseClause, newConjunctionWord, SecondClause):
    #For conjunctions like "before, after, once, since, when"
    #Insert "this + auxVerb + SecondClause"
    
    #auxVerb is the conjegation of "to be" -> except we ignore the plurality and the person -> we just take the tense of the verb
    #auxVerb can only be "was", "is", "will be?"
    tense = Determine_Tense_Advcl(spaciedSentence)
    auxVerb = Determine_AuxVerb_Advcl(tense)
    FirstSentence = BaseClause
    SecondSentence = "This" + auxVerb + SecondClause
    return (FirstSentence, SecondSentence)
def Determine_Tense_Advcl(spaciedSentence):
    #want the tense of the secondary clause verb
    #get the root
    #find the child marked advcl
    #get the tense
    root = [token for token in spaciedSentence if token.head == token][0]
    advcl = [child for child in root.children if child.dep_ == "advcl"][0]
    verb, tense = (advcl.text, advcl.tag_)
    return tense
def Determine_AuxVerb_Advcl(tense):
    if tense == "VB" or tense == "VBP" or tense == "VBZ": #present tense
        return "is"
    elif tense == "MD": #future tense
        return "will be"
    elif tense == "VBD": #past tense
        return "was"
    else: #don't know
        return "is"
#ccomp
def Contains_Conjoint_Clause_Ccomp(spaciedSentence):
    root = [token for token in spaciedSentence if token.head == token][0]
    ccomp = [child for child in root.children if child.dep_ == "ccomp"]
    if len(ccomp) > 0:
        return True
    return False
def Split_Conjoint_Clause_Ccomp(spaciedSentence):
    #secondary clause starts at the end of ccomp
    root = [token for token in spaciedSentence if token.head == token][0]
    RootSubtree = sorted(list(root.subtree), key=lambda x: x.i)
    ccomp = [child for child in root.children if child.dep_ == "ccomp"][0]
    CcompSubtree = sorted(list(ccomp.subtree), key=lambda x: x.i)
    #check if there is a mark/advmod at the start of the second clause -> delete if there is
    connectionWord = [child for child in ccomp.children if (child.dep_ == "mark" or child.dep_ == "advmod")]
    PureRootClause = []
    #may need to do some work on removing connection words tagged "mark" or "advmod"
    PureRootClause = []
    i = 0
    while i < len(RootSubtree):
        # Check if a sublist in list1 is equal to list2
        if RootSubtree[i:i + len(CcompSubtree)] == CcompSubtree:
            i += len(CcompSubtree)  # Skip the sublist in list1
        else:
            PureRootClause.append(RootSubtree[i])
            i += 1
    return PureRootClause, None, CcompSubtree
def Ccomp_Create_Sentences(BaseClause, connectionWord, SecondClause):
    return (BaseClause, SecondClause) 
def Correct_Sentence_Order(BaseClause, SecondClause):
    #Have the sentences appear in the order they are in
    #check which one has the earlier head
    if BaseClause[0].i > SecondClause[0].i:
        FirstSentence = SecondClause
        SecondSentence = BaseClause
    else:
        FirstSentence = BaseClause
        SecondSentence = SecondClause 
    return FirstSentence, SecondSentence
    # 
#addition functions
def Pure_Strings_List_From_Mixed_String_Token_List(list):
    return [item.text if isinstance(item, spacy.tokens.Token) else item for item in list]

