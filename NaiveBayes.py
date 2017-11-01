from os import listdir
from os.path import isfile, join
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import sys
import math
import  random as r
import logging
LOG_FILENAME = 'Accuracy.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

def subText(text):
    try:
        return text[text.index('lines:') + 10:]
    except ValueError:
        return text

def vocabList(fileList):
    uniqueVocabWords = set()
    for fileListEachClass in fileList:
        for filePaths in fileListEachClass:
          # read from file in lower case
          file = open(filePaths, 'r')
          data = file.read().lower()
          data = subText(data)
          replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
          words = word_tokenize(replaced_data)
          wordsFiltered = [w for w in words if w not in stopWords]
          for x in wordsFiltered:
              uniqueVocabWords.add(x)
    return uniqueVocabWords

def countDocs(fileList):
    count = 0;
    for fileListEachClass in fileList:
        count += len(fileListEachClass)
    return count

def concatAllDocsAndFindWordCountInTheClass(pathList):
    wordcount = Counter()
    for name in pathList:
        # read from file in lower case
        file = open(name, "r")

        data = file.read()
        data = subText(data)
        replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
        words = word_tokenize(replaced_data)
        wordsFiltered = [w for w in words if w not in stopWords]
        wordcount.update(wordsFiltered)
    return wordcount

def findConditionalProbabilityForVocab(className,wordCountInEachClass):
    wordsInClass=len(wordCountInEachClass)
    for word in vocabulary:
        if word in wordCountInEachClass.keys():
            prob_class = (wordCountInEachClass[word] + 1) / (wordsInClass + total_words)
        else:
            prob_class = (0 + 1) / (wordsInClass + total_words)


        if classNames.index(className) == 0:
            vocab_dict[word] = {className:prob_class }
        else:
            vocab_dict[word][className]=prob_class


def TrainMultinomialNB(classNames, fileList, vocabulary):

    documentCount = countDocs(fileList)

    for classIndex in range(len(classNames)):
        docCountOfTheClass = len(fileList[classIndex])
        prior[classIndex] = float(docCountOfTheClass)/float(documentCount)
        wordcounter[classIndex] = concatAllDocsAndFindWordCountInTheClass(fileList[classIndex])
        findConditionalProbabilityForVocab(classNames[classIndex],wordcounter[classIndex])

def applyMultinomialNB(classNames,vocab_dict,eachFilePath,prior):
    file = open(eachFilePath, "r")
    data = file.read()
    data = subText(data)
    replaced_data = re.sub('[^a-zA-Z\n]', ' ', data)
    words = word_tokenize(replaced_data)
    wordsFiltered = [w for w in words if w not in stopWords]
    score = [0 for x in range(len(classNames))]
    for eachClassIndex in range(len(classNames)):
       score[eachClassIndex] = math.log(prior[eachClassIndex])
       for eachword in wordsFiltered:
            if eachword in vocab_dict.keys():
               classvalue=classNames[eachClassIndex]
               score[eachClassIndex] += math.log(vocab_dict[eachword][classvalue])

    return classNames[score.index(max(score))]


pathForTrainingDataSet = sys.argv[1] + '/'
pathForTestingDataSet = sys.argv[2] +'/'
classNames = []
classWiseFileList = []
stopWords = set(stopwords.words('english'))
wordcounter={}

allTrainClassPaths = listdir(pathForTrainingDataSet)
randomFiveClass = r.sample(allTrainClassPaths,5)
for eachClass in randomFiveClass:
    #add class to the list
    classNames.append(eachClass)
    #add file names to the list and append it to the classWiseFileList
    pathForFiles = pathForTrainingDataSet + eachClass + '/'
    onlyfiles = [f for f in listdir(pathForFiles) if isfile(join(pathForFiles, f))]
    filePathList= [pathForFiles+x for x in onlyfiles]
    classWiseFileList.append(filePathList)

prior = [-1 for x in range(len(classNames))]
vocabulary = vocabList(classWiseFileList)
total_words=len(vocabulary)
vocab_dict={}
TrainMultinomialNB(classNames,classWiseFileList,vocabulary)



#********TESTING*******

testClassNames = []
testclassWiseFileList = []
for eachClass in classNames:
    #add class to the list
    testClassNames.append(eachClass)
    #add file names to the list and append it to the classWiseFileList
    pathForFiles = pathForTestingDataSet + eachClass + '/'
    onlyfiles = [f for f in listdir(pathForFiles) if isfile(join(pathForFiles, f))]
    filePathList = [pathForFiles+x for x in onlyfiles]
    testclassWiseFileList.append(filePathList)

correctCounter=0
totalTestingFiles = countDocs(testclassWiseFileList)
for eachClassIndex in range(len(testClassNames)):
    for eachfile in testclassWiseFileList[eachClassIndex]:
        predictClass = applyMultinomialNB(testClassNames,vocab_dict,eachfile,prior)
        if(predictClass == testClassNames[eachClassIndex]):
            correctCounter+=1

accuracy = float(correctCounter)/float(totalTestingFiles)
print("Accuracy: " + str(round(accuracy*100,2))+'%')
logging.info( '\n \t Class Names: ' + str(classNames) + ' , \n \t Accuracy : ' + str(round(accuracy*100,2))+'%')