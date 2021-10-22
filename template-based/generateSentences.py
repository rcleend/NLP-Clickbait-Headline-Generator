import pandas as ps
import numpy as np
nSentences = 10


sentStructures = ps.read_csv(
    "./template-based/sentenceStructuresByTag.csv", header=None, sep='/t')
wordList = ps.read_csv("./template-based/wordListByTag.csv")


def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]


def generateSentence():
    structure = sentStructures.sample()
    wordsInSentence = []
    for val in structure.values:
        tags = val[0].split(",")
        for tag in tags:
            words = wordList[wordList['tag'] == tag]['word-list']
            word = np.random.choice(words.values[0].split(), 1)
            wordsInSentence.append(word[0])
    sentence = ""
    for w in wordsInSentence:
        sentence += w+" "
    return sentence


for i in range(0, nSentences):
    sent = generateSentence()
    print(sent)
