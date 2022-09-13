import pandas as ps
import numpy as np
import random
from gensim.models.word2vec import Word2Vec

nSentences = 5
useWordEmbedding = True
model = Word2Vec.load("word2vec.model")


sentStructures = ps.read_csv(
    "./template-based/sentenceStructuresByTag.csv", header=None, sep='/t', engine="python")
wordList = ps.read_csv("./template-based/wordListByTag.csv")


def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]


def generateSentence():
    structure = sentStructures.sample().values
    for val in structure[0]:
        tags = val.split(",")
        firstWordList = wordList[wordList['tag'] ==
                                 tags[0]]['word-list'].values[0].split(", ")
        sentence = []
        sentence.append(random.choice(firstWordList))
        for i in range(1, len(tags)):
            words = wordList[wordList['tag'] ==
                             tags[i]]['word-list'].values[0].split(", ")
            if useWordEmbedding:
                similarityScores = dict()

                for word in words:
                    if word in model.wv:
                        score = model.wv.similarity(sentence[i-1], word)
                        if score != 1:
                            similarityScores.setdefault(word, score)

                orderedList = sorted(
                    similarityScores, key=similarityScores.get, reverse=True)

                sentence.append(random.choice(orderedList[0:5]))
            else:
                sentence.append(random.choice(words))
    return ' '.join(sentence)


for i in range(0, nSentences):
    sent = generateSentence()
    print(sent)
