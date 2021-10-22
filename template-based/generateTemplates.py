import pandas as ps
import numpy as np
import spacy
import nltk
import csv

data = ps.read_csv("clickbait_data.csv")
data = data[data['clickbait'] == 1]
nlp = spacy.load("en_core_web_lg")

# nlp.add_pipe("merge_subtokens")

# nlp.add_pipe("merge_entities")

# nlp.add_pipe("merge_noun_chunks")


structures = dict()
wordList = dict()

for i, sent in enumerate(data.headline.tolist()):

    doc = nlp(sent)
    sentStruct = []
    for token in doc:
        sentStruct.append(token.pos_+"-"+token.tag_+"-"+token.dep_)
        if(token.text not in wordList.get(token.pos_+"-"+token.tag_+"-"+token.dep_, [])):
            wordList.setdefault(token.pos_+"-"+token.tag_ +
                                "-"+token.dep_, []).append(token.text)
    sentStruct = tuple(sentStruct)
    structures[sentStruct] = structures.get(sentStruct, 0) + 1


with open('./template-based/sentenceStructuresByTag.csv', 'w', newline='') as f:
    w = csv.writer(f)
    for s in structures:
        if(structures[s] > 5):
            w.writerow(s)

with open('template-based\wordListByTag.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "word-list"])
    for data in wordList:
        writer.writerow([data, ', '.join(wordList[data])])
