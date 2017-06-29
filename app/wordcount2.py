#!/usr/bin/python
# -- coding: utf-8 --
from app.words import Words
from app.result2 import Result


def wordcountTask(block):
    partialResult = {}
    for text in block.itervalues():
        parsedWords = text[0].split(',')
        for word in parsedWords:
            partialResult[word] = partialResult.get(word, 0) + 1
    return partialResult


def reduceTask(localResults, result):
    for word, instances in localResults.iteritems():
        a = result.instances[word]
        b = result.instances[word][0]
        result.instances[word] = int(result.instances[word][0][0]) + instances


if __name__ == "__main__":
    words = Words('wordcount.tengb')
    result = Result('wordcount.result')

    localResults = {}
    for ind, block in enumerate(words.wordinfo.split()):
        localResults[ind] = wordcountTask(block)
        reduceTask(localResults[ind], result)

