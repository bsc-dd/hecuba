#!/usr/bin/python
# -*- coding: utf-8 -*-
#from pycompss.api.task import task
from app.words import Words
from app.result import Result


#@task(returns = dict)
def wordcountTask(block, words):
    from storage.api import start_task, end_task
    start_task([block, words])
    partialResult = {}
    for words in block.words.itervalues():
        parsedWords = words.split(',')
        for word in parsedWords:
            partialResult[word] = partialResult.get(word, 0) + 1
    end_task([block, words])
    return partialResult

#@task()
def reduceTask(localResults, result):
    total = 0
    for word, instances in localResults.iteritems():
        total += instances
    return total


def runme():

    words = Words('wordcount.wordobj')
    result = Result()
    result.make_persistent('result')
    result.delete_persistent()
    localResults = {}
    totals = []
    for ind, block in enumerate(words.split()):
        localResults[ind] = wordcountTask(block, words)
        totals.append(reduceTask(localResults[ind], result))
    print totals
    print "TOTAL",reduce(lambda a, b: a+b, totals)

#if __name__ == "__main__":
runme()
 #   from pycompss.api.api import compss_wait_on

