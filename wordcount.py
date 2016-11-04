#!/usr/bin/python
# -*- coding: utf-8 -*-
#from pycompss.api.task import task
from app.words import Words
from app.result import Result


#@task(returns = dict)
def wordcountTask(block, words):
    block.storageobj.init_prefetch(block)
    from storage.api import start_task, end_task
    start_task([block, words])
    partialResult = {}
    for words in block.itervalues():
        parsedWords = words.split(',')
        for word in parsedWords:
            partialResult[word] = partialResult.get(word, 0) + 1
    end_task([block, words])
    return partialResult

#@task()
def reduceTask(localResults, result):
    for word, instances in localResults.iteritems():
        result[word] = instances


def runme():

    words = Words('Words')
    result = Result()
    result.make_persistent('Result')
    result.empty_persistent()

    localResults = {}
    for ind, block in enumerate(words.split()):
        localResults[ind] = wordcountTask(block, words)
        reduceTask(localResults[ind], result)


#if __name__ == "__main__":
runme()
 #   from pycompss.api.api import compss_wait_on

