#!/usr/bin/python
# -*- coding: utf-8 -*-

from app.words import Words


def runme():

    words = Words()#'wordcount.wordobj')
    words.words.make_persistent("wordcount.words")
    totals = 0
    for key, row in words.words.iteritems():
        if row is None:
            print 'key', key, 'is none'
        else:
            parsedWords = row.split(',')
            for word in parsedWords:
                totals += 1

    print totals

#if __name__ == "__main__":
runme()
 #   from pycompss.api.api import compss_wait_on

