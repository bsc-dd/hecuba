#!/usr/bin/python
# -*- coding: utf-8 -*-

from app.words import Words


def runme():

    words = Words('wordcount.wordobj')
    totals = 0
    for words in words.words.itervalues():
        parsedWords = words.split(',')
        for word in parsedWords:
            totals += 1

    print totals

#if __name__ == "__main__":
runme()
 #   from pycompss.api.api import compss_wait_on

