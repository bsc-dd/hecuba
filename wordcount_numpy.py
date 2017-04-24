#!/usr/bin/python
# -*- coding: utf-8 -*-
from app.domainnumpy import WordsMeta
import uuid
import numpy as np


def runme():

    words = WordsMeta("wordcountnumpy.wordsmeta")
    words2 = WordsMeta("wordsmeta")
    words.metabibl[0] = ('1', uuid.UUID('8ee8a6cf-24da-3622-ac13-07c3f368f259'))
    words2.metabibl[0] = ('1', uuid.UUID('969ce9db-ffef-36d6-8d8d-2845d28e588c'))
    words2.metabibl[1] = words.metabibl[0]
    words2.metabibl[2] = words.metabibl[0]
    words.metabibl[1] = words2.metabibl[2]
    for i in range(0, 10):
        valuein = np.random.rand(3, 3)
        words.metabibl[0][1].mybibl[i] = valuein
        words2.metabibl[0][1].mybibl[i] = words.metabibl[0][1].mybibl[i]
    ind = 0
    ind2 = 0
    '''
    for key, row in words2.metabibl.iteritems():
        mySO = words2.metabibl[ind][1]
        print "###########################################################"
        for _ in mySO.mybibl.itervalues():
            mySO.mybibl[ind2] = 'bla' + str(ind2 + 1)
            ind2 += 1
        ind += 1
    '''

runme()

