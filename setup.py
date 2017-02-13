#!/usr/bin/env python

from distutils.core import setup

setup(name='Hecuba',
      version='1.0',
      description='Cache and prefetch for Hecuba',
      author='Guillem Alomar,Cesare Cugnasco, Pol Santamaria, Yolanda Becerra ',
      author_email='{guillem.alomar,cesare.cugnasco,pol.santamaria,yolanda.becerra}@bsc.es',
      url='https://www.bsc.es',
      install_requires=['nose', 'cassandra-driver', 'mock'],
      packages=['hecuba', 'storage', 'hfetch'],
      long_description='''Cache and prefetch for Hecuba.'''
      )
