#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.core import setup

config = {'name': 'hecuba',
          'version': '0.1',
          'author': 'Guillem Alomar',
          'author_email': 'guillem.alomar@bsc.es',
          'url': 'https://www.bsc.es',
          'install_requires': ['nose', 'cassandra-driver', 'mock'],
          'packages': ['hecuba']
          }

setup(**config)
