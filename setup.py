#!/usr/bin/env python
from __future__ import print_function

#use_cython = False

import os
from distutils.core import setup, Extension
from distutils.command.build import build as _build
import subprocess

home_path =str(os.environ['HOME'])
c11_flag = '-std=c++11'


def cmake_build():
    if subprocess.call(["cmake", "-H./hfetch",  "-B./hfetch/build"]) != 0:
        raise EnvironmentError("error calling cmake")

    if subprocess.call(["make","-C","./hfetch/build"]) != 0:
        raise EnvironmentError("error calling make")

class BuildWithCmake(_build):
    def run(self):
        cmake_build()
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)


setup(name='Hecuba',
      version='1.0',
      description='Cache and prefetch for Hecuba',
      author='Guillem Alomar,Cesare Cugnasco, Pol Santamaria, Yolanda Becerra ',
      author_email='{guillem.alomar,cesare.cugnasco,pol.santamaria,yolanda.becerra}@bsc.es',
      url='https://www.bsc.es',
      install_requires=['nose', 'cassandra-driver', 'mock'],
      packages=['hecuba', 'storage'],
      data_files=[('lib/python2.7/site-packages/', ['hfetch/build/hfetch.so'])],
      long_description='''Cache and prefetch for Hecuba.''',
      cmdclass={
          'build' : BuildWithCmake,
          }
      )
