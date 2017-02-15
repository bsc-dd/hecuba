#!/usr/bin/env python
from __future__ import print_function
from distutils.core import setup, Extension

import os


from distutils.core import setup
from distutils.command.build import build as _build
import subprocess

home_path =str(os.environ['HOME'])
c11_flag = '-std=c++11'

#if __cplusplus <= 199711L
  #error This library needs at least a C++11 compliant compiler
#endif


def cmake_build():
    if subprocess.call(["cmake", "."]) != 0:
        raise EnvironmentError("error calling cmake")

    if subprocess.call("make") != 0:
        raise EnvironmentError("error calling make")

class BuildWithCmake(_build):
    def run(self):
        cmake_build()
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)


hfetch_module = Extension('hfetch',
    define_macros = [('MAJOR_VERSION', '2'), ('MINOR_VERSION', '0')],
    include_dirs = ['/usr/local/include','/usr/include',home_path+'/local/include','./_install/lib'],
    libraries = ['cassandra','PocoFoundation','tbb'],
    library_dirs = ['/lib','/usr/local/lib','/usr/lib',home_path+'/local/lib','./_install/lib'],
    extra_compile_args=[c11_flag],
    headers =['HCache.h', 'CacheTable.h', 'Prefetch.h', 'TupleRow.h', 'TupleRowFactory.h'],
    sources = ['HCache.cpp', 'CacheTable.cpp','Prefetch.cpp', 'TupleRow.cpp', 'TupleRowFactory.cpp'],

)


setup (name = 'Hecuba',
    version = '1.0',
    description = 'Cache and prefetch for Hecuba',
    author = 'Guillem Alomar,Cesare Cugnasco, Pol Santamaria, Yolanda Becerra ',
    author_email = '{guillem.alomar,cesare.cugnasco,pol.santamaria,yolanda.becerra}@bsc.es',
    url = 'https://www.bsc.es',
    install_requires = ['nose', 'cassandra-driver', 'mock'],
    #packages = ['hecuba', 'api'],
    long_description = '''Cache and prefetch for Hecuba.''',
    ext_modules = [hfetch_module],
    cmdclass={
          'build' : BuildWithCmake,
          }
)
