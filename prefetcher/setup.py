#!/usr/bin/env python
from distutils.core import setup, Extension

import os

home_path =str(os.environ['HOME'])
c11_flag = '-std=c++11'

#if __cplusplus <= 199711L
  #error This library needs at least a C++11 compliant compiler
#endif


hfetch_module = Extension('hfetch',
    'version': '0.1',
    3
    define_macros = [('MAJOR_VERSION', '1'), ('MINOR_VERSION', '0')],
    include_dirs = ['/usr/local/include','/usr/include',home_path+'/local/include'],
    libraries = ['cassandra','PocoFoundation','tbb'],
    library_dirs = ['/lib','/usr/local/lib','/usr/lib',home_path+'/local/lib'],
    extra_compile_args=[c11_flag],
    headers =['prefetcher/HCache.h', 'prefetcher/CacheTable.h', 'prefetcher/Prefetch.h', 'prefetcher/TupleRow.h', 'prefetcher/TupleRowFactory.h'],
    sources = ['prefetcher/HCache.cpp', 'prefetcher/CacheTable.cpp','prefetcher/Prefetch.cpp', 'prefetcher/TupleRow.cpp', 'prefetcher/TupleRowFactory.cpp']
)


setup (name = 'Hecuba',
    version = '1.0',
    description = 'Cache and prefetch for Hecuba',
    author = 'Guillem Alomar,Cesare Cugnasco, Pol Santamaria, Yolanda Becerra ',
    author_email = '{guillem.alomar,cesare.cugnasco,pol.santamaria,yolanda.becerra}@bsc.es',
    url = 'https://www.bsc.es',
    install_requires = ['nose', 'cassandra-driver', 'mock'],
    packages = ['hecuba', 'api'],
    long_description = '''Cache and prefetch for Hecuba.''',
    ext_modules = [hfetch_module]
)
