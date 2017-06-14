#!/usr/bin/env python
from __future__ import print_function

import subprocess

from setuptools import setup, find_packages
import setuptools.command.build_py
import os
import sys


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if '.so' in filename:
                paths.append(os.path.join(path, filename))
    return paths


def cmake_build():
    if subprocess.call(["cmake", "-H./hfetch",  "-B./hfetch/build"]) != 0:
        raise EnvironmentError("error calling cmake")

    if subprocess.call(["make","-j4","-C","./hfetch/build"]) != 0:
        raise EnvironmentError("error calling make")



def setup_packages():

    # We first build C++ libraries
    if 'build' in sys.argv:
        cmake_build()

    extra_files = package_files('./hfetch/_install/lib') + package_files('./hfetch/_install/lib64')

    #TODO use some flag to detect that build has already been done instead of this
    if 'install' in sys.argv:
        cmake_build()
        extra_files = package_files('./hfetch/_install/lib') + package_files('./hfetch/_install/lib64')

    # compute which libraries were built

    metadata = dict(  name="Hecuba",
        version="0.1",

        packages=['hecuba', 'hecuba.qthrift', 'storage'], #find_packages(),

        #install_requires=['nose', 'cassandra-driver', 'mock'],
        zip_safe=False,
        data_files=[('.',extra_files)],
        include_package_data=True,

        # metadata for upload to PyPI
        license="PSF",
        keywords="hello world example examples",
        description='Hecuba',
        author='Guillem Alomar,Cesare Cugnasco, Pol Santamaria, Yolanda Becerra ',
        author_email='{guillem.alomar,cesare.cugnasco,pol.santamaria,yolanda.becerra}@bsc.es',
        url='https://www.bsc.es',
        long_description='''Hecuba.''',
     #   cmdclass={              'build_py' : BuildWithCmake,              }
    )

    setup(**metadata)
    return



if __name__ == '__main__':
    setup_packages()
