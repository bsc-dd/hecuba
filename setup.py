#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import os
import subprocess
import sys
import glob
import numpy


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if '.so' in filename:
                paths.append(os.path.join(path, filename))
    return paths


def cmake_build():
    try:
        if subprocess.call(["cmake", "-H./hecuba_core", "-B./build"]) != 0:
            raise EnvironmentError("error calling cmake")
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            # CMake not found error.
            raise OSError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), 'cmake')
        else:
            # Different error
            raise e

    if subprocess.call(["make", "-j4", "-C", "./build"]) != 0:
        raise EnvironmentError("error calling make build")


#    if subprocess.call(["make", "-j4", "-C", "./build", "install"]) != 0:
#        raise EnvironmentError("error calling make install")


extensions = [
    Extension(
        "hfetch",
        sources=glob.glob("hecuba_core/src/py_interface/*.cpp"),
        include_dirs=['hecuba_core/src/', 'build/include', numpy.get_include()],
        libraries=['hfetch', 'cassandra'],
        library_dirs=['build/lib', 'build/lib64'],
        extra_link_args=['-Wl,-rpath=$ORIGIN']
    ),
]


def setup_packages():
    # We first build C++ libraries
    if 'build' in sys.argv:
        cmake_build()

    extra_files = package_files('build/lib') + package_files('build/lib64')

    # TODO use some flag to detect that build has already been done instead of this
    if 'install' in sys.argv:
        cmake_build()
        extra_files = package_files('build/lib') + package_files('build/lib64')

    # compute which libraries were built
    metadata = dict(name="Hecuba",
                    version="0.1.3",
                    package_dir={'hecuba': 'hecuba_py/hecuba', 'storage': 'storage', 'pycompss_api': 'storage'},
                    packages=['hecuba', 'storage'] + find_packages(),  # find_packages(),
                    install_requires=['cassandra-driver>=3.7.1', 'numpy>=1.16'],
                    zip_safe=False,
                    data_files=[('', extra_files)],

                    # metadata for upload to PyPI
                    license="Apache License Version 2.0",
                    keywords="key-value, scientific computing",
                    description='Hecuba',
                    author='Guillem Alomar, Yolanda Becerra, Cesare Cugnasco, Pol Santamaria',
                    author_email='yolanda.becerra@bsc.es,cesare.cugnasco@bsc.es,pol.santamaria@bsc.es',
                    url='https://www.bsc.es',
                    long_description='''Hecuba.''',
                    #   test_suite='nose.collector',
                    #    tests_require=['nose'],
                    ext_modules=extensions

                    )

    setup(**metadata)
    return


if __name__ == '__main__':
    setup_packages()
