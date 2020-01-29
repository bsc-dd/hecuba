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
    import re
    c_bind_re = re.compile("--c_binding=*")
    try:
        c_binding_path_opt = next(arg for arg in sys.argv if (re.match("--c_binding=*", arg)))
        c_binding_path = re.search("--c_binding=(.*)", c_binding_path_opt).groups()[0]
        sys.argv.remove(c_binding_path_opt)
    except (IndexError, StopIteration):
        c_binding_path=None

    try:
        cmake_args=["cmake", "-H./hecuba_core", "-B./build"]
        if c_binding_path:
            cmake_args=cmake_args + [ "-DC_BINDING_INSTALL_PREFIX={}".format(c_binding_path)]
        if subprocess.call(cmake_args) != 0:
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

    if c_binding_path and subprocess.call(["make", "-j4", "-C", "./build", "install"]) != 0:
        raise EnvironmentError("error calling make install")

def get_var(var):
    value = os.environ.get(var,'')
    return [p for p in value.split(':') if p != '']


PATH_LIBS = get_var('LD_LIBRARY_PATH')
PATH_INCLUDE = get_var('CPATH') + get_var('CPLUS_INCLUDE_PATH') + get_var('C_INCLUDE_PATH')

extensions = [
    Extension(
        "hfetch",
        sources=glob.glob("hecuba_core/src/py_interface/*.cpp"),
        include_dirs=['hecuba_core/src/', 'build/include', numpy.get_include()] + PATH_INCLUDE,
        libraries=['hfetch', 'cassandra'],
        library_dirs=['build/lib', 'build/lib64'] + PATH_LIBS,
        extra_compile_args=['-std=c++11'],
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
                    package_dir={'hecuba': 'hecuba_py/hecuba', 'storage': 'storageAPI/storage'},
                    packages=['hecuba', 'storage'],  # find_packages(),
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
