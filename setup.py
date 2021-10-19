#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import os
import subprocess
import sys
import glob
import numpy

c_binding_path=None

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if '.so' in filename:
                paths.append(os.path.join(path, filename))
    return paths


def cmake_build():
    global c_binding_path
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
            cmake_args=cmake_args + [ "-DPYTHON_VERSION=python{}.{}".format(sys.version_info.major, sys.version_info.minor)]
        if subprocess.call(cmake_args) != 0:
            raise EnvironmentError("error calling cmake")
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            # CMake not found error.
            raise OSError(os.errno.ENOENT, os.strerror(os.errno.ENOENT), 'cmake')
        else:
            # Different error
            raise e

    jobs = get_var("CMAKE_BUILD_PARALLEL_LEVEL")
    jobs = "-j{}".format(jobs[0] if jobs else 1)
    print("JOBS={}".format(jobs), flush=True)
    if subprocess.call(["make", jobs, "-C", "./build"]) != 0:
        raise EnvironmentError("error calling make build")

    if c_binding_path and subprocess.call(["make", jobs, "-C", "./build", "install"]) != 0:
        raise EnvironmentError("error calling make install")

def get_var(var):
    value = os.environ.get(var,'')
    return [p for p in value.split(':') if p != '']




def setup_packages():
    # We first build C++ libraries
    if 'build' in sys.argv:
        cmake_build()

    # TODO use some flag to detect that build has already been done instead of this
    if 'install' in sys.argv:
        cmake_build()

    extra_files = package_files('build/lib') + package_files('build/lib64')

    PATH_LIBS = get_var('LD_LIBRARY_PATH')
    PATH_INCLUDE = get_var('CPATH') + get_var('CPLUS_INCLUDE_PATH') + get_var('C_INCLUDE_PATH')
    if c_binding_path:
        PATH_INCLUDE += [c_binding_path + "/include"]
        PATH_LIBS += [c_binding_path + "/lib"]
        extra_files = extra_files + package_files(c_binding_path + "/lib")

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

    # compute which libraries were built
    metadata = dict(name="Hecuba",
                    version="1.0",
                    package_dir={'hecuba': 'hecuba_py/hecuba', 'storage': 'storageAPI/storage'},
                    packages=['hecuba', 'storage'],  # find_packages(),
                    install_requires=['cassandra-driver>=3.7.1', 'numpy>=1.16'],
                    zip_safe=False,
                    data_files=[('', extra_files)],

                    # metadata for upload to PyPI
                    license="Apache License Version 2.0",
                    keywords="key-value, scientific computing",
                    description='Hecuba',
                    author='Guillem Alomar, Yolanda Becerra, Juanjo Costa, Cesare Cugnasco, Adrián Espejo, Pol Santamaria',
                    author_email='yolanda.becerra@bsc.es,jcosta@ac.upc.edu',
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
