#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import os
import errno
import subprocess
import sys
import glob
import numpy
import shutil

c_binding_path=None

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if '.so' in filename:
                paths.append(os.path.join(path, filename))
    return paths

def get_c_binding():
    import re
    c_bind_re = re.compile("--c_binding=*")
    try:
        c_binding_path_opt = next(arg for arg in sys.argv if (re.match("--c_binding=*", arg)))
        c_binding_path = re.search("--c_binding=(.*)", c_binding_path_opt).groups()[0]
        sys.argv.remove(c_binding_path_opt)
    except (IndexError, StopIteration):
        c_binding_path=None
    return c_binding_path

def cmake_build():
    global c_binding_path

    jobs = get_var("CMAKE_BUILD_PARALLEL_LEVEL")
    jobs = "-j{}".format(jobs[0] if jobs else 1)
    print("JOBS={}".format(jobs), flush=True)
    try:
        cmake_args=["cmake", "-H./hecuba_core", "-B./build"]
        #cmake_args=cmake_args + ["-DUSE_ARROW=TRUE"]
        #cmake_args=cmake_args + ["-DUSE_EXTRAE=TRUE"]
        if c_binding_path:
            cmake_args=cmake_args + [ "-DC_BINDING_INSTALL_PREFIX={}".format(c_binding_path)]
            cmake_args=cmake_args + [ "-DPYTHON_VERSION=python{}.{}".format(sys.version_info.major, sys.version_info.minor)]
        if subprocess.call(cmake_args) != 0:
            raise EnvironmentError("error calling cmake")
    except OSError as e:
        if e.errno == errno.ENOENT:
            # CMake not found error.
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), 'cmake')
        else:
            # Different error
            raise e

    if subprocess.call(["make", jobs, "-C", "./build"]) != 0:
        raise EnvironmentError("error calling make build")

    if subprocess.call(["make", jobs, "-C", "./build", "install"]) != 0:
        raise EnvironmentError("error calling make install")


def copy_files_to_dir(extra_files, destination_folder):
    os.makedirs(destination_folder, 0o755, exist_ok=True)
    # fetch all files
    for file_name in extra_files:
        # construct full file path
        destination = os.path.join(destination_folder , os.path.basename(file_name))
        # copy only files
        if os.path.isfile(file_name):
            filetooverwrite = os.path.join(destination_folder, os.path.basename(file_name))
            if  os.path.isfile(filetooverwrite):
                os.remove(filetooverwrite)
            shutil.copy(file_name, destination, follow_symlinks=False)
            print('copied {} to {}'.format(file_name, destination_folder))
        else:
            copy_files_to_dir(glob.glob('{}/*'.format(file_name)), destination)


def get_var(var):
    value = os.environ.get(var,'')
    return [p for p in value.split(':') if p != '']

def install_requirements():
    # https://stackoverflow.com/questions/12332975/how-can-i-install-a-python-module-within-code
    import subprocess
    import sys

    print("HECUBA REQUIREMENTS INSTALLATION")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("================================")

def do_build_process():
    ## avoid the compilation again in the isolated virtual environment (as it is already compiled)

    install_requirements()

    cmake_build()

    ## Copy 'jar' and 'arrow_helper' INSIDE package (hecuba and storage) so it gets included in the wheel
    copy_files_to_dir(['storageAPI/storageItf/target/StorageItf-1.0-jar-with-dependencies.jar'], "storageAPI/storage/ITF")
    copy_files_to_dir(glob.glob('build/bin/*'), "hecuba_py/hecuba/bin")
    ## Emulate lib and include hierarchy inside the python package to be compatible with C++ installation
    copy_files_to_dir(glob.glob('build/include/hecuba/*'), "hecuba_py/hecuba/include")
    copy_files_to_dir(glob.glob('includedep/*'), "hecuba_py/hecuba/include") # TODO: includedep contains header requeriments (manually created) automate the generation of this directory
    copy_files_to_dir(['build/lib/libhfetch.so'], "hecuba_py/hecuba/lib")
    ## Copy VERSION.txt to 'hecuba_py/hecuba' so it gets included in the python package
    copy_files_to_dir(['VERSION.txt'], 'hecuba_py/hecuba' )


def setup_packages():
    global c_binding_path

    c_binding_path = get_c_binding()

    print ("DEBUG: sys.argv {}<".format(sys.argv),flush=True)
    if 'build' in sys.argv or 'egg_info' in sys.argv or 'bdist_wheel' in sys.argv:
        ## We first build C++ libraries
        ## 'egg_info' is the parameter used for 'python -m build' the first time is invoked and we need the libraries created
        ## 'bdist_wheel' is the parameter used for 'spack' the first time is invoked and we need the libraries created
        do_build_process()

    if 'install' in sys.argv:
        # Check if library is already built and build it otherwise
        libhfetch_path='build/lib/libhfetch.so'
        if not os.path.isfile(libhfetch_path):
            print(' == Library {}/{} is not present. Forcing build process!'.format(os.getcwd(), libhfetch_path), flush=True)
            do_build_process()

        # Get our own instance of Distribution
        from distutils.dist import Distribution

        dist = Distribution()
        dist.parse_config_files()
        dist.parse_command_line()

        # Get prefix from either config file or command line
        try:
            prefix = dist.get_option_dict('install')['prefix'][1]
            print('  == --prefix {} used. Copy header files and libraries inside.'.format(prefix), flush=True)
            copy_files_to_dir(glob.glob('build/include/*'), prefix + "/include")
            copy_files_to_dir(glob.glob('build/include/hecuba/*'), prefix + "/include/hecuba")
            copy_files_to_dir(glob.glob('build/lib/*'), prefix + "/lib")
            copy_files_to_dir(['VERSION.txt'], prefix) # put a copy of VERSION.txt in the installation directory for easy finding
        except (KeyError):
            pass



    extra_link_args=[]
    PATH_LIBS = get_var('LD_LIBRARY_PATH')
    PATH_INCLUDE = get_var('CPATH') + get_var('CPLUS_INCLUDE_PATH') + get_var('C_INCLUDE_PATH')
    if c_binding_path:
        PATH_INCLUDE += [c_binding_path + "/include"]
        PATH_LIBS += [c_binding_path + "/lib"]
        # If --c_binding is used, hardcode the RPATH into the Extension so it
        # will work in the installed machine no matter where libraries are
        # located (and wheel generation will solve the dependencies anyway)
        extra_link_args += [ '-Wl,-rpath='+c_binding_path+'/lib']

    print ("DEBUG: numpy.get_include=>{}<".format(numpy.get_include()),flush=True)
    extensions = [
        Extension(
            "hecuba.hfetch",
            sources=glob.glob("hecuba_core/src/py_interface/*.cpp"),
            include_dirs=['hecuba_core/src/', 'build/include', numpy.get_include()] + PATH_INCLUDE,
            libraries=['hfetch'],
            library_dirs=['build/lib'] + PATH_LIBS,
            extra_compile_args=['-std=c++11'],
            extra_link_args=['-Wl,-rpath='+os.getcwd()+'/build/lib'] + extra_link_args,
        ),
    ]

    # compute which libraries were built
    metadata = dict(name="Hecuba",
                    version="2.1.3",
                    package_dir={'hecuba': 'hecuba_py/hecuba', 'storage': 'storageAPI/storage'},
                    packages=['hecuba', 'storage'],  # find_packages(),
                    install_requires=['cassandra-driver>=3.7.1', 'numpy>=1.16'],
                    zip_safe=False,
                    include_package_data=True, # REQUIRED
                    package_data={ # REQUIRED
                        "storage.ITF" : glob.glob("storageAPI/storage/ITF/*.jar"),
                        "hecuba.bin"  : glob.glob("hecuba_py/hecuba/bin/*"),
                        "hecuba.include"  : glob.glob("hecuba_py/hecuba/include/*.h"),
                        "hecuba.lib"  : glob.glob("hecuba_py/hecuba/lib/*"),
                                  },

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
