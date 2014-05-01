#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import os
import copy
from distutils.core import setup, Extension, Command
from distutils.util import get_platform
import numpy as np

import version

here = os.path.abspath(os.path.dirname(__file__))

def get_version(*file_paths):
    try:
        # read version from git tags
        ver = version.read_version_git()
    except:
        # read version from file
        ver = version.read_version_file(here, *file_paths)
    else:
        # write version to file if we got it successfully from git
        version.write_version_file(ver, here, *file_paths)
    return ver

try:
    from Cython.Build import cythonize
    from Cython.Compiler.Options import parse_directive_list
except ImportError:
    HAS_CYTHON = False
else:
    HAS_CYTHON = True

def no_cythonize(extensions, **_ignore):
    dupextensions = copy.deepcopy(extensions)
    for extension in dupextensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return dupextensions

# regular extension modules
ext_modules = []

# cython extension modules
cython_include_path = [] # include for cimport, different from compile include
ext_cython = [Extension('echolect.filtering.libfilters',
                        sources=['echolect/filtering/libfilters.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                        extra_link_args=['-O3', '-ffast-math', '-fopenmp']),
              Extension('echolect.filtering.libdopplerbanks',
                        sources=['echolect/filtering/libdopplerbanks.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                        extra_link_args=['-O3', '-ffast-math', '-fopenmp'])]
# add C-files from cython modules to extension modules
ext_modules.extend(no_cythonize(ext_cython))

# custom setup.py commands
cmdclass = dict()

if HAS_CYTHON:
    class CythonCommand(Command):
        """Distutils command to cythonize source files."""
        
        description = "compile Cython code to C code"
        
        user_options = [('annotate', 'a', 'Produce a colorized HTML version of the source.'),
                        ('directive=', 'X', 'Overrides a compiler directive.'),
                        ('timestamps', 't', 'Only compile newer source files.')]
        
        def initialize_options(self):
            self.annotate = False
            self.directive = ''
            self.timestamps = False
        
        def finalize_options(self):
            self.directive = parse_directive_list(self.directive)
        
        def run(self):
            cythonize(ext_cython,
                      include_path=cython_include_path,
                      force=(not self.timestamps),
                      annotate=self.annotate,
                      compiler_directives=self.directive)

    cmdclass['cython'] = CythonCommand

setup(name='echolect',
      version=get_version('echolect', '_version.py'),
      maintainer='Ryan Volz',
      maintainer_email='ryan.volz@gmail.com',
      url='http://github.com/ryanvolz/echolect',
      description='Radar Data Processing Tools',
      long_description='',
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Cython',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Topic :: Scientific/Engineering'],
      packages=['echolect',
                'echolect.clustering',
                'echolect.core',
                'echolect.estimators',
                'echolect.filtering',
                'echolect.jicamarca',
                'echolect.millstone',
                'echolect.sim',
                'echolect.tools'],
      cmdclass=cmdclass,
      ext_modules=ext_modules)
