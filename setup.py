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
import sys
import copy
from distutils.core import setup, Extension, Command
import numpy as np

try:
    from Cython.Compiler.Main import compile, CompilationResultSet
    from Cython.Compiler.Options import parse_directive_list
except ImportError:
    cython = False
else:
    cython = True

ext_modules = []

# cython extension modules, where it is assumed that all '.c' files have a corresponding
# '.pyx' file from which it is compiled using cython
cython_modules = [Extension('echolect.filtering.libfilters',
                            sources=['echolect/filtering/libfilters.c'],
                            include_dirs=[np.get_include()],
                            extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                            extra_link_args=['-O3', '-ffast-math', '-fopenmp']),
                  Extension('echolect.filtering.libdopplerbanks',
                            sources=['echolect/filtering/libdopplerbanks.c'],
                            include_dirs=[np.get_include(), 'echolect/include'],
                            extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                            extra_link_args=['-O3', '-ffast-math', '-fopenmp'])]
# add C-files from cython modules to extension modules
ext_modules.extend(cython_modules)

cython_extensions = []
for c_mod in cython_modules:
    cython_ext = copy.copy(c_mod)
    cython_ext.sources = [root + '.pyx' for root, ext 
                          in map(os.path.splitext, c_mod.sources) 
                          if ext.lower() == '.c']
    
    cython_extensions.append(cython_ext)

cmdclass = dict()

if cython:
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
            results = CompilationResultSet()
            
            for cython_ext in cython_extensions:                
                res = compile(cython_ext.sources,
                              include_path=cython_ext.include_dirs,
                              verbose=True,
                              timestamps=self.timestamps,
                              annotate=self.annotate,
                              compiler_directives=self.directive)
                if res:
                    results.update(res)
                    results.num_errors += res.num_errors
            
            if results.num_errors > 0:
                sys.stderr.write('Cython compilation failed!')

    cmdclass['cython'] = CythonCommand

setup(name='echolect',
      version='0.1-dev',
      maintainer='Ryan Volz',
      maintainer_email='ryan.volz@gmail.com',
      url='http://sess.stanford.edu',
      description='Radar Data Processing Tools',
      long_description='',
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
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
      package_data={'echolect': ['include/*']},
      cmdclass=cmdclass,
      ext_modules=ext_modules)
