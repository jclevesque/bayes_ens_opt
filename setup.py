# MIT License

# Copyright (c) 2017 Julien-Charles Lévesque

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import setuptools
from setuptools.command.build_py import build_py

ext_modules = [setuptools.Extension("beo.ens_cfuncs",
  ["beo/ensemble_cfuncs.c"])
]

setuptools.setup(
    name='beo',
    description="Bayesian Ensemble Optimization",
    version='0.1.0',
    packages=setuptools.find_packages(exclude=['examples', 'unit_tests',
        'scripts']),
    install_requires=['numpy',
                      'scipy',
                      'scikit-learn',
                      'psutil'
                      ],
    entry_points={
        'console_scripts': [
        ]
    },
    scripts=[
    ],
    ext_modules=ext_modules,
    cmdclass = {"build_py": build_py},
    include_package_data=False,
    author="Julien-Charles Lévesque",
    author_email="levesque.jc@gmail.com",
    license='MIT'
    )
