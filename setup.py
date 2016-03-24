import setuptools
import tia


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def read_file(path):
    with open(path, 'r') as f:
        return f.read()


AUTHOR = 'Brian P Smith'
AUTHOR_EMAIL = 'brian.p.smith@gmail.com'
PACKAGE = 'tia'
PACKAGE_DESC = 'Toolkit for integration and analysis'
VERSION = tia.__version__
URL = "https://github.com/bpsmith/tia"
REQUIRED = ['pandas', 'numpy']
REQUIRED_FOR_TESTS = []

LONG_DESC = """\
TIA is a toolkit that provides bloomberg data access, easier pdf generation, backtesting functionality,
technical analysis functionality, and return analysis.
"""

setuptools.setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=PACKAGE_DESC,
    include_package_data=True,
    install_requires=REQUIRED,
    long_description=LONG_DESC,
    name=PACKAGE,
    packages=['tia', 'tia.analysis', 'tia.bbg', 'tia.rlab', 'tia.tests',
        'tia.util', 'tia.analysis.model'],
    package_dir={'tia': 'tia'},
    test_suite='tia.tests',
    tests_require=REQUIRED_FOR_TESTS,
    url=URL,
    version=VERSION,
    keywords=['bloomberg', 'backtesting', 'technical analysis', 'pdf'],
    download_url=URL,
    license='BSD (3-clause)',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Utilities',
    ],
)