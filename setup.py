#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['mne', 'mne-bids', 'matplotlib']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Adina Wagner",
    author_email='adina.wagner@t-online.de',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Python package to analyze Memento MEG data",
    entry_points={
        'console_scripts': [
            'pymento=pymento_meg.cli:main',
            'pymento-restructure=pymento_meg.cli:restructure',
            'pymento-sss=pymento_meg.cli:sss',
            'pymento-epochandclean=pymento_meg.cli:epoch_and_clean',
            'pymento-srm=pymento_meg.cli:srm',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pymento',
    name='pymento_meg',
    packages=find_packages(include=['pymento_meg', 'pymento_meg.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/adswa/pymento',
    version='0.0.12',
    zip_safe=False,
)
