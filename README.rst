===========
pymento-meg
===========


.. image:: https://img.shields.io/pypi/v/pymento-meg.svg
        :target: https://pypi.org/project/pymento-meg/


This Python package contains various modules used to analyze "Memento" MEG data, a non-public dataset acquired by Kaiser et al. 2017.
The code is often tuned to this specific dataset, for example with preprocessing parameters tuned to subjects specifically, or data selections meant for specific experimental trial conditions or duration.
Most modules will not be out-of-the-box applicable to other dataset.
The modules and functions in this package were used for data analyses reported in Chapter 4 of Wagner (2023): "Reconptualizing neural function as high-dimensional brain-state dynamics", Inaugural Dissertation, Heinrich-Heine-Universität Düsseldorf.


In addition to individually importable functions, a range of functionality is available via command-line entrypoint as follows:

* ``pymento-restructure``: Restructures raw, original memento data into a raw MEG BIDS directory (MEG-BIDS version 1.4). See ``pymento-restructure --help`` for required and possible command parameters. Please note that this code is tuned to work with original memento data, which had an idiosyncratic file structure.
* ``pymento-sss``: Based on a raw BIDS directory, create and save SSS processed data BIDS-like. This function performs a temporal-spatial signal space separation including motion correction when provided with a subject identifier, a path to a raw BIDS directory and a path to where derivatives shall be saved. See ``pymento-sss --help`` for required and possible command parameters. Although untested, this function should generalize to other datasets, but it should contain the Elekta-Neuromag specific calibration and cross-talk files.
* ``pymento-epochandclean``: Epochs BIDS-compliant MEG data after a FastICA to remove artifacts. Using the autoreject algorithm, it rejects bad trials. See ``pymento-epochandclean --help`` for required and possible command parameters. The ``--event`` parameter is tuned for the event descriptions of the memento dataset specifically, and would need manual tuning in the source code to other events.
* ``pymento-srm``: Fits a shared response model on (parts) of the MEG data. Timespan selections are tuned to the memento dataset. See ``pymento-srm --help`` for required and possible command parameters.

The following module overview provides a summary of implemented functionality:

* ``pymento_meg/decoding`` contains functions for temporal decoding and temporal generalization analyses rooted in multivariate and machine-learning methodology.
* ``pymento_meg/orig`` contains functions for reading and restructuring original memento data and log files.
* ``pymento_meg/proc`` contains functions for preprocessing raw MEG data.
* ``pymento_meg/srm`` contains functions for applying shared response models to MEG data.

In addition, ``config.py`` contains dataset specific configurations for the memento dataset, e.g., pertaining to the structure of its logfiles, file naming, and event descriptions.

Installation
------------

The package is available on PyPi and can be installed using ``pip``::

    pip install pymento_meg

Credits
-------

This package contains the code created during a PhD project at the Heinrich Heine University Düsseldorf.
It was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
