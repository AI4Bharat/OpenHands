Datasets
========

This section describes the list of all supported pose-based ISLR datasets available for training, as well as instructions on how to add support for your own dataset for training.

Supported Datasets
------------------

The following pose datasets are available out-of-the-box.

.. csv-table::
   :file: ../_static/datasets.csv
   :header-rows: 1

Usage
-----

To use one of the above existing datasets, follow the steps below:

- Download the zip from above for the required sign langauge.
   - Extract it to any desired folder.
- Mention the dataset class and path to the extracted dataset in the config.
   - For example configs, `click here <https://github.com/AI4Bharat/OpenHands/tree/main/examples>`_.
   - Feel free to change any other parameters pertaining to the dataset usage in the config.

You can now directly proceed to train!

Custom Datasets
---------------

To add support for your own dataset, follow the steps below:

