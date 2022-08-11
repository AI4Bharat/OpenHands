ISLR Datasets
========

This section describes the list of all supported pose-based ISLR datasets available for training, as well as instructions on how to add support for your own dataset for training.

Supported Datasets
------------------

The following pose datasets are available out-of-the-box.

.. csv-table::
   :file: ../_static/islr_datasets.csv
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

To add support for your own dataset, create a class of the following structure:

.. code:: python

    from .base import BaseIsolatedDataset

    class MyDatasetDataset(BaseIsolatedDataset):
        def read_glosses(self):
            self.glosses = ... # Populate the list of all glosses

        def read_original_dataset(self):
            self.data = ... # Populate the list of all video files and gloss IDs as tuples

        def read_video_data(self, index):
            # Read the following ...
            return imgs, label, video_name

- For implementation examples, check `this folder in the source code <https://github.com/AI4Bharat/OpenHands/tree/main/openhands/datasets/isolated>`_
- This class can now be referenced in your config file appropriately, and used for training or inference.

Finger-spelling Datasets
========

This section describes the list of all supported pose-based finger-spelling datasets available.

Supported Datasets
------------------

The following pose datasets are available out-of-the-box.

.. csv-table::
   :file: ../_static/fs_datasets.csv
   :header-rows: 1
