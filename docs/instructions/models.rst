Models
======

This section describes the list of all models and trained checkpoints currently available in the library.

Usage
-----

The following ISLR models are supported currently in the library.

.. csv-table::
   :file: ../_static/networks.csv
   :header-rows: 1

For examples on how to use the models in configs, `click here <https://github.com/AI4Bharat/OpenHands/tree/main/examples>`_.

Trained checkpoints
-------------------

The following trained checkpoints are available for download across all the currently supported datasets and models.

.. csv-table::
   :file: ../_static/checkpoints.csv
   :header-rows: 1

Note:  

- Metadata is a smaller extract from the actual dataset, which contains the actual signs to ID mappings.
   - Extracted metadata path should be mentioned in the config.
- The zipped checkpoints has both the config used to train as well as the trained parameters.
   - For inference using the checkpoint, check the ``Inference`` section.
