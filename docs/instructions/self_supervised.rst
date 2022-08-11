Self-Supervised Learning
========================

This section is dedicated to explaining how to perform pre-training and fine-tuning using any required datasets for ISLR.

Pretraining Dataset
-------------------

The raw dataset used for pretraining is expected to be in HDF5 format, inorder to have faster random-access at frame-level for sampling random windows during pretraining.

HDF5 format
^^^^^^^^^^^

- For each YouTube channel/playlist, we have different ``.h5`` files.
- Each ``.h5`` file has 2 groups, namely ``keypoints`` (which has the actual pose data) and ``visibility`` (confidence scores for each keypoint returned)
- Each group has multiple datasets in it, with name as the YouTube video and data of shape ``(F, K, C)``, where ``F`` is number of frames in that video chunk, ``K`` is the number of keypoints (75 in our dataset) and ``C`` is the number of channels (3 for ``keypoints`` and 1 for ``visibility``)

Generating HDF5 datasets
^^^^^^^^^^^^^^^^^^^^^^^^

- `See this script <https://github.com/AI4Bharat/OpenHands/blob/main/scripts/mediapipe_extract.py>`_ to extract pose for all the given videos using MediaPipe Holistic
- Use `this script <https://github.com/AI4Bharat/OpenHands/blob/main/scripts/pkl_to_h5.py>`_ to convert all the above individual pose files (in ``.pkl``) to HDF5 format.

Download datasets
^^^^^^^^^^^^^^^^^

The following are the checkpoints scraped for Indian SL for raw pretraining (as mentioned in paper):

.. csv-table::
   :file: ../_static/raw_datasets.csv
   :header-rows: 1

For downloading data for the other 9 sign languages mentioned in our work, please `use this Azure container link <https://ai4bharatsignlanguage.blob.core.windows.net/archives?sp=r&st=2022-08-02T13:51:42Z&se=2023-01-30T21:51:42Z&spr=https&sv=2021-06-08&sr=c&sig=7L6rwZdRz8lFhtxR4llamHUJzifJbLDzm0f9cEVZL%2BU%3D>`_.

Pre-training
------------

Currently, the library supports pose-based pretraining based on the `dense predictive coding <https://www.robots.ox.ac.uk/~vgg/research/DPC/dpc.html>`_ (DPC) technique.

- To perform pre-training, `download the config from here <https://github.com/AI4Bharat/OpenHands/blob/main/examples/ssl/pretrain_dpc.yaml>`_
- Set the ``root_dir`` for train_dataset and val_dataset. Usually, HDF5 is used for training and a ISLR dataset like INCLUDE (from `Datasets` section) is used as validation set.

Finally, run the following snippet to perform the pretraining:

.. code:: python

    import omegaconf
    from openhands.apis.dpc import PretrainingModelDPC

    cfg = omegaconf.OmegaConf.load("path/to/config.yaml")
    trainer = PretrainingModelDPC(cfg=cfg)
    trainer.fit()

Fine-tuning
-----------

- Ensure that the model parameters and pretrained checkpoint path are specified in a new config as shown in `this fine-tuning example <https://github.com/AI4Bharat/OpenHands/blob/main/examples/configs/include/pose_finetune_dpc.yaml>`_.
- Finally, you can perform the fine-tuning using the same snippet from the `Training` section.

Checkpoints
-----------

The following are the checkpoints reported in the paper, which was pretrained using the above mentioned Indian raw SL data, and finetuned on different labeled datasets.

.. csv-table::
   :file: ../_static/ssl_checkpoints.csv
   :header-rows: 1
