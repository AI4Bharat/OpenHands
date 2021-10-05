Inference and Testing
=====================

This section describes how to use the trained models for testing using a test set or running inference for any given videos.

To test using our trained models, download the required checkpoint, config, and dataset metadata from the `Models` section.

Computing accuacy using test set
--------------------------------

- Add a sub-section called ``test_pipeline`` to the ``data`` section of the config.
- This section is of same format as ``valid_pipeline`` section.
- Finally, run the following snippet to compute accuracy for the given pose test set:

.. code:: python

    import omegaconf
    from openhands.core.inference import InferenceModel

    cfg = omegaconf.OmegaConf.load("path/to/config.yaml")
	model = InferenceModel(cfg=cfg)
    model.init_from_checkpoint_if_available()
    if cfg.data.test_pipeline.dataset.inference_mode:
        model.test_inference()
    else:
        model.compute_test_accuracy()

Predicting for any given videos
-------------------------------

- In the same config as described above, just change the ``root_dir`` to point to any desired folder containing videos (or pose files)
- Set an additional variable called ``inference_mode`` as ``true``, to indicate that there will be no ground truth.
- You can now run the inference using the same snippet above.
