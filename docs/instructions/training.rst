Training
========

This section explains how to train ISLR models using the existing datasets and models.

Config-based training
---------------------

For examples on how to use the datasets and models in configs, `click here <https://github.com/AI4Bharat/OpenHands/tree/main/examples>`_.

After you have a config ready, run the following python snippet:

.. code:: python

    import omegaconf
    from openhands.core.classification_model import ClassificationModel
    from openhands.core.exp_utils import get_trainer

    cfg = omegaconf.OmegaConf.load("path/to/config.yaml")
    trainer = get_trainer(cfg)
    
    model = ClassificationModel(cfg=cfg, trainer=trainer)
    model.init_from_checkpoint_if_available()
    model.fit()

- This will automatically do all the setup, and start the training for you!
- The best checkpoints will also be dumped based on validation from each epoch.
- Feel free to play with the different parameters in the existing configs
