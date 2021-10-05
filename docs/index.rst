.. 👐OpenHands documentation master file, created by
   sphinx-quickstart on Tue Oct  5 05:08:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to 👐OpenHands's documentation!
=======================================

**👐OpenHands** is an open source toolkit to democratize sign language research by making pose-based Sign Language Recognition (SLR) more accessible to everyone.

Features
--------

- Easily train pose-based isolated sign language recognizers (ISLR)
   - Currently supports 4 different models
- Support for many run-time augmentations for pose keypoints
   - Check for complete list of pose transforms
- Support for efficient inference
   - Check for more details: 
- Easily pre-train models using monolingual SL data
   - And fine-tune on any small ISLR dataset
- All Training and Inference is completely config-based
   - No-code required
   - Check for example configs: 
- Supports 6 sign languages out-of-the-box
   - Check for supported datasets: 

For detailed explanation, please check out our paper: `👐OpenHands: Making ... <https://arxiv.org>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   instructions/installation
   instructions/datasets
   instructions/models
   instructions/training
   instructions/self_supervised
   instructions/inference
   instructions/support
