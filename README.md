# 👐OpenHands: Sign Language Recognition Library

> _Making Sign Language Recognition Accessible_

Check the documentation on how to use the library:  
**[ReadTheDocs: 👐OpenHands](https://openhands.readthedocs.io)**

## Installation

- For stable version: `pip install --upgrade OpenHands`
- For latest development version: `pip install git+https://github.com/AI4Bharat/OpenHands`

## License

This project is released under the [Apache 2.0 license](LICENSE.txt).

## Datasets used

Please cite the respective datasets if you used them in your research. Also check the licensing terms for the dataset used.

| Dataset         | Link |
| --------------- | ----------- |
| AUTSL           | [Link](https://chalearnlap.cvc.uab.es/dataset/40/description/)       |
| CSL             | [Link](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)        |
| DEVISIGN         | [Link](http://vipl.ict.ac.cn/homepage/ksl/data.html)       |
| GSL             | [Link](https://vcl.iti.gr/dataset/gsl/)        |
| INCLUDE         | [Link](https://sign-language.ai4bharat.org/#/INCLUDE)       |
| LSA64           | [Link](http://facundoq.github.io/datasets/lsa64/)        |
| WLASL           | [Link](https://dxli94.github.io/WLASL/)        |

## Extraction of poses

For datasets without the pose data, poses can be extracted from the videos using [this script](scripts/mediapipe_extract.py). 
## Citation

If you find our work useful in your research, please consider citing us:

```BibTeX
@misc{2021_openhands_slr_preprint,
      title={OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages}, 
      author={Prem Selvaraj and Gokul NC and Pratyush Kumar and Mitesh Khapra},
      year={2021},
      eprint={2110.05877},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
