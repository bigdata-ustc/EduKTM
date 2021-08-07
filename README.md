<p align="center">
  <img width="300" src="docs/_static/EduKTM.png">
</p>

# EduKTM
[![PyPI](https://img.shields.io/pypi/v/EduKTM.svg)](https://pypi.python.org/pypi/EduKTM)
[![test](https://github.com/bigdata-ustc/EduKTM/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/bigdata-ustc/EduKTM/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/bigdata-ustc/EduKTM/branch/main/graph/badge.svg?token=B7gscOGQLD)](https://codecov.io/gh/bigdata-ustc/EduKTM)
[![Download](https://img.shields.io/pypi/dm/EduKTM.svg?style=flat)](https://pypi.python.org/pypi/EduKTM)
[![License](https://img.shields.io/github/license/bigdata-ustc/EduKTM)](LICENSE)
[![DOI](https://zenodo.org/badge/348569820.svg)](https://zenodo.org/badge/latestdoi/348569820)

The Model Zoo of Knowledge Tracing  Models.

Knowledge Tracing (KT), which aims to monitor students’ evolving knowledge state, is a fundamental and crucial task to support these intelligent services. Therefore, an increasing amount of research attention has been paid to this emerging area and considerable progress has been made[1]. However, the code of these works may use different program languages (e.g., python, lua) and different deep learning frameworks (e.g., tensorflow, torch and mxnet). Furthermore, some works did not well organize the codes systemly (e.g., the missing of running environments and dependencies), which brings difficulties in reproducing the models. To this end, we put forward the Model Zoo of Knowledge Tracing Models, named EduKTM, which collects most of concurrent popular works.

## Brief introduction to KTM

## List of models

* [KPT,EKPT](EduKTM/KPT) [[doc]](docs/KPT.md) [[example]](examples/KPT)
* [DKT](EduKTM/DKT) [[doc]](docs/DKT.md) [[example]](examples/DKT)
* [DKT+](EduKTM/DKTPlus) [[doc]](docs/DKT+.md) [[example]](examples/DKT+)
* [AKT](EduKTM/AKT) [[doc]](docs/AKT.md) [[example]](examples/AKT)

## Contribute

EduKTM is still under development. More algorithms and features are going to be added and we always welcome contributions to help make EduKTM better. If you would like to contribute, please follow this [guideline](CONTRIBUTE.md).

## Citation

If this repository is helpful for you, please cite our work

```
@misc{bigdata2021eduktm,
  title={EduKTM},
  author={bigdata-ustc},
  publisher = {GitHub},
  journal = {GitHub repository},
  year = {2021},
  howpublished = {\url{https://github.com/bigdata-ustc/EduKTM}},
}
```

## Reference

[1] [Liu Q, Shen S, Huang Z, et al. A Survey of Knowledge Tracing[J]. arXiv preprint arXiv:2105.15106, 2021.](https://arxiv.org/pdf/2105.15106.pdf)
