# Dynamic Key-Value Memory Networks for Knowledge Tracing(DKVMN)

## Introduction

Dynamic Key-Value Memory Networks (DKVMN) can exploit the relationships between underlying concepts and directly output a student’s mastery level of each concept. Unlike standard memory-augmented neural networks that facilitate a single memory matrix or two static memory matrices, DKVMN has one static matrix called key, which stores the knowledge concepts and the other dynamic matrix called value, which stores and updates the mastery levels of corresponding concepts.

## Model

![model](_static/DKVMN.png)


If the reader wants to know the details of DKVMN, please refer to the Appendix of the paper: *[Dynamic Key-Value Memory Networks for Knowledge Tracing](https://arxiv.org/pdf/1611.08108v1.pdf)*.

```bibtex
@inproceedings{10.1145/3038912.3052580,
author = {Zhang, Jiani and Shi, Xingjian and King, Irwin and Yeung, Dit-Yan},
title = {Dynamic Key-Value Memory Networks for Knowledge Tracing},
year = {2017},
isbn = {9781450349130},
publisher = {International World Wide Web Conferences Steering Committee},
address = {Republic and Canton of Geneva, CHE},
url = {https://doi.org/10.1145/3038912.3052580},
doi = {10.1145/3038912.3052580},
booktitle = {Proceedings of the 26th International Conference on World Wide Web},
pages = {765–774},
numpages = {10},
keywords = {massive open online courses, knowledge tracing, dynamic key-value memory networks, deep learning},
location = {Perth, Australia},
series = {WWW '17}
}
```

