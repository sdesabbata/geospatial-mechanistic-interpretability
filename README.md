# Geospatial Mechanistic Interpretability

This repository includes the code and data that support the findings of the *forthcoming* book chapter:

**De Sabbata, S., Mizzaro, S. and Roitero, K. (2025) “Geospatial mechanistic interpretability of large language models,” in Janowicz, K. et al. (eds.) Geography according to ChatGPT. IOS Press (Frontiers in artificial intelligence and applications).**

```
@incollection{desabbata2025geomechinterp,
  author    = {De Sabbata, Stef and Mizzaro, Stefano and Roitero, Kevin},
  title     = {Geospatial Mechanistic Interpretability of Large Language Models},
  booktitle = {Geography According to ChatGPT},
  publisher = {IOS Press},
  year      = {2025},
  editor    = {Janowicz, Krzysztof and Cai, Ling and Mai, Gengchen and Bennett, Lauren and Zhu, Rui and Gao, Song and Hu, Yingjie and Wang, Zhangyu},
  series    = {Frontiers in Artificial Intelligence and Applications}
}
```

## Abstract

Large language models (LLMs) have demonstrated unprecedented capabilities across various natural language processing tasks. Their ability to process and generate viable text and code has made them ubiquitous in many fields, while their deployment as knowledge bases and ``reasoning'' tools remains an area of ongoing research. In geography, a growing body of literature has been focusing on evaluating LLMs' geographical knowledge and their ability to perform spatial reasoning. However, very little is still known about the internal functioning of these models, especially about how they process geographical information.

In this chapter, we establish a novel framework for the study of geospatial mechanistic interpretability -- using spatial analysis to reverse engineer how LLMs handle geographical information. Our aim is to advance our understanding of the internal representations that these complex models generate while processing geographical information -- what one might call "how LLMs think about geographic information" if such phrasing was not an undue anthropomorphism.

We first outline the use of probing in revealing internal structures within LLMs. We then introduce the field of mechanistic interpretability, discussing the superposition hypothesis and the role of sparse autoencoders in disentangling polysemantic internal representations of LLMs into more interpretable, monosemantic features.

In our experiments, we use spatial autocorrelation to show how features obtained for placenames display spatial patterns related to their geographic location and can thus be interpreted geospatially, providing insights into how these models process geographical information. We conclude by discussing how our framework can help shape the study and use of foundation models in geography.

## License

This project uses [Free Gazetteer Data](https://download.geonames.org/export/dump/) made available by [GeoNames](https://www.geonames.org/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), a [file containing the names of the Italian provinces](https://figshare.com/articles/dataset/Italian_provinces_2018/12249575?file=22534718) made available by [Michele Tizzoni](https://micheletizzoni.github.io/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and data derived from them using [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), which was made available by the [Mistral AI Team](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#the-mistral-ai-team) under [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

The input and intermediate data used in this project are available to download [here](https://figshare.le.ac.uk/articles/dataset/Data_for_Geospatial_Mechanistic_Interpretability_of_Large_Language_Models_/28905197) and should be copied in the `storage` folder, while the outputs can be found in the `results` folder. The input, intermediate and output data and figures are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The code is released under MIT License (see `LICENSE` file in this repo).

## Environment

We raccomend using the [gds_env](https://darribas.org/gds_env/) python environment to execute most of the scripts included in this repo, and a separate environment including `pytorch`, `pytorch-lightning`, `transformers` and `pandas` to execute the scripts numbered `002`, `111` and `112`.

## Acknowledgments

The authors would like to thank Univ.-Prof. Dr. Krzysztof Janowicz, Dr. Rui Zhu and the anonymous reviewers for their valuable comments, which helped us shape this project. This research used the ALICE High Performance Computing Facility at the University of Leicester.
