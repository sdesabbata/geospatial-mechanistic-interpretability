# Geospatial Mechanistic Interpretability of LLMs

This repository includes the code and data that support the findings of a *forthcoming* book chapter (see reference below).
The Author Accepted Manuscript of "[Geospatial Mechanistic Interpretability of Large Language Models](https://arxiv.org/abs/2505.03368)" available on arXiv ([arXiv:2505.03368](https://arxiv.org/abs/2505.03368)).

**De Sabbata, S., Mizzaro, S. and Roitero, K. Geospatial mechanistic interpretability of large language models. In Janowicz, K. et al. editors. Geography according to ChatGPT. Frontiers in artificial intelligence and applications. IOS Press; forthcoming.**

## Abstract

Large language models (LLMs) have demonstrated unprecedented capabilities across various natural language processing tasks. Their ability to process and generate viable text and code has made them ubiquitous in many fields, while their deployment as knowledge bases and ``reasoning'' tools remains an area of ongoing research. In geography, a growing body of literature has been focusing on evaluating LLMs' geographical knowledge and their ability to perform spatial reasoning. However, very little is still known about the internal functioning of these models, especially about how they process geographical information.

In this chapter, we establish a novel framework for the study of geospatial mechanistic interpretability -- using spatial analysis to reverse engineer how LLMs handle geographical information. Our aim is to advance our understanding of the internal representations that these complex models generate while processing geographical information -- what one might call "how LLMs think about geographic information" if such phrasing was not an undue anthropomorphism.

We first outline the use of probing in revealing internal structures within LLMs. We then introduce the field of mechanistic interpretability, discussing the superposition hypothesis and the role of sparse autoencoders in disentangling polysemantic internal representations of LLMs into more interpretable, monosemantic features.

In our experiments, we use spatial autocorrelation to show how features obtained for placenames display spatial patterns related to their geographic location and can thus be interpreted geospatially, providing insights into how these models process geographical information. We conclude by discussing how our framework can help shape the study and use of foundation models in geography.

## Framework

![An example illustrating the extraction of the activations from an LLM (top); the use of the activations in a linear probe to predict the latitude and longitude of the place mentioned in the input (bottom-left) and a sparse autoencoder (bottom-right); and the use of spatial autocorrelation to analyse the activations and the sparse features (centre). Our approach encompasses the latter two components.](paper/img_probing-and-sae_v1-0.png)

## Results

### Spatial analysis of activations

![Activations captured for the input placenames at different layers of the LLM (left for each region) and their local spatial autocorrelation (local Moran's $I$ clusters, $p<.01$, right for each region), illustrating the polysemantic nature of its internal representations. Two neurons at layers 7 (a) and 15 (b) show high values for the State of New York and Northern Ireland and very low values for northern Italy. A neuron at layer (c) 15 shows high values for several UK cities. A neuron at layer 31 (d) shows high values for the State of New York and Northern Ireland and diverse values for provinces in Italy.](paper/img_results-probing_v1-0.png)


### Spatial analysis of SAE features

![Features extracted from layer 15 through a sparse autoencoder (left for each region) and their local spatial autocorrelation (local Moran's $I$ clusters, $p<.01$, right for each region): (a) Wales as a region part of prompt; (b) south of Italy as a region activating a seemingly monosemantic feature; (c) north-east of Italy and north-west of England as regions activating a seemingly polysemantic feature; and (d) a representation of ``city'' highlighting New York City and London, amongst others.](paper/img_results-sparse-autoencoder_v1-0.png)

## License

This project uses [Free Gazetteer Data](https://download.geonames.org/export/dump/) made available by [GeoNames](https://www.geonames.org/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), a [file containing the names of the Italian provinces](https://figshare.com/articles/dataset/Italian_provinces_2018/12249575?file=22534718) made available by [Michele Tizzoni](https://micheletizzoni.github.io/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and data derived from them using [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), which was made available by the [Mistral AI Team](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#the-mistral-ai-team) under [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

The input and intermediate data used in this project are available to download [here](https://figshare.le.ac.uk/articles/dataset/Data_for_Geospatial_Mechanistic_Interpretability_of_Large_Language_Models_/28905197) and should be copied in the `storage` folder, while the outputs can be found in the `results` folder. The input, intermediate and output data and figures are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The code is released under MIT License (see `LICENSE` file in this repo).

## Environment

We raccomend using the [gds_env](https://darribas.org/gds_env/) python environment to execute most of the scripts included in this repo, and a separate environment including `pytorch`, `pytorch-lightning`, `transformers` and `pandas` to execute the scripts numbered `002`, `111` and `112`.

## Acknowledgments

The authors would like to thank Univ.-Prof. Dr. Krzysztof Janowicz, Dr. Rui Zhu and the anonymous reviewers for their valuable comments, which helped us shape this project. This research used the ALICE High Performance Computing Facility at the University of Leicester.
