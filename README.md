# SDDN: Soft Knowledge-based Distilled Dehazing Networks

[![Weights](https://img.shields.io/badge/Weights-Hugging%20Face-gold)](https://huggingface.co/tranleanh/sddn)

This repo contains the official implementation of the paper "Soft Knowledge-based Distilled Dehazing Networks".

Authors: [Le-Anh Tran](https://scholar.google.com/citations?user=WzcUE5YAAAAJ&hl=en), [Dong-Chul Park](https://scholar.google.com/citations?user=VZUH4sUAAAAJ&hl=en)

## Updates
- [x] Results on benchmarks
- [ ] Pre-trained weights & Inference code
- [ ] Training code

## Introduction

Diagram of the framework:

<p align="center">
<img src="docs/SDDN_Framework.jpg" width="1000">
</p>

## Results

### Quantitative results on I-HAZE, O-HAZE, Dense-HAZE, and NH-HAZE:

<p align="center">
<img src="docs/results_ancuti.png" width="1000">
</p>

### Qualitative Results:

<p align="center">
<img src="docs/results_ihaze_ohaze.png" width="1000">
</p>

<p align="center">
<img src="docs/results_densehaze_nhhaze.png" width="1000">
</p>

<p align="center">
<img src="docs/results_sotsoutdoor_hsts.png" width="1000">
</p>

<p align="center">
<img src="docs/results_natural.png" width="1000">
</p>


## Citation

Please cite our work if you use the data in this repo. 

```bibtex
@misc{tran2024soft,
  author = {Tran, Le-Anh and Park, Dong-Chul},
  title = {Soft knowledge-based Distilled Dehazing Networks},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```

LA Tran
