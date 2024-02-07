# SDDN: Soft Knowledge-based Distilled Dehazing Networks

[![Models](https://img.shields.io/badge/Models-Hugging_Face-gold)](https://huggingface.co/tranleanh/sddn)
[![Paper](https://img.shields.io/badge/Paper-TechRxiv-white)](https://www.techrxiv.org/doi/full/10.36227/techrxiv.170723333.32153858/v1)

This repo contains the official implementation of the paper "Soft Knowledge-based Distilled Dehazing Networks".

Authors: [Le-Anh Tran](https://scholar.google.com/citations?user=WzcUE5YAAAAJ&hl=en), [Dong-Chul Park](https://scholar.google.com/citations?user=VZUH4sUAAAAJ&hl=en)

## Updates
- [ ] Training code
- [ ] Inference code
- [x] Pre-trained weights ([Hugging Face](https://huggingface.co/tranleanh/sddn))
- [x] Results on benchmarks (I-HAZE, O-HAZE, Dense-HAZE, NH-HAZE, SOTS-Outdoor, HSTS)



## Introduction

Diagram of the framework:

<p align="center">
<img src="docs/SDDN_Framework.jpg" width="1000">
</p>

## Results

### Quantitative results:

<p align="center">
<img src="docs/quantitative_results.png" width="1000">
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
