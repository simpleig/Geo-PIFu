# Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction

This repository is the official PyTorch implementation of [Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction](https://papers.nips.cc/paper/2020/file/690f44c8c2b7ded579d01abe8fdb6110-Paper.pdf), NeurIPS, 2020.
<p align="center">
<img src="https://github.com/simpleig/Geo-PIFu/blob/master/assests/pipeline.png" width="750">
</p>

If you find this code useful, please consider citing
```
@inproceedings{he2020geopifu,
  title     = {Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction},
  author    = {Tong He and John Collomosse and Hailin Jin and Stefano Soatto},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```

Development log
- Data preparation
  - [ ] DeepHuman dataset rendering
- Training
  - [ ] Train `PIFu` on the DeepHuman training dataset
  - [ ] Train `Geo-PIFu` on the DeepHuman training dataset
- Test
  - [ ] Test `PIFu` on the DeepHuman test dataset
  - [ ] Test `Geo-PIFu` on the DeepHuman test dataset
- Evaulation
  - [ ] Compute 4 metrics: CD, PSD, Normal Cosine, Normal L2

## Requirements

We provide a conda `yaml` environment file.

	conda env create -f geopifu_requirements.yaml
	conda activate geopifu

## Acknowledgements

This repository is built on: [DeepHuman](https://github.com/ZhengZerong/DeepHuman) and [PIFu](https://github.com/shunsukesaito/PIFu). Thank the authors for sharing their code!