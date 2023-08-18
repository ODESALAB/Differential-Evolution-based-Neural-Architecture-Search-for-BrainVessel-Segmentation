Differential evolution-based neural architecture search for brain vessel segmentation
===
Abstract: Brain vasculature analysis is critical in developing novel treatment targets for neurodegenerative diseases. Such an accurate analysis cannot be performed manually but requires a semi-automated or fully-automated approach. Deep learning methods have recently proven indispensable for the automated segmentation and analysis of medical images. However, optimizing a deep learning network architecture is another challenge. Manually selecting deep learning network architectures and tuning their hyper-parameters requires a lot of expertise and effort. To solve this problem, neural architecture search (NAS) approaches that explore more efficient network architectures with high segmentation performance have been proposed in the literature. This study introduces differential evolution-based NAS approaches in which a novel search space is proposed for brain vessel segmentation. We select two architectures that are frequently used for medical image segmentation, i.e. U-Net and Attention U-Net, as baselines for NAS optimizations. The conventional differential evolution and the opposition-based differential evolution with novel search space are employed as search methods in NAS. Furthermore, we perform ablation studies and evaluate the effects of specific loss functions, model pruning, threshold selection and generalization performance on the proposed models. The experiments are conducted on two datasets providing 335 single-channel 8-bit gray-scale images. These datasets are a public volumetric cerebrovascular system dataset (vesseINN) and our own dataset called KUVESG. The proposed NAS approaches, namely UNAS-Net and Attention UNAS-Net architectures, yield better segmentation performance in terms of different segmentation metrics. More specifically, UNAS-Net with differential evolution reveals high dice score/sensitivity values of 79.57/81.48, respectively. Moreover, they provide shorter inference times by a factor of 9.15 than the baseline methods.

## Paper Information
- Title:  [`Differential-Evolution-based-Neural-Architecture-Search-for-BrainVessel-Segmentation](https://doi.org/10.1016/j.jestch.2023.101502)`
- Authors:  `Zeki Kuş`,`Berna Kiraz`,`Tuğçe Koçak Göksu`, `Musa Aydın`, `Esra Özkan`, `Atay Vural`, `Alper Kiraz`, `Burhanettin Can`

## Dataset

KUVESG: [Link](https://zenodo.org/record/7383295)<br>
vesselNN: [Link](https://github.com/petteriTeikari/vesselNN)<br>
Train Test Splits: [Link](https://github.com/ODESALAB/Differential-Evolution-based-Neural-Architecture-Search-for-BrainVessel-Segmentation/tree/main/Vessel_2D)<br>
You should put the Vessel_2D folder into the ./UNAS-Net/DataSets/ or Attention UNAS-Net/DataSets/ folder to train/evaluation.

## Use
- for short-term evaluation
  ```
  please look into ./UNAS-Net/de_op.py
  please look into ./UNAS-Net/de.py
  please look into ./Attention UNAS-Net/de_op.py
  please look into ./Attention UNAS-Net/de.py
  ```
- for long-term evaluation
  ```
  please look into ./UNAS-Net/eval_model.py
  please look into ./Attention UNAS-Net/eval_model.py
  ```
  
To cite the paper or code:
```bibtex
@article{KUS2023101502,
title = {Differential evolution-based neural architecture search for brain vessel segmentation},
journal = {Engineering Science and Technology, an International Journal},
volume = {46},
pages = {101502},
year = {2023},
issn = {2215-0986},
doi = {https://doi.org/10.1016/j.jestch.2023.101502}
}
```
