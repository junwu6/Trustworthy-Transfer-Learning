## Trustworthy Transfer Learning: Transferability and Trustworthiness

The KDD 2023 tutorial titled "Trustworthy Transfer Learning: Transferability and Trustworthiness" can be found on the [official site](https://sites.google.com/view/kdd23-trustworthy-transfer).

```
@inproceedings{wu2023trustworthy,
  title={Trustworthy Transfer Learning: Transferability and Trustworthiness},
  author={Wu, Jun and He, Jingrui},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={5829--5830},
  year={2023}
}
```

### ABSTRACT

Deep transfer learning investigates the transfer of knowledge or information from a source domain to a relevant target domain via deep neural networks. In this tutorial, we dive into understanding deep transfer learning from the perspective of knowledge transferability and trustworthiness (e.g., privacy, robustness, fairness, transparency, etc.). To this end, we provide a comprehensive review of state-of-the-art theoretical analysis and algorithms for deep transfer learning. To be specific, we start by introducing the concepts of transferability and trustworthiness in the context of deep transfer learning. Then we summarize recent theories and algorithms for understanding knowledge transferability from two aspects: (1) IID transferability: the samples within each domain are independent and identically distributed (e.g., individual images), and (2) non-IID transferability: The samples within each domain are interdependent (e.g., connected nodes within a graph). In addition to knowledge transferability, we also review the impact of trustworthiness on deep transfer learning, e.g., whether the transferred knowledge is adversarially robust or algorithmically fair, how to transfer the knowledge under privacy-preserving constraints, etc. Finally, we highlight the open questions and future directions for understanding deep transfer learning in real-world applications. We believe this tutorial can benefit researchers and practitioners by rethinking the trade-off between knowledge transferability and trustworthiness in developing trustworthy transfer learning systems.

### OUTLINE

* Introduction

* Part I: Knowledge Transferability

  + Part I-1: IID Transferability

    - Domain Discrepancy (Data-level)

    - Task Diversity (Task-level)

    - Transferability Measures (Model-level)

  + Part I-2: Non-IID Transferability

    - Graph Transferability

    - Text Transferability

    - Time-series Transferability

* Part II: Knowledge Trustworthiness

  + Part II-1: Adversarial Robustness

    - Attack

    - Defense

  + Part II-2: Privacy

    - Source-free Hypothesis Transfer

    - Federated Learning

  + Part II-3: Fairness

    - Individual Fairness

    - Group Fairness

  + Part II-4: Transparency and Explainability

    - Feature-level Explainability

    - Prediction-level Uncertainty

* Part III: Open Questions and Future Trends

  + Multi-dimensional Distribution Shifts

  + Universal Trustworthiness

  + Tradeoff between Transferability and Trustworthiness


### Reference

1. [IID Transferability](#1-iid-transferability)
    * 1.1. [Distribution Discrepancy](#11-distribution-discrepancy)
    * 1.2. [Task Diversity](#12-task-diversity)
    * 1.3. [Transferability Measures](#13-transferability-measures)
2. [Non-IID Transferability](#2-non-iid-transferability)
    * 2.1. [Graph Transferability](#21-graph-transferability)
    * 2.2. [Text Transferability](#22-text-transferability)
    * 2.3. [Time Series Transferability](#23-time-series-transferability)
3. [Robustness](#3-robustness)
    * 3.1. [Poisoning Attacks](#31-poisoning-attacks)
    * 3.2. [Evasion Attacks](#32-evasion-attacks)
    * 3.3. [Defense](#33-defense)
    * 3.4. [Transferability vs. Robustness](#34-transferability-vs-robustness)
4. [Privacy](#4-privacy)
    * 4.1. [Federated Transferability](#41-federated-transferability)
    * 4.2. [Source-Free Transferability](#42-source-free-transferability)
5. [Fairness](#5-fairness)
    * 5.1. [Individual Fairness](#51-individual-fairness)
    * 5.2. [Group Fairness](#52-group-fairness)
6. [Transparency](#6-transparency)
    * 6.1. [Model-Level Explainability](#61-model-level-explainability)
    * 6.2. [Predition-Level Uncertainty](#62-prediction-level-uncertainty)


### 1. IID Transferability

#### 1.1. Distribution Discrepancy

- **Domain adaptation: Learning bounds and algorithms (COLT'09).** [[Paper]](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/35391.pdf)
- **A theory of learning from different domains (ML'10).** [[Paper]](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- **Learning bounds for importance weighting (NeurIPS'10).** [[Paper]](https://proceedings.neurips.cc/paper/2010/hash/59c33016884a62116be975a9bb8257e3-Abstract.html)
- **Impossibility theorems for domain adaptation (AISTATS'10)** [[Paper]](http://proceedings.mlr.press/v9/david10a/david10a.pdf)
- **Learning transferable features with deep adaptation networks (ICML'15).** [[Paper]](https://proceedings.mlr.press/v37/long15)
- **Unsupervised domain adaptation by backpropagation (ICML'15).** [[Paper]](http://proceedings.mlr.press/v37/ganin15.html)
- **Return of frustratingly easy domain adaptation (AAAI'16).** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/10306)
- **Wasserstein distance guided representation learning for domain adaptation (AAAI'18).** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11784)
- **Bridging theory and algorithm for domain adaptation (ICML'19).** [[Paper]](http://proceedings.mlr.press/v97/zhang19i.html?ref=https://codemonkey)
- **On learning invariant representations for domain adaptation (ICML'19).** [[Paper]](https://proceedings.mlr.press/v97/zhao19a.html)
- **Adaptation based on generalized discrepancy (JMLR'19).** [[Paper]](https://www.jmlr.org/papers/volume20/15-192/15-192.pdf?ref=https://githubhelp.com)
- **$f$-domain-adversarial learning: Theory and algorithms (ICML'21).** [[Paper]](http://proceedings.mlr.press/v139/acuna21a/acuna21a.pdf)
- **Distribution-informed neural networks for domain adaptation regression (NeurIPS'22).** [[Paper]](https://openreview.net/pdf?id=8hoDLRLtl9h)
- **KL guided domain adaptation (ICLR'22).** [[Paper]](https://arxiv.org/pdf/2106.07780.pdf)

#### 1.2. Task Diversity

- **The benefit of multitask representation learning (JMLR'16).** [[Paper]](https://www.jmlr.org/papers/volume17/15-242/15-242.pdf)
- **On the theory of transfer learning: The importance of task diversity (NeurIPS'20).** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/59587bffec1c7846f3e34230141556ae-Abstract.html)
- **Few-shot learning via learning the representation, provably (ICLR'21).** [[Paper]](https://openreview.net/forum?id=pW2Q2xLwIMD)

#### 1.3. Transferability Measures

- **An information-theoretic approach to transferability in task transfer learning (ICIP'19).** [[Paper]](https://ieeexplore.ieee.org/abstract/document/8803726)
- **Transferability and hardness of supervised classification tasks (ICCV'19).** [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Tran_Transferability_and_Hardness_of_Supervised_Classification_Tasks_ICCV_2019_paper.html)
- **LEEP: A new measure to evaluate transferability of learned representations (ICML'20).** [[Paper]](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf)
- **LogME: Practical assessment of pre-trained models for transfer learning (ICML'21).** [[Paper]](https://proceedings.mlr.press/v139/you21b.html) [[Code]](https://github.com/thuml/LogME)
- **Ranking neural checkpoints (CVPR'21).** [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Ranking_Neural_Checkpoints_CVPR_2021_paper.html)
- **Frustratingly easy transferability estimation (ICML'22).** [[Paper]](https://proceedings.mlr.press/v162/huang22d.html)
- **Transferability estimation using Bhattacharyya class separability (CVPR'22).** [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Pandy_Transferability_Estimation_Using_Bhattacharyya_Class_Separability_CVPR_2022_paper.html)
- **How stable are transferability metrics evaluations (ECCV'22).** [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940296.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/stable_transfer)

### 2. Non-IID Transferability

#### 2.1. Graph Transferability

- **Graphon neural networks and the transferability of graph neural networks (NeurIPS'20).** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/12bcd658ef0a540cabc36cdf2b1046fd-Abstract.html)
- **Transfer learning of graph neural networks with ego-graph information maximization (NeurIPS'21).** [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/0dd6049f5fa537d41753be6d37859430-Abstract.html) [[Code]](https://github.com/GentleZhu/EGI)
- **Non-IID transfer learning on graphs (AAAI'23).** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26231) [[Code]](https://github.com/jwu4sml/GRADE)
- **Graph domain adaptation via theory-grounded spectral regularization (ICLR'23).** [[Paper]](https://openreview.net/forum?id=OysfLgrk8mk)
- **Structural re-weighting improves graph domain adaptation (ICML'23).** [[Paper]](https://proceedings.mlr.press/v202/liu23u.html)
- **Graph-Structured Gaussian Processes for Transferable Graph Learning (NeurIPS'23).**

#### 2.2. Text Transferability

- **Investigating transferability in pretrained language models (EMNLP'20).** [[Paper]](https://aclanthology.org/2020.findings-emnlp.125/) [[Code]](https://github.com/dgiova/bert-lm-transferability)
- **On learning language-invariant representations for universal machine translation (ICML'20).** [[Paper]](https://proceedings.mlr.press/v119/zhao20b.html)
- **On the transferability of pre-trained language models: A study from artificial datasets (AAAI'22).** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21295) [[Code]](https://github.com/d223302/Transformer-Structure)

#### 2.3. Time Series Transferability

- **Variational recurrent adversarial deep domain adaptation (ICLR'2017).** [[Paper]](https://openreview.net/pdf?id=rk9eAFcxg) [[Code]](https://github.com/floft/vrada)
- **Domain adaptation for time series forecasting via attention sharing (ICML'22).** [[Paper]](https://proceedings.mlr.press/v162/jin22d/jin22d.pdf)
- **Contrastive learning for unsupervised domain adaptation of time series (ICLR'23).** [[Paper]](https://openreview.net/forum?id=xPkJYRsQGM)
- **Domain adaptation for time series under feature and label shifts (ICML'23).** [[Paper]](https://arxiv.org/pdf/2302.03133.pdf) [[Code]](https://github.com/mims-harvard/Raincoat)

### 3. Robustness

#### 3.1. Poisoning Attacks

- **Indirect invisible poisoning attacks on domain adaptation (KDD'21).** [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3447548.3467214?casa_token=YrCDqICKcvEAAAAA:w-cwOfOWlhLFgxGdpzaa2jvISMFKGqtojKdhE7JLwiLYeywAEPwrHvmz7R2PAORuR9HrX5MpFEwt) [[Code]](https://github.com/jwu4sml/I2Attack)
- **Understanding the limits of unsupervised domain adaptation via data poisoning (NeurIPS'21).** [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/90cc440b1b8caa520c562ac4e4bbcb51-Abstract.html) [[Code]](https://github.com/akshaymehra24/LimitsOfUDA)
- **A unified framework for adversarial attacks on multi-source domain adaptation (TKDE'22).** [[Paper]](https://ieeexplore.ieee.org/abstract/document/9994047?casa_token=fUknsVhBBrYAAAAA:ui0cUh6yiB-nSTkm3-koDDFxNQWNF_v77IkBtfpxti9-Tii62ul8jX-I7BQmeG9dMNm0n1bHng)

#### 3.2. Evasion Attacks

- **Two sides of the same coin: White-box and black-box attacks for transfer learning (KDD'20).** [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394486.3403349)
- **A target-agnostic attack on deep models: Exploiting security vulnerabilities of transfer learning (ICLR'20).** [[Paper]](https://arxiv.org/pdf/1904.04334.pdf) [[Code]](https://github.com/shrezaei/Target-Agnostic-Attack)

#### 3.3. Defense

- **Adversarially robust transfer learning (ICLR'20).** [[Paper]](https://arxiv.org/pdf/1905.08232.pdf)
- **Adversarial robustness for unsupervised domain adaptation (ICCV'21).** [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Awais_Adversarial_Robustness_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.pdf) [[Code]](https://awaisrauf.github.io/robust_uda)

#### 3.4. Transferability vs. Robustness

- **Using pre-training can improve model robustness and uncertainty (ICML'19).** [[Paper]](https://proceedings.mlr.press/v97/hendrycks19a.html) [[Code]](https://github.com/hendrycks/pre-training)
- **Do adversarially robust ImageNet models transfer better? (NeurIPS'20).** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/24357dd085d2c4b1a88a7e0692e60294-Abstract.html) [[Code]](https://github.com/Microsoft/robust-models-transfer)
- **Adversarial training helps transfer learning via better representations (NeurIPS'21).** [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/d3aeec875c479e55d1cdeea161842ec6-Abstract.html)
- **Adversarially-trained deep nets transfer better: Illustration on image classification (ICLR'21).** [[Paper]](https://arxiv.org/pdf/2007.05869.pdf(%5Bpaper%5D(https://arxiv.org/pdf/2007.05869.pdf))) [[Code]](https://github.com/utrerf/robust_transfer_learning)


### 4. Privacy

#### 4.1. Federated Transferability

- **Federated adversarial domain adaptation (ICLR'20).** [[Paper]](https://arxiv.org/pdf/1911.02054.pdf) [[Code]](https://drive.google.com/file/d/1OekTpqB6qLfjlE2XUjQPm3F110KDMFc0/view)
- **Personalized federated learning with parameter propagation (KDD'23).** [[Paper]](https://dl.acm.org/doi/abs/10.1145/3580305.3599464) [[Code]](https://github.com/jwu4sml/FEDORA)
- **Optimizing the collaboration structure in cross-Silo federated learning (ICML'23).** [[Paper]](https://arxiv.org/abs/2306.06508) [[Code]](https://github.com/baowenxuan/FedCollab)

#### 4.2. Source-Free Transferability
- **Stability and hypothesis transfer learning (ICML'13).** [[Paper]](https://proceedings.mlr.press/v28/kuzborskij13.html)
- **Hypothesis transfer learning via transformation functions (NeurIPS'17).** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2017/file/352fe25daf686bdb4edca223c921acea-Paper.pdf)
- **Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation (ICML'20).** [[Paper]](https://proceedings.mlr.press/v119/liang20a.html) [[Code]](https://github.com/tim-learn/SHOT)
- **Tent: Fully test-time adaptation by entropy minimization (ICLR'21).** [[Paper]](https://openreview.net/forum?id=uXl3bZLkr3c) [[Code]](https://github.com/DequanWang/tent)
- **On balancing bias and variance in unsupervised multi-source-free domain adaptation (ICML'23).** [[Paper]](https://proceedings.mlr.press/v202/shen23b.html)


### 5. Fairness

#### 5.1. Individual Fairness

- **Domain adaptation meets individual fairness. And they get along (NeurIPS'22).** [[Paper]](https://openreview.net/pdf?id=XSNfXG9HBAu)

#### 5.2. Group Fairness

- **Learning adversarially fair and transferable representations (ICML'18).** [[Paper]](http://proceedings.mlr.press/v80/madras18a.html) [[Code]](https://github.com/VectorInstitute/laftr)
- **Transfer of machine learning fairness across domains (NeurIPS Workshop'19).** [[Paper]](https://aiforsocialgood.github.io/neurips2019/accepted/track1/pdfs/57_aisg_neurips2019.pdf)
- **Robust fairness under covariate shift (AAAI'21).** [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17135) [[Code]](https://github.com/arezae4/fair_covariate_shift)
- **Transferring fairness under distribution shifts via fair consistency regularization (NeurIPS'22).** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d1dbaabf454a479ca86309e66592c7f6-Abstract-Conference.html) [[Code]](https://github.com/umd-huang-lab/transfer-fairness)
- **Fairness transferability subject to bounded distribution shift (NeurIPS'22).** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4937610670be26d651ecdb4f2206d95f-Abstract-Conference.html)


### 6. Transparency

#### 6.1. Model-Level Explainability

- **How transferable are features in deep neural networks? (NeurIPS'14)** [[Paper]](https://proceedings.neurips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html) [[Code]](https://github.com/yosinski/convnet_transfer)
- **Transfusion: Understanding transfer learning for medical imaging (NeurIPS'19).** [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html)
- **What is being transferred in transfer learning? (NeurIPS'20).** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/0607f4c705595b911a4f3e7a127b44e0-Abstract.html) [[Code]](https://github.com/google-research/understanding-transfer-learning)
- **Surgical fine-tuning improves adaptation to distribution shifts (ICLR'23).** [[Paper]](https://openreview.net/forum?id=APuPRxjHvZ)

#### 6.2. Prediction-Level Uncertainty

- **Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift (NeurIPS'19).** [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- **Transferable calibration with lower bias and variance in domain adaptation (NeurIPS'20).** [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html)
- **Unlabelled data improves Bayesian uncertainty calibration under covariate shift (ICML'20).** [[Paper]](http://proceedings.mlr.press/v119/chan20a.html?ref=https://githubhelp.com)
- **Calibrated prediction with covariate shift via unsupervised domain adaptation (AISTATS'20).** [[Paper]](http://proceedings.mlr.press/v108/park20b.html)
