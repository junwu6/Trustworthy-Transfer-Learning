# Trustworthy Transfer Learning

The KDD 2023 tutorial titled "Trustworthy Transfer Learning: Transferability and Trustworthiness" can be found on the [[official site]](https://sites.google.com/view/kdd23-trustworthy-transfer).

## ABSTRACT

Deep transfer learning investigates the transfer of knowledge or information from a source domain to a relevant target domain via deep neural networks. In this tutorial, we dive into understanding deep transfer learning from the perspective of knowledge transferability and trustworthiness (e.g., privacy, robustness, fairness, transparency, etc.). To this end, we provide a comprehensive review of state-of-the-art theoretical analysis and algorithms for deep transfer learning. To be specific, we start by introducing the concepts of transferability and trustworthiness in the context of deep transfer learning. Then we summarize recent theories and algorithms for understanding knowledge transferability from two aspects: (1) IID transferability: the samples within each domain are independent and identically distributed (e.g., individual images), and (2) non-IID transferability: The samples within each domain are interdependent (e.g., connected nodes within a graph). In addition to knowledge transferability, we also review the impact of trustworthiness on deep transfer learning, e.g., whether the transferred knowledge is adversarially robust or algorithmically fair, how to transfer the knowledge under privacy-preserving constraints, etc. Finally, we highlight the open questions and future directions for understanding deep transfer learning in real-world applications. We believe this tutorial can benefit researchers and practitioners by rethinking the trade-off between knowledge transferability and trustworthiness in developing trustworthy transfer learning systems.

## OUTLINE

- Introduction

- ** Part I: Knowledge Transferability

Part I-1: IID Transferability

Domain Discrepancy (Data-level)

Task Diversity (Task-level)

Transferability Measures (Model-level)

Part I-2: Non-IID Transferability

Graph Transferability

Text Transferability

Time-series Transferability

Part II: Knowledge Trustworthiness

Part II-1: Adversarial Robustness

Attack

Defense

Part II-2: Privacy

Source-free Hypothesis Transfer

Federated Learning

Part II-3: Fairness

Individual Fairness

Group Fairness

Part II-4: Transparency and Explainability

Feature-level Explainability

Prediction-level Uncertainty

Part III: Open Questions and Future Trends

Multi-dimensional Distribution Shifts

Universal Trustworthiness

Tradeoff between Transferability and Trustworthiness
