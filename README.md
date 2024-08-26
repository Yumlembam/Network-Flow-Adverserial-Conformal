# Comprehensive Botnet Detection by Mitigating Adversarial Attacks

## Navigating the Subtleties of Perturbation Distances and Fortifying Predictions with Conformal Layers

Botnets are computer networks controlled by malicious actors that present significant cybersecurity challenges. These networks autonomously infect, propagate, and coordinate to conduct cybercrimes, making robust detection methods essential. This research addresses sophisticated adversarial manipulations posed by attackers, which aim to undermine machine learning-based botnet detection systems.

## Overview

We introduce a flow-based detection approach that leverages machine learning and deep learning algorithms, trained on the ISCX and ISOT datasets. Our detection algorithms are optimized using Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) to establish a baseline detection method.

### Key Contributions:

1. **Adversarial Attacks**: We utilize Carlini & Wagner (C&W) and Generative Adversarial Network (GAN) attacks to generate deceptive data. These attacks apply subtle perturbations to each feature used for classification while maintaining semantic and syntactic relationships, ensuring the adversarial samples retain their meaningfulness and realism.

2. **Perturbation Analysis**: An in-depth analysis is conducted on the required L2 distance from the original sample for a malware sample to be misclassified. This analysis is performed across various iteration checkpoints, revealing different levels of misclassification at varying L2 distances of the perturbed sample from the original.

3. **Model Vulnerability and Retraining**: We explore the vulnerability of various models, specifically examining the transferability of adversarial examples from a Neural Network surrogate model to Tree-based algorithms. Models that initially misclassified the perturbed samples are retrained to enhance their resilience and detection capabilities.

4. **Conformal Prediction Layer**: In the final phase, a conformal prediction layer is integrated into the models. This layer significantly improves the rejection of incorrect predictions — achieving a 58.20% rejection rate in the ISCX dataset and 98.94% in the ISOT dataset.

## Datasets

The datasets used in this study are essential for validating our proposed methodologies and are publicly available, ensuring transparency and reproducibility of our results. Specifically, we utilized the following datasets:

- **ISOT Botnet Dataset**: This dataset is publicly available and can be accessed through the following reference:
  - S. Saad, I. Traore, A. Ghorbani, B. Sayed, D. Zhao, W. Lu, J. Felix, P. Hakimian, "Detecting P2P botnets through network behavior analysis and machine learning," in *Proceedings of the 9th Annual Conference on Privacy, Security and Trust, PST2011*, 2011. [DOI: 10.1109/pst.2011.5971980](http://dx.doi.org/10.1109/pst.2011.5971980)

- **ISCX 2014 Botnet Dataset**: This dataset can be requested from the dataset owners:
  - E. Beigi, H. Jazi, N. Stakhanova, A. Ghorbani, "Towards effective feature selection in machine learning-based botnet detection approaches," in *Communications and Network Security (CNS), 2014 IEEE Conference on*, 2014. [DOI: 10.1109/cns.2014.6997492](http://dx.doi.org/10.1109/cns.2014.6997492)

## Repository Structure

- **FeatureSelection**: Contains the implementation of feature selection using Mutual Information.
- **HyperParameterOptimization**: Implements hyperparameter optimization using Particle Swarm Optimization (PSO) and Genetic Algorithm (GA).
- **Notebooks**
  - **Adversarial Attack**: Subfolder containing implementations of GAN and C&W attacks.
  - **ConformalPrediction**: Subfolder with implementations of Conformal Prediction and Conformal Threshold Optimization.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries can be installed PIP
