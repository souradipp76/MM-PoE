---
title: 'MM-PoE: Multiple Choice Reasoning via. Process of Elimination using Multi-Modal models'
tags:
  - machine learning
  - large language models
  - multi-modal
  - python
  - multiple choice reasoning
  - visual question answering
authors:
  - name: Sayak Chakrabarty
    affiliation: 1
  - name: Souradip Pal
    orcid: 0000-0002-5781-3032
    affiliation: 2
affiliations:
 - name: Northwestern University
   index: 1
 - name: Purdue University
   index: 2
date: 16 October 2024
bibliography: paper.bib
---

# Summary

# Statement of Need

Language models (LMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. Same is true in case of visual question answering tasks with multiple choices. This discrepancy can limit the effectiveness of vision language models in accurately solving such tasks. To address this, we introduce Multi-Modal Process of Elimination (MM-PoE), a two-step scoring method designed to enhance VLM performance by mimicking human reasoning strategies in multi-modal settings. 

In the first step, the method evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the VLM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across three datasets demonstrate MM-PoE's effectiveness, particularly excelling in logical reasoning scenarios . Additionally, MM-PoE proves adaptable to few-shot settings and is compatible with large language models (LLMs) like ChatGPT.

By implementing MM-PoE, researchers and practitioners can experiment and significantly improve the accuracy and reliability of VLMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning models for visual reasoning.


# State of the Field
Do the authors describe how this software compares to other commonly-used packages?

## Abstract

This paper introduces the Process of Elimination (POE), a method to enhance language models' performance on multiple-choice reasoning by employing a two-step scoring system that first eliminates incorrect options and then predicts from the remaining ones. Our experiments across eight reasoning tasks show the method's effectiveness, particularly in logical reasoning tasks.

## 1. Introduction

Humans typically approach multiple-choice questions by eliminating wrong answers before selecting the correct one. We hypothesize that a similar approach can improve language model (LM) performance on these tasks. Our method, POE, adopts this two-step elimination and prediction strategy, showing promise in preliminary zero-shot experiments across various reasoning tasks [Brown et al., 2020].

## 2. Method

The POE operates in two phases:
1. **Elimination**: Score each option and eliminate those below the average score.
2. **Prediction**: Use a binary mask to ignore eliminated options and predict from the remaining ones.

This method leverages the existing capabilities of LMs in scoring options and enhances decision-making by focusing only on plausible answers.

## 3. Experiment Setup

We evaluated POE on eight diverse reasoning tasks using FLAN-T5-XL and compared it against five baseline scoring methods. Accuracy was the primary metric for evaluation.

## 4. Results

POE consistently outperformed or matched the best-performing baselines across all tasks, showing particular strength in logical reasoning. The method's effectiveness in separating elimination and prediction tasks was crucial to its success.

## 5. Analysis

Further analysis revealed that POE's strengths lie particularly in tasks requiring logical reasoning. It effectively applies a masking strategy to focus the model's attention on likely correct options, improving both interpretability and factual adherence.

## 6. Conclusion

POE demonstrates a significant improvement in handling multiple choice reasoning tasks by mimicking a human-like process of elimination approach. Future work will focus on enhancing its generalizability and efficiency, possibly extending to few-shot settings and other modalities.

## Limitations

The current implementation of POE does not completely disregard eliminated options, potentially limiting its effectiveness. Optimizing the prompt and testing in few-shot scenarios remain areas for future improvement.

## Ethics Statement

While this model uses publicly available tasks and models, users should be aware of potential biases in the data and model outputs.

# Acknowledgements

We have used the server provided by Northwestern University for building this software.
