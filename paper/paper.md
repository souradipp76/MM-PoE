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

# Acknowledgements

We have used the server provided by Northwestern University for building this software.
