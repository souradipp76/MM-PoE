---
title: 'MM-POE'
tags:
  - machine learning
  - Large Language Models
  - Multi-modal
  - python
  - Multiple Choice Question Answering
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

**Statement of Need**

Language models (LMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. This discrepancy can limit the effectiveness of LMs in accurately solving such tasks. To address this, we introduce the Process of Elimination (POE), a two-step scoring method designed to enhance LM performance by mimicking human reasoning strategies. 

In the first step, POE evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the LM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across eight reasoning tasks demonstrate POE's effectiveness, particularly excelling in logical reasoning scenarios. Additionally, POE proves adaptable to few-shot settings and is compatible with large language models (LLMs) like ChatGPT.

By implementing POE, researchers and practitioners can significantly improve the accuracy and reliability of LMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning model selection and evaluation.

Implemented in Python, the Yellowbrick visualization package achieves steering by extending both scikit-learn [@sklearn] and Matplotlib [@matplotlib]. Like Yellowbrick, both scikit-learn and Matplotlib are extensions of SciPy [@scipy], libraries intended to facilitate scientific computing. Scikit-learn provides a generalized API for machine learning by exposing the concept of an `Estimator`, an object that learns from data. Yellowbrick in turn extends this concept with the idea of a `Visualizer`, an object that both learns from data and visualizes the result. Visualizers wrap Matplotlib procedures to produce publication-ready figures and rich visual analytics.


![Feature Analysis](figures/feature_analysis.png)

Because “more data beats better algorithms” [@rajaraman2008more], the first step to creating valid, predictive models is to find the minimum set of features that predicts the dependent variable. Generally, this means finding features that describe data in high dimensional space that are *separable* (i.e., by a hyperplane). Tools like `RadViz`, `ParallelCoordinates`, and `Manifold` help visualize high dimensional data for quick diagnostics. Bayesian models and regressions suffer when independent variables are collinear (i.e., exhibit pairwise correlation). `Rank2D` visualizations show pairwise correlations among features and can facilitate feature elimination.

**State of the field: Do the authors describe how this software compares to other commonly-used packages?**

# Acknowledgements

We have used the server provided by Northwestern University for building this software.
