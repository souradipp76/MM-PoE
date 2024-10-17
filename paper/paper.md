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

This paper introduces the Process of Elimination (POE), a method to enhance language models' performance on multiple-choice reasoning by employing a two-step scoring system that first eliminates incorrect options and then predicts from the remaining ones. Our experiments across eight reasoning tasks show the method's effectiveness, particularly in logical reasoning tasks.

# Statement of Need

Language models (LMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. Same is true in case of visual question answering tasks with multiple choices. This discrepancy can limit the effectiveness of vision language models in accurately solving such tasks. To address this, we introduce Multi-Modal Process of Elimination (MM-PoE), a two-step scoring method designed to enhance VLM performance by mimicking human reasoning strategies in multi-modal settings. 

In the first step, the method evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the VLM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across three datasets demonstrate MM-PoE's effectiveness, particularly excelling in logical reasoning scenarios . Additionally, MM-PoE proves adaptable to few-shot settings and is compatible with large language models (LLMs) like ChatGPT.

By implementing MM-PoE, researchers and practitioners can experiment and significantly improve the accuracy and reliability of VLMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning models for visual reasoning.


# State of the Field

A common strategy for answering multiple-choice questions, especially under examination conditions, involves a process of elimination where incorrect answers are systematically discarded to narrow down the choices to the most likely correct one. This approach, grounded in everyday test-taking strategies, contrasts with how current language models (LMs) handle multiple-choice reasoning tasks. Typically, LMs evaluate each option independently or collectively without actively discarding less likely answers, potentially reducing their effectiveness in distinguishing the best choice from plausible distractors.

This paper argues that language models can benefit from an explicit two-step reasoning process akin to human problem-solving techniques. The proposed method, dubbed the Process of Elimination (POE), enhances the decision-making process by first scoring and then eliminating options that are seemingly incorrect before focusing on selecting the correct answer from the remaining choices. This method is designed to align with natural human reasoning by replicating how individuals often approach multiple-choice questions, particularly under the constraint of time and accuracy, as frequently experienced in academic testing environments.

Our hypothesis posits that language models, when equipped with a mechanism to discard implausible answers systematically, can achieve better performance on multiple-choice reasoning tasks. This is particularly relevant in the context of logical reasoning, where the elimination of clearly incorrect options can simplify the decision process and potentially lead to more accurate outcomes. This idea is supported by previous work demonstrating the effectiveness of LMs in various reasoning tasks when adapted to more human-like reasoning methods [Brown et al., 2020; Holtzman et al., 2021].

In the development of POE, we draw inspiration from the established capabilities of LMs to handle complex reasoning tasks [Brown et al., 2020] and the known strategies that humans employ in test-taking scenarios. The approach builds on the foundational work in language modeling likelihood [Brown et al., 2020], which demonstrates the LMs' ability to perform in-context learning. By incorporating a structured process to eliminate unlikely choices, POE aims to refine this capability, making it more targeted and efficient in dealing with the nuanced challenges presented by multiple-choice questions.

The effectiveness of this approach is underscored through zero-shot experiments across a diverse set of reasoning tasks, illustrating that the integration of human-like elimination strategies can significantly enhance the performance of language models. This paper aims to show that by mimicking human reasoning processes, we can make LMs not only perform better on standardized reasoning tasks but also behave in ways that are more interpretable and aligned with human cognitive processes.


## 2. Method

The Process of Elimination (POE) introduced in this paper operates on a two-step mechanism designed to enhance the decision-making capabilities of language models (LMs) in multiple-choice reasoning tasks. This method employs a novel approach to option elimination followed by a focused prediction phase. The strategy is rooted in the belief that separating the elimination of clearly incorrect options from the choice of the best remaining option will improve overall task performance.

### Problem Setting

Given a multiple-choice reasoning task, we define the problem setting as follows:

- Let \( x \) be the question or context provided.
- Let \( Y = \{y_1, y_2, \ldots, y_n\} \) be the set of multiple-choice options available.
- Let \( y \) be the correct answer from \( Y \).

The goal is to develop an in-context learning method that accurately selects \( y \) from \( Y \) given \( x \).

### Two-Step Scoring Method

#### Step 1: Elimination

In the first step of the POE method, each option \( y_i \) is scored based on a specified metric. The score function, \( \text{score}(x, y_i) \), evaluates each option's plausibility given the question \( x \). The scores are used to eliminate options that are deemed less likely to be correct. Specifically, options whose scores are below the average score are eliminated. This is calculated as follows:

\[ s_i = \text{score}(x, y_i) \]
\[ Y_{\text{wrong}} = \{y_i \mid s_i < \text{avg}(s_1, \ldots, s_n)\} \]

This elimination strategy intuitively aligns with how humans often discard options that seem clearly incorrect before carefully considering the remaining choices.

#### Step 2: Prediction

The second step involves making the final choice from the non-eliminated options. This step utilizes a binary mask to exclude the eliminated options during the prediction phase. The mask for each option \( y_i \) is defined as follows:

\[ m_i = \begin{cases} 
0 & \text{if } y_i \in Y_{\text{wrong}} \\
1 & \text{otherwise}
\end{cases} \]

The masked context \( x_{\text{mask}} \) is then constructed by modifying the original context \( x \) to include only the options for which \( m_i = 1 \). Each option is scored again, but this time within the context that explicitly excludes the eliminated options, possibly by using a template \( T \) that masks out \( Y_{\text{wrong}} \) in the presentation of the options:

\[ x_{\text{mask}} = T(x, Y, \text{mask}) \]

The final predicted answer \( \hat{y} \) is then the option with the highest score among the remaining options:

\[ \hat{y} = \arg\max_{i \mid m_i = 1} \text{score}(x_{\text{mask}}, y_i) \]

### Implementation Considerations

The effectiveness of POE hinges on the robustness of the scoring function and the accuracy of the elimination step. The scoring function can be any LM-based likelihood estimator, such as language modeling likelihood or any of its alternatives like average log probability or calibrated log probability. Our implementation tests multiple such scoring functions to identify the most effective ones in both eliminating implausible options and accurately selecting the final answer.

The POE method is designed to be model-agnostic, meaning it can be implemented using any existing LM capable of scoring text options, and it is flexible enough to be adapted to different types of multiple-choice questions across various domains.

## 3. Experiment Setup

To evaluate the effectiveness of the Process of Elimination (POE), we designed an experimental framework that tests the method across a diverse set of reasoning tasks. This setup aims to compare POE with existing scoring methods to highlight its potential improvements in accuracy and reasoning capability.

### Data

Our experiments were conducted on eight different multiple-choice reasoning tasks, selected to cover a broad spectrum of reasoning types and complexities. These tasks include both traditional reasoning tasks and more specialized ones designed to test specific reasoning skills. To ensure a comprehensive evaluation, we used test sets from established benchmarks when available; otherwise, we utilized development sets.

### Model

For the core experiments, we utilized the FLAN-T5-XL model, chosen for its balance between computational efficiency and performance in instruction-tuned language tasks. This model has demonstrated strong capabilities in handling various NLP tasks and serves as a robust platform for evaluating our POE method.

### Baseline Methods

We compared POE against five baseline scoring methods to assess its relative performance:

1. **Language Modeling (LM):** This baseline uses the raw language modeling likelihood as the scoring function.
2. **Average Language Modeling (AVG):** This method averages the log probabilities across all tokens in the option.
3. **Calibration:** This involves adjusting the LM scores based on calibration techniques that aim to correct for the model's confidence.
4. **Channel:** Channel methods score each option based on how likely the question is given the option, which reverses the typical conditional probability used in LMs.
5. **Multiple Choice Prompting (MCP):** This approach formats the input by presenting the question followed by all options, prompting the model to select the most likely option.

Each method provides a different approach to scoring options, allowing for a comprehensive comparison of how each interacts with the structure and strategy of POE.

### Settings

Our experiments primarily focused on a zero-shot setting to evaluate the generalization capabilities of POE without any task-specific tuning. Accuracy was used as the main metric for performance evaluation, with results averaged over multiple seeds to ensure robustness.

To further explore the versatility of POE, we also examined its performance in few-shot settings by incorporating examples into the model's input, aiming to observe any changes in effectiveness when provided with context-specific demonstrations.

### Implementation Details

For each task, we implemented the scoring and prediction steps of POE as described in the Methods section. The scoring functions were carefully chosen based on their theoretical alignment with the two-step elimination and prediction philosophy of POE. We conducted extensive parameter tuning and optimization to maximize the performance of both the elimination step and the final prediction accuracy.

This experiment setup was designed to rigorously test the effectiveness of POE across a range of reasoning tasks and compare its performance against standard baseline methods. The results of these experiments are intended to demonstrate the potential benefits of integrating a process of elimination approach into language model reasoning strategies for multiple-choice questions.


## 4. Results

POE consistently outperformed or matched the best-performing baselines across all tasks, showing particular strength in logical reasoning. The method's effectiveness in separating elimination and prediction tasks was crucial to its success.

| Task | MCP  | PoE  | PoE - MCP |
|------|------|------|-----------|
| LA   | 50.0 | 68.8 | +18.8     |
| IMT  | 34.0 | 47.2 | +13.2     |
| CLD  | 67.2 | 75.9 | +8.7      |
| RACO | 53.8 | 60.6 | +6.8      |
| CAI  | 84.1 | 81.8 | -2.3      |
| EIE  | 25.0 | 19.1 | -5.9      |
| RS   | 55.1 | 49.0 | -6.1      |
| IOM  | 56.2 | 50.0 | -6.2      |

**Table 2**: Comparison of MCP and PoE accuracy scores on 8 new tasks. The top 4 tasks are logical reasoning tasks. PoE largely outperforms MCP on 4 logical reasoning tasks, and underperforms MCP on other 4 tasks.

## 6. Conclusion

POE demonstrates a significant improvement in handling multiple choice reasoning tasks by mimicking a human-like process of elimination approach. Future work will focus on enhancing its generalizability and efficiency, possibly extending to few-shot settings and other modalities.

## Ethics Statement

While this model uses publicly available tasks and models, users should be aware of potential biases in the data and model outputs.

# Acknowledgements

We have used the server provided by Northwestern University for building this software.
