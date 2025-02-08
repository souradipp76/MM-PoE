# MM-PoE

[![codecov](https://codecov.io/gh/souradipp76/MM-PoE/branch/main/graph/badge.svg?token=77bb4fa8-804f-4498-a040-4d93886f32be)](https://codecov.io/gh/souradipp76/MM-PoE)
[![CI](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml)


**Multiple Choice Reasoning via. Process of Elimination using Multi-Modal Models**


## What is MM-PoE?

Multi-Modal Process of Elimination (MM-PoE) is a method to enhance vision language models' performance on multiple-choice visual reasoning by employing a two-step scoring system that first eliminates incorrect options and then predicts from the remaining ones. Our experiments across three question answering datasets show the method's effectiveness, particularly in visual reasoning tasks.

Large Language models (LLMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. Same is true for vision language models (VLMs) in case of visual question answering tasks with multiple choices. This discrepancy can limit the effectiveness of vision language models in accurately solving such tasks. To address this, we introduce Multi-Modal Process of Elimination (MM-PoE), a two-step scoring method designed to enhance VLM performance by mimicking human reasoning strategies in multi-modal settings.

In the first step, the method evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the VLM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across three datasets demonstrate MM-PoE's effectiveness, particularly excelling in logical reasoning scenarios. Additionally, MM-PoE proves adaptable to few-shot settings and is compatible with the current state-of-the-art vision language models (VLMs).

Using this tool, researchers and practitioners can experiment and significantly improve the accuracy and reliability of VLMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning models for visual reasoning.



## Installation

MM-PoE is available only on Linux/Windows. CUDA-compatible hardware is required to run the tool.

### Install it from PyPI

The simplest way to install MM-PoE and its dependencies is from PyPI with pip, Python's preferred package installer.

```bash
$ pip install mm_poe
```

In order to upgrade MM-PoE to the latest version, use pip as follows.

```bash
$ pip install -U mm_poe
```

### Install it from source

You can also install MM-PoE from source as follows.

```bash
$ git clone https://github.com/souradipp76/MM-PoE.git
$ cd MM-PoE
$ make install
```

To create a virtual environment before installing MM-PoE, you can use the command:
```bash
$ make virtualenv
$ source .venv/bin/activate
```

## Usage

Here are some typical examples of using MM-PoE.

### Running CLI

To run the CLI application, execute the following.

```bash
$ python -m mm_poe
#or
$ mm_poe
```

The application will prompt the user to provide relevant inputs for a multiple choice question e.g. a question, multiple answer choices for the question and the path to the image relevant to the question context. Once the inputs are provided, the predicted answer will be displayed based prompt outputs. Note that this application runs inference for only a single sample at a time.


<img src="paper/figures/cli.png" alt="Example" width="500">

### Running Experiments

For running experiments with MM-PoE on large datasets, follow the instructions below.

Install dependencies:
```
make install
```
Download datasets using the script in `data/`:
```
$ cd mm_poe
$ bash data/data_downloaders.sh
```
Download models using the script in `models/model_downloaders/`:
```
$ cd mm_poe
$ bash models/model_downloaders/model_downloaders.sh
```

Run scripts in `methods/` to get the results:

```bash
$ cd methods
$ bash 7_main_exp_vqa.sh
#or
$ bash 9_mask_vqa.sh
#or
$ bash 11_few_shot_vqa.sh
```

The numbers 7, 9 and 11 in the script names correspnds to the experiments related to MM-PoE using VLMs for multiple-choice visual question answering tasks and rest corresponds to experiments related to PoE using LLMs on logical resoning tasks. The results will be saved in `results/`. 

Alternatively, run the notebook `scripts/experiments.ipynb` on Google Colab: <a src="https://colab.research.google.com/assets/colab-badge.svg" href="https://colab.research.google.com/github/souradipp76/MM-PoE/blob/main/scripts/experiments.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

MM-PoE is compared against the following five baseline scoring methods to assess its relative performance:

- **Language Modeling (LM)**: This baseline uses the raw vision language modeling likelihood as the scoring function.
- **Average Language Modeling (AVG)**: This method averages the log probabilities across all tokens in the option.
- **Calibration**: This involves adjusting the VLM scores based on calibration techniques that aim to correct for the model's confidence.
- **Channel**: Channel methods score each option based on how likely the question is given the option, which reverses the typical conditional probability used in VLMs.
- **Multiple Choice Prompting (MCP)**: This approach formats the input by presenting the question followed by all options, prompting the model to select the most likely option.

| **Method**                | **Score**                                      | **Input**                                           | **Output** |
|---------------------------|-----------------------------------------------|-----------------------------------------------------|------------|
| LM                        | $\log P (y_i\|x,h)$                           | $x, \langle h \rangle$ the answer is:              | $y_1$      |
| AVG                       | $1/len(y_i)*\log P(y_i\|x,h)$                  | $x, \langle h \rangle$ the answer is:              | $y_1$      |
| Calibration               | $\log P(y_i\|x,h)$                            | $x, \langle h \rangle$ the answer is:              | $y_1$      |
| Channel                   | $\log P(x\|y_i,h)$                            | $y_1, \langle h \rangle$                           | $x$ the answer is: |
| MCP                       | $\log P (y_i\|x,h, Y )$                       | Question: $x$, Image: $\langle h \rangle$ <br>A. $y_1$ <br>B. $y_2$ <br>C. $y_3$ <br>Answer:      | A          |
| MM-POE: Elimination       | $\log P (y_i\|x,h, Y )$                       | Question: $x$, Image: $\langle h \rangle$ <br>A. $y_1$ <br>B. $y_2$ <br>C. $y_3$ <br>Answer:      | A          |
| MM-POE: Prediction        | $\log P (y_i\|x_{mask},h, Y \setminus Y_{wrong})$ | Select the most suitable option to answer the question. <br>Ignore [MASK] options. <br> Question: $x$, Image: $\langle h \rangle$ <br> A. [MASK] <br>B. $y_2$ <br>C. $y_3$ <br>Answer:            | B          |
|                           |                                               |                           |            |


Below are the details about the scripts running the experiments. By default, the experiments use `microsoft/git-base-vqav2` model and runs on all the supported datasets(`VQA`, `ScienceQA` and `AI2D`) for 5 random seeds. 
- `7_main_exp_vqa.sh` - Evaluates the accuracy for the scoring methods `LM`, `AVG`, `Calibration`, `Channel`, `MCP` and `MM-POE` in zero-shot settings. `MM-POE` uses `MCP` for `MM-POE Elimination` phase. 
- `9_mask_vqa.sh` - Evaluates the accuracy for the scoring method `MM-POE` with all `LM`, `AVG`, `Calibration`, `Channel` and `MCP` for `MM-POE Elimination` phase in zero-shot settings. It also uses two masking strategies `lowest` and `below_average` to mask the options.
- `11_few_shot_vqa.sh` - Evaluates the accuracy for the scoring method `MCP` and `MM-POE` with `MCP` in the `MM-POE Elimination` phase using `lowest` masking strategy for the $n$-shot case with $n = 0,1,3$.

### Supported Datasets
- `VQA(v1)` - https://visualqa.org
- `ScienceQA` - https://scienceqa.github.io
- `AI2D` - https://prior.allenai.org/projects/diagram-understanding

| **Dataset**  | **#Options** | **Train** | **Dev** | **Test** |
|-------------|-------------|----------|--------|--------|
| VQA         | 18          | 248,349  | 121,512| 244,302|
| ScienceQA   | 4           | 12,726   | 4,241  | 4,241  |
| AI2D        | 4           | 3,921    | 982    | -      |

**Using Custom Dataset**

To use your own custom dataset, save your data under `mm_poe/data/custom_dataset`. The dataset should contain a `mm_poe/data/custom_dataset/questions.json` file in the following format given below. All the images should be under `mm_poe/data/custom_dataset/images` directory. While running the experiements, set the argument as `datasets="custom_dataset"` in the scripts. Check out the example custom dataset under the `mm_poe/data/custom_datasets` directory.

*Questions file format:*
```json
{
   "COCO_train2014_000000000025": {
        "question": "What is the capital of France?",
        "choices": ["Paris", "London", "Berlin", "Madrid"],
        "answer": 0,
        "image": "COCO_train2014_000000000025.jpg"
    },
    "COCO_train2014_000000000026": {
        ...
        ...
    }
}
```

### Supported Models
- `BLIP2-OPT`
  - `Salesforce/blip2-opt-2.7b` - https://huggingface.co/Salesforce/blip2-opt-2.7b
- `BLIP2-T5`
  - `Salesforce/blip2-flan-t5-xl` - https://huggingface.co/Salesforce/blip2-flan-t5-xl
- `InstructBLIP`
  - `Salesforce/instructblip-vicuna-7b` - https://huggingface.co/Salesforce/instructblip-vicuna-7b
- `GIT`
  - `microsoft/git-base-vqav2` - https://huggingface.co/microsoft/git-base-vqav2
  - `microsoft/git-base-textvqa` - https://huggingface.co/microsoft/git-base-textvqa
- `PaliGemma`
  - `google/paligemma-3b-ft-science-qa-448` - https://huggingface.co/google/paligemma-3b-ft-science-qa-448
  - `google/paligemma-3b-ft-vqav2-448` - https://huggingface.co/google/paligemma-3b-ft-vqav2-448
  - `google/paligemma-3b-ft-ai2d-448` - https://huggingface.co/google/paligemma-3b-ft-ai2d-448
- `Idefics2`
  - `HuggingFaceM4/idefics2-8b` - https://huggingface.co/HuggingFaceM4/idefics2-8b

Any of the above models can be selected by changing the `model_family` and `checkpoints` argument in the `models/model_downloaders/model_downloaders.sh` script for downloading the model and in the three scripts for the experiments. Note that appropriate `loading_precision` needs to set based on the model and hardware used.

## Contributing

MM-PoE is an open-source project that is supported by a community who will gratefully and humbly accept any contributions you might make to the project.

If you are interested in contributing, read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

- Submit a bug report or feature request on [GitHub Issues](https://github.com/souradipp76/MM-PoE/issues).
- Add to the documentation or help with our website.
- Write unit or integration tests for our project under the `tests` directory.
- Answer questions on our issues, mailing list, Stack Overflow, and elsewhere.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved, and we would be very happy for you to join us!

## License

Read the [LICENSE](LICENSE) file.
