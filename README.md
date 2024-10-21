# MM-PoE

[![codecov](https://codecov.io/gh/souradipp76/MM-PoE/branch/main/graph/badge.svg?token=MM-PoE_token_here)](https://codecov.io/gh/souradipp76/MM-PoE)
[![CI](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml)


**Multiple Choice Reasoning via. Process of Elimination using Multi-Modal Models**


## What is MM-PoE?

Multi-Modal Process of Elimination (MM-PoE) is a method to enhance vision language models' performance on multiple-choice visual reasoning by employing a two-step scoring system that first eliminates incorrect options and then predicts from the remaining ones. Our experiments across three question answering datasets show the method's effectiveness, particularly in visual reasoning tasks.

**Statement of Need**

Large Language models (LLMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. Same is true for vision language models (VLMs) in case of visual question answering tasks with multiple choices. This discrepancy can limit the effectiveness of vision language models in accurately solving such tasks. To address this, we introduce Multi-Modal Process of Elimination (MM-PoE), a two-step scoring method designed to enhance VLM performance by mimicking human reasoning strategies in multi-modal settings.

In the first step, the method evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the VLM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across three datasets demonstrate MM-PoE's effectiveness, particularly excelling in logical reasoning scenarios. Additionally, MM-PoE proves adaptable to few-shot settings and is compatible with the current state-of-the-art vision language models (VLMs).

By implementing MM-PoE, researchers and practitioners can experiment and significantly improve the accuracy and reliability of VLMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning models for visual reasoning.

## Installing MM-PoE

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

The application will prompt the user to provide relevant inputs for a multiple choice question e.g. a question, multiple answer choices for the question and the path to the image relevant the question context. Once the inputs are provided, the predicted answer will be displayed based prompt outputs. Note that this application runs inference for only a single sample at a time.


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
$ bash 7_main_exp.sh
#or
$ bash 9_mask_vqa.sh
#or
$ bash 11_few_shot_vqa.sh
```

The numbers 7, 9 and 11 in the script names correspnds to the experiments related to MM-PoE using VLMs for multiple-choice visual question answering tasks and rest corresponds to experiments related to PoE using LLMs on logical resoning tasks. The results will be saved in `results/`.

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
