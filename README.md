# MM-PoE

[![codecov](https://codecov.io/gh/souradipp76/MM-PoE/branch/main/graph/badge.svg?token=MM-PoE_token_here)](https://codecov.io/gh/souradipp76/MM-PoE)
[![CI](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml)


**Multiple Choice Reasoning tool via. Process of Elimination using Multi-Modal models**


## What is MM-PoE?

**Statement of Need**

Language models (LMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. Same is true in case of visual question answering tasks with multiple choices. This discrepancy can limit the effectiveness of vision language models in accurately solving such tasks. To address this, we introduce Multi-Modal Process of Elimination (MM-PoE), a two-step scoring method designed to enhance VLM performance by mimicking human reasoning strategies in multi-modal settings. 

In the first step, the method evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the VLM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across three datasets demonstrate MM-PoE's effectiveness, particularly excelling in logical reasoning scenarios . Additionally, MM-PoE proves adaptable to few-shot settings and is compatible with large language models (LLMs) like ChatGPT.

By implementing MM-PoE, researchers and practitioners can experiment and significantly improve the accuracy and reliability of VLMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning models for visual reasoning.

## Installing MM-POE

### Install it from PyPI

The simplest way to install MM-PoE and its dependencies is from PyPI with pip, Python's preferred package installer.

```bash
$ pip install mm_poe
```

In order to upgrade MM-POE to the latest version, use pip as follows.

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

## Usage

Here is a typical example usage of MM-PoE.

### Running CLI

```bash
$ python -m mm_poe
#or
$ mm_poe
```

### Running Experiments

```bash
$ bash 7_main_exp.sh
#or
$ bash 9_mask_vqa.sh
#or
$ bash 11_few_shot_vqa.sh
```

## Contributing

MM-POE is an open-source project that is supported by a community who will gratefully and humbly accept any contributions you might make to the project.

If you are interested in contributing, read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

- Submit a bug report or feature request on [GitHub Issues](https://github.com/souradipp76/MM-PoE/issues).
- Add to the documentation or help with our website.
- Write unit or integration tests for our project.
- Answer questions on our issues, mailing list, Stack Overflow, and elsewhere.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved, and we would be very happy for you to join us!

## License

Read the [LICENSE](LICENSE) file.
