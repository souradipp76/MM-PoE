# MM-PoE

[![codecov](https://codecov.io/gh/souradipp76/MM-PoE/branch/main/graph/badge.svg?token=MM-PoE_token_here)](https://codecov.io/gh/souradipp76/MM-PoE)
[![CI](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml/badge.svg)](https://github.com/souradipp76/MM-PoE/actions/workflows/main.yml)



## License

Read the [LICENSE](LICENSE) file.

**Visual analysis and diagnostic tools to facilitate machine learning model selection.**


## What is MM-POE? Statement of Need

**Statement of Need**

Language models (LMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. This discrepancy can limit the effectiveness of LMs in accurately solving such tasks. To address this, we introduce the Process of Elimination (POE), a two-step scoring method designed to enhance LM performance by mimicking human reasoning strategies. 

In the first step, POE evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the LM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across eight reasoning tasks demonstrate POE's effectiveness, particularly excelling in logical reasoning scenarios. Additionally, POE proves adaptable to few-shot settings and is compatible with large language models (LLMs) like ChatGPT.

By implementing POE, researchers and practitioners can significantly improve the accuracy and reliability of LMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning model selection and evaluation.

## Installing MM-POE

### Install it from PyPI

```bash
pip install mm_poe
```
### Install it from source

```bash
$ git clone https://github.com/souradipp76/MM-PoE.git
$ cd MM-PoE
$ make install
```

In order to upgrade MM-POE to the latest version, use pip as follows.

```bash
$ pip install -U mm_poe
```

## Using Yellowbrick

Here is a typical example usage of MM-POE:

### Running the CLI

```bash
$ python -m mm_poe
#or
$ mm_poe
```

## Contributing to MM-POE

MM-POE is an open-source project that is supported by a community who will gratefully and humbly accept any contributions you might make to the project.

If you are interested in contributing, read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

- Submit a bug report or feature request on [GitHub Issues](https://github.com/souradipp76/MM-PoE/issues).
- Add to the documentation or help with our website.
- Write [unit or integration tests]() for our project.
- Answer questions on our issues, mailing list, Stack Overflow, and elsewhere.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved, and we would be very happy for you to join us!
