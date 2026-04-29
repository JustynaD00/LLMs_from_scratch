# Recreating LLMs from Scratch – Portfolio Project

This repository contains my recreated notebook work based on the open-source **LLMs-from-Scratch** repository by **Sebastian Raschka**.

The purpose of this project was to develop a deeper practical understanding of how large language models work by rebuilding the main coding stages step by step. Instead of only reading about LLMs or using existing models through APIs, I worked through the core implementation stages myself using **Python**, **PyTorch** and **Jupyter Notebooks**.

The original repository is the official code repository for the book **Build a Large Language Model (From Scratch)** by Sebastian Raschka. It explains how to develop, pretrain and finetune a GPT-style language model from the ground up.

**Original repository for further reading:**  
https://github.com/rasbt/LLMs-from-scratch

My version is a learning and portfolio-focused recreation of the main practical notebooks. I used the original repository as a guide, recreated the key code files, tested them in my own environment, fixed issues where needed, and added my own comments and reflections throughout the notebooks to show what I learned from each section.

---

## Project Overview

Large Language Models are often used as ready-made tools, but this project helped me understand what happens underneath. The work starts from basic text processing and builds up towards more advanced stages such as **attention mechanisms**, **GPT architecture**, **pretraining**, **classification finetuning** and **instruction finetuning**.

The project follows the main learning path of the original repository:

1. **Working with text data**
2. **Coding attention mechanisms**
3. **Implementing a GPT model from scratch**
4. **Pretraining on unlabeled data**
5. **Finetuning for text classification**
6. **Finetuning to follow instructions**

Each notebook represents one stage in the LLM development pipeline. Together, they show how raw text can be converted into tokens, passed through transformer-based architecture, trained using next-token prediction, and then adapted for more specific tasks.

---

## Project Aims

The main aims of this project were to:

- recreate the main practical coding chapters from the **LLMs-from-Scratch** repository
- understand how text is prepared for LLMs using **tokenisation** and **dataloaders**
- implement **attention mechanisms** and understand how they help models process context
- build a **GPT-style model architecture** using PyTorch
- understand how **pretraining** works using next-token prediction
- explore text generation strategies such as **temperature scaling** and **top-k sampling**
- load and use pretrained **GPT-2 weights**
- finetune a GPT model for **text classification**
- finetune a model to **follow instructions**
- document my learning through comments and reflections inside the notebooks
- create a clear GitHub portfolio project showing my practical understanding of LLMs

---

## Repository Contents

This repository contains the recreated notebook files only. The original project blogs and written portfolio updates are kept separately for assessment, while this repository focuses on the code and technical learning.

| Notebook | Chapter / Stage | Description |
|---|---|---|
| `01_working_with_text_data.ipynb` | **Working with Text Data** | Covers tokenisation, token IDs, text encoding, input-target pairs, sliding windows and dataloaders. |
| `02_coding_attention_mechanisms.ipynb` | **Coding Attention Mechanisms** | Covers self-attention, causal attention, attention masks and multi-head attention. |
| `03_implementing_gpt_model_from_scratch.ipynb` | **Implementing a GPT Model from Scratch** | Covers GPT architecture, embeddings, LayerNorm, GELU, feed-forward networks, shortcut connections and transformer blocks. |
| `04_pretraining_on_unlabeled_data.ipynb` | **Pretraining on Unlabeled Data** | Covers next-token prediction, training and validation loss, model training, text generation strategies and saving/loading weights. |
| `05_finetuning_for_text_classification.ipynb` | **Finetuning for Text Classification** | Covers adapting a pretrained GPT model for spam classification using supervised learning. |
| `06_finetuning_to_follow_instructions.ipynb` | **Finetuning to Follow Instructions** | Covers instruction dataset formatting, batching, instruction finetuning and response evaluation. |

---

## Suggested File Structure

The repository is intentionally kept simple and focused:

```text
LLMs_from_scratch/
│
├── README.md
│
├── 01_working_with_text_data.ipynb
├── 02_coding_attention_mechanisms.ipynb
├── 03_implementing_gpt_model_from_scratch.ipynb
├── 04_pretraining_on_unlabeled_data.ipynb
├── 05_finetuning_for_text_classification.ipynb
└── 06_finetuning_to_follow_instructions.ipynb
```

This structure keeps the project easy to navigate. The numbered notebooks show the learning journey in order, from basic text processing to instruction finetuning.

---

## Notebook Details

### 01 – Working with Text Data

This notebook focuses on the first stage of preparing text for a language model. I worked through how raw text is split into tokens and converted into token IDs that a model can process.

The notebook includes:

- **reading and preparing raw text**
- **basic tokenisation**
- **converting text into token IDs**
- **using the GPT-2 tokenizer**
- **creating input-target pairs**
- **using sliding windows for training examples**
- **creating dataloaders for batches**

This stage helped me understand that LLMs do not work with words directly. Text first has to be converted into numbers, and the way this is done affects how the model learns from the data.

---

### 02 – Coding Attention Mechanisms

This notebook focuses on **attention**, one of the most important ideas behind transformer models. I worked through how attention allows a model to focus on different parts of the input sequence when processing each token.

The notebook includes:

- **simple self-attention**
- **attention scores and attention weights**
- **Query, Key and Value vectors**
- **causal masking**
- **dropout in attention**
- **multi-head attention**

This stage helped me understand why attention is so powerful. Instead of treating every token separately, the model can learn which tokens are most relevant to each other. Multi-head attention also showed how the model can look at language from multiple perspectives at the same time.

---

### 03 – Implementing a GPT Model from Scratch

This notebook brings the earlier concepts together into a **GPT-style architecture**. I worked through the components that make up a GPT model and connected them into transformer blocks.

The notebook includes:

- **token embeddings**
- **positional embeddings**
- **Layer Normalization**
- **GELU activation**
- **feed-forward networks**
- **shortcut connections**
- **transformer blocks**
- **full GPT model architecture**
- **basic text generation using an untrained model**

This stage helped me understand that GPT is built from several smaller components working together. I also learned that an untrained GPT model can technically generate tokens, but the output will not be meaningful until the model has been trained.

While working on this chapter, I also dealt with import issues involving helper files such as `previous_chapters.py`. I fixed these by using package imports where possible or by adding the required classes directly into the notebook. This helped me understand the code more deeply because I could see how each component was implemented.

---

### 04 – Pretraining on Unlabeled Data

This notebook focuses on how a GPT-style model learns from raw text using **next-token prediction**. Instead of using manually labelled data, the model learns by predicting the next token in a sequence.

The notebook includes:

- **loading a text dataset**
- **splitting text into training and validation data**
- **calculating training and validation loss**
- **training a GPT model**
- **generating text during training**
- **plotting loss curves**
- **temperature scaling**
- **top-k sampling**
- **saving and loading model weights**
- **loading pretrained GPT-2 weights**

This stage helped me understand the pretraining process more clearly. I learned that LLMs can learn from unlabeled text because the target is naturally created by shifting the input sequence by one token.

I also learned that generation quality depends not only on the model itself, but also on the decoding strategy used. **Temperature** and **top-k sampling** showed how generated text can be made more predictable or more creative.

---

### 05 – Finetuning for Text Classification

This notebook focuses on adapting a pretrained GPT model for a supervised classification task. In this case, the model was finetuned to classify SMS messages as **spam** or **not spam**.

The notebook includes:

- **preparing a spam classification dataset**
- **balancing class labels**
- **tokenising and batching messages**
- **loading pretrained GPT-2 weights**
- **modifying the model for classification**
- **calculating classification loss and accuracy**
- **finetuning the model**
- **evaluating training, validation and test accuracy**
- **testing the model on new messages**

This stage helped me understand **transfer learning**. Instead of training a model from scratch, a pretrained GPT model can be adapted for a specific task using a smaller labelled dataset.

It also showed me that LLMs are not only useful for text generation. With the right output layer and training setup, they can also be used for decision-making tasks such as classification.

---

### 06 – Finetuning to Follow Instructions

This notebook focuses on **instruction finetuning**, where a pretrained language model is adapted to respond better to user instructions.

The notebook includes:

- **preparing instruction-response data**
- **formatting examples into instruction prompts**
- **batching and padding instruction data**
- **creating dataloaders**
- **loading a pretrained LLM**
- **finetuning the model on instruction data**
- **inspecting training and validation loss**
- **generating responses**
- **extracting model outputs**
- **qualitatively evaluating responses**

This final stage helped me understand how models can become more useful as assistant-style systems. The model is not just trained to continue text, but to respond to a given instruction in a relevant way.

This chapter also showed me that evaluating instruction-following models is more complex than simple accuracy. Loss values are useful, but the actual generated responses also need to be checked for quality, relevance and usefulness.

---

## Reflections Inside the Notebooks

A key part of my work was adding my own comments and reflections throughout the notebooks.

These reflections explain:

- **what each section of code does**
- **what I learned from the section**
- **why the concept is useful**
- **how the section connects to the wider LLM pipeline**
- **what issues I faced while running the code**
- **how I solved environment or import problems**

This was important because I wanted the notebooks to show more than just executed code. They also show my learning process and my understanding of each stage.

---

## Issues and Fixes

While recreating the notebooks, I faced some issues with missing helper files. Some original examples used imports such as:

```python
from previous_chapters import ...
```

or:

```python
from gpt_download import download_and_load_gpt2
```

In my environment, these files were not always available locally, which caused import errors.

To fix this, I used two approaches:

1. importing functions from the `llms_from_scratch` package where possible
2. adding the required classes and functions directly into the notebook

This helped me understand the code more clearly because I was able to see what the helper functions were doing instead of relying on hidden files.

---

## Key Learning Outcomes

By completing this project, I learned how the main stages of LLM development connect together.

The main learning outcomes were:

- **understanding how raw text is converted into tokens and token IDs**
- **learning how dataloaders prepare text data for model training**
- **understanding how self-attention and multi-head attention work**
- **learning how causal masks prevent a model from seeing future tokens**
- **understanding how GPT models are built from embeddings, transformer blocks and output layers**
- **learning why LayerNorm, GELU and shortcut connections are used**
- **understanding how pretraining works through next-token prediction**
- **learning how training and validation loss are used to evaluate progress**
- **exploring text generation strategies such as temperature scaling and top-k sampling**
- **understanding how pretrained weights can be loaded and reused**
- **learning how GPT models can be finetuned for classification**
- **understanding how instruction finetuning helps models follow prompts**
- **improving my debugging and code adaptation skills**

Overall, this project helped me move beyond using LLMs as black-box tools. I now have a clearer understanding of the technical pipeline behind how these models are built, trained and adapted.

---

## Technologies Used

The project uses:

- **Python**
- **PyTorch**
- **Jupyter Notebook**
- **tiktoken**
- **NumPy**
- **pandas**
- **matplotlib**
- **GPT-2 tokenizer**
- **pretrained GPT-2 weights**
- **Ollama for response evaluation in the instruction finetuning stage**

---

## How to Use This Repository

The notebooks are numbered in the order they should be read and run.

Recommended order:

1. `01_working_with_text_data.ipynb`
2. `02_coding_attention_mechanisms.ipynb`
3. `03_implementing_gpt_model_from_scratch.ipynb`
4. `04_pretraining_on_unlabeled_data.ipynb`
5. `05_finetuning_for_text_classification.ipynb`
6. `06_finetuning_to_follow_instructions.ipynb`

Some notebooks may require additional packages such as `tiktoken`, `torch`, `pandas` or `matplotlib`.

Example installation:

```bash
pip install torch tiktoken pandas numpy matplotlib
```

For chapters that use helper functions from the original repository, the `llms-from-scratch` package may also be useful:

```bash
pip install llms-from-scratch
```

---

## Hardware Notes

The recreated notebooks were mainly used for learning and experimentation. Some sections, such as loading pretrained GPT-2 weights or finetuning models, may take longer depending on the available hardware.

A GPU is helpful for training and finetuning, but several parts of the project can still be explored on CPU for educational purposes.

---

## Project Outcome

By completing this project, I gained a stronger practical understanding of how GPT-style large language models are built and adapted.

I started with basic text processing, then moved through attention mechanisms, GPT architecture, pretraining, classification finetuning and instruction finetuning. Each stage built on the previous one and helped me understand the full LLM development pipeline.

This repository is part of my AI portfolio and demonstrates my ability to:

- **follow and recreate complex technical material**
- **work with PyTorch and Jupyter Notebooks**
- **understand transformer-based language models**
- **debug code and fix environment issues**
- **document learning through comments and reflections**
- **connect theory with practical implementation**

---

## Credit and Further Reading

This project is based on the open-source repository:

**Sebastian Raschka – LLMs-from-Scratch**  
https://github.com/rasbt/LLMs-from-scratch

The original repository is the official code repository for the book:

**Build a Large Language Model (From Scratch)**  
by **Sebastian Raschka**

Further reading:

- **Original GitHub repository:** https://github.com/rasbt/LLMs-from-scratch
- **Book information:** https://www.manning.com/books/build-a-large-language-model-from-scratch

The original repository provides the main learning structure and source code. My version focuses on recreating the practical notebooks, adapting them to my own environment, adding comments and reflections, and documenting my learning as part of my portfolio.
