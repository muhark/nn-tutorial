---
title: "Demystifying Deep Learning"
subtitle: "An Introduction to Neural Networks for Social Scientists"
author: "Dr Musashi Jacobs-Harukawa, DDSS Princeton"
date: "23 Mar 2023"
mainfont: "IBM Plex Sans-Light"
aspectratio: 1610
theme: "white"
css: "minimal-theme.css"
---

# Introductions

- PoliSci/Data Science background
- Research on NLP, Interpretability:
	- Explanation and Validation for DL Language Models
	- Using LLMs for Corpus Description and Summary
	- How/when does domain-specific pre-training matter?

::: {.fragment}
Website: [`muhark.github.io`](https://muhark.github.io)
:::

# Motivation

## Who is this for?

:::.notes
- In short, me from a few years ago.
- Received training in quantitative methods, causal inference, some machine learning
- But deep learning always treated as a separate super-complicated thing
- Lots of resources for learning deep learning out there, but focus tends to be for (I assume) a CS audience.
- As quantitative social scientists, we're not seeking to avoid the math--on the contrary, we want to understand the tools and the resulting models. More on this in a bit.
:::

## What is Deep Learning?

- Approach to using computers to learn patterns in data
- based around a modular set of tools loosely inspired by biological neurons.
- Basis of many advanced systems seen today.

::: .notes
- "Machine learning" (it is in fact arguably a subfield)
:::

## Why learn about Deep Learning?

- _Incredibly_ powerful tool for modelling complex processes
	- Image processing (Computer Vision)
	- Language (Natural Language Processing)
- Increasingly applied in social sciences
	- Extracting visual features [(Torres and Cantú 2021)](https://www.cambridge.org/core/journals/political-analysis/article/learning-to-see-convolutional-neural-networks-for-the-analysis-of-social-science-data/7417AE7F021C92CA32DC04D4D4282B90); Multiple imputation [(Lall and Robinson 2021)](https://www.cambridge.org/core/journals/political-analysis/article/midas-touch-accurate-and-scalable-missingdata-imputation-with-deep-learning/5007854F57E88AF16D69BCCA4C5AF1FF); Multilingual embeddings [(Licht 2023)](https://www.cambridge.org/core/journals/political-analysis/article/crosslingual-classification-of-political-texts-using-multilingual-sentence-embeddings/30689C8798F097EEBA514ABE4891A71B); Automated coding of videos [(Tarr, Hwang and Imai 2023)](https://www.cambridge.org/core/journals/political-analysis/article/automated-coding-of-political-campaign-advertisement-videos-an-empirical-validation-study/7B60C86AAC9E71016F9397D2FD247F8C)
- Increasingly inherently affecting society
	- [Potential Labor Market Impact of LLMs](https://arxiv.org/pdf/2303.10130.pdf)

::: .notes
- 1st point is about research opportunities
- 2nd point is about need to engage with existing research
- 3rd point is about object of research
:::

## Why learn the fundamentals?

- Can increasingly get by with minimal understanding of "how" it works
- But we're also interested in characterizing scope of claims, predictions.
- DL is about scaling; fundamentals scale.

## Learning Objectives

- The basic building block of neural networks: neurons.
- How to learn from data with **optimization**.
- How and why "ensembling" neurons is powerful.






# About DDSS

- **Innovation** in data-/computationally-intensive research.
- **Support** faculty/students in changing technical landscape.
- **Community** building and production of public goods.
- **Position** Princeton as leader in quant soc sci.

::: {.fragment}
Website: [`ddss.princeton.edu`](https://ddss.princeton.edu)
:::

# Workshop Preview (Next Semester)

1. Demystifying Deep Learning
2. (Tools for) Interpretation and Explanation
3. Computational Social Science: Mapping the State of the Field

::: {.notes}
- Workshops aimed at quantitative social scientists interested in branching out into computational methods.
- 1 is hands-on workshop where we will build up a deep learning model from scratch and visualize what happens when we train and add complexity.
- 2; social scientists care about explanation more than prediction. Hands-on workshop about models for explanation and tools for interpreting the output of complex models.
- 3 is still tbc, but likely a seminar overviewing the state of the subfield, aiming to help graduate students who want to enter the fray and identify what areas are interesting/growing.
:::


<!--docker run --rm --volume "`pwd`:/data" --user `id -u`:`id -g` pandoc/latex:2.18 -t revealjs --slide-level 2 -V theme=white --mathjax --citeproc -i -s -o harukawa-cpeb-workshop-presentation.html presentation.md --self-contained-->
