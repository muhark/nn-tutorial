---
title: "Planning GPT for Social Research: How and Whether Large Language Models Can Help Social Scientists"
author: "Musashi Jacobs-Harukawa"
---

# Rough Planning

- _Context:_ You're a social scientist that's hearing a lot about LLMs and how big a deal they are.
- In this lecture, I want to take a step back and give a relatively hype-free explanation of:
	- What are LLMs/foundation models
		- Mechanically what do they do
		- How are they trained (and what does training mean?)
	- What is completion, zero-shot learning, in-context learning, few-shot learning


- Note: I still simplify at places. (Will try and highlight where).
- The choice of where to simplify is based on what I think is a useful level of abstraction that still allows us to engage


Structuring the explanation


## Rough Intro

- What's been going on in the world of AI?
- Here are my observations about how it's already impacting social science research.
	- Give some examples
- I think some applications are better than others (valid inference, less problematic)
- and I think that some of the worse applications are based in part on a misconception of what this technology is.
- So, I want to first do a deep-ish dive on what LLMs are and how they work so we can be more specific about what inferences we can make from them.
- Then I'll come back to some key pieces of research that are coming out in this area and discuss them within the context of the explanation I have just given.


- Context/Format - catching you up on what is happening, what it means, and how it relates to us
- What people are already doing
	- One model to rule them all?
	- Using LLMs to study society?
- How it works (in a relevant level of detail?)
	- Language as conditional probabilities / autoregressive language generation
	- Training a generative model
	- Completion -> everything else
		- Zero-shot and in-context learning
		- Prompt engineering
	- Instruction-tuning
	- What's coming next?...
		- Bigger
		- Smaller
- Considerations
	- Reproducibility
	- Transparency
	- Privacy
- Discussing papers in more detail?



## How it Works

1. What is it?
2. How is it trained? (and tuned?)
3. Is Completion Everything?

### What is it?

- Sequence-to-sequence model
	- Maps from one sequence to another
	- Language as a sequence?

- Language as a Conditional Probability Distribution

- Autoregressive Language Generation

- Tokenization and Vocabularies

### How is it trained?





# Reading Notes

## Social Science

- Using LLMs to learn about the world:
	- Argyle et al 2023
	- Wu et al 2023
	- OpinionQA
	- Also discuss precedence of neural LMs as tool for corpus summary/description, e.g. Rodman etc.

- Using LLMs for zero-shot classification
	- Ornstein et al 2022
	- Gilardi et al 2023

## Ethics

- Ethical Aspects of LLMs
	- Bender et al 2021 (?)

## Technical

### Brown et al 2020 (GPT-3)

Training Data:

- CommonCrawl (deduplicated, similarity to reference corpora), `https://commoncrawl.org/the-data/`
- WebText2 (Radford et al 2019) OpenAI's internal dataset. Starting point all outbound links from Reddit with at least 3 karma - heuristic indicating whether people found something interesting, educational or funny.
- Books1 and 2 (`bookcorpus` and a mystery)
- English Wikipedia (as it sounds)

Pre-training:

- Next word prediction


- Technical Papers on LLMs
	- Vaswani et al?
	- Bradford GPT-2
	- Foundation Models paper (Bommasani?)

# Drafting


## How it Works

# Graveyard

- Nov 11: "How to Train Your Stochastic Parrot: LLMs for Political Texts" [@ornstein2022]
- Feb 21: "Out of One, Many: Using LMs to Simulate Human Samples" [@argyle2023]
- Mar 7: "LLMs Can Argue in Convincing and Novel Ways About Politics" [@palmer2023]
- Mar 22: "Large Language Models Can Be Used to Estimate the Ideologies of Politicians in a Zero-Shot Learning Setting" [@wu2023]
- Mar 27: "ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks" [@gilardi2023]


_Applications of GPT (or other LLMs) in social science_:

- Nov 11: Text classification, scaling and topic modelling [@ornstein2022]
- Feb 21: Simulating survey responses for counterfactual persons [@argyle2023]
- Mar 7: Generating persuasive political arguments [@palmer2023]
- Mar 22: Estimate ideology of politicians [@wu2023]
- Mar 27: Out-perform crowd workers for annotation [@gilardi2023]
- Mar 30: Comparing the opinions of GPT to the public [@santurkar2023]
