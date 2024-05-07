# Modeling the Sequential Dependence for Targeted Sentiment Analysis

This is the PyTorch implementation for **PKM** proposed in the paper **Modeling the Sequential Dependence for Targeted Sentiment Analysis**, which is submitted to the Transactions on Knowledge and Data Engineering (TKDE).


## 1. Introduction

Aspect extraction and sentiment prediction are two representative sub-tasks in targeted sentiment analysis, with a certain logical sequential dependency, i.e., aspect extraction is performed first, followed by sentiment prediction. 
The inherent sequential relationship between these tasks plays a crucial role in implicitly conveying information within targeted sentiment analysis. Consequently, some researchers adopt the two-stage approach to sequentially extract aspect and sentiment features. 
Nevertheless, the later-extracted features often lack direct associations with their predecessors in training phase. To tackle this issue, many studies leverage multi-task learning (MTL) to model task relationships, enabling the joint prediction of aspects and sentiments. 
They encode task-specific features through some information-sharing modules (e.g., parameter sharing and expert sharing) in parallel. 
However, it is worth noting that the target task inherently unfolds progressively, and the parallel sharing of features may inadvertently ignore the explicit dependency information from the prior task, potentially resulting in insufficient interactions across tasks. 
In this paper, we propose a \underline{P}rior \underline{K}nowledge \underline{M}erged framework (\textbf{PKM}), which explicitly considers the logical correlations in targeted sentiment analysis. 
Specifically, the PKM simultaneously selects the true aspect information (explicit interaction) and the prediction expectation information (implicit interaction) to transfer to the downstream task during training, progressively enhancing task predictions. 
This ensures effective information propagation for the sentiment prediction task throughout training, and sentiment information can be back-propagated to aspect extraction in a fully differentiable manner. To minimize error accumulation during information transfer,
a semantic compatibility mechanism is designed to bolster aspect extraction capabilities.
 We conduct comprehensive experiments on three real-world datasets to demonstrate the superior performance of our model.

## 2. Running environment

We develop our codes in the following environment:

- Python 3
- [Pytorch 1.1](https://pytorch.org/) 
- numpy >= 1.13.3
  
Download the uncased [BERT-Large](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model and unzip it in the current directory. 

## 3. How to run the codes

You can See  more details in folder absa. 

If you want to run the joint model, run the command blow, since the parameters of each data-set have been set:

```
python run.py 
```

And the result are replaced in out/ 
