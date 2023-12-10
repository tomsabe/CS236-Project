# CS236 Deep Generative Modeling - Project

## Abstract
We introduce "Binary Generative Flows," a novel binary neural network (BNN) architecture. We propose a generative BNN that is designed to rely on simple bit-wise operations for both training and inference. Our approach is inspired by flow-based models but uniquely adapted to the binary domain. We constrain the model to a sequence of invertible transformations identified through layer-wise greedy optimization, thus circumventing the need for differentiation in backpropagation. As a proof of concept, we train and test the model on a binarized MNIST dataset, aiming to discover new techniques that can address critical issues of energy consumption and hardware constraints in modern neural network applications.

![Binary Generative Flows Diagram](https://github.com/tomsabe/cs236-project/Binary_Gen_Flow.png)

## Getting Started

To set up the project:

```bash
pip install -r requirements.txt
python main.py
