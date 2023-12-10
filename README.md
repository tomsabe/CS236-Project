# CS236-Project
Deep Generative Modeling

# Abstract
We introduce "Binary Generative Flows," a novel binary neural network (BNN) architecture. We propose a generative BNN that is designed to rely on simple bit-wise operations for both training and inference. Our approach is inspired by flow-based models but uniquely adapted to the binary domain. We constrain the model to a sequence of invertible transformations discovered through layer-wise greedy optimization, thus circumventing the need for differentiation in backpropagation. As a proof of concept, we train and test the model on a binarized MNIST dataset. Our hope is that over time further approaches can be found to increase the accuracy of digitally-native architectures that can address the critical issues of energy consumption and hardware constraints that limit modern neural network applications.

# Getting started

pip install requirements.txt
python main.py

Some hyperparameters are specified as globals at the start of main.py

