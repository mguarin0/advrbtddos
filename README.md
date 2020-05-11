# Adversarially Robust Network Analysis Classifier

This repo contains all a framework for training a classifier to classify ddos-attacks on the `marist ddos attack` dataset. Also contained in this work is ability to train a classifer that is robust to common gradient based ([Gradient Sign Attack](https://arxiv.org/abs/1412.6572) and [Projected Gradient Decent Attack](https://arxiv.org/pdf/1706.06083.pdf)) and gradient free([Single Pixel Attack](https://arxiv.org/pdf/1612.06299.pdf) and [Jacobian Saliency Map Attack](https://arxiv.org/abs/1511.07528v1)) adversarial attacks by using Robust Adversarial Training as outlined in [TOWARDS DEEP LEARNING MODELS RESISTANT TO ADVERSARIAL ATTACKS](https://openreview.net/pdf?id=rJzIBfZAb).

# Run Experiments
To run all experiments (5 replicas per experiment) please find `exp_runner.sh`.
