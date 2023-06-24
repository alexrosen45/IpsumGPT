import torch


def swish(x, beta=1.0):
    """
    Swish activation function: https://paperswithcode.com/method/swish
    This is a self-gated activation function introduced by Google.
    beta parameter controls the smoothness of our function. When beta
    is 1, our function behaves like ReLU.
    """
    sigmoid = torch.sigmoid(beta * x)
    return x * sigmoid