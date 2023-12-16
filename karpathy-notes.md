Mean squared error: sum((pred - actual)**2).

For n-gram models, if something isn't observed in training data, it will lead to infinite loss. Thus, you should "smooth" the model by adding a small amount to all probabilities.

Probabilities have to be positive numbers, and must sum to 1. Neural networks output positive or negative numbers, so you must normalize them. To do this, you interpret them as **log counts**, or logits. To get the "counts", you exponentiate the raw outputs (logits). Then to get probabilities, you normalize those "counts", averaging along the output dimension. This process is called a **softmax**.

Probability of data: n1 * n2 * ... * nk. This becomes vanishingly small. However, log(x1, x2, x3) == log(x1) + log(x2) + log(x3). Thus, we take sum of logs. This number is negative (log normalizes to negative), so we take the negative of this. Then we average by number of values. This is the "negative (average) log likelihood". This works because log is monotonic.

To prevent overfitting, you can add a regularization component to the loss. This can be as simple as a sum of the squared weights (scaled by some constant factor).

Study broadcasting.

Initialization is important. You want a rough idea of what the initial loss should be, and initialize the network close to that. You can start by scaling the biases and weights to be quite small. This helps the softmax avoid being "confidently wrong", which means more training time can be spent on productive gradient descent.

Activation functions like tanh, sigmoid, ReLU, and ELU are "squashing functions". If the network is initialized in a way such that the pre-activation values are far from zero, those get "squashed" to 0 or 1 (0 gradient). This means that those neurons are useless, and not used by the networks. Sometimes, a neuron will be unused across the entire dataset (a "dead neuron"). To fix this, 

However, scaling by a "magic number" is relatively un-principled.
- To normalize a gaussian distribution to unit gaussian, divide by the square root of the "fan-in" (number of inputs). This is the same as multiplying by sqrt(1/fan-in).
- In "Delving Deep into Rectifiers", Kaiming He describes a principled approach to this kind of normalization. His approach is called "Kaiming Normalization".
- Because you through away ~half of the distribution with a ReLU, you have to compensate for that with a *gain*. This means they must initialize with a gaussian distribution with a mean of zero and a std. deviation of **sqrt(2/fan-in)**. A properly intialized forward pass will also ~approximately properly initialize the backwards pass.
- Each activation function has a corresponding "gain". This is implemented in PyTorch: `torch.nn.init.kaiming_normal_(...)`. std. dev = gain / sqrt(fan-in). https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
- Generally, stddev of (gaussian * num) ~= stddev(gaussian) * num.

Batch Normalization just works.
- You have hidden states (pre-activations on hidden layers), and you want them to be roughly gaussian when fed into the activation function. So you can just normalize them to be gaussian!
- To do this, take the mean and standard deviation of any given mini-batch (collection of samples). To normalize, take the mini-batch outputs, then subtract the mean and divide by the standard deviation (plus a tiny factor to prevent dividing by zero).
- To allow the mean and deviation to adapt to data, you train a batch-norm gain (for scale, starting as ones) and bias (for offset, starting as zeroes). This is called the batch norm scale and shift.
- Expressed in PyTorch, this is: `hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias # batch norm`
- Batch normalization introduces some entropy, because each example is normalized relative to other examples in the batch, but this is actually a good thing – it's actually a regularizer (data augmentation)!
- People have tried other types of normalization, but batch normalization has stuck around because it works so well.
- If you're doing batch normalization after a layer, any bias in the layer is averaged out and subtracted. You can just remove the bias. The batch normalization bias will "bias" the distribution for the layer.
- It's very easy to introduce bugs with batchnorm. Karpathy says it's often better to use layer normalization or group normalization instead.
- To do inference at the end, calculate the mean/std across the whole training set and normalize inputs relative to that. You can calculate this explicitly, or get rough running averages like this:

```python
# update running batch norm stats
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```


Convolutions are basically linear layers, but applied to small patches of input. They are used for input with spatial structure (like images).

To initialize a linear layer, use `torch.nn.Linear`. You must know the fan-in, fan-out, and whether you want a bias. By default, PyTorch initializes the weights to a normal distribution over 1/sqrt(fan-in). This assures that if the input is roughly gaussian, the output will be roughly gaussian.

To initialize a batch normalization layer, use `torch.nn.BatchNorm1d`. The `momentum` is the small value used for updating the running mean/std dev. For larger batch sizes, you can use a larger momentum, but for smaller batch sizes, use a smaller one (it's more non-regular).
