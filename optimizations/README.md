Activation functions and error functions are important components in machine learning. Activation functions introduce
non-linearity into neural networks, while error functions measure how well a model is performing on a given task. The
choice of activation and error function depends on the specific problem being solved and the characteristics of the
data.

# Activation Functions

In machine learning, activation functions are used to introduce non-linearity into neural networks, which is necessary
for the network to learn complex patterns and relationships in data. An activation function takes the weighted sum of
the input data and bias term as input and produces an output value that is fed into the next layer of the neural
network.

- [x] Sigmoid Function
- [x] Rectified Linear Unit (ReLU)
- [x] Leaky Relu
- [x] Softmax Function
- [x] Hyperbolic Tangent Function (tanH)
- [x] Exponential Linear Units (ELU)

# Error Functions

In machine learning, an error function (also known as a loss function or cost function) is a measure of how well a model
is performing on a given task. The error function is typically defined mathematically and is used to evaluate the
difference between the predicted output of the model and the actual output.

- [ ] Mean Squared Error (MSE)
- [ ] Mean Absolute Error (MAE)
- [ ] Binary Cross-Entropy Loss
- [ ] Categorical Cross-Entropy Loss
- [ ] Hinge Loss
- [ ] Huber Loss
- [ ] Kullback-Leibler (KL) Divergence
- [ ] Mean Squared Logarithmic Error (MSLE)

## Sigmoid Function

The sigmoid function is a mathematical function that is often used in machine learning and artificial neural networks.
It is a type of activation function that maps any real-valued number to a value between 0 and 1, which makes it
particularly useful for tasks that involve binary classification.

$$ \alpha (x) = \frac{\mathrm{1} }{\mathrm{1} + e^-x } $$ 

## Rectified Linear Unit (ReLU)

The ReLU function is defined as f(x) = max(0,x), where x is the input to the function. In other words, if the input is negative, the output is 0, and if the input is positive, the output is equal to the input.



## Leaky Relu
Leaky ReLU (Rectified Linear Unit) is a modified version of the standard ReLU activation function used in deep learning. The Leaky ReLU function is defined as f(x) = max(αx, x), where x is the input to the function and α is a small positive constant.


## Softmax Function

The Softmax function is a mathematical function that is often used as an activation function in neural networks. It is a type of logistic function that maps a vector of real numbers to a probability distribution that sums up to 1.
\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

## Hyperbolic Tangent Function (tanH)
The Hyperbolic Tangent function, also known as tanh, is a common activation function used in neural networks. It is similar to the logistic sigmoid function, but has the advantage of being zero-centered and producing output values between -1 and 1.

y = (e^x - e^-x) / (e^x + e^-x)


## Exponential Linear Units (ELU)

Exponential Linear Units (ELUs) are a type of activation function used in neural networks. ELUs are similar to the Rectified Linear Unit (ReLU) function, but with some key differences that can improve the performance of deep neural networks.

The ELU function is defined as follows: given an input x, the ELU function computes the output y using the following formula:

y = x if x > 0
y = α*(e^x - 1) if x <= 0

where α is a small constant that controls the slope of the function for negative inputs.






