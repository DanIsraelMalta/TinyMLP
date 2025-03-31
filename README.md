# TinyMLP
TinyMLP is a header only itsy bitsy teeny weeny little dotted poked multi layered perceptron network.

I often use small MLPs to perform fuzzy pattern matchings or emulate certain functions, and its easier to plugin a small heaser rather than goin to python on use the big guns.

## Example 1 - train network to emulate 'not' logic:
```cpp
// define a network with ramp activation function and zero internal layers
TinyML::MultiLayeredPerceptron<double> nn({ 1, 1 }, TinyML::Activation::activation_type::ramp);

// initialize its weights with normally distributed random numbers
TinyML::randomize_normal(nn);

// create training data
std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
dataset.push_back({ {0.0}, {1.0} }); // not(false) = true
dataset.push_back({ {1.0}, {0.0} }); // not(true) = false

// train network
const std::size_t max_iter{ 256 };
const double max_error{ 1e-5 };
const double mu{ 1.0 };
const double error{ TinyML::batch_training(nn, dataset, mu, max_iter, max_error) };
assert(error < max_error);

// check network is accuracy
TinyML::MultiLayeredPerceptronEvaluator<double> eval_nn;
assert(std::abs(eval_nn(nn, dataset[0].first)[0] - dataset[0].second[0]) < 1e-5);
assert(std::abs(eval_nn(nn, dataset[1].first)[0] - dataset[1].second[0]) < 1e-5);
```

## Example 2 - train network to emulate 'or' logic:
```cpp
// define a network with sigmoid activation function and one internal layer with 4 perceptron's
TinyML::MultiLayeredPerceptron<double> nn({ 2, 4, 1 }, TinyML::Activation::activation_type::sigmoid );

// initialize its weights with normally distributed random numbers
TinyML::randomize_normal(nn);

// create training data
std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
dataset.push_back({{0.0, 0.0}, {1.0}}); // or(0, 0) = 1
dataset.push_back({{1.0, 0.0}, {0.0}}); // or(1, 0) = 0
dataset.push_back({{0.0, 1.0}, {0.0}}); // or(0, 0) = 0
dataset.push_back({{1.0, 1.0}, {1.0}}); // or(1, 1) = 1

// train network
const std::size_t max_iter{ 256 };
const double max_error{ 1e-2 };
const double mu{ 15.0 };
const double error{ TinyML::batch_training(nn, dataset, mu, max_iter, max_error) };
assert(error < max_error);

// check network accuracy
TinyML::MultiLayeredPerceptronEvaluator<double> eval_nn;
assert(static_cast<std::int32_t>(std::round(eval_nn(nn, dataset[0].first)[0]) == static_cast<std::int32_t>(dataset[0].second[0])));
assert(static_cast<std::int32_t>(std::round(eval_nn(nn, dataset[1].first)[0]) == static_cast<std::int32_t>(dataset[1].second[0])));
assert(static_cast<std::int32_t>(std::round(eval_nn(nn, dataset[2].first)[0]) == static_cast<std::int32_t>(dataset[2].second[0])));
assert(static_cast<std::int32_t>(std::round(eval_nn(nn, dataset[3].first)[0]) == static_cast<std::int32_t>(dataset[3].second[0])));
```


## Example 3 - train network to check if 2d coordinate is on specific tile:
```cpp
// define a network with sigmoid activation function and three internal layers with 20 perceptron's in each layer
TinyML::MultiLayeredPerceptron<double> nn({ 2, 20, 20, 20, 1 }, TinyML::Activation::activation_type::sigmoid);

// initialize its weights with uniformly distributed random numbers
TinyML::randomize_uniform(nn);

// return 1 or 0 according to {x, y} coordinate
/**
* y
* ^
* |
* 3	 1 1 0 0 0 0 
* 2	 0 1 1 0 0 0 
* 1	 0 0 0 1 1 1 
* 0	 0 0 0 0 1 1 
* 
*    0 1 2 3 4 5 -> x
**/
std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
dataset.push_back({ {0.0, 0.0}, {0.0} }); // |
dataset.push_back({ {0.0, 1.0}, {0.0} }); // |
dataset.push_back({ {0.0, 2.0}, {0.0} }); // |
dataset.push_back({ {0.0, 3.0}, {1.0} }); //  \ x = 0

dataset.push_back({ {1.0, 0.0}, {0.0} }); // |
dataset.push_back({ {1.0, 1.0}, {0.0} }); // |
dataset.push_back({ {1.0, 2.0}, {1.0} }); // |
dataset.push_back({ {1.0, 3.0}, {1.0} }); //  \ x = 1

dataset.push_back({ {2.0, 0.0}, {0.0} }); // |
dataset.push_back({ {2.0, 1.0}, {0.0} }); // |
dataset.push_back({ {2.0, 2.0}, {1.0} }); // |
dataset.push_back({ {2.0, 3.0}, {0.0} }); //  \ x = 2

dataset.push_back({ {3.0, 0.0}, {0.0} }); // |
dataset.push_back({ {3.0, 1.0}, {1.0} }); // |
dataset.push_back({ {3.0, 2.0}, {0.0} }); // |
dataset.push_back({ {3.0, 3.0}, {0.0} }); //  \ x = 3

dataset.push_back({ {4.0, 0.0}, {1.0} }); // |
dataset.push_back({ {4.0, 1.0}, {1.0} }); // |
dataset.push_back({ {4.0, 2.0}, {0.0} }); // |
dataset.push_back({ {4.0, 3.0}, {0.0} }); //  \ x = 4

dataset.push_back({ {5.0, 0.0}, {1.0} }); // |
dataset.push_back({ {5.0, 1.0}, {1.0} }); // |
dataset.push_back({ {5.0, 2.0}, {0.0} }); // |
dataset.push_back({ {5.0, 3.0}, {0.0} }); //  \ x = 5

// train network
const std::size_t max_iter{ 1024 };
const double max_error{ 1e-3 };
const double mu{ 1.0 };
const double error{ TinyML::batch_training(nn, dataset, mu, max_iter, max_error) };
assert(error < max_error);

// check network is accuracy
TinyML::MultiLayeredPerceptronEvaluator<double> eval_nn;
for (const auto& d : dataset) {
	assert(static_cast<std::int32_t>(std::round(eval_nn(nn, dataset[0].first)[0]) == static_cast<std::int32_t>(dataset[0].second[0])));
}
```
