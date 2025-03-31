#pragma once

#include <type_traits>
#include <concepts>
#include <vector>
#include <span>
#include <cmath>
#include <algorithm>
#include <cstdint>
#ifdef _DEBUG
#include <assert.h>
#include <iostream>
#endif

/**
* itsy bitsy teeny weeny little dotted poked multi layered perceptron network
**/
namespace TinyML {
	
	/**
	* helpers
	**/
	namespace helpers {

		/**
		* \brief generate a gaussian distributed random numbers with mean of 0 and variance of 1.
		*        taken from https://github.com/DanIsraelMalta/Numerics/blob/main/Hash.h
		* @param {size_t} a variable that defines the precision of the distribution.
		*                 default is 15 which gives the smallest distance between two numbers (C3= 1 / (2^15 / 3) = 1/10922 = 0.000091)
		**/
		template<std::size_t Q = 15, typename T = float>
		    requires(std::is_floating_point_v<T>)
		constexpr T normal_distribution() noexcept {
		    constexpr T C1{ static_cast<T>((1 << Q) - 1) };
		    constexpr T C2{ static_cast<T>((C1 / 3) + 1) };
		    constexpr T C3{ static_cast<T>(1.0) / static_cast<T>(C1) };

#define RAND01(x) (static_cast<T>(x)) * (static_cast<T>(rand())) /  (static_cast<T>(RAND_MAX))
		    return (static_cast<T>(2.0) * (RAND01(C2) + RAND01(C2) + RAND01(C2)) - static_cast<T>(3.0) * (C2 - static_cast<T>(1.0))) * C3;
#undef RAND01
		}

		/**
		* \brief generate a 32bit uniformly distributed random number with 24bit linear distance in range [0, 1]
		*        taken from https://github.com/DanIsraelMalta/Numerics/blob/main/Hash.h
		* @param {float, out} uniformly distributed random number in range [0, 1]
		**/
		std::float_t rand32() {
			// eps = 1.0f - 0.99999994f (0.99999994f is closest value to 1.0f from below)
			constexpr std::double_t eps{ 5.9604645E-8 };
			const std::uint_least32_t r{ static_cast<std::uint_least32_t>(rand() & 0xffff) +
										 static_cast<std::uint_least32_t>((rand() & 0x00ff) << 16) };
			return static_cast<std::float_t>(static_cast<double>(r) * eps);
		}
	};

	/**
	* activation related functions and types
	**/
	namespace Activation {
		/**
		* \brief type of activation function
		**/
		enum class activation_type : std::uint8_t {
			id      = 0,
			sigmoid = 1,
			tanh    = 2,
			relu    = 3,
			cubic   = 4,
			quintic = 5,
			ramp    = 6,
		};

		/**
		* \brief given activation function type, return appropriate value
		* @param {floating_point, in}  activation function parameter
		* @param {floating_point, out} activation function value
		**/
		template<activation_type type, typename T>
			requires(std::is_floating_point_v<T>)
		constexpr T activation(const T x) {
			if constexpr (type == activation_type::id) {
				return x;
			}
			else if constexpr (type == activation_type::sigmoid) {
				return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
			}
			else if constexpr (type == activation_type::tanh) {
				return static_cast<T>(2.0) / (static_cast<T>(1.0) + std::exp(static_cast<T>(-2.0) * x)) - static_cast<T>(1.0);
			}
			else if constexpr (type == activation_type::relu) {
				return std::max(T{}, x);
			}
			else if constexpr (type == activation_type::cubic) {
				return std::clamp(x * x * std::fma(-static_cast<T>(2.0), x, static_cast<T>(3.0)), T{}, static_cast<T>(1.0));
			}
			else if constexpr (type == activation_type::quintic) {
				return std::clamp(x * x * x * (static_cast<T>(10.0) + x * std::fma(static_cast<T>(6.0), x, static_cast<T>(-15.0))), T{}, static_cast<T>(1.0));
			}
			else if constexpr (type == activation_type::ramp) {
				return std::clamp(x, T{}, static_cast<T>(1.0));
			}
		}

		template<typename T>
			requires(std::is_floating_point_v<T>)
		constexpr T activation(const T x, const activation_type type) {
			switch (type) {
				case activation_type::id: return activation<activation_type::id>(x);
				case activation_type::sigmoid: return activation<activation_type::sigmoid>(x);
				case activation_type::tanh: return activation<activation_type::tanh>(x);
				case activation_type::relu: return activation<activation_type::relu>(x);
				case activation_type::cubic: return activation<activation_type::cubic>(x);
				case activation_type::quintic: return activation<activation_type::quintic>(x);
				case activation_type::ramp: return activation<activation_type::ramp>(x);
			};
		}

		/**
		* \brief given activation function type, return appropriate derivative value
		* @param {floating_point, in}  activation function parameter
		* @param {floating_point, out} activation function derivative value
		**/
		template<activation_type type, typename T>
			requires(std::is_floating_point_v<T>)
		constexpr T activation_derivative(const T x) {
			if constexpr (type == activation_type::id) {
				return static_cast<T>(1.0);
			}
			else if constexpr (type == activation_type::sigmoid) {
				return x * (static_cast<T>(1.0) - x);
			}
			else if constexpr (type == activation_type::tanh) {
				return static_cast<T>(1.0) - x * x;
			}
			else if constexpr (type == activation_type::relu) {
				return x == T{} ? T{} : static_cast<T>(1.0);
			}
			else if constexpr (type == activation_type::cubic) {
				return std::clamp(static_cast<T>(6.0) * x * (static_cast<T>(1.0) - x), T{}, static_cast<T>(1.0));
			}
			else if constexpr (type == activation_type::quintic) {
				const T minus{ x - static_cast<T>(1.0) };
				return std::clamp(static_cast<T>(30.0) * x * x * minus * minus, T{}, static_cast<T>(1.0));
			}
			else if constexpr (type == activation_type::ramp) {
				return (x < T{} || x > static_cast<T>(1.0)) ? T{} : static_cast<T>(1.0);
			}
		}

		template<typename T>
			requires(std::is_floating_point_v<T>)
		constexpr T activation_derivative(const T x, const activation_type type) {
			switch (type) {
				case activation_type::id: return activation_derivative<activation_type::id>(x);
				case activation_type::sigmoid: return activation_derivative<activation_type::sigmoid>(x);
				case activation_type::tanh: return activation_derivative<activation_type::tanh>(x);
				case activation_type::relu: return activation_derivative<activation_type::relu>(x);
				case activation_type::cubic: return activation_derivative<activation_type::cubic>(x);
				case activation_type::quintic: return activation_derivative<activation_type::quintic>(x);
				case activation_type::ramp: return activation_derivative<activation_type::ramp>(x);
			};
		}
	};

	/**
	* loss functions
	**/
	namespace Loss {
		/**
		* \brief type of loss function
		**/
		enum class loss_type : std::uint8_t {
			L2              = 0,	// L2 norm (Euclidean)
			MeanSquareError = 1,	// mean square error
			CrossEntropy    = 2,	// cross entropy
		};

		/**
		* \brief given loss function type, return appropriate value
		* @param {vector<floating_point>, in}  actual data
		* @param {vector<floating_point>, in}  predicted data
		* @param {floating_point,         out} loss function value
		**/
		template<loss_type type, typename T>
			requires(std::is_floating_point_v<T>)
		constexpr T loss(const std::vector<T>& actual, const std::vector<T>& predicted) {
			assert(actual.size() == predicted.size());

			if constexpr (type == loss_type::L2) {
				T loss{};
				for (std::size_t i{}; i < actual.size(); ++i) {
					const T diff{ actual[i] - predicted[i] };
					loss += diff * diff / static_cast<T>(2.0);
				}
				return loss;
			}
			else if constexpr (type == loss_type::MeanSquareError) {
				T loss{};
				for (std::size_t i{}; i < actual.size(); ++i) {
					const T diff{ actual[i] - predicted[i] };
					loss += diff * diff;
				}
				return loss / static_cast<T>(actual.size());
			}
			else if constexpr (type == loss_type::CrossEntropy) {
				T loss{};
				for (std::size_t i{}; i < actual.size(); ++i) {
					loss += actual[i] + std::log(predicted[i] + std::numeric_limits<T>::epsilon());
				}
				return -loss / static_cast<T>(actual.size());
			}
		}

		template<typename T>
			requires(std::is_floating_point_v<T>)
		constexpr T loss(const std::vector<T>& actual, const std::vector<T>& predicted, const loss_type type) {
			switch (type) {
				case loss_type::L2: return loss<loss_type::L2>(actual, predicted);
				case loss_type::MeanSquareError: return loss<loss_type::MeanSquareError>(actual, predicted);
				case loss_type::CrossEntropy: return loss<loss_type::CrossEntropy>(actual, predicted);
			}
		}
	};

	/**
	* \brief multi-layered fully-connected network.
	*  notice that first layer defines inputs, last layer defines outputs and the rest are fully connected internal layers.
	**/
	template<typename T>
		requires(std::is_floating_point_v<T>)
	struct MultiLayeredPerceptron {
		// aliases
		using value_type = T;

		/**
		* \brief construct MLP by specifying size of each layer and its activation functions
		* @param {vector<size_t>,                      in} sizes of each layer
		* @param {vector<Activation::activation_type>, in} activation types of each layer
		**/
		constexpr explicit MultiLayeredPerceptron(std::vector<std::size_t> layer_sizes,
			                                      std::vector<Activation::activation_type> activation_types) :
			                                      layer_sizes_(layer_sizes), activation_types_(activation_types) {
			assert(!layer_sizes_.empty());
			assert(layer_sizes_.size() == activation_types_.size() + 1);
			this->initialize_weights();
		}

		/**
		* \brief construct MLP by specifying size of each layer and activation function for all layers
		* @param {vector<size_t>,              in} sizes of each layer
		* @param {Activation::activation_type, in} activation type of each layer
		**/
		constexpr explicit MultiLayeredPerceptron(std::pair<std::vector<std::size_t>,
			                                      std::vector<Activation::activation_type>> structure) :
			                                      MultiLayeredPerceptron(structure.first, structure.second) {}

		/**
		* \brief construct MLP by specifying size of each layer and its activation functions as a pair
		* @param {pair<vector<size_t>, Activation::activation_type>, in} {sizes of each layer, activation types of each layer}
		**/
		constexpr explicit MultiLayeredPerceptron(std::vector<std::size_t> layer_sizes,
			                                      Activation::activation_type activation_type) :
			                                      layer_sizes_(layer_sizes) {
			this->activation_types_.resize(layer_sizes_.size() - 1, activation_type);
			this->initialize_weights();
		}

		/**
		* @param {bool, out} true if network is empty, false otherwise
		**/
		constexpr bool empty() const noexcept {
			return this->layer_sizes_.empty();
		}

		/**
		* @param {span<size_t>, out} span over layer sized
		**/
		std::span<const std::size_t> layer_sizes() const noexcept {
			return this->layer_sizes_;
		}

		/**
		* @param {span<size_t>, out} span over perceptron activation types
		**/
		std::span<const Activation::activation_type> activation_types() const noexcept {
			return this->activation_types_;
		}

		/**
		* \brief return span over MLP weights.
		*        weights are stored row major, from first layer to the last.
		* @param {span<floating_point, in} span of MLP weights
		**/
		std::span<const value_type> weights() const noexcept { return this->weights_; }
		std::span<      value_type> weights()       noexcept { return this->weights_; }

		// internals
		private:
			std::vector<std::size_t> layer_sizes_; // amount of neurons in every layer
			std::vector<Activation::activation_type> activation_types_; // activation type in each layer
			std::vector<value_type> weights_; // neuron weights

			/**
			* \brief initialize MLP weights
			**/
			constexpr void initialize_weights() {
				std::size_t weight_count{};
				for (std::size_t i{}; i < this->layer_sizes_.size() - 1; ++i) {
					weight_count += (this->layer_sizes_[i] + 1) * this->layer_sizes_[i + 1];
				}
				this->weights_.resize(weight_count);
			}
	};

	template<typename>   struct is_mlp                            : std::false_type {};
	template<typename T> struct is_mlp<MultiLayeredPerceptron<T>> : std::true_type {};
	template<typename T> constexpr bool is_mlp_v = is_mlp<T>::value;

	////////////////////////////////
	// functions operating on MLP //
	////////////////////////////////

	/**
	* \brief given MLP and its weights, call a given function (taking these weights as mutable input)
	* @param {MLP,      in} mlp to operate upon
	* @param {callable, in} function which takes layer size and weight (as mutable)
	**/
	template<typename MLP, class F, class T = typename MLP::value_type>
		requires(is_mlp_v<MLP> && std::is_invocable_v<F, std::size_t, std::span<T>>)
	void for_each_layer(MLP& mlp, F&& func) {
		const auto layer_sizes = mlp.layer_sizes();
		const auto weights = mlp.weights();

		std::size_t offset{};
		for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
			const std::size_t count{ (layer_sizes[i] + 1) * layer_sizes[i + 1] };
			func(i, std::span{ weights.data() + offset, weights.data() + offset + count });
			offset += count;
		}
	}

	/**
	* \brief initialize MLP weights with uniformly distributed random numbers
	* @param {MLP, in} mlp
	**/
	template<typename MLP>
		requires(is_mlp_v<MLP>)
	void randomize_uniform(MLP& mlp) {
		using T = typename MLP::value_type;

		const std::span<const std::size_t> layer_sizes{ mlp.layer_sizes() };
		for_each_layer(mlp, [&layer_sizes](std::size_t i, std::span<T> weights) {
			const T length{ std::sqrt(static_cast<T>(6.0) / static_cast<T>(layer_sizes[i] + layer_sizes[i + 1])) };
			for (T& w : weights) {
				w = length * static_cast<T>(helpers::rand32() - static_cast<T>(0.5));
			}
		});
	}

	/**
	* \brief initialize MLP weights with normally distributed random numbers
	* @param {MLP, in} mlp
	**/
	template<typename MLP>
		requires(is_mlp_v<MLP>)
	void randomize_normal(MLP& mlp) {
		using T = typename MLP::value_type;

		const std::span<const std::size_t> layer_sizes{ mlp.layer_sizes() };
		for_each_layer(mlp, [&layer_sizes](std::size_t i, std::span<T> weights) {
			const T std{ std::sqrt(static_cast<T>(2.0) / static_cast<T>(layer_sizes[i] + layer_sizes[i + 1])) };
			for (T& w : weights) {
				w = std * helpers::normal_distribution();
			}
		});
	}

	/**
	* \brief object which perform MLP evaluation
	**/
	template<typename T>
		requires(std::is_floating_point_v<T>)
	struct MultiLayeredPerceptronEvaluator {
		// aliases
		using value_type = T;

		/**
		* \brief given MLP and input, return the output
		**/
		template<class MLP>
			requires(is_mlp_v<MLP> && std::is_same_v<value_type, typename MLP::value_type>)
		std::vector<value_type> operator()(const MLP& mlp, std::vector<value_type> input) {
			// housekeeping
			const std::span<const std::size_t> layer_sizes{ mlp.layer_sizes() };
			auto weights = mlp.weights().begin();
			assert(layer_sizes[0] == input.size());

			// evaluate MLP
			for (std::size_t l{}; l < layer_sizes.size() - 1; ++l) {
				this->temp_.resize(layer_sizes[l + 1]);

				for (std::size_t i{}; i < layer_sizes[l + 1]; ++i) {
					this->temp_[i] = *weights++;

					for (std::size_t j{}; j < layer_sizes[l]; ++j) {
						this->temp_[i] += (*weights++) * input[j];
					}

					this->temp_[i] = Activation::activation(this->temp_[i], mlp.activation_types()[l]);
				}

				std::swap(this->temp_, input);
			}

			// output
			return input;
		}

		private:
			std::vector<T> temp_;
	};

	/**
	* \brief object which encompasses MLP training operations
	**/
	template<typename T>
		requires(std::is_floating_point_v<T>)
	struct MultiLayeredPerceptronTrainer {
		// aliases
		using value_type = T;

		/**
		* \brief update given MLP based on input data
		* @param {MLP,                    in}  MLP
		* @param {vector<floating_point>, in}  input data
		* @param {vector<floating_point>, out} MLP layers after MLP was run on input data
		**/
		template<class MLP>
			requires(is_mlp_v<MLP> && std::is_same_v<value_type, typename MLP::value_type>)
		const std::vector<value_type>& apply(const MLP& mlp, std::vector<value_type> input) {
			assert(!mlp.empty());

			// housekeeping
			auto layer_sizes{ mlp.layer_sizes() };
			auto weights{ mlp.weights().begin() };
			this->layers_.resize(layer_sizes.size());
			assert(layer_sizes[0] == input.size());
			this->layers_[0] = input;

			// apply data
			for (std::size_t l{}; l < layer_sizes.size() - 1; ++l) {
				auto& layer{ this->layers_[l + 1] };
				layer.resize(layer_sizes[l + 1]);

				for (std::size_t i{}; i < layer_sizes[l + 1]; ++i) {
					layer[i] = *weights++;

					for (std::size_t j{}; j < layer_sizes[l]; ++j) {
						layer[i] += (*weights++) * this->layers_[l][j];
					}

					layer[i] = Activation::activation(layer[i], mlp.activation_types()[l]);
				}
			}

			// output
			return this->layers_.back();
		}

		/**
		* @param {vector<floating_point>, out} MLP layers
		**/
		const std::vector<value_type>& result() const noexcept {
			return this->layers_.back();
		}

		/**
		* \brief compute local errors before calling 'backpropagate'
		* @param {MLP,                    in} MLP
		* @param {vector<floating_point>, in} output
		**/
		template<class MLP>
			requires(is_mlp_v<MLP>&& std::is_same_v<value_type, typename MLP::value_type>)
		void backpropagate(const MLP& mlp, const std::vector<value_type>& output) {
			assert(!mlp.empty());

			// housekeeping
			const auto layer_sizes = mlp.layer_sizes();
			assert(output.size() == layer_sizes.back());

			// error update (will be used as gradient in back propagation)
			this->error_tmp_.resize(output.size());
			for (std::size_t i{}; i < output.size(); ++i) {
				this->error_tmp_[i] = this->layers_.back()[i] - output[i];
			}

			// call back propagation
			backpropagate_inrernal(mlp, this->error_tmp_);
		}

		/**
		* \brief return L2 norm of gradients
		* @param {floating_point, in} gradient L2 norm
		**/
		value_type gradient_norm() const {
			T norm{};
			for (auto g : this->gradient_) {
				norm += g * g;
			}
			return std::sqrt(norm);
		}

		/**
		* \brief perform one gradient descent step in the direction of computed gradient, scaled by given amount
		* @param {floating_point, in} gradient L2 norm
		**/
		template<class MLP>
			requires(is_mlp_v<MLP> && std::is_same_v<value_type, typename MLP::value_type>)
		void descend(MLP& mlp, value_type factor) const {
			auto gradient = this->gradient_.data();
			for (auto& w : mlp.weights()) {
				w -= factor * (*gradient++);
			}
		}

		/**
		* \brief nullify gradient
		**/
		void clear() noexcept {
			std::fill(this->gradient_.begin(), this->gradient_.end(), value_type{});
		}

		/**
		* \brief return span over gradients.
		* @param {span<floating_point, in} span of gradients
		**/
		std::span<const value_type> gradient() const noexcept { return this->gradient_; }
		std::span<      value_type> gradient()       noexcept { return this->gradient_; }

		/**
		* \brief return span over errors.
		* @param {span<floating_point, in} span of errors
		**/
		std::span<const value_type> arg_gradient() const noexcept {
			return this->error_;
		}

		// internals
		private:
			std::vector<std::vector<value_type>> layers_; // mlp layers
			std::vector<value_type> gradient_;            // gradient
			std::vector<value_type> error_, error_tmp_;   // weights

			/**
			* \brief given mlp and gradient, calculate errors
			* @param {MLP,                    in}  MLP
			* @param {vector<floating_point>, in}  gradient
			**/
			template<class MLP>
				requires(is_mlp_v<MLP>&& std::is_same_v<value_type, typename MLP::value_type>)
			void backpropagate_inrernal(const MLP& mlp, const std::vector<value_type>& gradient) {
				assert(!mlp.empty());

				// housekeeping
				const auto layer_sizes = mlp.layer_sizes();
				assert(layer_sizes.back() == gradient.size());
				const auto weights = mlp.weights();
				const auto activation_types = mlp.activation_types();

				// calculate gradient and errors
				this->error_.resize(gradient.size());
				for (std::size_t i{}; i < gradient.size(); ++i) {
					const value_type value{ this->layers_.back()[i] };
					this->error_[i] = gradient[i] * Activation::activation_derivative(value, activation_types.back());
				}
				this->gradient_.resize(mlp.weights().size());

				for (std::size_t offset{ this->gradient_.size() }, l{ layer_sizes.size() - 1 }; l-- > 0;) {
					offset -= (layer_sizes[l] + 1) * layer_sizes[l + 1];

					for (std::size_t i{}; i < layer_sizes[l + 1]; ++i) {
						const std::size_t row_offset{ offset + i * (layer_sizes[l] + 1) };
						this->gradient_[row_offset] += this->error_[i];
						for (std::size_t j{}; j < layer_sizes[l]; ++j) {
							this->gradient_[row_offset + j + 1] += this->error_[i] * this->layers_[l][j];
						}
					}

					this->error_tmp_.assign(layer_sizes[l], value_type{});
					for (std::size_t i{}; i < layer_sizes[l + 1]; ++i) {
						for (std::size_t j{}, row_offset{ offset + i * (layer_sizes[l] + 1) }; j < layer_sizes[l]; ++j) {
							this->error_tmp_[j] += weights[row_offset + j + 1] * this->error_[i];
						}
					}

					if (l > 0) {
						for (std::size_t i{}; i < this->error_tmp_.size(); ++i) {
							this->error_tmp_[i] *= Activation::activation_derivative(this->layers_[l][i], activation_types[l - 1]);
						}
					}

					std::swap(this->error_, this->error_tmp_);
				}
			}
	};


	/**
	* \brief train a given mlp on batch of data
	* @param {MLP,                                              in}  mlp to train
	* @param {vector<{vector<value_type>, vector<value_type>}>, in>  pairs of {input data, expected output data}
	* @param {value_type,                                       in}  backpropagation, gradient descent step, scale, i.e. - adaptation variable
	* @param {size_t,                                           in}  maximal amount of training iterations
	* @param {value_type,                                       in}  maximal error for training session to end
	* @param {Loss::loss_type,                                  in}  LOSS function type (for error calculation)
	* @param {value_type,                                       out} training error at end of the training
	**/
	template<class MLP, class T = typename MLP::value_type>
		requires(is_mlp_v<MLP>)
	constexpr T batch_training(MLP& mlp, const std::vector<std::pair<std::vector<T>, std::vector<T>>>& batch,
		                       const T mu, const std::size_t max_iter, const T max_error,
		                       const Loss::loss_type type = Loss::loss_type::L2) {
		// housekeeping
		const T len{ static_cast<T>(batch.size()) };
		MultiLayeredPerceptronTrainer<T> learner;
		T error{ static_cast<T>(2.0) * max_error };

#ifdef _DEBUG
		const std::size_t debug_freq{ max_iter / 10 };
		std::cout << "batch_learning (mu = " << mu << ", max iterations = " << max_iter << ", max error = " << max_error << "):\n";
#endif

		// training
		for (std::size_t i{}; i < max_iter && (error > max_error); ++i)  {
			// evaluate error and backpropagate
			learner.clear();
			error = T{};
			for (const auto& data : batch) {
				learner.apply(mlp, data.first);
				error += Loss::loss(data.second, learner.result(), type) / len;
				learner.backpropagate(mlp, data.second);
			}

#ifdef _DEBUG
			if ((i % debug_freq) == 0) {
				std::cout << "Iteration " << i << " error: " << error << '\n';
			}
#endif

			// gradient descent
			learner.descend(mlp, mu);
		}

		// error
#ifdef _DEBUG
		std::cout << "last Iteration error: " << error << '\n';
#endif
		return error;
	}
};
