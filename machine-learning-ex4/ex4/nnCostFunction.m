function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

thetas = {Theta1, Theta2};

a = {};
z = {};

% Map vector y to binar vector Y
Y = eye(num_labels);
Y = Y(:, y);

a{1} = X;

thetas_squares = 0;

no_of_layers = length(thetas) +1;


% Part 1: Feed Forward

for i = 1:no_of_layers-1
    
    a{i} = [ones(m, 1) a{i}];
    z{i + 1} = a{i} * thetas{i}';
    a{i + 1} = sigmoid(z{i + 1});
    
    thetas_squares = thetas_squares + sum ( sum (thetas{i}(:,2:end) .^ 2));
    
end

h = a{no_of_layers};

J = ( 1 / m ) * sum( sum( -Y' .* log(h) - (1 - Y)' .* log(1 - h) )) + (lambda / (2 * m)) * thetas_squares;


% Part 2: Back Propagation using for loop

% Small delta
delta = {};
delta{no_of_layers} = h - Y';

for i = (no_of_layers - 1) : -1: 2
%     fprintf("\n%d x %d \n", size(delta{i+1}));
%     fprintf("\n%d x %d \n", size(thetas{i}));
    delta{i} = (delta{i+1} * (thetas{i}(:,2:end))) .* sigmoidGradient(z{i});
end

% Capital Delta
Delta = {};

for i = 1 : no_of_layers - 1
    
    thetas{i}(:,1) = 0;    
    Delta{i} = ( 1 / m ) * (delta{i + 1}' * a{i} + lambda * thetas{i} );
    
end

Theta1_grad = Delta{1};

Theta2_grad = Delta{2};
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
