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

% disp(size(Theta1))
% disp(size(Theta2))
% disp(size(X))
% disp(size(y))
% disp(size(a3))
% >> 25   401
% >> 10   26
% >> 5000    400
% >> 5000      1
% >> 5000      10

% Forward pass
X = [ones(m, 1) X]; % extend one for bias
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2), 1) a2];
z3 = (Theta2 * a2')';
a3 = sigmoid(z3);

for k = 1:num_labels
    yk = y == k; % one hot if highest y == k --> 5000x1, for all 5000 rows 1 if y matches k right now
    col = a3(:, k); % get the heuristic for all 5000 rows at current col
    Jk = 1 / m * sum(-yk .* log(col) - (1 - yk) .* log(1 - col));
    J = J + Jk;
end

tnb_1 = Theta1(:, 2:end); % get all weights except for bias
tnb_2 = Theta2(:, 2:end);
% disp(size(tnb_1))
% disp(size(tnb_2))
tnb_1_sqr = tnb_1 .^ 2; % square all weights
tnb_2_sqr = tnb_2 .^ 2;
% disp(lambda)
lambda_error = lambda/(2*m); % get scaling factor
lambda_error = lambda_error * (sum(sum(tnb_1_sqr))+sum(sum(tnb_2_sqr))); % sum all weights of all rows and columns and scale them back down

J = J + lambda_error; % add regularization weight

% -------------------------------------------------------------

for t = 1:m
  % setup 1 hot encoded array where 1 signales the target index. 10x1
  y_hot = ([1:num_labels]==y(t))';
  % get the hepothesis of the forward pass and subtract the target value
  delta_3 = (a3(t,:) - y_hot');

  % delta 3 now contains the error for each output row

  % delta 2 (of the first hidden layer) is the error of the ouput layer, scaled by the weights and scaled by the gradient of the first, not yet activated values of the first hidden layer
  delta_2 = Theta2' * delta_3' .* sigmoidGradient([1, z2(t, :)])';
  delta_2 = delta_2(2:end);
  
  % gradient with respect to X
  Theta1_grad = Theta1_grad + delta_2 * X(t,:);
  % gradient first respect to first activation
  Theta2_grad = Theta2_grad + delta_3' * a2(t,:);
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% add regulularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * Theta2(:,2:end); 



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
