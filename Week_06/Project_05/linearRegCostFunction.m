function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% disp(size(X)) % 12x2
% disp(size(y)) % 12x1
% disp(size(theta)) % 2x1

num_features = size(theta);

h = X * theta; % 12x2 * 2x1 --> 12x1
diff = h-y; % 12x1 - 12x1 --> 12x1 target diff 
sqr_diff = diff .^ 2; % elementwise square to penalize high values
J = 1/(2*m) * sum(sqr_diff); % --> downscaling to per element level

lambda_penalty = lambda/(2*m)*sum(theta(2:end) .^ 2); % regularization by penalizing high weight values through squaring

J += lambda_penalty; % final, single error value J


grad = 1 / m * X' * (h - y); % base gradient for all values, including bias
grad(2:num_features) += lambda / m * theta(2:num_features); % add regularization for all features starting at 2


% =========================================================================

grad = grad(:);

end
