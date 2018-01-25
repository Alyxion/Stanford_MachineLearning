function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

m = size(y,1);

% predict logits and call sigmoid actication function
h = sigmoid(X*theta);

% calculate error
J = 1/m * sum(-y.*log(h) - (1-y).*log(1-h)) + lambda/(2*(m))*sum(theta.^2);

% we want to generalize so calcutate theta offset for all elements
lambda_adj = lambda/(m) * theta;
% and then zero to modifications in the first row to keep bias untouched
lambda_adj(1) = 0;
% calculate typical gradients
grad = 1/(m) * sum((h - y) .* X)';
% adjust gradients by regularization
grad = grad + lambda_adj;

% =============================================================

end
