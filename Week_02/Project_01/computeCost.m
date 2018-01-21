function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = size(y)(1); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% predict values
predict = X*theta;

% disp("Y: ")
% disp(y(1:10))
% disp("H: ")
% disp(h(1:10))

% get error
diff = y-predict;
% get squared error
diff_sqr = diff .^2;

% return mean squared via J(x) = 1/(2*m)*sum((y-predictions)ˆ2)
J = 1/(2*size(y)(1))*sum(diff_sqr);

% =========================================================================

end
