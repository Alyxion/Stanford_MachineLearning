function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = size(y)(1) % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % gradient descent formular here:
    % t(j) = t(j) - alpha * (1/m) * sum((h(i)-y(i)) * x(i)(j))
    % is solvable as
    % t = t - alpha * (1/m) * sum((X*theta - y) .* X)'

    % predict costs
    predict = X * theta;
    % scale error by feature of each row
    diff = (predict - y) .* X;
    % downscale error with respect to input
    diff_avg = sum(diff)/m;
    
    % adjust theta using given learning rate
    theta = theta - alpha * diff_avg';


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
