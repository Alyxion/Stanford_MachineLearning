function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% sigmoid converts negative values into a range from 0 to 0.49999
% and positive values into a range from 0.5 to 1.0. This way a range
% between 0 and 1 is guaranteed, either for combined boolean matches
% or a final classification.
g = 1 ./ (1 + exp(-z));

% =============================================================

end
