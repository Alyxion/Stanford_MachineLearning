function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Personal notes
% Inputs:
%
% X = 5000x400
% Theta1 = 25x401
% Theta2 = 10x26
% 
% Add bias column so that X = 5000x401
% hidden1 = sigmoid( 5000x401 * 401 x 25 )
% --> 5000x25
% sigmoid for each value of hidden1
% hidden1 = 5000x25
% Add bias column so that hidden1 = 5000x26
% y = 5000x26 * 26*10
% y = 5000x10
% activate sigmoid
% get maximum value for each row:
% p = 5000x1
% 
X = [ones(m, 1) X];

h1 = sigmoid(X * Theta1');
h1 = [ones(size(h1), 1) h1];

y = sigmoid((Theta2 * h1')');

[predict_max, index_max] = max(y, [], 2);
disp(size(index_max))
p = index_max;

% =========================================================================


end
