function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% for example if reducing from 50x2, so 50 datasets with 2 features to
% a 1 dimensional set with 50 rows and 1 feature`
% disp(size(X)) % 50x2
% disp(size(K)) % 1x1, single scalar
% disp(size(Z)) % 50x1
% disp(size(U)) % 2x2, always features x features
% disp(size(U_reduce)) % 2x1, reduced dimensions

% 2x2 -> 2x1
U_reduce = U(:, 1:K);
% 50x2 x 2x1 -> 50x1
Z = X * U_reduce;

% =============================================================

end
