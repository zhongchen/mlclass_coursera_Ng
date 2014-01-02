function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% J(theta) = 1 / m \sum [ ( 1-yi) * theta' * xi + log(1+exp(-theta' * xi))]
temp = X * theta;

%\sum [(1-yi) * theta' * xi]
temp2 = sum(temp .* (1 - y)) ;

%\sum [log(1+exp(-theta'*xi))]
% log(1 + exp(-theta' * xi)) = - log(h_theta(xi))
temp3 = sum(-log(sigmoid(temp))) ;

J = (temp2 + temp3) / m;

size(X);
%grad = 1 / n \sum [ h_theta(xi) - yi ] * xi
%gradient has the same dimension as x!!!
% X = [x1';x2';...;xm']
grad  = sum( bsxfun(@times, X', sigmoid(temp') - y'), 2) / m ;

% =============================================================

end
