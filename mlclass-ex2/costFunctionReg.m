function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%!!! Don't regularize theta_0!!!!
% J(theta) = 1 / m \sum [ ( 1-yi) * theta' * xi + log(1+exp(-theta' * xi))]
% + lambda / (2m) * theta' * theta
temp = X * theta;

%\sum [(1-yi) * theta' * xi]
temp2 = sum(temp .* (1 - y)) ;

%\sum [log(1+exp(-theta'*xi))]
% log(1 + exp(-theta' * xi)) = - log(h_theta(xi))
temp3 = sum(-log(sigmoid(temp))) ;

J = (temp2 + temp3) / m + lambda * (theta(2:n)' * theta(2:n)) / ( 2*m);

size(X);
%grad = 1 / n \sum [ h_theta(xi) - yi ] * xi + lambda / m * theta
%gradient has the same dimension as x!!!
% X = [x1';x2';...;xm']
grad  = sum( bsxfun(@times, X', sigmoid(temp') - y'), 2) / m;
grad(2:n) = grad(2:n)  + lambda / m * theta(2:n);


% =============================================================

end
