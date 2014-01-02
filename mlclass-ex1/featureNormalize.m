function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% same as mu = mean(X)
mu = mean(X,1); 
sigma = std(X);

% In fact, for loops _are_ optimal, in terms of memory consumption, if you want to the new matrix to overwrite the old matrix. For example
% 
% M=bsxfun(@minus,M,vector);
% 
% will cause bsxfun() to generate an intermediate matrix the same size as M and then overwrite M with this intermediate matrix . Conversely, the following for-loop will allocate no new memory:
% 
% for ii=1:n
%   M(:,ii)=M(:,ii)-vector;
% end 

X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
