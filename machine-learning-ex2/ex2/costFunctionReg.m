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
%t1 = theta(2:length(theta));
%size(t1), size(X)
%x1 = X(2:length(X), :);
%size(x1)

hX = sigmoid(X * theta);

% original functions
J = sum((-y' * sum(log(hX),2)))/m ...
 - sum((1 .- y)' * sum(log(1 .- hX),2)/m);
 grad = (X' * (sum(hX,2) - y))./m;

% 'normalize' 
theta(1) = 0;
J = J + (lambda/(2*m) * (theta' *theta));
grad = grad + sum((lambda/m).*theta, 2);

% =============================================================

end
