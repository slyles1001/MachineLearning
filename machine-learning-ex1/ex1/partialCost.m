function T = partialCost(X, y, theta, i)

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


T =  (1/m) * sum((sum((theta' .* X),2) - y).*X(:,i));



end