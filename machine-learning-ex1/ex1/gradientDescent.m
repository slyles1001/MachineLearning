function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%X2 = X(:,2);
%norm = (X2.-(mean(X2)))./(max(X2)-min(X2));
%noX = horzcat(X(:,1), norm);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ============================================================

    % Save the cost J in every iteration

    J_history(iter) = computeCost(X, y, theta);
    %J_history(iter), iter
    thetaChange =  (alpha/m) * sum((sum((theta' .* X),2) - y).*X)';
    theta = theta - thetaChange;

end

end
