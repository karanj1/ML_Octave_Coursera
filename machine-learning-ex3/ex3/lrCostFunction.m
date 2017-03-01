function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% First calculate the unregularized cost (J) and gradient (grad) for logistic regression
% (Taken straight from Coding Exercise 2 (costFunction.m))

hx = sigmoid(X * theta);

c1 = log(hx);
c2 = log(1-hx);

J = -1/m * (y' * c1 + (1-y)' * c2);

diff = hx - y;
grad = 1/m * (X' * diff);


% Next calculate the regularized cost (J) and gradient (grad) for logistic regression
% (Taken straight from Coding Exercise 2 (costFunctionReg.m))

% this effectively ignores "theta zero" in the following calculations
theta_zeroed_first = [0; theta(2:length(theta));];

J = J + lambda/(2*m) * (theta_zeroed_first' * theta_zeroed_first);
grad = grad .+ (lambda / m) * theta_zeroed_first;


% =============================================================

grad = grad(:);

end