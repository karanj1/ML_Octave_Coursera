function cost = cost(theta, X, y, lambda)

%cost Compute cost and gradient for regularized linear 
%regression with multiple variables

% Your code goes here.
m = length(y);

%hx = sigmoid(X * theta);
%prediction = 1 ./ (1 + e.^(-(X*theta)));  %prediction of hypothesis on all m examples for Logitic regression

prediction = X*theta;    %for linear regression
sqrErrors = (prediction - y) .^ 2;

% excluded the first theta value
theta1 = [0 ; theta(2:end, :)];

cost = 1/(2*m) * (sum(sqrErrors) + lambda * sum(theta1 .^2));

endfunction