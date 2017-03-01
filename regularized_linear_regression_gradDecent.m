function grad = grad(theta, X, y, lambda)

% Your code goes here.
m = length(y);

diff = X*theta - y; 

% excluded the first theta value
theta1 = [0 ; theta(2:end, :)];

grad = (X'*diff + lambda*theta1)/m;


endfunction