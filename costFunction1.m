function J = costFunction1(X, y, theta)

%X is design matrix containing training examples
%y is class labels

m = size(X,1);
prediction = X*theta;  %prediction of hypothesis on all m examples
sqrErrors = (prediction - y) .^ 2;

J = 1/(2*m) * sum(sqrErrors);