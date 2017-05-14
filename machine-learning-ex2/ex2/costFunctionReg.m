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


h=sigmoid(X*theta);

%J = (1/m)*(-y'*log(h) - (1 - y)'*log(1-h));
%grad = (1/m)*X'*(h-y);


%st = theta(2:size(theta));
%theta_new = [0;st];
%theta_new = 0.01*theta; %reducing the value of theta to ~ 0

%J = (1/m)*(-y'*log(h) - (1 - y)'*log(1-h)) +%(lambda/(2*m)*theta_new'*theta_new);

%grad = (1/m)*X'*(h-y) + ((lambda/m)*theta_new);




[J, grad] = costFunction(theta, X, y);
% 
regularization_parameter = sum(theta(2:end).^2);

J = J + (lambda/(2*m))*regularization_parameter;

%we should not regularize the parameter θ0.
%In Octave/MAT- LAB, recall that indexing starts from 1, hence, you %should not be regularizing the theta(1) parameter (which corresponds to %θ0) in the code
%

grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);




% =============================================================

end
