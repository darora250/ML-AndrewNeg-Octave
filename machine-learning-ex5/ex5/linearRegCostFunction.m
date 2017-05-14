function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%fprintf('Size of X \n');
%size(X)

%fprintf('Size of theta \n');
%size(theta)

% X is 12x2 matrix
% theta is 2x1 vector

%theta

h = X*theta;

% h is 12x1 vector
%fprintf('Size of h \n');
%size(h)

%%%%Regularized linear regression cost function
%%%%===========================================
%% complete J including the regularizations (with peanalization of theta)
%% used theta(2:end) coz theta0(theta(1) in octave) should not be regularized

J = ( (1/(2*m)) * (sum((h - y).^2)) ) + ( (lambda / (2*m)) * (sum(theta(2:end).^2)) );

%size(J)

%%%%Regularized linear regression gradient
%%%%======================================

x0 = X(:,1);	   %% x0 is now 12x1 vector with just the first column
%x0
x = X(:,2:end);    %% x is now 12x1 vector (initial data) with second column

%% since x0 or x is 12x1 and (h-y) is also 12x1, 
%% doing x' * (h - y) to do correct matrix multiplication. 
%% NOTE: when doing this, no need to sum up the results

theta0 = ( (1 /m) * (x0' * (h - y) ) );
theta1 = ( (1 /m) * ( x' * (h - y) ) ) +  ( (lambda/m) * theta(2:end)  );

grad = [theta0;theta1];

%% another way done in just one command
%grad = (1/m)*(X'*(h-y)) + [0; (lambda/m)*theta(2:end)];






% =========================================================================

grad = grad(:);

end
