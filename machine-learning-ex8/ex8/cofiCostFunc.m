function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%num_movies
%num_users 	

%X is 5 x 3
%fprintf('Size of X \n');
%size(X)

%Theta is 4 x 3
%fprintf('Size of Theta \n');
%size(Theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Collaborative filtering cost function%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Non Vectorized Working Implementation%%%%%%

%for i = 1 : num_movies
%	for j = 1 : num_users

%		if (R(i,j) == 1)

			%doing X * theta' bcoz X is 1 x 3 nd theta is 3 x 1
			% so the net result will be 1 x 1
%			J = J + ( (X(i,:) * Theta(j,:)') - Y(i,j) ) ^ 2; 
%		end

%	endfor
%endfor

%J = J/2;

%%%%%%% Vectorized Working Implementation%%%%%%

J = sum(sum((R==1) .* ((X * Theta' - Y) .^ 2))) / 2; %'	


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Collaborative filtering gradient%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%size((X * Theta') - Y)
%size(Theta)
%size(R)

% removing sum() bcoz with it the size of both was 1 x 3
%% still have doubts around this...so check again...
X_grad = ((R==1) .* ((X * Theta') - Y  ) * Theta); %'

Theta_grad = (((R==1) .* ((X * Theta') - Y  ) )' * X); %'

%fprintf('Size of X_grad \n');
%size(X_grad)
%fprintf('Size of Theta_grad \n');
%size(Theta_grad)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%Regularized Cost Function%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regularize_Theta = (lambda/2) * sum(sum(Theta .^ 2)  );

regularize_X = (lambda/2) * sum(sum(X .^ 2)  );

J = J + regularize_Theta + regularize_X;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%Regularized Gradient%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


X_grad = X_grad_regularized = X_grad + (lambda*X);

Theta_grad = Theta_grad_regularized = Theta_grad + (lambda*Theta);


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
