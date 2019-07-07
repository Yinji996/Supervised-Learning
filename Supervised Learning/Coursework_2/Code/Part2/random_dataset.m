% genenrate a dataset that contains m patterns x1, x2, ...xm which are sampled uniformly ar randowm from {-1,1}^n
function [X, Y] = random_dataset(m, n)
X = randi([0 1],m,n);  % creat a m*n random matrix from {0,1}
X(X==0) = -1; % change 0 to -1
Y = X(:,1);
end
