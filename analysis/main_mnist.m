

testing_data = mnist_import('../data/mnist_testing.json');
nn = nn_import('../networks/mnist/model_admm_paul_01.json');

[L,T] = nn_lipschitz(nn, 1e-9);



result = nn_query(nn, @tanh, testing_data.x);
[~, argsp] = max(result,[],1);
[~, argsr] = max(testing_data.y,[],1);
 
acc = sum( argsr == argsp ) / size(testing_data.x,2);

str =  sprintf('Result =>  lipschitz: %0.5f    accuracy: %0.4f\n', L, acc);
disp( str );




% n = 30;
% 
% [XX,YY] = meshgrid( 2*(0:n)/n - 1.0 );
% 
% omega = [ reshape(XX, (n+1)^2, 1), reshape(YY, (n+1)^2, 1) ]';
% 
% res = softmax( nn_query(nn, @tanh, omega) );
% 
% hold on
% mesh(XX,YY, reshape(res(1,:)', n+1, n+1))
% mesh(XX,YY, reshape(res(2,:)', n+1, n+1))
% mesh(XX,YY, reshape(res(3,:)', n+1, n+1))
% hold off