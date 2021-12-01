
function [L, T] = lipschitz( nn, eps )


      weights = nn.weights;
      

      %addpath(genpath('./mosek/9.2/toolbox/r2015a'))
      %addpath(genpath('./yalmip'))

      ops = sdpsettings('solver','mosek','verbose',0,'debug',0);

      
      layers = size(weights,2);
      dims = cell(1,layers);
      
      for i = 1:layers
            dims{i} = size(weights{i},2);
      end
      

      rho = sdpvar(1,1);
      T = sdpvar( sum([dims{2:layers}]) ,1);

      A = [blkdiag(weights{1:layers-1}), zeros(sum([dims{2:layers}]), dims{layers})];
      B = [zeros(sum([dims{2:layers}]), dims{1}), eye(sum([dims{2:layers}]))];
      
      
      Q = blkdiag(-rho*eye(dims{1}),zeros(sum([dims{2:layers-1}])), ...
            weights{layers}'*weights{layers});
   
      P = [A; B]' * [zeros(sum([dims{2:layers}])), diag(T); diag(T), diag(-2*T)] * [A; B] + Q;
      
      
      optimize( [P <=  -eps*eye(sum([dims{1:layers}]))] , rho, ops );
      
      L = sqrt(value(rho));
      T = value(T);
end