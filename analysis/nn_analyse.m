
function [feasible, res, lams, alphas] = nn_analyse( nn, sys, activation, vhead )
    
    dims = network_dims( nn );
    
    [alphas,betas] = sector_bounds( nn, vhead, activation );
    
    P = sdpvar( dims(1), dims(1), 'symmetric' );
    L = cell({});
    for i = 1:length(nn.weights)-1
        L{i} = sdpvar( size(nn.weights{i},1), 1 ); 
    %    L{i} = nn.lambdas{i};
    end
     
    j11 = cell({}); j22 = cell({}); j12 = cell({});
    for i = 1:length(L)
        Lalpha = diag(alphas{i});
        Lbeta = diag(betas{i});
        j11{i} = -2*Lalpha*Lbeta*diag(L{i});
        j22{i} = -2*diag(L{i});
        j12{i} = (Lalpha+Lbeta)*diag(L{i});
    end
    
   
    N = blkdiag( nn.weights{:} );
    
    Nux = N( sum(dims(2:end-1))+1:end, 1:dims(1) );
    Nuw = N( sum(dims(2:end-1))+1:end, dims(1)+1:end);
    Nvx = N( 1:sum(dims(2:end-1)), 1:dims(1));
    Nvw = N( 1:sum(dims(2:end-1)), dims(1)+1:end);
    
    
    
    
   
    Rv = [ eye(dims(1)), zeros(dims(1), size(Nuw,2));  Nux, Nuw  ];
    Rphi = [ Nvx, Nvw; zeros(size(Nvw, 2), size(Nvx, 2)), eye(size(Nvw, 2)) ];
    
    
    Z = [ sys.A'*P*sys.A-P, sys.A'*P*sys.B;
          sys.B'*P*sys.A, sys.B'*P*sys.B];
      
    M = [ blkdiag(j11{:}), blkdiag(j12{:}); 
          blkdiag(j12{:}), blkdiag(j22{:})];

      
    Y = Rv'*Z*Rv + Rphi'*M*Rphi;
    
    F = [ P >= 1e-6 , Y <= -1e-6 ];
    for i = 1:length(L)
       F = [F, diag(L{i}) >= 1e-6]; 
    end
    
    H = eye( dims(1) );
    for i = 1:size(vhead,1)
        D = [ vhead(i)^2, H(i,:);
              H(i,:)', P];
        F = [F, D >= 1e-6];    
    end
    
    options = sdpsettings('verbose',0,'solver','mosek');
    diagnostics = optimize( F, trace(P), options );
    feasible = diagnostics.problem == 0;
    
    res = value(P);
    
    lams = cell({});
    for i = 1:length(nn.weights)-1
        lams{i} = value(L{i});
    end
end

function dims = network_dims(nn)
    dims = [ size(nn.weights{1},2) ];
     
    for i = 1:length(nn.weights)
        dims(i+1) = size( nn.weights{i}, 1 ); 
    end
end
