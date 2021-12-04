function [alphas,betas] = sector_bounds(nn, x, act)
    alphas = cell({});
    betas = cell({});
    
    wstar = 0*x;
    vstar = 0;
    
    wupper = x;
    wlower = -x;
    
    vupper = 0;
    vlower = 0;
    
    for i = 1:length(nn.weights)-1
        vstar = nn.weights{i} * wstar;
        
        vupper = 1/2*nn.weights{i}*(wupper+wlower) + 1/2*abs(nn.weights{i})*abs(wupper-wlower);
        vlower = 1/2*nn.weights{i}*(wupper+wlower) - 1/2*abs(nn.weights{i})*abs(wupper-wlower);
       
        
        if ~isempty(nn.biases)
            vstar = nn.biases{i}; vupper = vupper + nn.biases{i}; vlower = vlower + nn.biases{i};
        end
        
        alphas{i} = min( (act(vupper) - act(vstar) ) ./ (vupper - vstar), ...
                         (act(vstar) - act(vlower) ) ./ (vstar  - vlower));
        
                     
        %% PASST WAHRSCHEINLICH NICHT!!!!
        % beats{i}  = 1 + 0*alphas{i}; 
        % 1.0*(vstar > 0).*(vlower < -2/3) + 1.0*(vstar < 0).*(vupper > 2/3)
        %betas{i}  = max( (act(vupper) - act(vstar) ) ./ (vupper - vstar), ...
        %                 (act(vstar) - act(vlower) ) ./ (vstar  - vlower) );
        
        betas{i} = 1 + 0*alphas{i};
        
        %betas{i}  = max( betas{i}, 1.0*(vstar > 0).*(vlower < 0) + 1.0*(vstar < 0).*(vupper > 0) );
        
        wstar  = act(vstar);
        wupper = act(vupper);
        wlower = act(vlower);
    end
end

