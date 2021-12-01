function [t,y,g] = simulate( sys, u, x0, T, range)
    t = range(1):T:range(2);
    x = x0;
    y = [];
    g = [];
    
    for i = t
        y = [y x];
        g = [g u(x,i)];
        x = sys.A*x+sys.B*u(x,i);
        %y = [y x];
    end
end
