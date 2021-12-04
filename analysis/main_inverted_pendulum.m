

close all;
clear all;

dt = 0.02;

Ar = [1.0000    0.0200;
      0.4000    0.9733 ];
Br = [ 0 ;   0.5333];

Kr = [2.1437    0.4102];

system =  struct('A', Ar, 'B', Br);



bounds = [2.5 6]';

nn1 = nn_import('../networks/stability/model_ipc_find.json');
nn2 = nn_import('../networks/stability/model_ipc_final.json');
nn3 = nn_import('../networks/stability/rcontroller_ipc.json');

[f1,ly1,~,~] = nn_analyse( nn1, system, @tanh, bounds );
[f2,ly2,~,~] = nn_analyse( nn2, system, @tanh, bounds );
[f3,ly3,~,~] = nn_analyse( nn3, system, @tanh, bounds );
%[f4,ly4,~,~] = nn_analyse( nn4, system, @tanh, bounds );

x0 = [0.5, 0.1]';
range = [0 4];

[t1,y1,u1] = simulate(system, @(x,t) nn_query(nn1,@tanh,x), x0, dt, range);
[t2,y2,u2] = simulate(system, @(x,t) nn_query(nn2,@tanh,x), x0, dt, range);
[t3,y3,u3] = simulate(system, @(x,t) nn_query(nn3,@tanh,x), x0, dt, range);
%[t4,y4,u4] = simulate(system, @(x,t) nn_query(nn4,@tanh,x), x0, dt, range);

figure;
for i = 1:2
    subplot(2,1,i);
    plot(t1, y1(i,:), t2, y2(i,:), t3, y3(i,:));
    legend('find', 'final', 'ref')
end

datacsv = table;

figure
hold on;
plot_ellipse( nn1.lyapu );
plot_ellipse( nn2.lyapu );
plot_ellipse( nn3.lyapu );
line1 = plot_ellipse( ly1 );
line2 = plot_ellipse( ly2 );
line3 = plot_ellipse( ly3 );

legend('find (train)', 'final (train)', 'ref (train)', 'find', 'final', 'ref')
hold off;


datacsv.findx = line1(1,:)';
datacsv.findy = line1(2,:)';
datacsv.finalx = line2(1,:)';
datacsv.finaly = line2(2,:)';
datacsv.refx = line3(1,:)';
datacsv.refy = line3(2,:)';

writetable(datacsv, 'ivp_ellipse.csv', 'Delimiter', ',');