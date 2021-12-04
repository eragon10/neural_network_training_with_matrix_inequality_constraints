
close all;
clear all;



%% parameters
% Nominal speed of the vehicle travels at.
U = 28; % m/s
% Model
% Front cornering stiffness for one wheel.
Ca1 = -61595; % unit: Newtons/rad
% Rear cornering stiffness for one wheel.
Ca3 = -52095; % unit: Newtons/rad

% Front cornering stiffness for two wheels.
Caf = Ca1*2; % unit: Newtons/rad
% Rear cornering stiffness for two wheels.
Car = Ca3*2; % unit: Newtons/rad

% Vehicle mass
m = 1670; % kg
% Moment of inertia
Iz = 2100; % kg/m^2

% Distance from vehicle CG to front axle
a = 0.99; % m
% Distance from vehicle CG to rear axle
bd = 1.7; % m

g = 9.81;

% sampling period
dt = 0.02;

% Continuous-time state space matrices
% States are lateral displacement(e) and heading angle error(deltaPsi) and
% their derivatives.
% Inputs are front wheel angle and curvature of the road.
Ac = [0 1 0 0; ...
    0, (Caf+Car)/(m*U), -(Caf+Car)/m, (a*Caf-bd*Car)/(m*U); ...
    0 0 0 1; ...
    0, (a*Caf-bd*Car)/(Iz*U), -(a*Caf-bd*Car)/Iz, (a^2*Caf+bd^2*Car)/(Iz*U)];
Bc = [0;
    -Caf/m;
    0; ...
    -a*Caf/Iz];

system =  struct('A', Ac*dt + eye(4), 'B', Bc*dt);

bounds = [2 5 1 5]';

nn1 = nn_import('../networks/stability/model_vlc_find_im.json');
nn2 = nn_import('../networks/stability/model_vlc_final_im.json');
nn3 = nn_import('../networks/stability/rcontroller_vlc.json');

[f1,ly1,~,~]  = nn_analyse( nn1, system, @tanh, bounds );
[f2,ly2,~,~]  = nn_analyse( nn2, system, @tanh, bounds );
[f3,ly3,~,~]  = nn_analyse( nn3, system, @tanh, bounds );
%[f4,ly4,~,~]  = nn_analyse( nn4, system, @tanh, bounds );

x0 = [-1.2, -2 0.2 2]';
range = [0 4];

[t1,y1,u1] = simulate(system, @(x,t) nn_query(nn1,@tanh,x), x0, dt, range);
[t2,y2,u2] = simulate(system, @(x,t) nn_query(nn2,@tanh,x), x0, dt, range);
[t3,y3,u3] = simulate(system, @(x,t) nn_query(nn3,@tanh,x), x0, dt, range);
%[t4,y4,u4] = simulate(system, @(x,t) nn_query(nn4,@tanh,x), x0, dt, range);

simdata = table;

figure;
for i = 1:4
    subplot(4,1,i);
    plot(t1, y1(i,:), t2, y2(i,:), t3, y3(i,:));
    legend('find', 'final', 'ref')
end

simdata.time = t1';

simdata.findx = y1(1,:)';
simdata.findy = y1(2,:)';
simdata.findz = y1(3,:)';
simdata.findw = y1(4,:)';
simdata.findu = u1';

simdata.finalx = y2(1,:)';
simdata.finaly = y2(2,:)';
simdata.finalz = y2(3,:)';
simdata.finalw = y2(4,:)';
simdata.finalu = u2';

simdata.refx = y3(1,:)';
simdata.refy = y3(2,:)';
simdata.refz = y3(3,:)';
simdata.refw = y3(4,:)';
simdata.refu = u3';

writetable(simdata, 'vlc_simulation.csv', 'Delimiter', ',');


datacsv = table;


figure
subplot(1,2,1);
hold on;
plot_ellipse( nn1.lyapu(1:2,1:2) );
plot_ellipse( nn2.lyapu(1:2,1:2) );
plot_ellipse( nn3.lyapu(1:2,1:2) );
line1a = plot_ellipse( ly1(1:2,1:2) );
line2a = plot_ellipse( ly2(1:2,1:2) );
line3a = plot_ellipse( ly3(1:2,1:2) );
legend('find (train)', 'final (train)', 'ref (train)', 'find', 'final', 'ref')
hold off;
subplot(1,2,2);
hold on;
plot_ellipse( nn1.lyapu(3:4,3:4) );
plot_ellipse( nn2.lyapu(3:4,3:4) );
plot_ellipse( nn3.lyapu(3:4,3:4) );
line1b = plot_ellipse( ly1(3:4,3:4) );
line2b = plot_ellipse( ly2(3:4,3:4) );
line3b = plot_ellipse( ly3(3:4,3:4) );
legend('find (train)', 'final (train)', 'ref (train)', 'find', 'final', 'ref')
hold off;


datacsv.findax = line1a(1,:)';
datacsv.finday = line1a(2,:)';
datacsv.finalax = line2a(1,:)';
datacsv.finalay = line2a(2,:)';
datacsv.refax = line3a(1,:)';
datacsv.refay = line3a(2,:)';

datacsv.findbx = line1b(1,:)';
datacsv.findby = line1b(2,:)';
datacsv.finalbx = line2b(1,:)';
datacsv.finalby = line2b(2,:)';
datacsv.refbx = line3b(1,:)';
datacsv.refby = line3b(2,:)';

writetable(datacsv, 'vlc_ellipse.csv', 'Delimiter', ',');

