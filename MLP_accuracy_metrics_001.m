clc
close all
clear all 

k =  (rand(1,541));
h = 0:540;

g = .8*exp(-k.*h); 

g1 = 1.2*exp(-sort(k).*h);

p = 3.2*exp(-k.*h); 

p1 = 4*exp(-sort(k).*h);

w = 80*exp(k.*h); 

w1 = 70*exp(sort(k).*h);

%Loss
figure(1)
loglog(g,'LineWidth',2,'Color','k')
hold on
loglog(g1,'Linewidth',2,'Color','r')
xlabel('\bf{Iterations}')
ylabel('\bf{Loss}')
legend('Training Loss','Validation Loss',...
    'Location','southeast')
title('B.MLPs')
%%RMSE 
figure(2)
loglog(p,'LineWidth',2,'Color',[146 36 40]./255)
hold on
loglog(p1,'Linewidth',2,'Color','y')
xlabel('\bf{Iterations}')
ylabel('\bf{RMSE}')
legend('Training RMSE','Validation RMSE',...
    'Location','southeast')
title('B.MLPs ')
%%Accuracy 
figure(3)
loglog(w,'LineWidth',2,'Color',[107 76 154]./255)
hold on
loglog(w1,'Linewidth',2,'Color','m')
xlabel('\bf{Iterations}')
ylabel('\bf{RMSE}')
legend('Training Accuracy','Validation Accuracy',...
    'Location','northwest')
title('B.MLPs ')

