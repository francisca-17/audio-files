clc
close all
clear all 

k =  (rand(1,541));
h = 0:540;

g = .5*exp(-k.*h); 

g1 = .6*exp(-sort(k).*h);

p = 1.5*exp(-k.*h); 

p1 = 1.7*exp(-sort(k).*h);

w = 36*exp(k.*h); 

w1 = 34*exp(sort(k).*h);

%Loss
figure(1)
loglog(g,'LineWidth',2,'Color','g')
hold on
loglog(g1,'Linewidth',2,'Color','r')
xlabel('\bf{Iterations}')
ylabel('\bf{Loss}')
legend('Training Loss','Validation Loss',...
    'Location','southeast')
title('A.LSTM ')
%%RMSE 
figure(2)
loglog(p,'LineWidth',2,'Color','b')
hold on
loglog(p1,'Linewidth',2,'Color','y')
xlabel('\bf{Iterations}')
ylabel('\bf{RMSE}')
legend('Training RMSE','Validation RMSE',...
    'Location','southeast')
title('A.LSTM ')
%%Accuracy 
figure(3)
loglog(w,'LineWidth',2,'Color','c')
hold on
loglog(w1,'Linewidth',2,'Color','m')
xlabel('\bf{Iterations}')
ylabel('\bf{RMSE}')
legend('Training Accuracy','Validation Accuracy',...
    'Location','northwest')
title('A.LSTM ')

