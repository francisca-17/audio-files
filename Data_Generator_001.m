
clc 
clear all
close all

fs = 44.1e3;
duration = 0.5;
N = duration*fs;
M=1000;
wFake = sin(2*rand([N,M]) - 1)+...
    cos(2*rand([N,M]) - 1);
wFake = wFake./max(abs(wFake),[],'all');
wLabels = repelem(categorical("fake"),1000,1);

bReal = filter(1,[1,-0.999],wFake);
bReal = bReal./max(abs(bReal),[],'all');
%bReal = sin(bReal)+cos(bReal); 
bLabels = repelem(categorical("real"),1000,1);

classNames = ["fake", "real"];

figure(1)
for i = 1: 3
    
 q1 = [cellstr('First signal') cellstr('Second signal ') cellstr('Third signal')];
 q = [cellstr('r') cellstr('g') cellstr('c')];
hold on
loglog(bReal(:,i),'LineWidth',4, 'Color',q{i})
hold on
legend(q1,'Location','southeast')
end
xlabel('\bf{Time(s)}')
ylabel('\bf{f*}')
grid on


figure(2)
for i = 1: 3
    
 q = [cellstr('r') cellstr('g') cellstr('c')];
hold on
loglog(wFake(1:200,i),'LineWidth',4, 'Color',q{i})
hold on
legend(q1,'Location','southeast')
end
xlabel('\bf{Time(s)}')
ylabel('\bf{f*}')
grid on

%% Save the data

save("Data.mat","bReal","wFake")

