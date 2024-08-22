clc
clear all
close all 

X = [1 0]; 

labels = {'LSTM' 'MissClassified'}; 

subplot(2,1,1)



pie(X)


legend(labels,'Location','eastoutside')

X = [98 2]; 

labels = {'MLPs' 'MissClassified'}; 

subplot(2,1,2)
pie(X)

legend(labels, 'Location','eastoutside')