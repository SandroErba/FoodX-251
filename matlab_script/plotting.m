train = readtable('../dataset/train_info_dirty.csv', 'ReadVariableNames', false);
% rinonima le colonne
train.Properties.VariableNames = {'imagePath', 'label'};
% converti le lable in categorical

distribuzione_classi = countcats(categorical(train.label));
classi_uniche = categories(categorical(train.label));

figure;
histogram(train.label, 'NumBins', max(train.label) - min(train.label) + 1, 'EdgeColor', 'black', 'FaceColor', [0.5, 0.5, 0.8]);
title('Distribuzione delle Label nel Training Set');
xlabel('Label');
ylabel('Frequenza');
grid on;
