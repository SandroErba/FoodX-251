%%% fine tuning di una rete pre-addestrata. 
%%% il modello dovrebbe essere più efficiente
clear;

%% load network
net = alexnet;
analyzeNetwork(net); 

%% load CSV
disp("read validation daset")
dataset = readtable('../dataset/val_info.csv', 'ReadVariableNames', false);

% rinonima le colonne
dataset.Properties.VariableNames = {'imagePath', 'label'};
% converti le lable in categorical
dataset.label = categorical(dataset.label);
classi = unique(dataset.label);

dataset = dataset(ismember(dataset.label, categorical(1:20)), :);


%% load sub-sample
% Trova il conteggio di immagini per ogni classe
% imageCounts = grpstats(dataset, 'label', 'numel');
% 
% % Limita il numero di immagini per ogni classe a 10
% maxImagesPerClass = 10;
% 
% % Itera attraverso tutte le classi
% uniqueClasses = categories(dataset.label);
% trainFiltered = table();
% for i = 1:length(uniqueClasses)
%     currentClass = uniqueClasses(i);
% 
%     % Seleziona le prime 200 immagini per la classe corrente
%     classSubset = dataset(dataset.label == currentClass, :);
%     if size(classSubset, 1) > maxImagesPerClass
%         classSubset = classSubset(1:maxImagesPerClass, :);
%     end
% 
%     % Aggiungi il subset al risultato
%     trainFiltered = [trainFiltered; classSubset];
% end
% 
% dataset = trainFiltered;

%% create datastore
imageTrainFolder = './dataset/val_set';
imds = imageDatastore(fullfile(imageTrainFolder, dataset.imagePath), 'Labels', dataset.label);

%% split train / validation
% il validation serve per il tuning degli iperparametri
disp('Split train / val')
rng(1); % Per riproducibilità
fractionTrain = 0.9; % Percentuale di immagini da utilizzare per il training set
[imdsTrain, imdsValidation] = splitEachLabel(imds, fractionTrain, 'randomized');

newSize = [227, 227];
imdsTrain = augmentedImageDatastore(newSize, imdsTrain);
imdsValidation = augmentedImageDatastore(newSize, imdsValidation);


%% data augmentation

%% cut layers 
disp('cut layers')
layersTransfer = net.Layers(1:end-3);

%% replace layers 
disp('replace layers')
numClasses = 20;
layers=[layersTransfer  
    fullyConnectedLayer(numClasses, ...
    'WeightLearnRateFactor', 20,... 
    'BiasLearnRateFactor', 20); 
    softmaxLayer 
    classificationLayer];

%% Training options
disp('training options')
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 20,...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% fine-tuning training
disp('training')
netTransfer = trainNetwork(imdsTrain, layers, options);

%% Valuta la rete
disp("read training set")
train = readtable('./dataset/train_info_dirty.csv', 'ReadVariableNames', false);

% rinonima le colonne
dataset.Properties.VariableNames = {'imagePath', 'label'};
% converti le lable in categorical
train.label = categorical(train.label);

imageTrainFolder = './dataset/train_set';
imds = imageDatastore(fullfile(imageTrainFolder, dataset.imagePath), 'Labels', dataset.label);

YPred = classify(netTransfer, imds);
YpredictTraining = imdsValidation.Labels;
accuracy = sum(YPred == YpredictTraining) / numel(YpredictTraining);
disp(['Accuracy: ' num2str(accuracy)]);

%% find bad predicts
incorrectIndices = find(YPred ~= YpredictTraining);

incorrectImagePaths = imdsValidation.Files(incorrectIndices);
outputFolder = './dataset/incorrect_predictions';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for i = 1:numel(incorrectIndices)
    imgPath = incorrectImagePaths{i};
    img = imread(imgPath);
    
    % Copia l'immagine nella nuova cartella
    [~, imgName, imgExt] = fileparts(imgPath);
    newImgPath = fullfile(outputFolder, [imgName imgExt]);
    imwrite(img, newImgPath);
end

disp("Immagini errate copiate in: " + outputFolder);