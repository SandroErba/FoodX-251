clear;

%% load CSV
disp("read dataset")
train = readtable('../dataset/train_info_clean.csv', 'ReadVariableNames', false);
datasetPath = '../dataset/cleaned_train_set';

% rinonima le colonne
train.Properties.VariableNames = {'imagePath', 'label'};
% converti le lable in categorical
train.label = categorical(train.label);
classi = unique(train.label);


%% modifica il dataset
newTrain = train;

% TBD con il tuo - non so perché non mi andava con quello relativo
newDatasetPath = '../dataset/new_sub_train';


for i = 1:height(newTrain)
    imagePath = fullfile('../dataset/cleaned_train_set/', newTrain.imagePath{i});

    disp(newTrain.imagePath{i});

    % Ottieni l'etichetta corrente
    label = char(newTrain.label(i));
    
    % Crea il percorso della nuova cartella basata sull'etichetta
    newFolderPath = fullfile(newDatasetPath, label);

    % Verifica se la cartella esiste, altrimenti creala
    if ~isfolder(newFolderPath)
        mkdir(newFolderPath);
    end

    % Estrai il nome del file senza estensione
    [~, fileName, fileExt] = fileparts(imagePath);

    % Crea il percorso completo per la nuova immagine di destinazione
    newImagePath = fullfile(newFolderPath, [fileName, fileExt]);

    % Copia l'immagine nella nuova cartella
    copyfile(imagePath, newImagePath);

    % Aggiorna il percorso nell'array di percorsi del nuovo dataset
    newTrain.newImagePath{i} = newImagePath;
end