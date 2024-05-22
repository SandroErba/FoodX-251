clear;
close all;

%% create clean dataset
train = readtable('../dataset/train_info_dirty.csv', 'ReadVariableNames', false);
% rinonima le colonne
train.Properties.VariableNames = {'imagePath', 'label'};
% converti le lable in categorical
train.label = categorical(train.label);

training_set_files = dir('../dataset/new_clean_train_set/');
training_set_files = {training_set_files.name}';
training_set_files = training_set_files(3:end, :);
rows_to_remove = ~ismember(train.imagePath, training_set_files);

train(rows_to_remove, :) = [];

writetable(train, '../dataset/train_info_clean.csv');

%% augmented 162

csv_file_path = '../dataset/train_info_clean.csv';
folder_input = '../dataset/cleaned_train_set/';

folder_output = '../dataset/augm_162/';

% Carica il file CSV
train_info = readtable(csv_file_path);

% Seleziona solo le righe con label 162
label_162_indices = train_info.label == 162;
train_info_162 = train_info(label_162_indices, :);

figure;
histogram(train_info.label, 'NumBins', max(train_info.label) - min(train_info.label) + 1, 'EdgeColor', 'black');
title('Distribuzione delle Label nel Training Set');
xlabel('Label');
ylabel('Frequenza');
grid on;

% Crea l'oggetto ImageDataAugmenter
imageAugmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandRotation', [-10, 10], ...
    'RandXShear',[-20 20], ...
    'RandXTranslation',[-2 2], ...
    'RandYTranslation',[-2 2], ...
    'RandScale', [1, 1.3], ...
    'FillValue', 0);  % valore di riempimento per le trasformazioni

%% loop
% Imposta il numero massimo di iterazioni per la data augmentation
num_iterations = 8;

% Itera sulle immagini con label 162
for i = 1:size(train_info_162, 1)
    % Ottieni il nome del file e la label
    filename = train_info_162.imagePath{i};
    label = train_info_162.label(i);

    % Carica l'immagine
    img = imread(fullfile(folder_input, filename));
    disp(fullfile(folder_input, filename));
    % Rimuovi l'estensione ".jpg" dal nome del file
    [~, filebase, ~] = fileparts(filename);
    % Applica la data augmentation e salva le immagini
    for j = 1:num_iterations
        % Applica la data augmentation
        augmented_img = augment(imageAugmenter, img);

        % Crea un nuovo nome per l'immagine
        new_filename = sprintf('%s_augm_%d.jpg', filebase, j);

        % Salva l'immagine nella cartella di output
        %imwrite(augmented_img, fullfile(folder_output, new_filename));

        % Aggiorna il file CSV con le nuove informazioni
        new_row = table({new_filename}, label, 'VariableNames', {'imagePath', 'label'});
        train_info = [train_info; new_row];
    end
end

figure;
histogram(train_info.label, 'NumBins', max(train_info.label) - min(train_info.label) + 1, 'EdgeColor', 'black');
title('Distribuzione delle Label nel Training Set');
xlabel('Label');
ylabel('Frequenza');
grid on;

%% Salva il file CSV aggiornato
writetable(train_info, csv_file_path);