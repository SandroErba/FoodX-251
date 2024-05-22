clear;

%% Impostazioni
datasetPath = '../dataset/sub_clean_train_set';
newDatasetPath = '../dataset/new_clean_train_set';

% Creare la cartella di destinazione se non esiste
if ~isfolder(newDatasetPath)
    mkdir(newDatasetPath);
end

%% Trova tutte le immagini nelle sottocartelle e spostale nella cartella generica
subfolders = dir(datasetPath);
subfolders = subfolders([subfolders.isdir] & ~strcmp({subfolders.name}, '.') & ~strcmp({subfolders.name}, '..'));

for i = 1:length(subfolders)
    disp(i)
    subfolderPath = fullfile(datasetPath, subfolders(i).name);
    
    % Trova tutte le immagini nella sottocartella
    images = dir(fullfile(subfolderPath, '*.jpg'));  % Assumendo che le immagini siano in formato jpg
    
    for j = 1:length(images)
        imagePath = fullfile(subfolderPath, images(j).name);        
        % Estrai il nome del file senza estensione
        [~, fileName, fileExt] = fileparts(imagePath);
        
        % Crea il percorso completo per la nuova immagine di destinazione
        newImagePath = fullfile(newDatasetPath, [fileName, fileExt]);
        
        % Copia l'immagine nella nuova cartella
        copyfile(imagePath, newImagePath);
    end
end

disp('Il processo di unione delle immagini è completato.');


%% Leggi la tabella CSV
csvFilePath = '../dataset/train_info_dirty.csv';
newCsvFilePath = '../dataset/provaprovaprova.csv';
disp('Lettura del file CSV...');
train = readtable(csvFilePath, 'ReadVariableNames', false);
train.Properties.VariableNames = {'imagePath', 'label'};
train.label = categorical(train.label);

%% Trova i nomi dei file nella nuova cartella
disp('Ricerca dei nomi dei file nella nuova cartella...');
newImages = dir(fullfile(newDatasetPath, '*.jpg'));  % Assumendo che le immagini siano in formato jpg

% Estrai i nomi dei file con estensione
newFileNames = {newImages.name};

%% Rimuovi i nomi dei file non presenti nella nuova cartella dalla tabella CSV
disp('Rimozione dei nomi dei file non presenti nella nuova cartella dalla tabella CSV...');
toRemove = ~ismember(train.imagePath, newFileNames);
train(toRemove, :) = [];

%% Salva la tabella CSV aggiornata
disp('Salvataggio della tabella CSV aggiornata...');
writetable(train, newCsvFilePath, 'WriteVariableNames', false);

disp('Il processo di rimozione è completato.');