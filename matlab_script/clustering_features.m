clear;

%% load CSV
disp("read dataset")
train = readtable('../dataset/train_info_dirty.csv', 'ReadVariableNames', false);
datasetPath = '../dataset/train_set';
OutliersPath = '../dataset/outliers_train_set/';

% rinonima le colonne
train.Properties.VariableNames = {'imagePath', 'label'};
% converti le lable in categorical
train.label = categorical(train.label);
classi = unique(train.label);
net = mobilenetv2;

%% estrazione features

% variabili
total_outliers = [];
features_flattened = zeros(1, 1001);
num_clusters = 5;

for i=1:length(classi)
    actual_class = train(train.label == classi(i), :); 
    label = char(actual_class.label(1));
    fprintf('\nClasse attuale %s\n', label);
    
    % numero di immagini nella classe
    length_class = length(actual_class.label); 
    
    % pre-allocazione dati
    data = zeros(length_class, 1001);
    data_hog = zeros(length_class, 26245);
    data_texture = zeros(length_class, 13);
    tot_mean = zeros(length_class, 1);
    tot_var = zeros(length_class, 1);
    media = 0;
    varianza = 0;
    
    tic;
    for j=1:length_class
        % Leggi l'immagine
        imagePath = fullfile(datasetPath, actual_class.imagePath{j});
        %disp(actual_class.imagePath{j});
        img = imread(imagePath);

        % calcola media e varianza
        m = mean(img(:));
        tot_mean(j, :) = m;
        v = var(double(img(:)));
        tot_var(j, :) = v;
        media = media + m;
        varianza = varianza + var(double(img(:)));
        
        % imresize per mobile net
        img = imresize(img, net.Layers(1).InputSize(1:2));  % Dimensioni di input di MobileNetV2
        img = im2single(img);  % Converte l'immagine in tipo singolo

        % Estrai le features con CNN mobilenet
        layer = 'Logits';
        features = activations(net, img, layer);

        features_flattened(1, 1) = j; % prima colonna con l'indice dell'immagine
        features_flattened(1, 2:end) = reshape(features, [1, 1000]);
        data(j, :) = features_flattened; % Aggiungi le features al vettore di features

        % feature HOG
        % Calcola le HOG features
        hogFeatures = extractHOGFeatures(img);
        data_hog(j, 1) = j;
        data_hog(j, 2:end) = hogFeatures;

        % feature texture GLCM
        % Converte l'immagine in scala di grigi, se necessario
        grayImg = rgb2gray(img);
        
        % Calcola la matrice di co-occorrenza dei livelli di grigio
        glcm = graycomatrix(grayImg, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
        % Calcola le feature di texture dalla matrice GLCM
        stats = graycoprops(glcm);
        texture_features = [stats.Contrast, stats.Correlation, stats.Energy];
        data_texture(j, 1) = j;
        data_texture(j, 2:end) = texture_features;

    end
    toc;
    media = media / length_class;
    varianza = varianza / length_class;
    
    %% outliers mean and var
    treshold_mean = 50;
    threshold_var = 6000; 
    
    outlier_mean_var_indices = find(abs(tot_mean - media) > treshold_mean | abs(tot_var - varianza) > threshold_var);
    outlier_percentage_mean = height(outlier_mean_var_indices) / length_class * 100;
    maxIter = 100; % per evitare loop

    % facciamo in modo di rimuovere tra il 20 e il 30% delle immagini
    while (outlier_percentage_mean < 5 || outlier_percentage_mean > 10) && maxIter > 1
        maxIter = maxIter - 1;
        outlier_mean_var_indices = find(abs(tot_mean - media) > treshold_mean | abs(tot_var - varianza) > threshold_var);
        outlier_percentage_mean = height(outlier_mean_var_indices) / length_class * 100;
        %fprintf('media %d var %d outliers %.2f%%\n', treshold_mean, threshold_var, outlier_percentage_mean);
        if(outlier_percentage_mean < 5)
            treshold_mean = treshold_mean - 2;
            threshold_var = threshold_var - 100;
        elseif(outlier_percentage_mean) > 10
            treshold_mean = treshold_mean + 2;
            threshold_var = threshold_var + 100;
        end
    end
    fprintf('MEAN classe %s con media %d var %d outliers rimossi %.2f%%\n', label, treshold_mean, threshold_var, outlier_percentage_mean);
    outliersClassTable_mean_var = actual_class(outlier_mean_var_indices, :);
    
    %% outliers HOG features - clustering
    
    data_cluster_hog = data_hog(:, 2:end); % escludi la prima colonna
    [idx_hog, C_hog] = kmeans(data_cluster_hog, num_clusters);
        
    % Calcola la distanza euclidea da ciascun punto al suo centroide
    distances_hog = sqrt(sum((data_cluster_hog - C_hog(idx_hog, :)).^2, 2));
    
    % Calcola la media e la deviazione standard delle distanze
    mean_distance_hog = mean(distances_hog);
    std_distance_hog = std(distances_hog);
    
    % parametri da settare per trovare gli outliers
    outlier_threshold_hog = 1.2; 
    outlier_percentage_hog = 0;

    % rimuoviamo tra il 5 e il 10% delle immagini
    while outlier_percentage_hog < 5 || outlier_percentage_hog > 10
        outliers_hog = find(distances_hog > (mean_distance_hog + outlier_threshold_hog * std_distance_hog));
        outlier_percentage_hog = length(outliers_hog) / length_class * 100;
        %fprintf('treshold %.2f outliers %.2f%%\n', outlier_threshold, outlier_percentage);
        % modifica automatica dei parametri
        if(outlier_percentage_hog < 5)
            outlier_threshold_hog = outlier_threshold_hog - 0.1;
        elseif(outlier_percentage_hog > 10)
            outlier_threshold_hog = outlier_threshold_hog + 0.1;
        end
    end
     fprintf('HOG classe %s con distanza centroide %.2f outliers rimossi %.2f%%\n', label, outlier_threshold_hog, outlier_percentage_hog);

    % recuperiamo i path degli outliers
    outlier_indices_hog = data(outliers_hog, 1);
    outliersClassTable_hog = actual_class(outlier_indices_hog, :);
    
    %% outliers Texture features - clustering
    rows_with_nan = any(isnan(data_texture(:,:)), 2);
    
    % potrebbero esserci NaN in caso di immagini senza edge (es: sfondo nero)
    ind_NaN_out = find(rows_with_nan);
    if(~isempty(ind_NaN_out) == 1)
        outliersClassTable_texture_NaN = actual_class(ind_NaN_out, :);
        data_texture(ind_NaN_out, :) = [];
    end

    data_cluster_texture = data_texture(:, 2:end); % escludi la prima colonna
    [idx_texture, C_texture] = kmeans(data_cluster_texture, num_clusters);    
    nan_indices = isnan(data_texture);
    idx_texture(nan_indices) = 1;
        
    % Calcola la distanza euclidea da ciascun punto al suo centroide
    distances_texture = sqrt(sum((data_cluster_texture - C_texture(idx_texture, :)).^2, 2));
    
    % Calcola la media e la deviazione standard delle distanze
    mean_distance_texture = mean(distances_texture);
    std_distance_texture = std(distances_texture);
    
    % Soglia per gli outliers
    outlier_threshold_texture = 0.6; 
    outlier_percentage_texture = 0;

    % Rimuoviamo tra il 10 e il 15% delle immagini
    while outlier_percentage_texture < 10 || outlier_percentage_texture > 15
        outliers_texture = find(distances_texture > (mean_distance_texture + outlier_threshold_texture * std_distance_texture));
        outlier_percentage_texture = length(outliers_texture) / length_class * 100;
        %fprintf('treshold %.2f outliers %.2f%%\n', outlier_threshold, outlier_percentage);
        if(outlier_percentage_texture < 10)
            outlier_threshold_texture = outlier_threshold_texture - 0.1;
        elseif(outlier_percentage_texture > 15)
            outlier_threshold_texture = outlier_threshold_texture + 0.1;
        end
    end
     fprintf('TEXTURE classe %s con distanza centroide %.2f outliers rimossi %.2f%%\n', label, outlier_threshold_texture, outlier_percentage_texture);

    % recuperiamo i path degli outliers
    outlier_indices_texture = data(outliers_texture , 1);
    outliersClassTable_texture = actual_class(outlier_indices_texture , :);
    
    if(~isempty(outliersClassTable_texture) == 1)
        outliersClassTable_texture = [outliersClassTable_texture; outliersClassTable_texture_NaN];
        outliersClassTable_texture_NaN = [];
    end

    %% outliers cnn features - clustering
    data_cluster = data(:, 2:end); % escludi la prima colonna
    [idx, C] = kmeans(data_cluster, num_clusters);
    
    % Calcola la distanza euclidea da ciascun punto al suo centroide
    distances = sqrt(sum((data_cluster - C(idx, :)).^2, 2));
    
    % Calcola la media e deviazione standard delle distanze
    mean_distance = mean(distances);
    std_distance = std(distances);
    
    % Soglia per gli outliers
    outlier_threshold = 0.4; 
    outlier_percentage = 0;

    % facciamo in modo di rimuovere tra il 20 e il 25% delle immagini
    while outlier_percentage < 20 || outlier_percentage > 25
        outliers = find(distances > (mean_distance + outlier_threshold * std_distance));
        outlier_percentage = length(outliers) / length_class * 100;
        %fprintf('treshold %.2f outliers %.2f%%\n', outlier_threshold, outlier_percentage);
        if(outlier_percentage < 20)
            outlier_threshold = outlier_threshold - 0.1;
        elseif(outlier_percentage > 25) 
            outlier_threshold = outlier_threshold + 0.1;
        end
    end
     fprintf('CNN classe %s con distanza centroide %.2f outliers rimossi %.2f%%\n', label, outlier_threshold, outlier_percentage);

    % recuperiamo i path degli outliers
    outlier_indices = data(outliers, 1);
    outliersClassTable = actual_class(outlier_indices, :);
    
    %% matching di tutti gli outliers
    % aggiungiamoli tutti a un'unica tabella, eliminando le sovrapposizioni
    matchedTable = [outliersClassTable_mean_var; outliersClassTable_hog; outliersClassTable_texture; outliersClassTable];
    matchedTable = unique(matchedTable, 'rows');
    matchedTable = sortrows(matchedTable, 'imagePath');
    totale_removed_outliers = height(matchedTable) / length_class * 100;
    fprintf('Outliers totali rimossi classe %s è %.2f%%\n', label, totale_removed_outliers);

    %% outliers folder

    % conterrà tutti gli outliers alla fine
    total_outliers = [total_outliers; matchedTable];

    % Crea il percorso della nuova cartella basata sull'etichetta
    actualOutliersPath = fullfile(OutliersPath, label);
    
    % Crea una cartella per ogni label nella cartella degli outliers
    if ~exist(actualOutliersPath, 'dir')
        mkdir(actualOutliersPath);
    end
    
    % Copia le immagini degli outliers in una nuova cartella
    for k = 1:height(matchedTable)
        % CAMBIA IN BASE AL TUO PERCORSO - quello relativo non va boh
        imagePath = fullfile('../dataset/train_set/', matchedTable.imagePath{k});
        [~, imageName, imageExt] = fileparts(imagePath);
        newImagePath = fullfile(actualOutliersPath, [imageName imageExt]);
        movefile(imagePath, newImagePath);
    end
end