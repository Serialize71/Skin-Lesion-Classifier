%% 1. Data Loading and Rebalancing (No Change)
clc; clear; close all;

rootData = 'data'; 
imds = imageDatastore(rootData, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds = shuffle(imds);
inputSize = [224 224 3];

%% 70% training, 15% validation, 15% testing
[imdsTrain, imdsRest] = splitEachLabel(imds, 0.70, 'randomized');
[imdsVal, imdsTest]   = splitEachLabel(imdsRest, 0.50, 'randomized');

fprintf("Initial Training images:   %d\n", numel(imdsTrain.Files));
fprintf("Validation images: %d\n", numel(imdsVal.Files));
fprintf("Test images:       %d\n", numel(imdsTest.Files));


%% Oversampling for Training Set
tbl = countEachLabel(imdsTrain);

% Choose target count (e.g. median or second-largest class)
sortedCounts = sort(tbl.Count,'descend');
targetCount = sortedCounts(2);   % NOT max, NOT min → balanced

newFiles = {};
newLabels = categorical();

rng(0); % reproducibility

for i = 1:height(tbl)
    label = tbl.Label(i);
    idx = find(imdsTrain.Labels == label);
    files = imdsTrain.Files(idx);
    n = numel(files);

    if n < targetCount
        % Oversample minority class
        extraIdx = randi(n, targetCount - n, 1);
        filesOS = [files; files(extraIdx)];
    else
        % Undersample dominant class (nv)
        filesOS = files(randperm(n, targetCount));
    end

    newFiles  = [newFiles; filesOS];
    newLabels = [newLabels; repmat(label, numel(filesOS), 1)];
end

imdsTrain = imageDatastore(newFiles, 'Labels', newLabels);

disp('After oversampling:');
disp(countEachLabel(imdsTrain));


%% Data Augmentation to solve overfitting
augmenter = imageDataAugmenter( ...
    'RandRotation',[-30 30], ...
    'RandXReflection',true, ...
    'RandYTranslation',[-20 20], ...
    'RandXTranslation',[-20 20], ...
    'RandScale',[0.8 1.2]);
%% Translating is shifting in X or Y axis

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);
numClasses = numel(categories(imdsTrain.Labels));

%% Custom CNN Structure
layers = [
    imageInputLayer(inputSize,'Normalization','rescale-zero-one','Name','input')
    % ---- Block 1: 32 Filters ----
    convolution2dLayer(6,32,'Padding','same','Name','conv1_1')
    batchNormalizationLayer('Name','bn1_1')
    reluLayer('Name','relu1_1')
    convolution2dLayer(6,32,'Padding','same','Name','conv1_2')
    batchNormalizationLayer('Name','bn1_2')
    reluLayer('Name','relu1_2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')
    
    % ---- Block 2: 64 Filters ----
    convolution2dLayer(6,64,'Padding','same','Name','conv2_1')
    batchNormalizationLayer('Name','bn2_1')
    reluLayer('Name','relu2_1')
    convolution2dLayer(6,64,'Padding','same','Name','conv2_2')
    batchNormalizationLayer('Name','bn2_2')
    reluLayer('Name','relu2_2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')
    
    % ---- Block 3: 128 Filters ----
    convolution2dLayer(6,128,'Padding','same','Name','conv3_1')
    batchNormalizationLayer('Name','bn3_1')
    reluLayer('Name','relu3_1')
    convolution2dLayer(6,128,'Padding','same','Name','conv3_2')
    batchNormalizationLayer('Name','bn3_2')
    reluLayer('Name','relu3_2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool3')
    
     % ---- Block 4: 256 Filters ----
    convolution2dLayer(6,256,'Padding','same','Name','conv4_1')
    batchNormalizationLayer('Name','bn4_1')
    reluLayer('Name','relu4_1')
    convolution2dLayer(6,256,'Padding','same','Name','conv4_2')
    batchNormalizationLayer('Name','bn4_2')
    reluLayer('Name','relu4_2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool4') % Fixed Name
    
    
    
    %% Global Pooling + Dropout
    globalAveragePooling2dLayer('Name','gap')
    dropoutLayer(0.2,'Name','dropout2') % Final Regularization
    
    %% Final Classifier
    fullyConnectedLayer(numClasses,'Name','fc') %Raw Scores Relative Confidence for every class
    softmaxLayer('Name','softmax') %Probabilites
    classificationLayer('Name','class') %Loss & Class Labels
];

%% 3. Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ... % Keeping it low for stable training
    'MaxEpochs',10, ...          % Increased epochs for the deeper net
    'MiniBatchSize',25, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',floor(numel(imdsTrain.Files)/32), ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',true, ...
    'LearnRateSchedule','piecewise', ...
    'L2Regularization', 1e-4, ... % ADDED: Weight Decay (L2) for regularization
    'Plots','training-progress');

%% Train the Network
net = trainNetwork(augTrain, layers, options);

%% Evaluate on Test Set
YPred = classify(net, augTest);
YTrue = imdsTest.Labels;
testAccuracy = mean(YPred == YTrue) * 100;
fprintf("\nFINAL TEST ACCURACY = %.2f%%\n", testAccuracy);

figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix - Custom Deep CNN (Test Set)');