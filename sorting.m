clc; clear; close all;

% Paths to your downloaded files
part1 = 'dataset/HAM10000_images_part_1/';
part2 = 'dataset/HAM10000_images_part_2/';

% Read metadata
metadata = readtable('dataset/HAM10000_metadata.csv');

% Create output folders
categories = unique(metadata.dx);
outputDir = 'data/';

for i = 1:length(categories)
    folderName = fullfile(outputDir, categories{i});
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
end

% Combine both image folders
allImageFolders = {part1, part2};

disp('Sorting images...');

% Loop through metadata rows
for i = 1:height(metadata)
    imgName = strcat(metadata.image_id{i}, '.jpg');
    label = metadata.dx{i};

    % Look for image in part1 or part2
    found = false;
    for f = 1:length(allImageFolders)
        src = fullfile(allImageFolders{f}, imgName);
        if exist(src, 'file')
            dest = fullfile(outputDir, label, imgName);
            copyfile(src, dest);
            found = true;
            break;
        end
    end

    if ~found
        fprintf('Image not found: %s\n', imgName);
    end
end

disp('Dataset sorting completed!');