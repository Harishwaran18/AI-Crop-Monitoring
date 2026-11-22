%% crop_health_demo.m
% NDVI -> patch extraction -> CNN spatial features -> LSTM-ready sequences.
% Mirrors the Python pipeline in src/cropmon for teams working in MATLAB.
%
% Requirements: Image Processing Toolbox, Deep Learning Toolbox,
% Hyperspectral Imaging Library (for real .mat cubes).
clc; clear; close all;

%% Paths (repo-relative, override with an absolute path if you keep data elsewhere)
repo_root = fileparts(fileparts(mfilename('fullpath')));
data_path = fullfile(repo_root, 'data');
sensor_file = fullfile(data_path, 'sensors.csv');
cube_file = fullfile(data_path, 'demo_spectral_cube.npz');

%% Step 1: Load a spectral cube (fall back to a synthetic cube)
if isfile(cube_file)
    S = load(cube_file);
    cube = double(S.cube);            % rows x cols x bands
else
    warning('No cube found at %s; using a synthetic field.', cube_file);
    cube = synthetic_cube(96, 96, 12);
end
[H, W, n_bands] = size(cube);

% Indian Pines band indices used when the real dataset is available:
%   NIR ~ 50, Red ~ 30 (1-based, 220 bands).
nir_idx = min(10, n_bands);
red_idx = min(6, n_bands);
NIR = cube(:, :, nir_idx);
Red = cube(:, :, red_idx);

denom = NIR + Red;
NDVI = zeros(H, W);
valid = abs(denom) > 1e-6;
NDVI(valid) = (NIR(valid) - Red(valid)) ./ denom(valid);
NDVI = max(min(NDVI, 1), -1);

figure('Name', 'NDVI');
imagesc(NDVI); colorbar; colormap jet; axis image;
title('NDVI map'); xlabel('column'); ylabel('row');

%% Step 2: Sensor features (optional)
sensor_features = [];
if isfile(sensor_file)
    sensor_table = readtable(sensor_file);
    numeric = sensor_table{:, varfun(@isnumeric, sensor_table, 'OutputFormat', 'uniform')};
    sensor_features = mean(numeric, 1, 'omitnan');
    fprintf('Loaded %d sensor rows, %d numeric features.\n', height(sensor_table), numel(sensor_features));
else
    warning('Sensor file not found: %s', sensor_file);
end

%% Step 3: Patch extraction (index-safe)
patch_size = 16;
row_starts = 1:patch_size:(H - patch_size + 1);
col_starts = 1:patch_size:(W - patch_size + 1);
num_patches = numel(row_starts) * numel(col_starts);

X_img = zeros(patch_size, patch_size, 1, num_patches);
count = 1;
for i = row_starts
    for j = col_starts
        X_img(:, :, 1, count) = NDVI(i:i + patch_size - 1, j:j + patch_size - 1);
        count = count + 1;
    end
end
fprintf('Extracted %d patches of %dx%d.\n', num_patches, patch_size, patch_size);

%% Step 4: CNN for spatial features
layers_cnn = [
    imageInputLayer([patch_size patch_size 1], 'Normalization', 'none')
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(16, 'Name', 'fc')
    ];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 3, ...
    'MiniBatchSize', min(16, num_patches), ...
    'Verbose', false, ...
    'Plots', 'none');

% Demo labels: replace with scouting labels before drawing conclusions.
Y_dummy = categorical(randi([0 1], num_patches, 1));
net_cnn = trainNetwork(X_img, Y_dummy, layers_cnn, options);

%% Step 5: CNN features -> sequence for the LSTM head
cnn_features = activations(net_cnn, X_img, 'fc');   % num_patches x 16
input_lstm = cnn_features;
if ~isempty(sensor_features)
    input_lstm = [cnn_features, repmat(sensor_features, num_patches, 1)];
end
fprintf('LSTM input ready: %d sequences x %d features.\n', size(input_lstm, 1), size(input_lstm, 2));

%% Local helper
function cube = synthetic_cube(rows, cols, bands)
% Simple leaf/soil mixing model, matching the Python simulator's shape.
rng(42);
[yy, xx] = ndgrid(1:rows, 1:cols);
stress = 0.5 * exp(-(((yy - rows * 0.7) .^ 2 + (xx - cols * 0.4) .^ 2) / (2 * 12 ^ 2)));
stress = stress + 0.3 * (yy ./ rows);
stress = min(max(stress ./ max(stress(:)), 0), 1);

healthy = linspace(0.04, 0.48, bands);
stressed = linspace(0.07, 0.26, bands);
leaf = healthy .* (1 - stress) + stressed .* stress;
fvc = max(min(0.95 - 0.55 * stress, 1), 0.05);
cube = fvc .* leaf + (1 - fvc) .* linspace(0.12, 0.29, bands);
cube = cube + 0.02 * randn(size(cube));
cube = min(max(cube, 0.01), 0.95);
end
