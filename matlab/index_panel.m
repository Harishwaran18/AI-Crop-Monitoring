%% index_panel.m
% Two-row index panel: NDVI, GNDVI, SAVI, EVI / MSAVI, NDRE and a class map.
% This is the corrected version of the original index montage.
%
% Why the original figure looked wrong
%   * EVI spanned -5000..1000 because raw 16-bit digital numbers were used and
%     the +1 (L) term was missing from the denominator. Divide by the sensor
%     scale and keep L = 1 and EVI stays inside [-1, 1].
%   * The class map was a single colour because every pixel was assigned to the
%     last class in an if/elseif chain, so CropHealthSummary.csv read
%     21024 x PestRisk out of 21025 pixels.
%
% Requirements: base MATLAB only (no toolboxes).

clc; clear; close all;

repo_root = fileparts(fileparts(mfilename('fullpath')));
cube_file = fullfile(repo_root, 'data', 'demo_spectral_cube.npz');
out_dir = fullfile(repo_root, 'docs', 'figures');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

if isfile(cube_file)
    S = load(cube_file);
    cube = double(S.cube);
else
    fprintf('No cube at %s; generating a synthetic field.\n', cube_file);
    cube = synthetic_cube(96, 96, 12);
end

% Digital numbers (0-10000) vs reflectance (0-1): pick the right scale once.
if max(cube(:)) > 2
    cube = cube / 10000;
end
cube = min(max(cube, 0), 1);

[~, ~, nb] = size(cube);
blue  = cube(:, :, min(1, nb));
green = cube(:, :, min(2, nb));
red   = cube(:, :, min(6, nb));
re    = cube(:, :, min(8, nb));
nir   = cube(:, :, min(11, nb));

ndvi  = safe_ratio(nir - red, nir + red);
gndvi = safe_ratio(nir - green, nir + green);
savi  = safe_ratio(1.5 * (nir - red), nir + red + 0.5);
msavi = ((2 * nir + 1) - sqrt(max((2 * nir + 1) .^ 2 - 8 * (nir - red), 0))) / 2;
evi   = safe_ratio(2.5 * (nir - red), nir + 6 * red - 7.5 * blue + 1);
ndre  = safe_ratio(nir - re, nir + re);

% Classification: NDVI sets the class, low red edge downgrades it one step.
classes = discretize(ndvi, [-Inf 0.35 0.55 Inf]) - 1;
downgrade = ndre < 0.18;
classes(downgrade) = max(classes(downgrade) - 1, 0);
classes(~isfinite(ndvi)) = -1;

names = {'NDVI', 'GNDVI', 'SAVI', 'EVI', 'MSAVI', 'NDRE'};
layers = {ndvi, gndvi, savi, evi, msavi, ndre};
limits = {[0 0.9], [0 0.9], [0 1.1], [0 1.1], [0 1.1], [0 0.8]};

figure('Name', 'Spectral index panel', 'Color', 'w', 'Position', [80 80 1500 720]);
for k = 1:numel(names)
    subplot(2, 4, k);
    imagesc(layers{k});
    axis image off;
    colorbar;
    title(names{k});
    colormap(gca, 'jet');
    caxis(limits{k});
end

subplot(2, 4, 7);
imagesc(classes);
axis image off;
colorbar;
title('Classified Map');
colormap(gca, [0.78 0.16 0.16; 0.98 0.66 0.15; 0.18 0.49 0.20]);
caxis([-0.5 2.5]);
text(0.02, 0.05, 'Stressed / Moderate / Healthy', 'Units', 'normalized', 'FontSize', 7, 'Color', 'w');

subplot(2, 4, 8);
axis off;

exportgraphics(gcf, fullfile(out_dir, 'matlab_index_panel.png'), 'Resolution', 130);

counts = [sum(classes(:) == 0), sum(classes(:) == 1), sum(classes(:) == 2)];
fprintf('Stressed %d | Moderate %d | Healthy %d | nodata %d\n', ...
    counts(1), counts(2), counts(3), sum(classes(:) == -1));

T = table({'Stressed'; 'Moderate'; 'Healthy'}, counts.', 'VariableNames', {'Class', 'PixelCount'});
writetable(T, fullfile(repo_root, 'data', 'class_summary.csv'));
fprintf('Wrote %s\n', fullfile(repo_root, 'data', 'class_summary.csv'));

%% Helpers
function out = safe_ratio(numerator, denominator)
% Division that yields NaN instead of Inf where the denominator vanishes.
out = numerator ./ denominator;
out(abs(denominator) < 1e-6) = NaN;
end

function cube = synthetic_cube(rows, cols, bands)
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
cube = min(max(cube + 0.02 * randn(size(cube)), 0.01), 0.95);
end
