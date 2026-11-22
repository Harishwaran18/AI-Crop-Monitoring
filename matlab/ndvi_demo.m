%% ndvi_demo.m
% Minimal, self-contained NDVI walkthrough.
% If data/demo_spectral_cube.npz exists (scripts/generate_demo_data.py) it is
% used; otherwise a synthetic field is generated so the script always runs.
clc; clear; close all;

repo_root = fileparts(fileparts(mfilename('fullpath')));
cube_file = fullfile(repo_root, 'data', 'demo_spectral_cube.npz');

if isfile(cube_file)
    S = load(cube_file);
    cube = double(S.cube);
    fprintf('Loaded %s (%dx%dx%d).\n', cube_file, size(cube, 1), size(cube, 2), size(cube, 3));
else
    fprintf('No cube found, generating a synthetic field.\n');
    cube = synthetic_cube(96, 96, 12);
end

nir = cube(:, :, end);            % last band is the NIR shoulder
red = cube(:, :, 6);              % red absorption band

denom = nir + red;
ndvi = nan(size(nir));
valid = abs(denom) > 1e-6;
ndvi(valid) = (nir(valid) - red(valid)) ./ denom(valid);

fprintf('NDVI min %.3f | mean %.3f | max %.3f\n', ...
    min(ndvi(:)), mean(ndvi(:), 'omitnan'), max(ndvi(:)));

figure('Name', 'NDVI');
imagesc(ndvi, [-0.2 0.9]); colorbar; colormap jet; axis image;
title('NDVI (synthetic field)'); xlabel('column'); ylabel('row');

figure('Name', 'NDVI histogram');
histogram(ndvi(~isnan(ndvi)), 30);
xlabel('NDVI'); ylabel('pixels'); title('NDVI distribution'); grid on;

function cube = synthetic_cube(rows, cols, bands)
rng(42);
[yy, xx] = ndgrid(1:rows, 1:cols);
stress = 0.5 * exp(-(((yy - rows * 0.7) .^ 2 + (xx - cols * 0.4) .^ 2) / (2 * 12 ^ 2)));
stress = min(max(stress ./ max(stress(:)), 0), 1);
healthy = linspace(0.04, 0.48, bands);
stressed = linspace(0.07, 0.26, bands);
leaf = healthy .* (1 - stress) + stressed .* stress;
fvc = max(min(0.95 - 0.55 * stress, 1), 0.05);
cube = fvc .* leaf + (1 - fvc) .* linspace(0.12, 0.29, bands);
cube = min(max(cube + 0.02 * randn(size(cube)), 0.01), 0.95);
end
