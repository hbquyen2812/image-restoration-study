function benchmark_upstream(repo_root, case_dir)
% Run UNMODIFIED upstream numerical functions on Python-exported inputs.
% Example:
% addpath('D:/QuantumLab/python-denoising/matlab');
% benchmark_upstream('D:/QuantumLab/repos', 'D:/QuantumLab/outputs/python_denoising/cameraman');
% Requires base MATLAB; no Image Processing Toolbox needed by this harness.
% Does not regenerate noise or use broken/incomplete upstream demo scripts.

old_path = path;
cleanup_path = onCleanup(@() path(old_path)); %#ok<NASGU>
cases = dir(fullfile(case_dir, 'case_*.mat'));
for index = 1:numel(cases)
    filename = cases(index).name;
    if contains(filename, '_matlab'), continue; end
    c = load(fullfile(case_dir, filename));
    result_file = fullfile(case_dir, strrep(filename, '.mat', '_matlab.mat'));
    if isfile(result_file)
        error('Refusing to overwrite existing measurement: %s', result_file);
    end
    old_threads = maxNumCompThreads(c.threads);
    cleanup_threads = onCleanup(@() maxNumCompThreads(old_threads));
    result = struct('input_sha256', c.input_sha256, 'matlab_version', version, ...
                    'matlab_computer', computer, 'threads', maxNumCompThreads, ...
                    'timestamp', datestr(now, 30));
    result.clean = c.clean;
    result.noisy = c.noisy;
    if isfield(c, 'qab_python')
        cfg = c.qab_config;
        result.qab_config = cfg;
        assert(cfg.patch_size == 64 && cfg.stride == 32 && ...
               cfg.threshold_step == 12 && cfg.patience == 20, ...
               'Unmodified QAB requires upstream patch/search settings.');
        assert(size(c.clean, 1) == size(c.clean, 2) && ...
               mod(size(c.clean, 1), 32) == 0 && size(c.clean, 1) >= 64);
        path(old_path);
        addpath(fullfile(repo_root, 'QAB-denoising', 'Image'));
        call = @() image_denoising_QAB(c.clean, c.noisy, cfg.slopes, ...
                                      cfg.weight, cfg.gaussian_variance);
        [result.qab_matlab, result.qab_seconds, result.qab_warmup_seconds] = ...
            measure_call(call, c.warmups, c.repeats);
    end
    if isfield(c, 'dequip_python')
        cfg = c.dequip_config;
        result.dequip_config = cfg;
        path(old_path);
        addpath(fullfile(repo_root, 'De-QuIP-Denoising', 'De-QuIP'));
        call = @() run_dequip(c.noisy, cfg);
        [result.dequip_matlab, result.dequip_seconds, result.dequip_warmup_seconds] = ...
            measure_call(call, c.warmups, c.repeats);
    end
    if isfield(c, 'qab_signal_python')
        cfg = c.qab_signal_config;
        result.qab_signal_config = cfg;
        assert(cfg.threshold_step == 2);
        path(old_path);
        addpath(fullfile(repo_root, 'QAB-denoising', 'Signal'));
        call = @() signal_denoising_QAB(c.clean(:)', c.noisy(:)', cfg.slopes, ...
                                       cfg.weight, cfg.gaussian_variance);
        [result.qab_signal_matlab, result.qab_signal_seconds, result.qab_signal_warmup_seconds] = ...
            measure_call(call, c.warmups, c.repeats);
    end
    save(result_file, '-struct', 'result', '-v7');
    fprintf('Saved measured MATLAB result: %s\n', result_file);
    clear cleanup_threads;
end
end

function output = run_dequip(noisy, cfg)
patches = spatial_patchization(noisy, cfg.patch_size);
filtered = DeQuIP_denoising(patches, cfg.window_radius, cfg.stride, ...
    cfg.interaction, cfg.retained_vectors, cfg.weight_factor, cfg.threshold, 10);
% del=10 is accepted by upstream but unused inside hardthresholding.m.
output = reprojection_UWA(filtered);
end

function [output, times, warmups] = measure_call(call, num_warmups, num_repeats)
warmups = zeros(1, num_warmups);
times = zeros(1, num_repeats);
for k = 1:num_warmups
    start = tic;
    captured = evalc('output = call();'); %#ok<NASGU>
    warmups(k) = toc(start);
end
for k = 1:num_repeats
    start = tic;
    captured = evalc('output = call();'); %#ok<NASGU>
    times(k) = toc(start);
    fprintf('MATLAB repeat %d/%d: %.6f seconds\n', k, num_repeats, times(k));
end
end

