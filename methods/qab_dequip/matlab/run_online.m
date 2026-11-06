function run_online(group)
% Upload/extract matlab_online_bundle.zip, cd to its folder, run run_online.
% Default measures De-QuIP64 only. Other groups: dequip_all, qab_ms3,
% qab_ms20, signal, all. Outputs are actual case_*_matlab.mat files.
if nargin == 0, group = 'dequip64'; end
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'));
valid = {'dequip64', 'dequip_all', 'qab_ms3', 'qab_ms20', 'signal'};
if strcmp(group, 'all')
    groups = {'dequip_all', 'qab_ms3', 'qab_ms20', 'signal'};
elseif any(strcmp(group, valid))
    groups = {group};
else
    error('Unknown group. Use dequip64, dequip_all, qab_ms3, qab_ms20, signal, all.');
end
for k = 1:numel(groups)
    benchmark_upstream(fullfile(root, 'repos'), fullfile(root, 'cases', groups{k}));
end
end
