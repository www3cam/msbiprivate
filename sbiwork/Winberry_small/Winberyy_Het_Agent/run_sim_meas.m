% Simulate data

global rng_seed is_data_gen

set_dynare_seed(rng_seed);  % Seed Dynare RNG
rng(rng_seed, 'twister');   % Seed Matlab RNG

if ~is_data_gen
    % Load previous data
    load_mat('simul_data_micro');
else
    % Simulate
    simul_data_meas;
end
