function [micro_data] = sim_hh_2(vecin)
% Simulate and estimate heterogeneous firm model
tic
global model_name ts_micro N_micro param_names transf_to_param param_to_transf ...
    prior_logdens_transf num_smooth_draws num_interp rng_seed dynare_model is_run_dynare...
    is_data_gen likelihood_type serial_id T num_burnin_periods ssigmaMeasmicro

model_name = 'hh';

addpath(genpath('/home/cameron/Dynare/Winberyy_Het_Agent-20221009T010206Z-001/Winberyy_Het_Agent/het_agents_bayes-master/program/functions'));
addpath(genpath(['/home/cameron/Dynare/Winberyy_Het_Agent-20221009T010206Z-001/Winberyy_Het_Agent/het_agents_bayes-master/program/' model_name '_model/auxiliary_functions']));


%% Settings

% Decide what to do
is_run_dynare = true;   % Process Dynare model?
is_data_gen = true;     % Simulate data?
likelihood_type = 1;    % =1: macro + full-info micro; =2: macro only;
                        % =3: macro + 3 micro moments; =4: macro + 2 micro moments; =5: macro + 1 micro moment

% ID
serial_id = 1;          % ID number of current run (used in file names and RNG seeds)

% Model/data settings
T = 174;                % Number of periods of simulated macro data
ts_micro = [71,75,79,83,91,95,99,103,123,127,131,139,143,147,151,155,159,163,167];     % Time periods where we observe micro data
N_micro = 100;          % Number of households per non-missing time period

% File names
global mat_suff;
mat_suff = sprintf('%s%d%s%d%s%02d', '_N', N_micro, '_liktype', likelihood_type, '_', serial_id); % Suffix string for all saved .mat files
save_folder = fullfile(pwd, 'results'); % Folder for saving results

% Parameter transformation
if ismember(likelihood_type,[1 3 4]) % When mu_l is identified
    param_names = {'bbeta', 'ssigmaMeas', 'mu_l'};                      % Names of parameters to estimate
    transf_to_param = @(x) [1/(1+exp(-x(1))) exp(x(2)) -exp(x(3))];     % Function mapping transformed parameters into parameters of interest
    param_to_transf = @(x) [log(x(1)/(1-x(1))) log(x(2)) log(-x(3))];   % Function mapping parameters of interest into transformed parameters
else % When mu_l is not identified
    param_names = {'bbeta', 'ssigmaMeas'};                   % Names of parameters to estimate
    transf_to_param = @(x) [1/(1+exp(-x(1))) exp(x(2))];     % Function mapping transformed parameters into parameters of interest
    param_to_transf = @(x) [log(x(1)/(1-x(1))) log(x(2))];   % Function mapping parameters of interest into transformed parameters
end

% Prior
prior_logdens_transf = @(x) sum(x) - 2*log(1+exp(x(1)));    % Log prior density of transformed parameters

% Optimization settings
% is_optimize = true;                             % Find posterior mode?
% if ismember(likelihood_type,[1 3 4]) % When mu_l is identified
%     [aux1, aux2, aux3] = meshgrid(linspace(0.8,0.99,5),linspace(0.001,0.05,5),linspace(-1,-0.01,5));
%     optim_grid = [aux1(:), aux2(:), aux3(:)];   % Optimization grid
% else % When mu_l is not identified
%     [aux1, aux2] = meshgrid(linspace(0.8,0.99,5),linspace(0.001,0.05,5));
%     optim_grid = [aux1(:), aux2(:)];
% end
clearvars aux*;

% % MCMC settings
% if likelihood_type ~= 2
%     mcmc_init = param_to_transf([.9 .06 -1]); % Initial transformed draw (will be overwritten if is_optimize=true)
% else % mu_l is not identified with macro data only
%     mcmc_init = param_to_transf([.9 .06]);
% end
% mcmc_num_iter = 1e4;                    % Number of MCMC steps (total)
% mcmc_thin = 1;                          % Store every X draws
% mcmc_stepsize_init = 1e-2;              % Initial MCMC step size
% mcmc_adapt_iter = [50 200 500 1000];    % Iterations at which to update the variance/covariance matrix for RWMH proposal; first iteration in list is start of adaptation phase
% mcmc_adapt_diag = false;                % =true: Adapt only to posterior std devs of parameters, =false: adapt to full var/cov matrix
% mcmc_adapt_param = 10;                  % Shrinkage parameter for adapting to var/cov matrix (higher values: more shrinkage)
% 
% % Adaptive RWMH
% mcmc_c = 0.55;                          % Updating rate parameter
% mcmc_ar_tg = 0.3;                       % Target acceptance rate
% mcmc_p_adapt = .95;                     % Probability of non-diffuse proposal

% Likelihood settings
num_smooth_draws = 500;                 % Number of draws from the smoothing distribution (for unbiased likelihood estimate)
num_interp = 100;                       % Number of interpolation grid points for calculating density integral

% Numerical settings
num_burnin_periods = 100;               % Number of burn-in periods for simulations
rng_seed = 20200813+serial_id;          % Random number generator seed
% if likelihood_type == 1
%     delete(gcp('nocreate'));    
%     poolobj = parpool(1);                  % Parallel computing object
% end

% Dynare settings
dynare_model = 'firstOrderDynamics_polynomials'; % Dynare model file


%% Calibrate parameters, execute initial Dynare processing

run_calib_dynare_func(vecin);
% Approximates cross-sec distribution of assets conditional on employment
% using third-order exponential family density, as in Winberry (2018)


%% Simulate data

run_sim_meas;

micro1 = sort(simul_data_micro(:,:,2)+ ssigmaMeasmicro*randn(19,100),2);
smicro1 = sum(micro1,2);
micro1 = micro1./smicro1;

simmicrodata = reshape(permute(micro1,[2,1]),[1900,1]) ;

micro_data = [sim_struct.logAggregateOutput;sim_struct.logAggregateInvestment;
    sim_struct.logAggregateConsumption;sim_struct.logWage;simmicrodata];
toc
end

