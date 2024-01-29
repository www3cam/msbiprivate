function [a] = run_calib_dynare_func(vecin)

% Calibrate model and run Dynare


%% Calibrate parameters and set numerical settings
a = '1';

global model_name is_run_dynare dynare_model

calibrate_func(vecin);

cd(['/home/cameron/Dynare/Winberyy_Het_Agent-20221009T010206Z-001/Winberyy_Het_Agent/het_agents_bayes-master/program/' model_name '_model/dynare']);
saveParameters;
economicParameters_true = load_mat('economicParameters'); % Store true parameters


%% Initial Dynare processing

    if is_run_dynare
        dynare(dynare_model, 'noclearall', 'nopathchange'); % Run Dynare once to process model file
    else
        load(strcat(dynare_model, '_results'));
        check_matlab_path(false);
        dynareroot = dynare_config(); % Add Dynare sub-folders to path
    end

end 