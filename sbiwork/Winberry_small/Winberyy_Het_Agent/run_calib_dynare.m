% Calibrate model and run Dynare


%% Calibrate parameters and set numerical settings

calibrate;

cd(['/home/cameron/Dynare/Winberyy_Het_Agent-20221009T010206Z-001/Winberyy_Het_Agent/dynare']);
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