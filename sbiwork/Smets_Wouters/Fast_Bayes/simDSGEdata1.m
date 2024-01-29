function data = simDSGEdata1(vector)
    global labobs robs pinfobs dy dc dinve dw
    cd '/home/cameron/Dynare/Fast_Bayes'
    % vector = [1,1];
    r = length(vector);
    hh = zeros(r,r);
    parameter_names = {'ea';'eb';'eg';'eqs';'em';'epinf';'ew';'crhoa';'crhob';'crhog';'crhoqs';'crhoms';'crhopinf';'crhow';'cmap';'cmaw';'csadjcost';'csigma';'chabb';'cprobw';'csigl';'cprobp';'cindw';'cindp';'czcap';'cfc';'crpi';'crr';'cry';'crdy';'constepinf';'constebeta';'constelab';'ctrend';'cgy';'calfa'};
    % disp(vector)
    xparam1 = vector;
    save('usmodel_shock_decomp_mode_2.mat')
    % save('usmodel_shock_2.mat')
    dynare usmodel_camtest_3
    data = [dc, dinve, dw, dy, labobs, pinfobs, robs];
end

