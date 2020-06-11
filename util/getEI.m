function EI = getEI(max_of_targets,nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets, x )
    sigma1 = 0.01;
%     SwSp_z = x(:,1) + x(:,7) + x(:,9) + x(:,11) - x(:,6) - x(:,8)-x(:,10);
%     test_x = [SwSp_z, x];
    
%     test_x = NaN(size(x,1),size(vars2keep_index,1));
%     test_x(:, vars2keep_index) = max2keep;
%     test_x(:, ~vars2keep_index) = x;
    test_x = x;
    [gp_output, sample_std] = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,test_x);
%     sample_std = sqrt(sample_var);
    Z1 = (gp_output - max_of_targets-sigma1)./(sample_std);
    EI = ((-max_of_targets+gp_output-sigma1).*normcdf(Z1) + (sample_std).*normpdf(Z1));
    EI = -1*EI;
end