function sgm = getSigma(max2keep, vars2keep_index, nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets, x )
    
%     vars2keep_index = repmat(vars2keep_index, size(x,1),1);
    max2keep = repmat(max2keep, size(x,1),1);
    test_x = NaN(size(x,1),size(vars2keep_index,2));
    test_x(:, vars2keep_index) = max2keep;
    test_x(:, ~vars2keep_index) = x;
    
    [sample_mean, sample_std] = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,test_x);
    sample_var = sample_std.^2;
    sgm = sample_mean./sample_var;
end