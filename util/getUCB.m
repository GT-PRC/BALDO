% Copyright (c) 2020 
% School of Electrical and Computer Engineering
% 3D Packaging Research Center (PRC)
% Georgia Institute of Technology

%Upper confidence bound acquisition function
function UCB = getUCB(M, nu, nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets, x)
    test_x = x;
    GP_varsigma = @(M) sqrt(2*log(pi^2*(M)^2/(12*nu))); 
    [gp_output, sample_std] = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,test_x);
%     sample_std = sqrt(sample_var);
    UCB = (gp_output + GP_varsigma(M)*(sample_std));
    UCB = -1*UCB;
end