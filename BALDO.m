% Copyright (c) 2020 
% School of Electrical and Computer Engineering
% 3D Packaging Research Center (PRC)
% Georgia Institute of Technology

% Bayesian Active Learning using Dropout (BAL-DO) for creating a predictive
% model of a black-box function while finding its global optimum.

%This material is based on work supported by NSF I/UCRC Center for Advanced
%Electronics Through Machine Learning (CAEML)
%For questions and queries, please contact: htorun3@gatech.edu

%Please cite our paper if you use the code:
%H. M. Torun, J. A. Hejase, J. Tang, W. D. Becker and M. Swaminathan, 
%"Bayesian Active Learning for Uncertainty Quantification of High Speed Channel Signaling," 
%2018 IEEE 27th Conference on Electrical Performance of Electronic Packaging and Systems (EPEPS), San Jose, CA, USA, 2018, pp. 311-313.

clear all
close all
clc

addpath(genpath('gpml-matlab-v4.1-2017-10-19/'));
addpath(genpath('util/'));
addpath(genpath('Python/'));
addpath(genpath('test_function/'));

%% Function Definition.
% This section is for defining the objective function. The objective
% function should be a seperate ".m" file or a closed form expression.
% "-1" in front of function is to minimize the function, if maximum is
% desired, delete "-1".


sin1 = @(value) 1*(sin(13 .* value) .* sin(27 .* value) ./ 2.0 + 0.5);
peak2D = @(value) 1*(3*(1-value(:,2)).^2.*exp(-(value(:,2).^2) - (value(:,1)+1).^2) - 10*(value(:,2)/5 - value(:,2).^3 - value(:,1).^5).*exp(-value(:,2).^2-value(:,1).^2) - 1/3*exp(-(value(:,2)+1).^2 - value(:,1).^2));
Branin = @(value) -1*braninf(value);
Hart6 = @(value) -1*hart6f(value);
Hart3 = @(value) -1*hart3f(value);

%% Function Selection
% Select which function (defined above) to be used by BAL-DO algorithm.
% Also define number of input variables as "dimension".

% f = sin1; dimension = 1; sample_space = [0 1]; optima = 0.975599;
f = peak2D; dimension = 2; sample_space = [-2 2;-2 2]; optima = 8.106213;
% f = Hart6; dimension=6; sample_space = [0 1; 0 1; 0 1; 0 1; 0 1; 0 1]; optima = 3.322368011391339;
% f = Hart3; dimension=3; sample_space = [0 1; 0 1; 0 1]; optima= 3.862782147819745;
%% Generate all possible combinations of inputs here.
% This part is application dependent. Here, there are 3 variables considered
% After the variables and the values they can take are defined, use
% "allcomb" function to generate the variable "test_x", which contains all
% possible combination of inputs.

% If you want to replace a parameter sweep over discrete values, then the
% sample space is not continous anymore. Here, you should enter the values
% each parameter can take as in "else" part below.

% For continous values, Latin Hypercube Sampling with "N_candidates" points
% is considered. This can be made better with usual auxiliary optimization
% process of Bayesian Optimization, but this is observed to be good enough
% for now. We are working on finding better way to do this discretization.
continous_sample_space = 1;

if(continous_sample_space)
    N_candidates = 10000;
    test_x = sample_space(:,1) + (sample_space(:,2)-sample_space(:,1)).*lhsdesign(N_candidates,dimension)';
    test_x = test_x';
%     test_x = test_x(:,2:end);
%     test_x(:,6:end) = test_x(:,6:end)*0.1;
%     SwSp_z = test_x(:,1) + test_x(:,7) + test_x(:,9) + test_x(:,11) - test_x(:,6) - test_x(:,8)-test_x(:,10);
%     test_x = [SwSp_z, test_x];
%     test_x(:,7:end) = test_x(:, 7:end)*10; 
else
    input_parameters{1} = 1:1:3;
    input_parameters{2} = -1:1:2;
    input_parameters{3} = -0.5:0.1:0;
    test_x = allcomb(input_parameters{:});  
end

       


%% Initialization
% Define initial sample here. If a previous dataset is present, import it
% here with the (NxD) input variable vector as "total_samples" and (Nx1)
% output vector as "total_targets". Here, N is number of samples and D is
% dimensionality.

data_ALL = [];
%No dataset available. Define inititial sample
% total_samples = [1,1,2,1,1,2,3];
total_samples = test_x(10,:); %Candidate at 10th index is initial, can also be random.
total_targets = f(total_samples);


gains = zeros(3,1);

%Define maximum number of simulations.
count_max = 50;

[max_of_targets,xxx] = max(total_targets);
max_sample = total_samples(xxx,:);

BALDO_max_targets = [];
BALDO_max_variance = [];
%Parameters of BALDO. Doesn't need to be changed.
M = 1;
nu = 0.1;
GP_varsigma = @(M) sqrt(2*log(pi^2*(M)^2/(12*nu)));
ucb_count = 0;
ei_count = 0;
pi_count = 0;
sample_indices = find(ismember(test_x,total_samples(1,:),'rows') ~= 0);
previous_max_of_targets = max_of_targets;
%% Initialize the Gaussian Process
% Define mean and covariance function of the GP. This part can be kept same
% except the value of "hyp.mean = 0". This is the ball-park value of the
% objective function value, taken as "0" here.
% Covariance function used here is "Matern 3/2 with ARD". 
% If you want to use other covariance functions, see
% "usageCov" under the "gpml..." folder.

% create likelihood, mean and covariance functions.
likfunc = {'likGauss'}; hyp.lik = log(0.001);
meanfunc = {'meanConst'}; hyp.mean = 0;
covfunc = {'covMaternard',5};

D = size(total_samples,2);
nCov = eval(feval(covfunc{:}));

% Parameters for Slice Sampling, Markov Chain Monte Carlo (MCMC) based
% integration to be used to derive hyperparameter posterior and training of
% the GP as explained in EPEPS paper.

% "sls_opts.nomit" is the burn-in samples for MCMC training, can leave as
% default but decrease for faster analysis. Don't decrease it below 150 as
% this is required for Markov Chain to converge to a stationary
% distribution.

sls_opts = sls_opt;
sls_opts.nsamples = 100;
sls_opts.nomit = 200;
sls_opts.method = 'minmax';

sls_opts.display = 1;
sls_opts.wsize = 15;
sls_opts.plimit = 7;
sls_opts.unimodal = 0;

%Bounds of GP hyperparameters used in training.
%Don't change the first value. It is the numerical noise used to avoid
%numerical errors when inverting ill-conditioned covariance matrices of GP.
%The second is [-10, 10] is for mean function. Modify this appropriately to
%contain mean of the OBJECTIVE FUNCTION values you expect. This can be a
%large sample space.
%-10 and 10 is chosen for bounds of hyperparameters in covariance function.
%This is very large as the bound is in log domain. This should be good for
%any application, but can also be modified to be a smaller bound.

sls_opts.mmlimits = [-10 -10 -5*ones(1,nCov); -5 10 5*ones(1,nCov)];
sls_opts.maxiter = 500;
thinning = 2;

tic

for count = 1:count_max
    previous_max_of_targets = max_of_targets;
    
    MI_indices = 1:1:size(test_x,1)';
    MI_indices(sample_indices) = [];
    test_MII = test_x(MI_indices,:);
    %TRAINING OF GP using Slice Sampling (SLS) - A MCMC INTEGRATION METHOD.
    hyp.cov = -3+6*rand(1,nCov);
    hyp.cov = hyp.cov';
    nll_MCMC_func = @(x) nll_func(x,hyp,meanfunc,covfunc,likfunc,total_samples,total_targets);
    initial = [hyp.lik;hyp.mean;hyp.cov];
    [nsamples, energies, diagns] = sls(nll_MCMC_func, initial, sls_opts);
    nsamples_thinned = nsamples(1:thinning:end,:);
    hyp2_MCMC = mean(nsamples,1);
    hyp2.lik = hyp2_MCMC(1);
    hyp2.mean = hyp2_MCMC(2);
    hyp2.cov = hyp2_MCMC(3:end);
    
    final_lik = hyp2.lik;
    noise_variance = exp(2*final_lik);
    
    %IF YOU WANT BAL-DO, set "optimMode = rem(count,2)"
    %IF YOU WANT JUST BO, set "optimMode = 1"
    %IF YOU WANT JUST BAL, set "optimMode = 0"
    optimMode = rem(count,2);
    %     optimMode = 0;
%         optimMode = 1;
    if(count > 1 && optimMode == 1)
        gainTest = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,[test_UCB;test_EI;test_PI]);
        gains(1) = gains(1) + gainTest(1);
        gains(2) = gains(2) + gainTest(2);
        gains(3) = gains(3) + gainTest(3);
    end
    kfcn = @(XM,XN) feval(covfunc{:},hyp2.cov,XM,XN);
    if(optimMode == 0)
        %DROPOUT PART
        all_vars = 1:1:dimension;
        
        %SELECT HOW MANY VARIABLES TO KEEP (between [0, D-1]). If you want
        %to learn more about the function, decrease the upper bound.
        %If you want to prioritirize optimization more, increase lower bound
        %and set upper bound to D-1.
        %If you want to prioritize model building more, set lower bound to
        %zero an decrease upper bound.
        %Here, bound is [0,D-2].
%         num2keep = randi([1,dimension])-1;
        num2keep = 0;
        vars2keep = randsample(all_vars,num2keep);
        vars2keep = sort(vars2keep,'ascend');
        vars2keep_index = logical(zeros(1,dimension));
        vars2keep_index(vars2keep) = 1;
        
        max2keep = max_sample(vars2keep_index);
        MI_indices2 = ismembertol(test_MII(:,vars2keep),max2keep,1e-2,'ByRows',true);
        % MODEL BUILDING (LEARNING) PART
        test_MI = test_MII(MI_indices2,:);
        if(isempty(test_MI))
           test_MI = test_x(MI_indices,:); 
        end
        MI_y = zeros(size(test_MI,1),1);
        [~,std_MI] = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,test_MI);
        
        eta = std_MI.^2;
        candidates = test_MI;
        
    else
        %BO PART, this is very similary to TSBO.m, please see the comments
        %there for explanations.
        sigma1 = 0.01;
        sigma2 = 0.01;
        test_BO = test_x(MI_indices,:);
        [candidates_mean,candidates_std] = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,test_BO);
        Z1 = (candidates_mean - max_of_targets-sigma1)./(candidates_std);
        EI = (candidates_mean-max_of_targets-sigma1).*normcdf(Z1) + (candidates_std).*normpdf(Z1);
        UCB = candidates_mean + GP_varsigma(M)*(candidates_std);
        M = M+1;
        
        PI = normcdf((candidates_mean - max_of_targets-sigma2)./candidates_std);
        if(rem(count,3) == 2)
            eta = UCB;
            ucb_count = ucb_count + 1;
        elseif(rem(count,3) == 0)
            eta = EI;
            ei_count = ei_count + 1;
        elseif(rem(count,3) == 1)
            eta = PI;
            pi_count = pi_count +1;
        end
        [u_x,u_i] = max(UCB);
        test_UCB = test_BO(u_i,:);
        [e_x,e_i] = max(EI);
        test_EI = test_BO(e_i,:);
        [p_x,p_i] = max(PI);
        test_PI = test_BO(p_i,:);
        
        if(count > 50)
            [a,sel] = max(gains);
            if(sel == 1)
                eta = UCB;
                ucb_count = ucb_count + 1;
            elseif sel == 2
                eta = EI;
                ei_count = ei_count + 1;
            else
                eta = PI;
                pi_count = pi_count +1;
            end
        end
        candidates = test_BO;
    end
    
    %SELECT NEW PARAMETERS TO BE SIMULATED
    [max_MI,mindex] = max(eta);
    ymax = candidates(mindex,:);
    sample_new_idx = ismember(test_x,ymax,'rows');
    sample_new_idx = find(sample_new_idx ~= 0);
    sample_indices = [sample_new_idx;sample_indices];
    sample_new = test_x(sample_new_idx,:);
    
    target_new = f(sample_new);
    
    
    total_samples = [total_samples;sample_new];
    total_targets = [total_targets;target_new];
    
    
    
    [max_of_targets,xxx] = max(total_targets);
    max_sample = total_samples(xxx,:);
    
    
    [gp_all,std_all] = predictGP_MCMC(nsamples_thinned,meanfunc,covfunc,likfunc,total_samples,total_targets,test_x);
    max_of_variances = max(std_all).^2;
    
    BALDO_max_targets = [BALDO_max_targets;max_of_targets];
    BALDO_max_variance = [BALDO_max_variance;max_of_variances];
    
    %THIS IS FOR PRINTING AT EACH ITERATION. MODIFY ACCORDINGLY IF NEEDED
    count
    target_new
    max_of_targets
    previous_max_of_targets

    
    %THIS IS FOR SAVING THE WORKSPACE. 
    %If something goes wrong and the code terminates, this can be used as
    %checkpoint to start from. Modify the filename accordingly (include the
    %path if necessary as: 
    
    %savename = sprintf(".../.../GP_BALDO_checkpoint_%d.mat",count);
    savename = sprintf('GP_BALDO_checkpoint.mat');
%     save(savename);
end
%%
%Wrapper function used for training with SLS. 
function nll = nll_func(x,hyp,meanfunc,covfunc,likfunc,total_samples,total_targets)
nLik = length(hyp.lik);
nMean = length(hyp.mean);
hyp.lik = x(1:nLik);
hyp.mean = x(nLik+1:nLik + nMean);
hyp.cov = x(nLik + nMean + 1:end);
nll = gp_call(hyp,@infGaussLik,meanfunc,covfunc,likfunc,total_samples,total_targets);
end

