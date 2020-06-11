% Copyright (c) 2020 
% School of Electrical and Computer Engineering
% 3D Packaging Research Center (PRC)
% Georgia Institute of Technology

%Wrapper for predicting with GP to avoid numerical errors.
%Numerical noise is added to diagonal of covariance matrix if its
%eigenvalues are less than 0, which should not happen as the matrix is
%positive semi-definite by definition.
function [nll,dll,final_lik] = gp_call(hyp2, infFunc, meanfunc, covfunc, likfunc, total_samples, total_targets,allCombinations)
done = false;

while (~done)
    try
        if (nargin > 7)
            [nll,dll] = gp(hyp2, infFunc, meanfunc, covfunc, likfunc, total_samples, total_targets,allCombinations);
        else
            if(nargout > 1)
                [nll,dll] = gp(hyp2, infFunc, meanfunc, covfunc, likfunc, total_samples, total_targets);
            else
                nll = gp(hyp2, infFunc, meanfunc, covfunc, likfunc, total_samples, total_targets);
            end
        end
        done = true;
        if(nargout>2)
            final_lik = hyp2.lik;
        end
    catch
        if(hyp2.lik < -15)
            hyp2.lik = -15;
        else
            hyp2.lik = hyp2.lik + 0.5;
        end
        done = false;
    end
    
    
    
    
end
