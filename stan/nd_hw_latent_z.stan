// Hammerstein-Wiener system model utilising hinges models on the input and output non-linearities.
functions {
    real saturation(real nu, real maxi, real mini) {
        real y;
        y = nu > maxi ? maxi : nu;
        y = nu < mini ? mini : y;
        return y;
    }

    real deadzone(real nu, real left, real right) {
        real y;
        y = nu > right ? nu - right : 0;
        return nu < left ? nu - left : y;
    }
}

data {
    int<lower=0> N; // length of discretised time window
    int<lower=0> n_u; // number of inputs
    int<lower=0> n_y; // number of outputs
    int<lower=0> n_x; // number of states of linear model

    vector[n_x] x0; // initial system state
    matrix[n_x,n_x] Q0; // covariance of initial state 

    vector[n_u] u[N]; // inputs
    vector[n_y] y[N]; // outputs
}

parameters {
    // memory-less non-linear block parameters
    vector[4] alpha;
    vector[4] beta;

    // linear block parameters
    matrix[n_x,n_x] A;
    matrix[n_x,n_u] B;
    matrix[n_y,n_x] C;
    matrix[n_y,n_u] D;

    vector[n_y] z[N];
    // matrix[n_y,N+1] z;
    
    vector[n_x] x0_p;

    // horseshoe covariance parameters for the e output noise and v driving noise processes
    // vector<lower=1e-9>[n_x] sq; // diagonal elements of diagonal sQ matrix 
    cholesky_factor_corr[n_y] Q_corr_chol; // n_y by n_y lower cholesky of correlation matrix
    vector<lower=0.0>[n_y] Q_tau; // scale vector, give weakly informative cauchy prior
    cholesky_factor_corr[n_y] R_corr_chol; // n_y by n_y lower cholesky of correlation matrix
    vector<lower=0.0>[n_y] R_tau; // scale vector, give weakly informative cauchy prior
}

transformed parameters {
    // tform for the full covariance matrix
    matrix[n_y,n_y] Q;
    matrix[n_y,n_y] R;

    matrix[n_x,N+1] x;
    matrix[n_u,N] w;
    matrix[n_y,N] z_hat_mat; // i really hate stan
    vector[n_y] z_hat[N]; // seriously, what the heck
    vector[n_y] y_hat[N]; // 

    Q = multiply_lower_tri_self_transpose(diag_pre_multiply(Q_tau,Q_corr_chol));    
    R = multiply_lower_tri_self_transpose(diag_pre_multiply(R_tau,R_corr_chol));
    x[:,1] = x0_p;
    for (i in 1:N){
        w[1,i] = saturation(u[i,1],alpha[1],alpha[2]);
        w[2,i] = deadzone(u[i,2],alpha[3],alpha[4]);
        x[:,i+1] = A*x[:,i] + B*w[:,i];
    }
    z_hat_mat = C*x[:,1:N] + D*w; // do not update x first
    for (i in 1:N){
        z_hat[i,1] = z_hat_mat[1,i]; // can you believe a for loop is needed for this transformation
        z_hat[i,2] = z_hat_mat[2,i];
        y_hat[i,1] = deadzone(z[i,1],beta[1],beta[2]);
        y_hat[i,2] = saturation(z[i,2],beta[3],beta[4]);
    }
}

model {
    real tau0 = 0.25; // cauchy hyperprior parameter
    // eta > 1, expect more correlation, eta < 1, less. eta = 1 is a unifrom prior over correlation matrices
    real eta = 1; // lkj hyperprior parameter

    // LKJ prior on noise covariance
    Q_tau ~ cauchy(0,tau0);
    Q_corr_chol ~ lkj_corr_cholesky(eta);
    R_tau ~ cauchy(0,tau0);
    R_corr_chol ~ lkj_corr_cholesky(eta);
    // prior for process variance
    // sq ~ cauchy(0, tau0);

    x0_p ~ multi_normal(x0,Q0);
    // x[1,2:N+1] ~ normal(x_hat[1,:],sq[1]);
    // x[2,2:N+1] ~ normal(x_hat[2,:],sq[2]);
    // x[3,2:N+1] ~ normal(x_hat[3,:],sq[3]);
    // x[4,2:N+1] ~ normal(x_hat[4,:],sq[4]);
    z ~ multi_normal(z_hat,Q);
    y ~ multi_normal(y_hat,R);

    // regularising prior 
    // alpha ~ normal(0,1);
    // beta ~ normal(0,1);

}
generated quantities {
    vector[n_y] y_hat_out[N];
    for (k in 1:N) {
        y_hat_out[k,:] = multi_normal_rng(y_hat[k,:],R);
    }
}