data{
    int<lower=1> N;
    int<lower=1> N_id;
    real changehr[N];
    real monitoring[N];
    real monitoring_lagged[N];
    real shaming[N];
    real polity[N];
    real forinv[N];
    real conflict[N];
    real poplog[N];
    real gdplog[N];
    real time[N];
    real timesq[N];
    int id[N];
    real meanmon[N];
}
parameters{
    vector[N_id] bM_id;
    vector[N_id] bS_id;
    vector[N_id] alpha_id;
    real alpha;
    real bM;
    real bS;
    real bP;
    real bFI;
    real bCon;
    real bPop;
    real bGDP;
    real bT;
    real bT2;
    real bA;
    real bE;
    corr_matrix[2] Rho;
    real<lower=0> sigma;
    vector<lower=0>[2] sigma_id;
}
transformed parameters{
    vector[2] v_alpha_idbM_id[N_id];
    vector[2] Mu_00;
    cov_matrix[2] SRS_sigma_idRho;
    for ( j in 1:N_id ) {
        v_alpha_idbM_id[j,1] = alpha_id[j];
        v_alpha_idbM_id[j,2] = bM_id[j];
    }
    for ( j in 1:2 ) {
        Mu_00[1] = 0;
        Mu_00[2] = 0;
    }
    SRS_sigma_idRho = quad_form_diag(Rho,sigma_id);
}
model{
    vector[N] mu;
    sigma_id ~ inv_gamma(1, 1);
    sigma ~ exponential( 1 );
    Rho ~ lkj_corr( 2 );
    bE ~ normal( 0 , 1 );
    bA ~ normal( 0 , 0.5 );
    bT2 ~ normal( 0 , 0.25 );
    bT ~ normal( 0 , 0.25 );
    bS ~ normal(0, .5);
    bP~ normal(0, .5);
    bFI~ normal(0, .5);
    bCon~ normal(0, .5);
    bPop~ normal(0, .5);
    bGDP~ normal(0, .5);    
    bM ~ normal( 0 , 0.5 );
    bS_id~normal(0 , .5); 
    
    alpha ~ normal( 0 , 0.25 );
    v_alpha_idbM_id ~ multi_normal( Mu_00 , SRS_sigma_idRho );
    for ( i in 1:N ) {
        mu[i] = alpha + alpha_id[id[i]] + (bM + bM_id[id[i]]) * monitoring_lagged[i] + (bS + bS_id[id[i]]) * shaming[i]+
        bP*polity[i]+ bFI*forinv[i]+ bCon*conflict[i] + bPop*poplog[i]+bGDP*gdplog[i]
        +bT * time[i] + bT2 * timesq[i] +      bE * meanmon[i] + bA * monitoring[i];
    }
    changehr ~ normal( mu , sigma );
}

