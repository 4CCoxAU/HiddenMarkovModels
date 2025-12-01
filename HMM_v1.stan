data {
  int<lower=0> N; // number of states
  int<lower=1> T; // length of data set
  real y[T]; // observations
}

parameters {
  simplex[N] theta[N]; // N x N tpm, this is our Gamma from before, simplex means each row must sum to 1 (they're probabilities!),
                      //but we don't know its values, means each row must sum to 1 (they're probabilities!)
  ordered[N] mu; // state-dependent parameters, These are like our mu = c(1, 5) from before,
                 //ordered means μ₁ < μ₂ < μ₃... (prevents label switching)
}

transformed parameters{
  matrix[N, N] ta; //
  simplex[N] statdist; // stationary distribution
  
  // Copy theta into a matrix format
  for(j in 1:N){
    for(i in 1:N){
      ta[i,j]= theta[i,j];
    }
  }
      // Calculate stationary distribution, this is computing the same thing as our earlier delta calculation, 
      // the long-run proportion of time in each state, given the transition probabilities.
      statdist = to_vector((to_row_vector(rep_vector(1.0, N))/(diag_matrix(rep_vector(1.0, N)) - ta + rep_matrix(1, N, N)))) ;
}

model {
  vector[N] log_theta_tr[N]; // log of transposed transition matrix
  vector[N] lp; // "log probability" - forward variable
  vector[N] lp_p1; // temporary storage for next time step
  
  // prior for mu
  mu ~ student_t(3, 0, 1);
  
  // transpose the tpm and take natural log of entries, 
  // takes the log of transition probabilities (for numerical stability), 
  // transposes the matrix (rows become columns)
  for (n_from in 1:N)
  for (n in 1:N)
    log_theta_tr[n, n_from] = log(theta[n_from, n]);

  // forward algorithm implementation
  for(n in 1:N) // first observation, initialize at time t=1
    lp[n] = log(statdist[n]) + normal_lpdf(y[1] | mu[n], 2); // For each state n, calculate: "What's the probability 
                                                            // we started in state n AND observed y[1]?
  for (t in 2:T) { // looping over observations, 
    for (n in 1:N) // looping over states, 
      lp_p1[n] = log_sum_exp(log_theta_tr[n] + lp) + normal_lpdf(y[t] | mu[n], 2);
                //"probability of transitioning INTO state n from any previous state"
                //efficiently compute the total probability (summing over all possible paths)
                //multiply by probability of observing y[t] in state n
    lp = lp_p1; // Think of lp as a running tally of "how likely is it we're in each state at time t, 
                // given all observations so far?"
  }
  
  target += log_sum_exp(lp);
    // sum across all possible final states to get the total likelihood of the observed data.
}
