//
// HMM_v2 for movement data
//
data {
  int<lower=0> T; // length of the time series
  int ID[T]; // track identifier, tells the model when one animal's track ends and another begins
  vector[T] steps; // step lengths, (distances between consecutive locations)
  vector[T] angles; // turning angles, (change in direction), modeling two variables now, not just one!
  int<lower=1> N; // Number of hidden states (still N = 2)
  int nCovs; // Number of environmental covariates (e.g., 3: temperature, slope, slope²)
  matrix[T,nCovs+1] covs; // NEW! Design matrix with covariates at each time point
}

parameters {
  positive_ordered[N] mu; // mean of gamma - ordered, - means of step length distributions for each state 
                          // `ordered` prevents label switching (μ₁ < μ₂)
                          // State 1 = short steps, State 2 = long steps
  vector<lower=0>[N] sigma; // SD of gamma, standard deviations of step lengths for each state
  // unconstrained angle parameters
  vector[N] xangle; 
  vector[N] yangle;
  // This is a **reparameterization trick** for von Mises distributions, 
   // Von Mises has parameters: mean angle μ ∈ [-π, π] and concentration κ > 0,
   // Sampling μ is hard because it's bounded (gets stuck at ±π)
   // **The Solution:**
   // - Use **Cartesian coordinates** instead of polar coordinates!
   // - Convert (μ, κ) → (x, y) where:
   // - `x = κ × cos(μ)`
   // - `y = κ × sin(μ)`
   // - Now x and y are **unbounded** (can be any real number)
   // - Stan samples x and y, then converts back to μ and κ,
   // think of it like representing a point on a circle: Polar: "go 3 units at angle 45°", Cartesian: "go to point (2.1, 2.1)"
  matrix[N*(N-1),nCovs+1] beta; // regression coefficients for transition probabilities
  // This contains **regression coefficients** for how covariates affect transitions.
  // **Why N*(N-1) rows?**
  //- With N=2 states, transition matrix has 4 entries: γ₁₁, γ₁₂, γ₂₁, γ₂₂
  //- But rows must sum to 1, so: γ₁₂ = 1 - γ₁₁ and γ₂₁ = 1 - γ₂₂
  //- We only need to model **off-diagonal** entries: γ₁₂ and γ₂₁
  //- N*(N-1) = 2*1 = **2 rows** (one for each off-diagonal transition)
  
  //**Why (nCovs+1) columns?**
  //*- One coefficient per covariate + intercept
  //*//- Example with 3 covariates: `[β₀, β_temp, β_slope, β_slope²]`
  //***What this means:**
  //*beta[1,] = coefficients for transition 1→2 (leaving State 1)
  //*beta[2,] = coefficients for transition 2→1 (leaving State 2)
}

transformed parameters {
  vector<lower=0>[N] shape;
  vector<lower=0>[N] rate;
  vector<lower=-pi(),upper=pi()>[N] loc;
  vector<lower=0>[N] kappa;
  // derive turning angle mean and concentration
  for(n in 1:N) {
    loc[n] = atan2(yangle[n], xangle[n]); //*- Converts Cartesian (x,y) back to **mean angle** μ, 
                                          //`atan2()` handles all four quadrants correctly
    kappa[n] = sqrt(xangle[n]*xangle[n] + yangle[n]*yangle[n]); 
    } // - Converts to **concentration** κ (how focused the angles are), This is just the Euclidean distance: √(x² + y²)
      //- Higher κ = more directional movement
    
  // transform mean and SD to shape and rate
  for(n in 1:N)
    shape[n] = mu[n]*mu[n]/(sigma[n]*sigma[n]); // Stan's gamma() uses shape and rate parameters, 
                                                //but we estimate mean and SD (more interpretable).
  
  for(n in 1:N)
    rate[n] = mu[n]/(sigma[n]*sigma[n]);
}

model {
  vector[N] logp; // forward variable (probability of being in each state)
  vector[N] logptemp; // temporary storage for next time step
  matrix[N,N] gamma[T]; // array of T transition matrices (one per time point!)
  matrix[N,N] log_gamma[T]; // log probabilities
  matrix[N,N] log_gamma_tr[T]; // transposed log probabilities

  // priors
  mu ~ normal(0, 5); // means are probably between -10 and 10 (weakly informative)
  sigma ~ student_t(3, 0, 1); //  SDs are positive, probably small
  xangle[1] ~ normal(-0.5, 1); // equiv to concentration when yangle = 0
  xangle[2] ~ normal(2, 2); 
  yangle ~ normal(0, 0.5); // zero if mean angle is 0 or pi, y-components near 0 (means angles are near 0 or π)

  // derive array of (log-)transition probabilities
  for(t in 1:T) {
    int betarow = 1; // Counter to track which regression we're using
    
    for(i in 1:N) {
      for(j in 1:N) {
        if(i==j) {gamma[t,i,j] = 1;} else { 
          // Diagonal entries start at 1 (will be normalized later), 
          gamma[t,i,j] = exp(beta[betarow] * to_vector(covs[t])); //Off-diagonal entries use logistic regression!
          betarow = betarow + 1;
        }
      }
    }

  // each row must sum to 1
  for(i in 1:N)
  log_gamma[t][i] = log(gamma[t][i]/sum(gamma[t][i]));
  }
  // transpose
  for(t in 1:T)
  for(i in 1:N)
  for(j in 1:N)
  log_gamma_tr[t,j,i] = log_gamma[t,i,j];
  
  // likelihood computation
  for (t in 1:T) {
    // initialise forward variable if first obs of track
    if(t==1 || ID[t]!=ID[t-1]) // CRITICAL: Reset when starting a new animal's track!
    logp = rep_vector(-log(N), N); // uniform initial distribution [log(0.5), log(0.5)]
    
    for (n in 1:N) {
        logptemp[n] = log_sum_exp(to_vector(log_gamma_tr[t,n]) + logp); //Sum over all ways to reach state n (same as before!)
                                                                        //Uses time-specific transition matrix log_gamma_tr[t]
        if(steps[t]>=0) //Check if step is observed (not missing)
          logptemp[n] = logptemp[n] + gamma_lpdf(steps[t] | shape[n], rate[n]); 
          //gamma_lpdf() = log probability of observing this step length in state n
          
        if(angles[t]>=(-pi())) // Check if angle is observed
          logptemp[n] = logptemp[n] + von_mises_lpdf(angles[t] | loc[n], kappa[n]);
          //log probability of observing this turning angle in state n
          // Missing data handling: If step or angle is NA (coded as -10), we skip adding its likelihood!
    }
    
    logp = logptemp; // Save current forward variable for next iteration
      // add log forward variable to target at the end of each track
      if(t==T || ID[t+1]!=ID[t]) // End of track: either last observation OR switching to new animal
      
      target += log_sum_exp(logp); // Add final likelihood to target
  } //Sum across final states to get total likelihood for this track
}

