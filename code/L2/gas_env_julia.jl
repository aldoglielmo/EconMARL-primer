using POMDPs
using Random
using Distributions
using Parameters


struct GasStorageState
    # Time tracking
    t::Int                    # Current time step
    sim_index::Int           # Simulation index for seasonal tracking

    # Economic state variables
    u_d::Float32             # Demand shifter (persistent shock)
    u_s::Float32             # Supply shifter (persistent shock)
    p_d::Float32             # Log demand price index (EWMA)
    p_s::Float32             # Log supply price index (EWMA)
    I::Float32               # Current inventory level
    g::Float32               # Bank account balance
    p_prev::Float32          # Previous log price

    # Pre-generated shock sequences (for reproducibility)
    demand_shocks::Vector{Float32}   # Demand shock sequence
    supply_shocks::Vector{Float32}   # Supply shock sequence

    # Cumulative tracking variables
    p_var_cum::Float32       # Cumulative price variance
    penalties_cum::Float32   # Cumulative penalties

    # Reward
    reward::Float32

    # market clearing
    market_does_not_clear::Bool
end

function initial_gas_storage_state(gas_params; rng = Random.GLOBAL_RNG)
    # Generate shock sequences for the entire episode
    demand_shocks = Float32.(gas_params.sigma_d * randn(rng, gas_params.T_max))
    supply_shocks = Float32.(gas_params.sigma_s * randn(rng, gas_params.T_max) .+ gas_params.mu_s)

    return GasStorageState(
        1,                              # t = 1 (Julia 1-indexed)
        0,                              # sim_index = 0 (for November tracking)
        0.0f0,                          # u_d = 0
        0.0f0,                          # u_s = 0
        0.0f0,                          # p_d = 0
        0.0f0,                          # p_s = 0
        0.8f0 * gas_params.I_max,             # I = 80% of max capacity
        0.0f0,                          # g = 0
        0.0f0,                          # p_prev = 0
        demand_shocks,
        supply_shocks,
        0.0f0,                          # p_var_cum = 0
        0.0f0,                           # penalties_cum = 0
        0.0f0,                           # reward
        false                           # market_does_not_clear = false
    )
end


struct GasStorageParameters
    # Time parameters
    N::Int                   # Months per year (12)
    T_max::Int              # Total time steps (360 = 30 years)

    # Demand parameters
    eta_d::Float32          # Elasticity of demand (0.20)
    lambda_d::Float32       # Stickiness of demand (0.975)
    rho_d::Float32          # Persistence of demand shocks (0.98)
    sigma_d::Float32        # Volatility of demand shocks (0.01)

    # Supply parameters
    eta_s::Float32          # Elasticity of supply (0.30)
    lambda_s::Float32       # Stickiness of supply (0.95)
    rho_s::Float32          # Persistence of supply shocks (0.75)
    sigma_s::Float32        # Volatility of supply shocks (0.04)
    mu_s::Float32           # Mean of supply shocks (0.0)

    # Storage parameters
    tau::Float32            # Storage cost per month (0.005)
    I_max::Float32          # Storage capacity (3.0)
    r::Float32              # Interest rate on cash (1.0025)
    theta::Float32          # Volatility penalty weight (100.0)
    h::Float32              # Market clearing penalty (1000.0)
    penalty_thresh::Float32 # Threshold penalty (0.0)
    U::Float32              # Maximum price (100.0)
    L::Float32              # Minimum price (0.01)

    # Pre-computed seasonal patterns
    seasonal_demand::Vector{Float32}
    cos_seasonal::Vector{Float32}
    sin_seasonal::Vector{Float32}
    november_indices::Set{Int}

    # Derived parameters
    lambda_d_compl::Float32
    lambda_s_compl::Float32
end

"""
    GasStorageParameters()

Constructor for the gas storage MDP with default parameters.
"""
function GasStorageParameters()
    # Default parameters
    N = 12
    T_max = N * 30  # 30 years

    # Demand parameters
    eta_d = 0.2f0
    lambda_d = 0.975f0
    rho_d = 0.98f0
    sigma_d = 0.01f0

    # Supply parameters
    eta_s = 0.3f0
    lambda_s = 0.95f0
    rho_s = 0.75f0
    sigma_s = 0.04f0
    mu_s = 0.0f0

    # Storage parameters
    tau = 0.005f0
    I_max = 3.0f0
    r = 1.0025f0
    U = 100.0f0
    L = 0.01f0

    # Reward parameters
    theta = 20.0f0
    h = 1000.0f0
    penalty_thresh = 0.0f0

    # Derived parameters
    lambda_d_compl = 1.0f0 - lambda_d
    lambda_s_compl = 1.0f0 - lambda_s

    # Pre-compute seasonal components
    phi = 2 * π / N
    seasonal_demand = Float32[]
    cos_seasonal = Float32[]
    sin_seasonal = Float32[]

    for t in 1:T_max
        # Seasonal demand pattern (Fourier series)
        seasonality = (
            0.4276f0 * cos(phi * (t - 1)) + -0.0122f0 * sin(phi * (t - 1)) +
                0.1074f0 * cos(phi * 2 * (t - 1)) + -0.0003f0 * sin(phi * 2 * (t - 1)) +
                -0.0391f0 * cos(phi * 3 * (t - 1)) + -0.0023f0 * sin(phi * 3 * (t - 1)) +
                0.0126f0 * cos(phi * 4 * (t - 1)) + -0.0347f0 * sin(phi * 4 * (t - 1)) +
                0.0302f0 * cos(phi * 6 * (t - 1)) + 0.0f0 * sin(phi * 6 * (t - 1))
        )

        push!(seasonal_demand, seasonality)
        push!(cos_seasonal, cos(phi * (t - 1)))
        push!(sin_seasonal, sin(phi * (t - 1)))
    end

    # Pre-compute November indices for fast lookup
    november_indices = Set(10:12:T_max)

    return GasStorageParameters(
        N, T_max,
        eta_d, lambda_d, rho_d, sigma_d,
        eta_s, lambda_s, rho_s, sigma_s, mu_s,
        tau, I_max, r, theta, h, penalty_thresh, U, L,
        seasonal_demand, cos_seasonal, sin_seasonal, november_indices,
        lambda_d_compl, lambda_s_compl
    )
end

@with_kw mutable struct GasStorageEnv <: POMDP{Vector{Float32}, Vector{Float32}, Vector{Float32}}
    gas_params = GasStorageParameters()
    gas_state = initial_gas_storage_state(gas_params)
    γ::Float32 = 0.99f0
    obs_buffer::Vector{Float32} = Vector{Float32}(undef, 9)
end

Crux.state_space(p::GasStorageEnv) = Crux.ContinuousSpace((1,), Float32)
POMDPs.actions(p::GasStorageEnv) = ContinuousSpace((1,), Float32)  # Continuous action space
POMDPs.observations(p::GasStorageEnv) = Crux.ContinuousSpace((9,), Float32)  # Continuous observation space

function POMDPs.initialstate(p::GasStorageEnv)
    # Initialize the model with the given worker, firm, and capital parameters
    p.gas_params = GasStorageParameters()
    p.gas_state = initial_gas_storage_state(p.gas_params)
    return Dirac(zeros(Float32, 9))
end

function POMDPs.initialobs(p::GasStorageEnv, s)
    # Return the initial observation as a Dirac distribution with proper shape
    o = Float32[0, 0, 0, 0, 0, 0, 0, 0, 0]
    return Dirac(o)
end


function POMDPs.observation(p::GasStorageEnv, s_dummmy)
    gas_params = p.gas_params
    gas_state = p.gas_state
    t_idx = gas_state.t

    log_inventory = log(0.5f0 + gas_state.I)

    # Use pre-allocated buffer to avoid allocation on every call
    obs = p.obs_buffer
    obs[1] = gas_params.seasonal_demand[t_idx]
    obs[2] = gas_params.cos_seasonal[t_idx]
    obs[3] = gas_params.sin_seasonal[t_idx]
    obs[4] = gas_state.u_d
    obs[5] = gas_state.u_s
    obs[6] = gas_state.p_d
    obs[7] = gas_state.p_s
    obs[8] = log_inventory
    obs[9] = gas_state.p_prev

    return Dirac(copy(obs))
end

function POMDPs.transition(p::GasStorageEnv, s_dummmy::AbstractVector{Float32}, a)

    # Extract action
    log_price = a[1]
    P = exp(log_price)

    gas_params = p.gas_params
    gas_state = p.gas_state

    # Price variance calculation
    delta_p = log_price - gas_state.p_prev
    p_var = delta_p^2

    # Store old bank account for delta calculation
    old_g = gas_state.g

    # Update bank account with interest and storage cost
    new_g = old_g * gas_params.r - gas_state.I * gas_params.tau

    # Update price indices (EWMA)
    new_p_d = log(gas_params.lambda_d * exp(gas_state.p_d) + gas_params.lambda_d_compl * P)
    new_p_s = log(gas_params.lambda_s * exp(gas_state.p_s) + gas_params.lambda_s_compl * P)

    # Compute demand and supply with bounds checking eliminated
    @inbounds begin
        d = gas_state.u_d + gas_params.seasonal_demand[gas_state.t] - gas_params.eta_d * new_p_d
        supply = gas_state.u_s + gas_params.eta_s * new_p_s
        excess_demand = exp(d) - exp(supply)

        # Update shifters using pre-generated shocks
        new_u_d = gas_params.rho_d * gas_state.u_d + gas_state.demand_shocks[gas_state.t]
        new_u_s = gas_params.rho_s * gas_state.u_s + gas_state.supply_shocks[gas_state.t]
    end

    # Storage capacity calculations
    spare_storage_capacity = gas_params.I_max - gas_state.I

    # Market clearing checks
    demand_is_not_satisfied = excess_demand > gas_state.I
    supply_goes_to_waste = -excess_demand > spare_storage_capacity
    market_does_not_clear = demand_is_not_satisfied || supply_goes_to_waste
    market_clears = !market_does_not_clear

    # Market clearing penalty
    penalty = gas_params.h * market_does_not_clear *
        (
        1.0f0 + supply_goes_to_waste * (-excess_demand - spare_storage_capacity) +
            demand_is_not_satisfied * (excess_demand - gas_state.I)
    )

    # Inventory update
    delta_I = if market_clears
        -excess_demand
    elseif demand_is_not_satisfied
        -gas_state.I
    else  # supply_goes_to_waste
        spare_storage_capacity
    end

    new_I = gas_state.I + delta_I

    # Update bank account with transaction costs
    new_g = new_g - P * delta_I
    delta_g = new_g - old_g

    # Calculate reward
    reward = delta_g - gas_params.theta * p_var - penalty

    # November inventory penalty check
    new_sim_index = gas_state.sim_index + 1
    is_november = new_sim_index in gas_params.november_indices
    inventory_threshold = 0.83f0 * gas_params.I_max

    new_penalties_cum = gas_state.penalties_cum
    november_penalty = 0.0f0
    if is_november && new_I < inventory_threshold
        november_penalty = gas_params.penalty_thresh + (inventory_threshold - new_I) * gas_params.penalty_thresh
        reward = reward - november_penalty
        new_penalties_cum += gas_params.penalty_thresh
    end

    # Check if terminal and add terminal value
    new_t = gas_state.t + 1
    is_terminal = new_t >= gas_params.T_max
    terminal_value = 0.0f0
    if is_terminal
        # Terminal liquidation value
        market_value = 0.5f0 * (exp(new_p_d) + exp(new_p_s))
        terminal_value = new_I * market_value
        new_g = new_g + terminal_value
        delta_g = new_g - old_g
        reward = delta_g - gas_params.theta * p_var - penalty
    end

    next_state = GasStorageState(
        new_t,
        new_sim_index,
        new_u_d,
        new_u_s,
        new_p_d,
        new_p_s,
        new_I,
        new_g,
        log_price,  # p_prev = current p
        gas_state.demand_shocks,  # Keep same shock sequences
        gas_state.supply_shocks,
        gas_state.p_var_cum + p_var,
        new_penalties_cum,
        reward,
        market_does_not_clear
    )

    # Store additional variables for sampling
    # These will be accessible via the info mechanism in Crux
    p.gas_state = next_state

    return Dirac(s_dummmy)
end

POMDPs.discount(p::GasStorageEnv) = p.γ

function POMDPs.reward(p::GasStorageEnv, s_dummmy::AbstractVector{Float32}, a)
    reward = p.gas_state.reward
    return reward
end

function POMDPs.isterminal(p::GasStorageEnv, s_dummmy::AbstractVector{Float32})
    # Check if the maximum number of steps has been reached
    return p.gas_state.t >= p.gas_params.T_max
end
