"""
Minimal RBC simulators (KL & KS variants).

Change the baseline parameters once; all models inherit them.
"""

import numpy as np

# ------------------------------------------------------------------
# Baseline parameters ---------------------------
# ------------------------------------------------------------------
alpha          = 0.36   # Capital share in Cobb–Douglas production
delta          = 0.025  # Capital depreciation per period
rho            = 0.9    # Persistence of log‑TFP (AR(1))
sigma          = 0.01   # Std. dev. of TFP shock
gamma          = 1.0    # CRRA coefficient
u_g            = 0.04   # Unemployment in good state (KS)
u_b            = 0.10   # Unemployment in bad  state (KS)
l_bar          = 1.11   # Labour endowment per employed worker (KS)
zg             = 1.01   # TFP level, good state
zb             = 0.99   # TFP level, bad  state
low_cap_prod   = 0.5   # κ for low‑productivity group (KS‑het)
mid_cap_prod   = 1.0    # κ for middle group (KS‑het)
high_cap_prod  = 1.5    # κ for high‑productivity group (KS‑het)
Prod_GRID = np.array([0.98, 1.0, 1.02])  # κ, λ grid for KL‑het

def make_centered_grid_1d(side: int, half_width: float = 0.02, center: float = 1.0):
    """
    Create a 1D grid of 'side' points centered at 'center'
    with total width = 2 * half_width.
    Example: side=3, half_width=0.02 -> [0.98, 1.00, 1.02]
    """
    lo, hi = center - half_width, center + half_width
    return np.linspace(lo, hi, side)


def make_kappa_lambda_grid(side: int, half_width: float = 0.02):
    """
    Build a square grid of (kappa, lambda) pairs.
    Returns two arrays of length side*side with all combinations.
    """
    g = make_centered_grid_1d(side=side, half_width=half_width, center=1.0)
    kap, lab = np.meshgrid(g, g)         # 2D grid
    return kap.ravel(), lab.ravel()      # Flatten to vectors

# ------------------------------------------------------------------
# Utility functions ------------------------------------------------
# ------------------------------------------------------------------

def _CRRA_utility_single(c, gamma=1.0):
    """Single‑argument CRRA utility with a heavy penalty for c ≤ 0."""
    if c <= 0:
        return -1e9
    if np.isclose(gamma, 1.0):
        return np.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def _CRRA_utility_cl(c, l, gamma=1.0, b=5.0):
    """Utility from consumption and leisure."""
    return _CRRA_utility_single(c, gamma) + b * _CRRA_utility_single(1.0 - l, gamma)

# ==================================================================
# KL MODELS  (capital + labour, mean‑field) -------------------------
# ==================================================================

class RBCSimulator_KL:
    """Homogeneous agents, endogenous labour supply (baseline KL)."""

    def __init__(self, alpha=alpha, delta=delta, rho=rho, sigma=sigma,
                 n_agents=1, gamma=1.0):
        # Parameters
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        self.sigma = sigma
        self.n = n_agents
        self.gamma = gamma
        self.reset()

    # --------------------------------------------------------------
    # Initialisation / reset ---------------------------------------
    # --------------------------------------------------------------
    def reset(self):
        # Individual state: capital k_i, labour l_i, consumption c_i
        self.ks = np.ones(self.n)
        self.ls = np.ones(self.n)
        self.cs = np.zeros(self.n)
        # Aggregate technology
        self.z= 0.0  # Log TFP
        self.A = 1.0  # TFP level
        self._update_prod()  # Compute initial prices & output

    # --------------------------------------------------------------
    # Technology and prices ----------------------------------------
    # --------------------------------------------------------------
    def _update_tfp(self):
        """AR(1) shock to log‑TFP."""
        self.z = self.rho * self.z + self.sigma * np.random.randn()
        self.A = np.exp(self.z)

    def _update_prod(self):
        """Aggregate production Y and factor demand."""
        self.K = np.mean(self.ks)
        self.L = np.mean(self.ls)
        self.Y = self.A * self.K ** self.alpha * self.L ** (1.0 - self.alpha)

    def factor_prices(self):
        """Compute factor prices r (capital) and w (wages)."""
        self.r = self.alpha * (self.Y / self.K)
        self.w = (1.0 - self.alpha) * (self.Y / self.L)

    def update_hh(self, actions):
        """Update household choices given actions = [c_frac_i … l_i]."""
        # Split action vector
        self.c_fracs = actions[:self.n]
        self.ls = actions[self.n:]

        # Perfect competition ⇒ zero profits
        self.incomes = self.r * self.ks + self.w * self.ls
        self.wealths = self.incomes + (1 - self.delta) * self.ks
        self.cs = self.wealths * self.c_fracs
        self.ks = self.wealths * (1 - self.c_fracs)

    # --------------------------------------------------------------
    # One‑period transition ----------------------------------------
    # --------------------------------------------------------------
    def step(self, actions):
        """Advance one period (actions length = 2n)."""
        self._update_tfp()      # 1. Draw TFP shock
        self._update_prod()     # 2. Update production
        self.factor_prices()    # 3. Compute prices
        self.update_hh(actions) # 4. Update households

    def get_utilities_old(self):
        """Return current‑period utilities for all agents."""
        return np.array([_CRRA_utility_cl(c, l, self.gamma) for c, l in zip(self.cs, self.ls)])

    def get_utilities(self):
        mask = self.cs <= 0
        safe_cs = np.where(mask, 1.0, self.cs)
        u = np.where(mask, -1e9, np.log(safe_cs))  # gamma=1 case
        return u
# ------------------------------------------------------------------
# KL Variants -------------------------------------------------------
# ------------------------------------------------------------------
class RBCSimulator_KL_theory(RBCSimulator_KL):
    """Full depreciation variant (δ = 1)."""
    def __init__(self, **kw):
        kw.setdefault('delta', 1.0)
        super().__init__(**kw)


# class RBCSimulator_KL_Heterogeneous(RBCSimulator_KL):
#     """Fixed heterogeneous productivities in capital (κ) and labour (λ)."""

#     prod_grid = Prod_GRID  # Common grid for (κ, λ)

#     def __init__(self, **kw):
#         self.n = kw.get('n_agents', 2)
#         # Assign (κ, λ) from 3×3 grid
#         kap, lab = np.meshgrid(self.prod_grid, self.prod_grid)
#         pairs = np.column_stack([kap.ravel(), lab.ravel()])
#         if self.n <= 9:
#             self.kappa, self.lambda_ = pairs[: self.n].T
#         else:
#             idx = np.random.choice(len(pairs), self.n, replace=True)
#             self.kappa, self.lambda_ = pairs[idx].T
#         super().__init__(**kw)

#     def _update_prod(self):
#         """Aggregate production with heterogeneous (κ, λ)."""
#         self.KK = np.mean(self.ks * self.kappa)
#         self.LL = np.mean(self.ls * self.lambda_)
#         self.Y = self.A * self.KK ** self.alpha * self.LL ** (1.0 - self.alpha)

#     def factor_prices(self):
#         """Individual factor prices under heterogeneity."""
#         self.r = (self.alpha / self.n) * (self.Y / self.KK) * self.kappa
#         self.w = ((1 - self.alpha) / self.n) * (self.Y / self.LL) * self.lambda_


class RBCSimulator_KL_Heterogeneous(RBCSimulator_KL):
    """
    KL model with heterogeneous productivities (kappa, lambda).
    Each agent is assigned a unique pair from a square grid.

    Parameters
    ----------
    grid_side : int
        Side length of the grid (e.g. 3, 5, 7, 10, 15, 23).
        Total number of agents = grid_side^2.
    half_width : float
        Half-width around 1 for the grid. 
        Example: half_width=0.02 -> values in [0.98, 1.02].
    """

    def __init__(self, *args, grid_side=3, half_width=0.02, **kw):
        # Required number of agents = grid_side^2
        n_expected = grid_side * grid_side
        if 'n_agents' in kw:
            if kw['n_agents'] != n_expected:
                raise ValueError(f"n_agents must equal grid_side^2 = {n_expected}, "
                                 f"but got {kw['n_agents']}")
        else:
            kw['n_agents'] = n_expected

        # Build the kappa/lambda grid (unique pair for each agent)
        self.kappa, self.lambda_ = make_kappa_lambda_grid(side=grid_side,
                                                          half_width=half_width)

        # Store number of agents
        self.n = kw['n_agents']

        # Call parent init
        super().__init__(*args, **kw)

    def _update_prod(self):
        """
        Aggregate production with heterogeneous inputs.
        Uses effective capital and effective labor.
        """
        self.KK = np.mean(self.ks * self.kappa)
        self.LL = np.mean(self.ls * self.lambda_)
        self.Y  = self.A * self.KK ** self.alpha * self.LL ** (1.0 - self.alpha)

    def factor_prices(self):
        """Individual factor prices under heterogeneity."""
        self.r = self.alpha * (self.Y / self.KK) * self.kappa
        self.w = (1 - self.alpha)* (self.Y / self.LL) * self.lambda_

# ---------------------------------------------------------------------------------
# KL Variants for impulse‑response analysis ---------------------------------------
# ---------------------------------------------------------------------------------

class RBCSimulator_KL_IRF(RBCSimulator_KL):
    """Impulse‑response: one‑off TFP jump followed by AR(1) decay."""

    def __init__(self, t_shock=100, shock_size=0.1, **kw):
        self.t_shock, self.shock = t_shock, shock_size
        super().__init__(**kw)
        self.t = 0

    def _update_tfp(self):
        """Deterministic TFP path for IRF."""
        self.t += 1
        if self.t == self.t_shock:
            self.z = self.shock
        elif self.t > self.t_shock:
            self.z = self.rho * self.z
        else:
            self.z = 0.0
        self.A = np.exp(self.z)


class RBCSimulator_KL_TH_IRF(RBCSimulator_KL_IRF):
    """Full depreciation version."""
    def __init__(self, **kw):
        kw.setdefault('delta', 1.0)
        super().__init__(**kw)

# ==================================================================
# KS MODELS  (aggregate & idiosyncratic shocks) ---------------------
# ==================================================================

class RBCSimulator_KS_Heterogeneous:
    """KS model with heterogeneous κ: 15% low, 70% mid, 15% high."""

    def __init__(self, alpha=0.36, delta=0.025, n_agents=20, gamma=1.0,
                 zg=zg, zb=zb, u_g=u_g, u_b=u_b, l_bar=l_bar,
                 low_cap_prod=low_cap_prod, mid_cap_prod=mid_cap_prod, high_cap_prod=high_cap_prod,
                 P_z=None):

        # Core parameters
        self.alpha = alpha
        self.delta = delta
        self.n = n_agents
        self.gamma = gamma

        # Aggregate technology
        self.z_vals = np.array([zg, zb])
        self.P_z = P_z if P_z is not None else np.array([[0.875, 0.125],
                                                         [0.125, 0.875]])

        # Labour market
        self.u_g, self.u_b = u_g, u_b
        self.l_bar = l_bar

        # Idiosyncratic employment transition matrices
        self.P_eps = {
            (0, 0): np.array([[0.97222222, 0.02777778],
                              [0.66666667, 0.33333333]]),
            (0, 1): np.array([[0.92708333, 0.07291667],
                              [0.25      , 0.75      ]]),
            (1, 0): np.array([[0.98333333, 0.01666667],
                              [0.75      , 0.25      ]]),
            (1, 1): np.array([[0.95555556, 0.04444444],
                              [0.4       , 0.6       ]])
        }

        # Heterogeneous capital productivity (κ)
        n_low  = max(1, int(np.ceil(0.80 * n_agents)))
        n_high = max(1, int(np.ceil(0.02 * n_agents)))
        n_mid  = n_agents - n_low - n_high
        self.cap_prods = np.concatenate([
            np.full(n_low,  low_cap_prod),
            np.full(n_mid,  mid_cap_prod),
            np.full(n_high, high_cap_prod)
        ])

        # self._capital_initialized = False

        self.reset()  # Initialise state

    def reset(self):
        """Initialise or reset the economy."""
        self.ks = np.random.uniform(10, 70, size=self.n)  # Capital stocks
        # if not self._capital_initialized:
        #     self.ks = np.random.uniform(10, 70, size=self.n)  # Capital stocks
        #     self._capital_initialized = True
                
        # self.ks = np.random.uniform(10, 70, size=self.n)  # Capital stocks
        self.cs = np.zeros(self.n)
        self.incomes = np.zeros(self.n)
        self.wealths = self.ks.copy()

        self.eps = None  # Employment status
        self.z_flag = None  # Aggregate state index
        self.prev_z_flag = None  # Previous aggregate state index
        self.z = None
        self.K = None
        self.KK = None
        self.L = None
        self.Y = None
        self.r = None
        self.w = None
        self.u_z = None
        self.done = False  # Episode termination flag
        self._update_shocks()

    # ------------------------ internal helpers --------------------
    def _adjust_employment(self, eps, target_rate):
        """Force employment to match target unemployment rate."""
        target = int(round((1 - target_rate) * self.n))
        gap = target - eps.sum()
        if gap > 0:
            zeros = np.where(eps == 0)[0]
            eps[np.random.choice(zeros, gap, replace=False)] = 1
        elif gap < 0:
            ones = np.where(eps == 1)[0]
            eps[np.random.choice(ones, -gap, replace=False)] = 0
        return eps

    def _update_shocks(self):
        """Draw aggregate and idiosyncratic shocks."""
        # Aggregate TFP state
        if self.z_flag is None:
            p01, p10 = self.P_z[0,1], self.P_z[1,0]
            pi = [p10/(p01+p10), p01/(p01+p10)]
            self.z_flag = np.random.choice([0,1], p=pi)
            self.prev_z_flag = self.z_flag  # Initialize previous state
        else:
            self.prev_z_flag = self.z_flag  # Store current as previous
            self.z_flag = np.random.choice([0,1], p=self.P_z[self.prev_z_flag])
        self.z = self.z_vals[self.z_flag]

        # Employment status
        if self.eps is None:
            u0 = self.u_g if self.z_flag == 0 else self.u_b
            p_emp = 1 - u0
            eps = (np.random.rand(self.n) < p_emp).astype(int)
            self.eps = self._adjust_employment(eps, u0)
        else:
            # Use aggregate state transition to select employment transition matrix
            # Key: (previous_aggregate_state, current_aggregate_state)
            if not hasattr(self, 'prev_z_flag'):
                # Fallback if prev_z_flag not set (shouldn't happen with fixed init)
                self.prev_z_flag = self.z_flag
            
            key = (self.prev_z_flag, self.z_flag)
            Pmat = self.P_eps[key]
            
            # Row index: current employment status (unemployed=1, employed=0)
            row_idx = 1 - self.eps
            p_emp_vec = Pmat[row_idx, 0]  # Probability of becoming employed
            new_eps = (np.random.rand(self.n) < p_emp_vec).astype(int)
            u_now = self.u_g if self.z_flag == 0 else self.u_b
            self.eps = self._adjust_employment(new_eps, u_now)

    def update_production(self):
        """Compute aggregates and effective capital."""
        self.K = np.mean(self.ks)
        self.KK = np.mean(self.ks * self.cap_prods)  # Effective capital
        self.u_z = self.u_g if self.z_flag == 0 else self.u_b
        self.L = self.l_bar * (1 - self.u_z)
        self.Y = self.z * self.KK**self.alpha * self.L**(1 - self.alpha)

    def factor_prices(self):
        """Compute factor prices r and w."""
        self.r = self.alpha * (self.Y / self.KK) * self.cap_prods
        self.w = (1 - self.alpha) * (self.Y / self.L)

    def update_hh(self, actions):
        """Update individual states given consumption fractions."""
        self.c_fracs = actions
        self.incomes = self.r * self.ks + self.w * self.l_bar * self.eps
        self.wealths = self.incomes + (1 - self.delta) * self.ks
        self.cs = self.wealths * self.c_fracs
        self.ks = self.wealths * (1 - self.c_fracs)

    def step(self, actions):
        """Advance one period."""
        self._update_shocks()
        self.update_production()
        self.factor_prices()
        self.update_hh(actions)
        
    def get_utilities_old(self):
        """Return CRRA utilities."""
        return np.array([_CRRA_utility_single(c, self.gamma) for c in self.cs])

    def get_utilities(self):
        mask = self.cs <= 0
        safe_cs = np.where(mask, 1.0, self.cs)
        u = np.where(mask, -1e9, np.log(safe_cs))  # gamma=1 case
        return u
# --------------------

class RBCSimulator_KS(RBCSimulator_KS_Heterogeneous):
    """KS with homogeneous κ = 1."""
    def __init__(self, alpha=0.36, delta=0.025, n_agents=20, gamma=1.0,
                 zg=zg, zb=zb, u_g=u_g, u_b=u_b, l_bar=l_bar,
                 P_z=None):
        super().__init__(alpha=alpha, delta=delta, n_agents=n_agents,
                         gamma=gamma, zg=zg, zb=zb,
                         u_g=u_g, u_b=u_b, l_bar=l_bar,
                         low_cap_prod=1.0, mid_cap_prod=1.0, high_cap_prod=1.0,
                         P_z=P_z)

