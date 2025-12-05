# hyperparameter optimization algorithms

this document explains how the hyperparameter optimization (hpo) algorithms work in hpoea.

## overview

hyperparameter optimization finds the best settings for an evolutionary algorithm. instead of manually tuning parameters like mutation rate or population size, an outer optimizer searches for values that produce the best results.

the process works in two layers:
1. outer loop: hpo algorithm proposes hyperparameter configurations
2. inner loop: evolutionary algorithm runs with those hyperparameters and returns fitness

```
hpo algorithm
    |
    +-- proposes hyperparameters (F=0.8, CR=0.9, ...)
    |
    +-- runs EA with those hyperparameters
    |
    +-- receives fitness score
    |
    +-- proposes new hyperparameters based on result
    |
    (repeat until budget exhausted)
```

## implemented algorithms

### cma-es (covariance matrix adaptation evolution strategy)

maintains a multivariate gaussian distribution over the hyperparameter space. adapts the covariance matrix based on successful steps to learn correlations between parameters.

**algorithm:**

the search distribution at generation $g$ is:

$$\mathbf{x} \sim \mathcal{N}(\mathbf{m}^{(g)}, (\sigma^{(g)})^2 \mathbf{C}^{(g)})$$

where:
- $\mathbf{m}$ is the distribution mean (current best estimate)
- $\sigma$ is the global step size
- $\mathbf{C}$ is the covariance matrix encoding parameter correlations

**sampling:** generate $\lambda$ offspring by sampling from the distribution:

$$\mathbf{x}_k = \mathbf{m} + \sigma \mathbf{y}_k, \quad \mathbf{y}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{C})$$

**selection:** sort offspring by fitness and select the best $\mu$ individuals.

**mean update:** move toward weighted average of selected individuals:

$$\mathbf{m}^{(g+1)} = \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}$$

where $w_i$ are recombination weights (higher for better individuals) and $\mathbf{x}_{i:\lambda}$ is the $i$-th best individual.

**evolution path:** accumulate successful steps to detect correlated directions:

$$\mathbf{p}_c^{(g+1)} = (1 - c_c)\mathbf{p}_c^{(g)} + \sqrt{c_c(2 - c_c)\mu_{\text{eff}}} \frac{\mathbf{m}^{(g+1)} - \mathbf{m}^{(g)}}{\sigma^{(g)}}$$

where $c_c \approx 4/n$ controls the path length and $\mu_{\text{eff}} = 1/\sum w_i^2$ is the effective selection mass.

**covariance update:** combine rank-one update (from evolution path) and rank-$\mu$ update (from current generation):

$$\mathbf{C}^{(g+1)} = (1 - c_1 - c_\mu)\mathbf{C}^{(g)} + c_1 \mathbf{p}_c \mathbf{p}_c^T + c_\mu \sum_{i=1}^{\mu} w_i \mathbf{y}_{i:\lambda} \mathbf{y}_{i:\lambda}^T$$

where:
- $c_1 \approx 2/n^2$ is the rank-one learning rate
- $c_\mu \approx \mu_{\text{eff}}/n^2$ is the rank-$\mu$ learning rate

**step size adaptation:** uses cumulative step-size adaptation (csa) with a conjugate evolution path:

$$\sigma^{(g+1)} = \sigma^{(g)} \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_\sigma\|}{E\|\mathcal{N}(\mathbf{0},\mathbf{I})\|} - 1\right)\right)$$

if steps are consistently longer than expected under random selection, increase $\sigma$; if shorter, decrease it.

**parameters:**
- `generations`: number of cma-es generations (default: 100)
- `sigma0`: initial step size $\sigma^{(0)}$ (default: 0.5)
- `cc`: evolution path decay rate (default: -1 = auto, $\approx 4/n$)
- `cs`: step-size cumulation rate (default: -1 = auto)
- `c1`: rank-one update rate (default: -1 = auto, $\approx 2/n^2$)
- `cmu`: rank-$\mu$ update rate (default: -1 = auto)
- `ftol`: fitness tolerance for convergence (default: 1e-6)
- `xtol`: decision vector tolerance for convergence (default: 1e-6)
- `memory`: retain adapted parameters between calls (default: false)
- `force_bounds`: enforce box bounds during search (default: false)

**when to use:** continuous hyperparameters, moderate dimensionality (5-50 parameters), smooth fitness landscape.

---

### simulated annealing

single-point search that accepts worse solutions with decreasing probability over time. based on metallurgical annealing where controlled cooling produces low-energy crystal structures.

**algorithm:**

at each iteration, propose a neighbor $\mathbf{x}'$ of current solution $\mathbf{x}$ and accept it with probability:

$$P(\text{accept}) = \begin{cases} 1 & \text{if } f(\mathbf{x}') < f(\mathbf{x}) \\ \exp\left(-\frac{f(\mathbf{x}') - f(\mathbf{x})}{T}\right) & \text{otherwise} \end{cases}$$

where $T$ is the temperature parameter.

**neighbor generation:** perturb current hyperparameters within a range:

$$x'_i = x_i + r_i \cdot \text{range}_i \cdot U(-1, 1)$$

where $r_i$ is the current range for dimension $i$ and $U(-1,1)$ is uniform random.

**cooling schedule:** temperature decreases from $T_s$ to $T_f$ over the annealing process. the schedule is controlled by `n_T_adj` temperature adjustments per cycle:

$$T_{k+1} = T_k \cdot r_T$$

where the reduction factor $r_T$ is computed to reach $T_f$ from $T_s$ in `n_T_adj` steps:

$$r_T = \left(\frac{T_f}{T_s}\right)^{1/\text{n\_T\_adj}}$$

this gives:
- high $T$ early: accept most moves, explore broadly
- low $T$ late: accept only improvements, exploit local region

**adaptive range:** the perturbation range adjusts based on acceptance ratio. if too many moves accepted, increase range; if too few, decrease it:

$$\text{range}_i \leftarrow \text{range}_i \cdot \begin{cases} 1 + c(a_i - 0.6)/0.4 & \text{if } a_i > 0.6 \\ 1/(1 + c(0.4 - a_i)/0.4) & \text{if } a_i < 0.4 \\ 1 & \text{otherwise} \end{cases}$$

where $a_i$ is the acceptance ratio in dimension $i$ and $c$ is a constant.

**parameters:**
- `iterations`: number of annealing cycles/evolve calls (default: 1000)
- `ts`: starting temperature $T_s$ (default: 10.0)
- `tf`: final temperature $T_f$ (default: 0.1)
- `n_T_adj`: temperature adjustments per cycle (default: 10)
- `n_range_adj`: range adjustments per temperature level (default: 1)
- `bin_size`: samples per bin for acceptance ratio (default: 10)
- `start_range`: initial perturbation range as fraction of bounds (default: 1.0)

note: each cycle performs `n_T_adj * n_range_adj * bin_size * dim` function evaluations.

**when to use:** rough fitness landscapes, avoiding local optima, limited budget.

---

### pso (particle swarm optimization)

swarm of particles explore hyperparameter space. each particle maintains position, velocity, and memory of its best position. inspired by bird flocking behavior.

**algorithm:**

each particle $i$ has:
- position $\mathbf{x}_i$ (current hyperparameters)
- velocity $\mathbf{v}_i$ (search direction)
- personal best $\mathbf{p}_i$ (best position found by this particle)
- global best $\mathbf{g}$ (best position found by any particle)

**velocity update:** combine inertia, cognitive (personal), and social (swarm) components:

$$\mathbf{v}_i^{(t+1)} = \omega \mathbf{v}_i^{(t)} + \eta_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^{(t)}) + \eta_2 r_2 (\mathbf{g} - \mathbf{x}_i^{(t)})$$

where:
- $\omega$ is inertia weight (momentum from previous velocity)
- $\eta_1$ is cognitive coefficient (attraction to personal best)
- $\eta_2$ is social coefficient (attraction to global best)
- $r_1, r_2 \sim U(0,1)$ are random factors for stochasticity
- $\mathbf{p}_i$ is particle's personal best position
- $\mathbf{g}$ is global best position

the constriction coefficient variant (default, variant 5) uses:

$$\mathbf{v}_i^{(t+1)} = \omega \left( \mathbf{v}_i^{(t)} + \eta_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^{(t)}) + \eta_2 r_2 (\mathbf{g} - \mathbf{x}_i^{(t)}) \right)$$

**position update:**

$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$

**velocity clamping:** prevent explosion by limiting velocity magnitude:

$$v_{ij} \leftarrow \text{sign}(v_{ij}) \cdot \min(|v_{ij}|, v_{\max})$$

**constriction coefficients:** the default values $\omega = 0.7298$, $\eta_1 = \eta_2 = 2.05$ come from clerc's constriction factor:

$$\chi = \frac{2}{\phi - 2 + \sqrt{\phi^2 - 4\phi}}, \quad \phi = \eta_1 + \eta_2 > 4$$

this ensures convergence while maintaining search ability.

**variants:** different topologies for information sharing:
1. global best (gbest): all particles share one global best
2. local best (lbest): particles only know neighbors' bests
3. von neumann: grid topology
4-6: variations with different update rules

**parameters:**
- `generations`: number of pso generations (default: 100)
- `omega`: inertia weight $\omega$ (default: 0.7298)
- `eta1`: cognitive coefficient $\eta_1$ (default: 2.05)
- `eta2`: social coefficient $\eta_2$ (default: 2.05)
- `max_velocity`: velocity clamp $v_{\max}$ (default: 0.5)
- `variant`: algorithm variant 1-6 (default: 5)

**when to use:** parallel evaluation available, multimodal landscapes, fast convergence needed.

---

### nelder-mead (simplex method)

derivative-free simplex optimization. uses nlopt's implementation of the nelder-mead downhill simplex algorithm.

**algorithm:**

nelder-mead maintains a simplex of $n+1$ points in $n$-dimensional space. at each iteration, the algorithm tries to improve the worst point through geometric operations:

1. **reflection**: reflect worst point through centroid of remaining points
   $$\mathbf{x}_r = \mathbf{x}_c + \alpha(\mathbf{x}_c - \mathbf{x}_w)$$
   where $\mathbf{x}_c$ is centroid, $\mathbf{x}_w$ is worst point, $\alpha = 1$

2. **expansion**: if reflection is best so far, try expanding further
   $$\mathbf{x}_e = \mathbf{x}_c + \gamma(\mathbf{x}_r - \mathbf{x}_c)$$
   where $\gamma = 2$

3. **contraction**: if reflection is still worst, contract toward centroid
   $$\mathbf{x}_t = \mathbf{x}_c + \beta(\mathbf{x}_w - \mathbf{x}_c)$$
   where $\beta = 0.5$ (inside) or $\beta = -0.5$ (outside)

4. **shrink**: if contraction fails, shrink entire simplex toward best point
   $$\mathbf{x}_i \leftarrow \mathbf{x}_b + \sigma(\mathbf{x}_i - \mathbf{x}_b)$$
   where $\mathbf{x}_b$ is best point, $\sigma = 0.5$

**decision logic:**
```
if f(x_r) < f(x_best):
    try expansion
elif f(x_r) < f(x_second_worst):
    accept reflection
else:
    try contraction
    if contraction fails:
        shrink simplex
```

**convergence:** stops when:
- function value changes less than `ftol_rel`
- simplex size (in x) shrinks below `xtol_rel`
- maximum evaluations reached

**parameters:**
- `max_fevals`: maximum function evaluations (default: 1000)
- `xtol_rel`: relative tolerance on simplex size (default: 1e-8)
- `ftol_rel`: relative tolerance on function value (default: 1e-8)

**when to use:** local refinement, few hyperparameters, smooth landscape, polishing after global search.

## comparison

| algorithm | type | parallelizable | best for |
|-----------|------|----------------|----------|
| cma-es | population | yes | correlated continuous params |
| simulated annealing | single-point | no | rough landscapes |
| pso | swarm | yes | multimodal problems |
| nelder-mead | local search | no | local refinement |

## usage example

```cpp
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

// problem to optimize
wrappers::problems::SphereProblem problem{10};

// ea factory (de will have its hyperparameters tuned)
pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;

// hpo algorithm
pagmo_wrappers::PagmoCmaesHyperOptimizer hpo;

// budget for hpo
core::Budget budget;
budget.generations = 50;

// run optimization
auto result = hpo.optimize(de_factory, problem, budget, 42);

// result.best_parameters contains optimal hyperparameters
// result.best_objective is the best fitness achieved
```

## references

- hansen, n. (2006). the cma evolution strategy: a comparing review.
- kirkpatrick, s. et al. (1983). optimization by simulated annealing.
- kennedy, j. & eberhart, r. (1995). particle swarm optimization.
- nelder, j. & mead, r. (1965). a simplex method for function minimization.
