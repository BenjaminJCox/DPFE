using LinearAlgebra
using Distributions

abstract type PFTransitionModel end
abstract type PFObservationModel end
abstract type PFResamplingScheme end
abstract type PFProposal end

#==

all functions take the struct as first input
define functions outside of structs and reuse, for example if using bootstrap proposal the proposal reuses the transition function
I think (read: hope) structs only point to functions...
struct _example_t_model <: PFTransitionModel
    d::Integer #size of state
    f::Function #takes state, yields transition without noise term
    n::Function #takes state, yields noise distribution
    ll::Function #takes previous state, current state, yields log likelihood of current state given previous state (i.e. log(p(x_t|x_t-1)))
end

struct _example_o_model <: PFObservationModel
    d::Integer #size of observation
    f::Function #takes state, yields observation without noise term
    n::Function #takes state, observation, yields noise distribution
    ll::Function #takes state, observation, yields log likelihood of current observation given current state (i.e. log(p(y_t|x_t)))
end

struct _example_r_scheme <: PFResamplingScheme
    criterion::Function #takes weights, tests criterion, returns true is resampling is to occur
    perform::Function #takes states, weights, performs resampling and returns like states. 
end

struct _example_proposal <: PFProposal
    prop::Function #takes state, observation, returns proposal for state. MUST BE PARAMETERISED TO BE DIFFERENTIABLE
    # prop must also implement prior (i.e. x ∼ p(x_0)), will be given prior = true flag when required
    ll::Function #takes sample, state, observation, returns log likelihood of sample
end
==#


Base.length(m::PFTransitionModel) = m.d
Base.length(m::PFObservationModel) = m.d

Distributions.logpdf(m::PFTransitionModel, previous_state, state) = m.ll(m, previous_state, state)
Distributions.pdf(m::PFTransitionModel, previous_state, state) = exp.(m.ll(m, previous_state, state))

Distributions.logpdf(m::PFObservationModel, state, observation) = m.ll(m, state, observation)
Distributions.pdf(m::PFObservationModel, state, observation) = exp.(m.ll(m, state, observation))

test(s::PFResamplingScheme, weights) = s.criterion(s, weights)
resample(s::PFResamplingScheme, states, weights) = s.perform(s, states, weights)

Base.rand(p::PFProposal, state, observation, prior::Bool) = p.prop(p, state, observation, prior)
Distributions.logpdf(p::PFProposal, sample, state, observation) = p.ll(p, sample, state, observation)
Distributions.pdf(p::PFProposal, sample, state, observation) = exp.(p.ll(p, sample, state, observation))

function particle_filter(; transition_model, observation_model, proposal_method, resampling_scheme, observations, n_particles)
    # Initialisation
    _state_dim = length(transition_model)
    _obs_dim = length(observation_model)
    T = size(observations, 2)
    N = n_particles

    # Remember, no mutation, only reassignment
    weights = zeros(n_particles, T + 1)
    states = zeros(_state_dim, n_particles, T + 1)
    lθ = 0.0

    for n = 1:N
        states[:, n, 1] = rand(proposal_method, nothing, nothing, prior = true)
        weights[n, 1] = inv(N)
    end

    for t = 1:T
        weights[:, t] = weights[:, t] ./ sum(weights[:, t])
        if test(resampling_scheme, weights[:, t])
            states[:, :, t] = resample(resampling_scheme, states[:, :, t], weights[:, t])
        end

        for n = 1:N
            states[:, n, t+1] = rand(proposal_method, states[:, n, t], observations[:, t])
            weights[n, t+1] = exp(
                logpdf(transition_model, states[:, n, t], states[:, n, t+1]) +
                logpdf(observation_model, states[:, n, t+1], observations[:, t]) -
                logpdf(proposal_method, states[:, n, t+1], states[:, n, t], observations[:, t]),
            )
        end
        lθ = lθ + log(inv(N) * sum(weights[:, t+1]))
    end
    return (states, weights, lθ)
end
