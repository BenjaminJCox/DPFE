using DrWatson
using Random
using LinearAlgebra
using CairoMakie

include("../src/dpf_stochad.jl")

#==

==#

Random.seed!(12)

prior(θ) = MvNormal(zeros(2), 0.1 .* collect(I(2)))

f_dyn(x, θ) = θ[1] .* x .+ (θ[2] .* x[1].*x[2])

g_obs(x, θ) = x

state_model(x, θ) = MvNormal(f_dyn(x, θ), 0.1 .* collect(I(length(x))))
obs_model(x, θ) = MvNormal(g_obs(x, θ), 0.1 .* collect(I(length(x))))

Zygote.gradient(θ -> mean(rand(state_model([0.1, 0.5], θ))), [0.5, 0.1])

prop_model(x, y, θ) = MvNormal(f_dyn(x, θ), collect(I(length(x)))) # bootstrap pf

T = 50

θ = [0.9, 0.1]

test_SSM = StateSpaceModel(T, prior, state_model, obs_model, prop_model)

test_xs, test_ys = generate_fake_trajectory(test_SSM, θ)

p_x1s = getindex.(test_xs, 1)

pfig = Figure()
pax = Axis(pfig[1, 1])

scatter!(pax, collect(1:T), p_x1s)

pfig

K = 200

test_pf = ParticleFilter(K, test_SSM, test_ys, strat_sample)

tpfo = test_pf(θ, _store = true, _bpf = false)

msx = [zeros(2) for i in 1:T]

for i in 1:K
    local xs = [tpfo[1][t][i] for t in 1:T]
    msx .+= xs
end

msx ./= K


# _vov = tpfo[1][2:end]
# _vov = getindex.(_vov, 1)

pf_x1s = getindex.(msx, 1)

scatter!(pax, collect(1:T), pf_x1s)

pfig


ll_grad(θ, test_pf, _bpf = false)

θ_in = rand(2)
tδ = 5e-4

ll_grad(θ_in, test_pf, _bpf = false)
for i = 1:120
    global θ_in = θ_in + tδ * ll_grad(θ_in, test_pf, _bpf = false)
    println(θ_in)
end

tpfin = test_pf(θ_in, _store = true, _bpf = false)

msxo = [zeros(2) for i in 1:T]

for i in 1:K
    local xs = [tpfin[1][t][i] for t in 1:T]
    msxo .+= xs
end

msxo ./= K


pfo_x1s = getindex.(msxo, 1)

scatter!(pax, collect(1:T), pfo_x1s)

pfig

_trmse = sqrt(mean((p_x1s - pf_x1s).^2))
_ermse = sqrt(mean((p_x1s - pfo_x1s).^2))

_plot_θ = collect(-1:0.05:1)
_plot_grads = [ll_grad([th, 0.1], test_pf, _bpf = false) for th in _plot_θ]

_pgs = getindex.(_plot_grads, 1)

lines(_plot_θ, _pgs)
