using DrWatson
using Random
using LinearAlgebra
using ForwardDiff
using Zygote
using StatsBase

include("../src/nn_helpers.jl")

Random.seed!(12)

m1 = collect(rand(10, 128)')
m2 = collect(rand(128, 256)')
m3 = collect(rand(256, 100)')

b1 = rand(128)
b2 = rand(256)
b3 = zeros(100)

afs = (identity, identity, identity)

test_x = rand(10)

test_fx = afs[1](m1 * test_x + b1)
test_fx = afs[2](m2 * test_fx + b2)
test_fx = afs[3](m3 * test_fx + b3)

test_packed = pack_known([m1, m2, m3], [b1, b2, b3])

test_ep_x = evaluate_pack(test_x, test_packed..., afs)

_empty = generate_empty_pack(test_packed[2], test_packed[3])
_empty += rand(length(_empty))

sum(abs.(test_ep_x - test_fx))

loss(x, y) = sqrt(mean((x .- y).^2))

gr_fw = ForwardDiff.gradient(θ -> loss(evaluate_pack(test_x, θ, test_packed[2], test_packed[3], afs), test_fx), test_packed[1] .+ 0.01)
gr_zy = Zygote.gradient(θ -> loss(evaluate_pack(test_x, θ, test_packed[2], test_packed[3], afs), test_fx), _empty)[1]

# opt_me = copy(_empty)
#
# lr = 5e-4
# steps = 1500
# for step in steps
#     opt_me = opt_me - lr * Zygote.gradient(θ -> loss(evaluate_pack(test_x, θ, test_packed[2], test_packed[3], afs), test_fx), opt_me)[1]
# end
