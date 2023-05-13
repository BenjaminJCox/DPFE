using LinearAlgebra
using Zygote
using ForwardDiff
using Distributions
using ProtoStructs
using DistributionsAD
using BenchmarkTools
using FiniteDiff
using ReverseDiff
using StaticArrays
using Enzyme

function f(a, b)
    m = Normal(a,b)
    s = rand(m, 5)
    return mean(s)
end

g = Zygote.gradient(f, 1, 1)

function f_reparam(a, b)
    m = Normal(0,1)
    s = a .+ sqrt(b) .* rand(m, 5)
    return sum(s)
end

g_reparam = Zygote.gradient(f_reparam, 1, 1)

@proto struct function_containing_struct
    f::Function
    x::Vector{Float64}
    y::Integer
end

k = function_containing_struct(x -> sum(0.5 .* x .^ 2), [1., 2., 3., 4.], 432)

g_fcs = Zygote.gradient(k.f, k.x)

@proto struct con_struct
    x::Vector{Float64}
    y::Integer
end

cs = con_struct([1., 2., 3., 4.], 4)
visf(x,y) = mean(y .* x .^ 2)

function tf(cs, f)
    fcs = function_containing_struct(f, cs.x, cs.y)
    return fcs.f(fcs.x,fcs.y)
end

tf(cs, visf)

tf_g = Zygote.gradient(tf, cs, visf)

kf(x,y) = [x * y, y^2]

j_kf = Zygote.jacobian(kf, 1, 6)

function talloc(x, y)
    N = length(x)
    rv = Zygote.Buffer(x, N)
    for n = 1:N-1
        rv[n] = x[n] + x[n+1]
    end
    rv[N] = y
    return mean(rv)
end

tf_g = Zygote.withgradient(talloc, [1., 2., 3.], 1.0)

tf_comp(x) = sum(x[1:50].^2 ./ sqrt.(x[51:100]))

txs, tys = randn(50), rand(50)

ForwardDiff.gradient(tf_comp, vcat(txs, tys))
Zygote.gradient(tf_comp, vcat(txs, tys))
Enzyme.gradient(Reverse, tf_comp, vcat(txs, tys))
rd1 = ReverseDiff.gradient(tf_comp, vcat(txs, tys))

rdtape = ReverseDiff.compile(ReverseDiff.GradientTape(tf_comp, vcat(txs, tys)))
@btime ReverseDiff.gradient!(rd1, rdtape, vcat(txs, tys))
@btime ForwardDiff.gradient(tf_comp, vcat(txs, tys))
@btime Zygote.gradient(tf_comp, vcat(txs, tys))
@btime FiniteDiff.finite_difference_gradient(tf_comp, vcat(txs, tys))
@btime Enzyme.gradient(Reverse, tf_comp, vcat(txs, tys))
@btime Enzyme.gradient(Forward, tf_comp, vcat(txs, tys))

npdf(mu, sig, x) = inv(sig * sqrt(2π)) * exp(-0.5 * ((x - mu)/sig)^2)

function testcles(x)
    N = length(x)
    X = collect(1:N) .* x ./ 4
    X2 = abs.(X .+ 0.1)
    X = X .+ X2
    # X = map((x, y) -> npdf(x, 1, y), X, X2)
    X = map((x, y) -> pdf(MvNormal([x, x], [1. 0.; 0. 1.]), [y, y]), X, X2)
    return sum(X)
end

testcles(collect(-2:3))

# enzyme fails to compile here

fwd = ForwardDiff.gradient(testcles, collect(-2:200))
zgd = Zygote.gradient(testcles, collect(-2:200))[1]
fid = FiniteDiff.finite_difference_gradient(testcles, 1. .* collect(-2:200))
rvd = ReverseDiff.gradient(testcles, 1. .* collect(-2:200))
rvd_t = ReverseDiff.compile(ReverseDiff.GradientTape(testcles, 1. .* collect(-2:200)))

@btime ReverseDiff.gradient!(rvd, rvd_t, collect(-2:200)) seconds = 1
@btime ForwardDiff.gradient(testcles, collect(-2:200)) seconds = 1
@btime Zygote.gradient(testcles, collect(-2:200)) seconds = 1
@btime FiniteDiff.finite_difference_gradient(testcles, 1. .* collect(-2:200)) seconds = 1

t1s = [ones(5) for i in 1:2]
t2s = [rand(5) for i in 1:2]
tv = map((i,j) -> i+j, t1s, t2s)
