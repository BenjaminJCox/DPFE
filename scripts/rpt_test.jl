using LinearAlgebra
using Zygote
using Distributions

function f(a, b)
    m = Normal(a,b)
    s = rand(m, 5)
    return mean(s)
end

g = gradient(f, 1, 1)

function f_reparam(a, b)
    m = Normal(0,1)
    s = a .+ sqrt(b) .* rand(m, 5)
    return sum(s)
end

g_reparam = gradient(f_reparam, 1, 1)
