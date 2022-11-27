struct AutoModelingToolkit <: AbstractADType
    obj_sparse::Bool
    cons_sparse::Bool
end

AutoModelingToolkit() = AutoModelingToolkit(false, false)

function instantiate_function(f, x, adtype::AutoModelingToolkit, p,
                              num_cons = 0)
    p = isnothing(p) ? SciMLBase.NullParameters() : p
    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, x, p))

    hess_prototype, cons_jac_prototype, cons_hess_prototype = nothing, nothing, nothing

    if f.grad === nothing
        grad_oop, grad_iip = ModelingToolkit.generate_gradient(sys, expression = Val{false})
        grad(J, u) = (grad_iip(J, u, p); J)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess_oop, hess_iip = ModelingToolkit.generate_hessian(sys, expression = Val{false},
                                                              sparse = adtype.obj_sparse)
        hess(H, u) = (hess_iip(H, u, p); H)
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = adtype.obj_sparse ? hess_prototype : ArrayInterfaceCore.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    expr = symbolify(ModelingToolkit.Symbolics.toexpr(ModelingToolkit.equations(sys)))
    pairs_arr = if p isa SciMLBase.NullParameters
        [Symbol(_s) => Expr(:ref, :x, i)
         for (i, _s) in enumerate(ModelingToolkit.states(sys))]
    else
        vcat([Symbol(_s) => Expr(:ref, :x, i)
              for (i, _s) in enumerate(ModelingToolkit.states(sys))],
             [Symbol(_p) => Expr(:ref, :p, i)
              for (i, _p) in enumerate(ModelingToolkit.parameters(sys))])
    end
    rep_pars_vals!(expr, pairs_arr)

    if f.cons === nothing
        cons = nothing
        cons_exprs = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x, p) -> (_res = zeros(eltype(x), num_cons); f.cons(_res, x, p); _res)

        cons_sys = ModelingToolkit.modelingtoolkitize(NonlinearProblem(cons_oop, x, p)) # 0 == f(x)
        cons_eqs_lhss = ModelingToolkit.lhss(ModelingToolkit.Symbolics.canonical_form.(ModelingToolkit.equations(cons_sys))) # -f(x) == 0
        cons_exprs = map(cons_eqs_lhss) do lhs
            e = symbolify(ModelingToolkit.Symbolics.toexpr(-lhs)) # 0 == f(x)
            rep_pars_vals!(e, pairs_arr)
            return Expr(:call, :(==), e, :0)
        end
    end

    if f.cons !== nothing && f.cons_j === nothing
        jac_oop, jac_iip = ModelingToolkit.generate_jacobian(cons_sys,
                                                             expression = Val{false},
                                                             sparse = adtype.cons_sparse)
        cons_j = function (J, θ)
            jac_iip(J, θ, p)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if f.cons !== nothing && f.cons_h === nothing
        cons_hess_oop, cons_hess_iip = ModelingToolkit.generate_hessian(cons_sys,
                                                                        expression = Val{
                                                                                         false
                                                                                         },
                                                                        sparse = adtype.cons_sparse)
        cons_h = function (res, θ)
            cons_hess_iip(res, θ, p)
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if adtype.obj_sparse
        _hess_prototype = ModelingToolkit.hessian_sparsity(sys)
        hess_prototype = convert.(eltype(x), _hess_prototype)
    end

    if adtype.cons_sparse
        _cons_jac_prototype = ModelingToolkit.jacobian_sparsity(cons_sys)
        cons_jac_prototype = convert.(eltype(x), _cons_jac_prototype)
        _cons_hess_prototype = ModelingToolkit.hessian_sparsity(cons_sys)
        cons_hess_prototype = [convert.(eltype(x), _cons_hess_prototype[i])
                               for i in 1:num_cons]
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = hess_prototype,
                                      cons_jac_prototype = cons_jac_prototype,
                                      cons_hess_prototype = cons_hess_prototype,
                                      expr = expr, cons_expr = cons_exprs)
end

function instantiate_function(f, cache::ReInitCache,
                              adtype::AutoModelingToolkit, num_cons = 0)
    p = isnothing(cache.p) ? SciMLBase.NullParameters() : cache.p
    sys = ModelingToolkit.modelingtoolkitize(OptimizationProblem(f, cache.u0, cache.p))

    hess_prototype, cons_jac_prototype, cons_hess_prototype = nothing, nothing, nothing

    if f.grad === nothing
        grad_oop, grad_iip = ModelingToolkit.generate_gradient(sys, expression = Val{false})
        grad(J, u) = (grad_iip(J, u, cache.p); J)
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        hess_oop, hess_iip = ModelingToolkit.generate_hessian(sys, expression = Val{false},
                                                              sparse = adtype.obj_sparse)
        hess(H, u) = (hess_iip(H, u, cache.p); H)
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            res = adtype.obj_sparse ? hess_prototype : ArrayInterfaceCore.zeromatrix(θ)
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    expr = symbolify(ModelingToolkit.Symbolics.toexpr(ModelingToolkit.equations(sys)))
    pairs_arr = if cache.p isa SciMLBase.NullParameters
        [Symbol(_s) => Expr(:ref, :x, i)
         for (i, _s) in enumerate(ModelingToolkit.states(sys))]
    else
        vcat([Symbol(_s) => Expr(:ref, :x, i)
              for (i, _s) in enumerate(ModelingToolkit.states(sys))],
             [Symbol(_p) => Expr(:ref, :p, i)
              for (i, _p) in enumerate(ModelingToolkit.parameters(sys))])
    end
    rep_pars_vals!(expr, pairs_arr)

    if f.cons === nothing
        cons = nothing
        cons_exprs = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, cache.p)
        cons_oop = (x, p) -> (_res = zeros(eltype(x), num_cons); f.cons(_res, x, p); _res)

        cons_sys = ModelingToolkit.modelingtoolkitize(NonlinearProblem(cons_oop, cache.u0,
                                                                       cache.p)) # 0 = f(x)
        cons_eqs_lhss = ModelingToolkit.lhss(ModelingToolkit.Symbolics.canonical_form.(ModelingToolkit.equations(cons_sys))) # -f(x) == 0
        cons_exprs = map(cons_eqs_lhss) do lhs
            e = symbolify(ModelingToolkit.Symbolics.toexpr(-lhs)) # 0 == f(x)
            rep_pars_vals!(e, pairs_arr)
            return Expr(:call, :(==), e, :0)
        end
    end

    if f.cons !== nothing && f.cons_j === nothing
        jac_oop, jac_iip = ModelingToolkit.generate_jacobian(cons_sys,
                                                             expression = Val{false},
                                                             sparse = adtype.cons_sparse)
        cons_j = function (J, θ)
            jac_iip(J, θ, cache.p)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    if f.cons !== nothing && f.cons_h === nothing
        cons_hess_oop, cons_hess_iip = ModelingToolkit.generate_hessian(cons_sys,
                                                                        expression = Val{
                                                                                         false
                                                                                         },
                                                                        sparse = adtype.cons_sparse)
        cons_h = function (res, θ)
            cons_hess_iip(res, θ, cache.p)
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end

    if adtype.obj_sparse
        _hess_prototype = ModelingToolkit.hessian_sparsity(sys)
        hess_prototype = convert.(eltype(cache.u0), _hess_prototype)
    end

    if adtype.cons_sparse
        _cons_jac_prototype = ModelingToolkit.jacobian_sparsity(cons_sys)
        cons_jac_prototype = convert.(eltype(cache.u0), _cons_jac_prototype)
        _cons_hess_prototype = ModelingToolkit.hessian_sparsity(cons_sys)
        cons_hess_prototype = [convert.(eltype(cache.u0), _cons_hess_prototype[i])
                               for i in 1:num_cons]
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
                                      cons = cons, cons_j = cons_j, cons_h = cons_h,
                                      hess_prototype = hess_prototype,
                                      cons_jac_prototype = cons_jac_prototype,
                                      cons_hess_prototype = cons_hess_prototype,
                                      expr = expr, cons_expr = cons_exprs)
end
