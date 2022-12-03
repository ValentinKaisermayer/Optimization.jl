mutable struct MOIOptimizationNLPEvaluator{T, F <: OptimizationFunction, RC, LB, UB,
                                           I,
                                           JT <: DenseOrSparse{T}, HT <: DenseOrSparse{T},
                                           CHT <: DenseOrSparse{T}, S} <:
               MOI.AbstractNLPEvaluator
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    int::I
    lcons::Vector{T}
    ucons::Vector{T}
    sense::S
    J::JT
    H::HT
    cons_H::Vector{CHT}
end

function Base.getproperty(evaluator::MOIOptimizationNLPEvaluator, x::Symbol)
    if x in fieldnames(Optimization.ReInitCache)
        return getfield(evaluator.reinit_cache, x)
    end
    return getfield(evaluator, x)
end

struct MOIOptimizationNLPCache{E <: MOIOptimizationNLPEvaluator, O} <:
       SciMLBase.AbstractOptimizationCache
    evaluator::E
    opt::O
    solver_args::NamedTuple
end

function Base.getproperty(cache::MOIOptimizationNLPCache{E}, name::Symbol) where {E}
    if name in fieldnames(E)
        return getfield(cache.evaluator, name)
    elseif name in fieldnames(Optimization.ReInitCache)
        return getfield(cache.evaluator.reinit_cache, name)
    end
    return getfield(cache, name)
end
function Base.setproperty!(cache::MOIOptimizationNLPCache{E}, name::Symbol, x) where {E}
    if name in fieldnames(E)
        return setfield!(cache.evaluator, name, x)
    elseif name in fieldnames(Optimization.ReInitCache)
        return setfield!(cache.evaluator.reinit_cache, name, x)
    end
    return setfield!(cache, name, x)
end

function SciMLBase.get_p(sol::SciMLBase.OptimizationSolution{T, N, uType, C}) where {T, N,
                                                                                     uType,
                                                                                     C <:
                                                                                     MOIOptimizationNLPCache
                                                                                     }
    sol.cache.evaluator.p
end
function SciMLBase.get_observed(sol::SciMLBase.OptimizationSolution{T, N, uType, C}) where {
                                                                                            T,
                                                                                            N,
                                                                                            uType,
                                                                                            C <:
                                                                                            MOIOptimizationNLPCache
                                                                                            }
    sol.cache.evaluator.f.observed
end
function SciMLBase.get_syms(sol::SciMLBase.OptimizationSolution{T, N, uType, C}) where {T,
                                                                                        N,
                                                                                        uType,
                                                                                        C <:
                                                                                        MOIOptimizationNLPCache
                                                                                        }
    sol.cache.evaluator.f.syms
end
function SciMLBase.get_paramsyms(sol::SciMLBase.OptimizationSolution{T, N, uType, C}) where {
                                                                                             T,
                                                                                             N,
                                                                                             uType,
                                                                                             C <:
                                                                                             MOIOptimizationNLPCache
                                                                                             }
    sol.cache.evaluator.f.paramsyms
end

function MOIOptimizationNLPCache(prob::OptimizationProblem, opt; kwargs...)
    reinit_cache = Optimization.ReInitCache(prob.u0, prob.p) # everything that can be changed via `reinit`

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = Optimization.instantiate_function(prob.f, reinit_cache, prob.f.adtype, num_cons)
    T = eltype(prob.u0)
    n = length(prob.u0)

    J = if isnothing(f.cons_jac_prototype)
        zeros(T, num_cons, n)
    else
        convert.(T, f.cons_jac_prototype)
    end
    H = if isnothing(f.hess_prototype)
        zeros(T, n, n)
    else
        convert.(T, f.hess_prototype)
    end
    cons_H = if isnothing(f.cons_hess_prototype)
        Matrix{T}[zeros(T, n, n) for i in 1:num_cons]
    else
        [convert.(T, f.cons_hess_prototype[i]) for i in 1:num_cons]
    end
    lcons = prob.lcons === nothing ? fill(-Inf, num_cons) : prob.lcons
    ucons = prob.ucons === nothing ? fill(Inf, num_cons) : prob.ucons

    evaluator = MOIOptimizationNLPEvaluator(f,
                                            reinit_cache,
                                            prob.lb,
                                            prob.ub,
                                            prob.int,
                                            lcons,
                                            ucons,
                                            prob.sense,
                                            J,
                                            H,
                                            cons_H)
    return MOIOptimizationNLPCache(evaluator, opt, NamedTuple(kwargs))
end

SciMLBase.has_reinit(cache::MOIOptimizationNLPCache) = true
function SciMLBase.reinit!(cache::MOIOptimizationNLPCache; p = nothing, u0 = nothing)
    if !isnothing(p) && eltype(p) <: Pair || (!isnothing(u0) && eltype(u0) <: Pair)
        defs = Dict{Any, Any}()
        if hasproperty(cache.f, :sys)
            if hasfield(typeof(cache.f.sys), :ps)
                defs = mergedefaults(defs, cache.p, parameters(cache.f.sys))
            end
            if hasfield(typeof(cache.f.sys), :states)
                defs = SciMLBase.mergedefaults(defs, cache.u0, states(cache.f.sys))
            end
        end
    else
        defs = nothing
    end

    if isnothing(p)
        p = cache.p
    else
        if eltype(p) <: Pair
            if hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :ps)
                p = varmap_to_vars(p, parameters(cache.f.sys);
                                   defaults = defs)
                defs = mergedefaults(defs, p, parameters(cache.f.sys))
            else
                throw(ArgumentError("This problem does not support symbolic parameter maps with `reinit!`, i.e. it does not have a symbolic origin. Please use `reinit!` with the `p` keyword argument as a vector of values, paying attention to parameter order."))
            end
        end
    end

    length(cache.p) == length(p) ||
        error("Something went wrong! The new parameter vector does not match the old one.")
    cache.p = p

    if isnothing(u0)
        u0 = cache.u0
    else
        if eltype(u0) <: Pair
            if hasproperty(cache.f, :sys) && hasfield(typeof(cache.f.sys), :states)
                u0 = varmap_to_vars(u0, states(cache.f.sys);
                                    defaults = defs)
                defs = mergedefaults(defs, u0, states(cache.f.sys))
            else
                throw(ArgumentError("This problem does not support symbolic state maps with `reinit!`, i.e. it does not have a symbolic origin. Please use `reinit!` with the `u0` keyword argument as a vector of values, paying attention to state order."))
            end
        end
    end

    length(cache.u0) == length(u0) ||
        error("Something went wrong! The new state vector does not match the old one.")
    cache.u0 = u0

    return cache
end

function MOI.features_available(evaluator::MOIOptimizationNLPEvaluator)
    features = [:Grad, :Hess, :Jac]
    # Assume that if there are constraints and expr then cons_expr exists
    if evaluator.f.expr !== nothing
        push!(features, :ExprGraph)
    end
    return features
end

function MOI.initialize(evaluator::MOIOptimizationNLPEvaluator,
                        requested_features::Vector{Symbol})
    available_features = MOI.features_available(evaluator)
    for feat in requested_features
        if !(feat in available_features)
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
    return
end

function MOI.eval_objective(evaluator::MOIOptimizationNLPEvaluator, x)
    return evaluator.f(x, evaluator.p)
end

function MOI.eval_constraint(evaluator::MOIOptimizationNLPEvaluator, g, x)
    evaluator.f.cons(g, x)
    return
end

function MOI.eval_objective_gradient(evaluator::MOIOptimizationNLPEvaluator, G, x)
    evaluator.f.grad(G, x)
    return
end

# This structure assumes the calculation of moiproblem.J is dense.
function MOI.jacobian_structure(evaluator::MOIOptimizationNLPEvaluator)
    if evaluator.J isa SparseMatrixCSC
        rows, cols, _ = findnz(evaluator.J)
        inds = Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols)]
    else
        rows, cols = size(evaluator.J)
        inds = Tuple{Int, Int}[(i, j) for j in 1:cols for i in 1:rows]
    end
    return inds
end

function MOI.eval_constraint_jacobian(evaluator::MOIOptimizationNLPEvaluator, j, x)
    if isempty(j)
        return
    elseif evaluator.f.cons_j === nothing
        error("Use OptimizationFunction to pass the derivatives or " *
              "automatically generate them with one of the autodiff backends")
    end
    evaluator.f.cons_j(evaluator.J, x)
    if evaluator.J isa SparseMatrixCSC
        nnz = nonzeros(evaluator.J)
        @assert length(j) == length(nnz)
        for (i, Ji) in zip(eachindex(j), nnz)
            j[i] = Ji
        end
    else
        for i in eachindex(j)
            j[i] = evaluator.J[i]
        end
    end
    return
end

function MOI.hessian_lagrangian_structure(evaluator::MOIOptimizationNLPEvaluator)
    sparse_obj = evaluator.H isa SparseMatrixCSC
    sparse_constraints = all(H -> H isa SparseMatrixCSC, evaluator.cons_H)
    if !sparse_constraints && any(H -> H isa SparseMatrixCSC, evaluator.cons_H)
        # Some constraint hessians are dense and some are sparse! :(
        error("Mix of sparse and dense constraint hessians are not supported")
    end
    N = length(evaluator.u0)
    inds = if sparse_obj
        rows, cols, _ = findnz(evaluator.H)
        Tuple{Int, Int}[(i, j) for (i, j) in zip(rows, cols) if i <= j]
    else
        Tuple{Int, Int}[(row, col) for col in 1:N for row in 1:col]
    end
    if sparse_constraints
        for Hi in evaluator.cons_H
            r, c, _ = findnz(Hi)
            for (i, j) in zip(r, c)
                if i <= j
                    push!(inds, (i, j))
                end
            end
        end
    elseif !sparse_obj
        # Performance optimization. If both are dense, no need to repeat
    else
        for col in 1:N, row in 1:col
            push!(inds, (row, col))
        end
    end
    return inds
end

function MOI.eval_hessian_lagrangian(evaluator::MOIOptimizationNLPEvaluator{T},
                                     h,
                                     x,
                                     σ,
                                     μ) where {T}
    fill!(h, zero(T))
    k = 0
    evaluator.f.hess(evaluator.H, x)
    sparse_objective = evaluator.H isa SparseMatrixCSC
    if sparse_objective
        rows, cols, _ = findnz(evaluator.H)
        for (i, j) in zip(rows, cols)
            if i <= j
                k += 1
                h[k] = σ * evaluator.H[i, j]
            end
        end
    else
        for i in 1:size(evaluator.H, 1), j in 1:i
            k += 1
            h[k] = σ * evaluator.H[i, j]
        end
    end
    # A count of the number of non-zeros in the objective Hessian is needed if
    # the constraints are dense.
    nnz_objective = k
    if !isempty(μ) && !all(iszero, μ)
        evaluator.f.cons_h(evaluator.cons_H, x)
        for (μi, Hi) in zip(μ, evaluator.cons_H)
            if Hi isa SparseMatrixCSC
                rows, cols, _ = findnz(Hi)
                for (i, j) in zip(rows, cols)
                    if i <= j
                        k += 1
                        h[k] += μi * Hi[i, j]
                    end
                end
            else
                # The constraints are dense. We only store one copy of the
                # Hessian, so reset `k` to where it starts. That will be
                # `nnz_objective` if the objective is sprase, and `0` otherwise.
                k = sparse_objective ? nnz_objective : 0
                for i in 1:size(Hi, 1), j in 1:i
                    k += 1
                    h[k] += μi * Hi[i, j]
                end
            end
        end
    end
    return
end

function MOI.objective_expr(evaluator::MOIOptimizationNLPEvaluator)
    expr = deepcopy(evaluator.f.expr)
    repl_getindex!(expr)
    _replace_parameter_indices!(expr, evaluator.p)
    _replace_variable_indices!(expr)
    return expr
end

function MOI.constraint_expr(evaluator::MOIOptimizationNLPEvaluator, i)
    # expr has the form f(x,p) == 0 or f(x,p) <= 0
    cons_expr = deepcopy(evaluator.f.cons_expr[i].args[2])
    repl_getindex!(cons_expr)
    _replace_parameter_indices!(cons_expr, evaluator.p)
    _replace_variable_indices!(cons_expr)
    lb, ub = Float64(evaluator.lcons[i]), Float64(evaluator.ucons[i])
    return :($lb <= $cons_expr <= $ub)
end

function _add_moi_variables!(opt_setup, evaluator::MOIOptimizationNLPEvaluator)
    num_variables = length(evaluator.u0)
    θ = MOI.add_variables(opt_setup, num_variables)
    if evaluator.lb !== nothing
        @assert eachindex(evaluator.lb) == Base.OneTo(num_variables)
    end
    if evaluator.ub !== nothing
        @assert eachindex(evaluator.ub) == Base.OneTo(num_variables)
    end

    for i in 1:num_variables
        if evaluator.lb !== nothing && evaluator.lb[i] > -Inf
            MOI.add_constraint(opt_setup, θ[i], MOI.GreaterThan(evaluator.lb[i]))
        end
        if evaluator.ub !== nothing && evaluator.ub[i] < Inf
            MOI.add_constraint(opt_setup, θ[i], MOI.LessThan(evaluator.ub[i]))
        end
        if evaluator.int !== nothing && evaluator.int[i]
            if evaluator.lb !== nothing && evaluator.lb[i] == 0 &&
               evaluator.ub !== nothing &&
               evaluator.ub[i] == 1
                MOI.add_constraint(opt_setup, θ[i], MOI.ZeroOne())
            else
                MOI.add_constraint(opt_setup, θ[i], MOI.Integer())
            end
        end
    end

    if MOI.supports(opt_setup, MOI.VariablePrimalStart(), MOI.VariableIndex)
        @assert eachindex(evaluator.u0) == Base.OneTo(num_variables)
        for i in 1:num_variables
            MOI.set(opt_setup, MOI.VariablePrimalStart(), θ[i], evaluator.u0[i])
        end
    end
    return θ
end

function SciMLBase.__solve(cache::MOIOptimizationNLPCache)
    maxiters = Optimization._check_and_convert_maxiters(cache.solver_args.maxiters)
    maxtime = Optimization._check_and_convert_maxtime(cache.solver_args.maxtime)
    opt_setup = __map_optimizer_args(cache,
                                     cache.opt;
                                     abstol = cache.solver_args.abstol,
                                     reltol = cache.solver_args.reltol,
                                     maxiters = maxiters,
                                     maxtime = maxtime,
                                     cache.solver_args...)

    θ = _add_moi_variables!(opt_setup, cache.evaluator)
    MOI.set(opt_setup,
            MOI.ObjectiveSense(),
            cache.evaluator.sense === Optimization.MaxSense ? MOI.MAX_SENSE : MOI.MIN_SENSE)
    if cache.evaluator.lcons === nothing
        @assert cache.evaluator.ucons === nothing
        con_bounds = MOI.NLPBoundsPair[]
    else
        @assert cache.evaluator.ucons !== nothing
        con_bounds = MOI.NLPBoundsPair.(Float64.(cache.evaluator.lcons),
                                        Float64.(cache.evaluator.ucons))
    end
    MOI.set(opt_setup,
            MOI.NLPBlock(),
            MOI.NLPBlockData(con_bounds, cache.evaluator, true))
    MOI.optimize!(opt_setup)
    if MOI.get(opt_setup, MOI.ResultCount()) >= 1
        minimizer = MOI.get(opt_setup, MOI.VariablePrimal(), θ)
        minimum = MOI.get(opt_setup, MOI.ObjectiveValue())
        opt_ret = __moi_status_to_ReturnCode(MOI.get(opt_setup, MOI.TerminationStatus()))
    else
        minimizer = fill(NaN, length(θ))
        minimum = NaN
        opt_ret = SciMLBase.ReturnCode.Default
    end
    return SciMLBase.build_solution(cache,
                                    cache.opt,
                                    minimizer,
                                    minimum;
                                    original = opt_setup,
                                    retcode = opt_ret)
end
