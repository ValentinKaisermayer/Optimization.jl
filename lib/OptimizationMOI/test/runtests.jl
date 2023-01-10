using OptimizationMOI, Optimization, Ipopt, NLopt, Zygote, ModelingToolkit
using AmplNLWriter, Ipopt_jll, HiGHS
using Test

function _test_sparse_derivatives_hs071(backend, optimizer)
    function objective(x, ::Any)
        return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    end
    function constraints(res, x, ::Any)
        res .= [
            x[1] * x[2] * x[3] * x[4],
            x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2,
        ]
    end
    prob = OptimizationProblem(OptimizationFunction(objective, backend; cons = constraints),
                               [1.0, 5.0, 5.0, 1.0];
                               sense = Optimization.MinSense,
                               lb = [1.0, 1.0, 1.0, 1.0],
                               ub = [5.0, 5.0, 5.0, 5.0],
                               lcons = [25.0, 40.0],
                               ucons = [Inf, 40.0])
    sol = solve(prob, optimizer)
    @test isapprox(sol.objective, 17.014017145179164; atol = 1e-6)
    x = [1.0, 4.7429996418092970, 3.8211499817883077, 1.3794082897556983]
    @test isapprox(sol.minimizer, x; atol = 1e-6)
    @test prod(sol.minimizer) >= 25.0 - 1e-6
    @test isapprox(sum(sol.minimizer .^ 2), 40.0; atol = 1e-6)
    return
end

@testset "NLP" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 100.0]
    l1 = rosenbrock(x0, _p)

    optprob = OptimizationFunction((x, p) -> -rosenbrock(x, p), Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MaxSense)

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.objective < l1

    # cache interface
    cache = init(prob, Ipopt.Optimizer())
    sol = solve!(cache)
    @test 10 * sol.minimum < l1

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoZygote())
    prob = OptimizationProblem(optprob, x0, _p; sense = Optimization.MinSense)

    opt = Ipopt.Optimizer()
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1
    sol = solve(prob, opt) #test reuse of optimizer
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                            "max_cpu_time" => 60.0))
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
                                                            "algorithm" => :LN_BOBYQA))
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
                                                            "algorithm" => :LD_LBFGS))
    @test 10 * sol.objective < l1

    opt = OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer,
                                                      "algorithm" => :LD_LBFGS)
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1
    sol = solve(prob, opt)
    @test 10 * sol.objective < l1

    cons_circ = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
    optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(true, true);
                                   cons = cons_circ)
    prob = OptimizationProblem(optprob, x0, _p, ucons = [Inf], lcons = [0.0])

    sol = solve(prob, Ipopt.Optimizer())
    @test 10 * sol.objective < l1

    sol = solve(prob,
                OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                            "max_cpu_time" => 60.0))
    @test 10 * sol.objective < l1
end

@testset "backends" begin
    backends = (Optimization.AutoModelingToolkit(false, false),
                Optimization.AutoModelingToolkit(true, false),
                Optimization.AutoModelingToolkit(false, true),
                Optimization.AutoModelingToolkit(true, true))
    for backend in backends
        @testset "$backend" begin
            _test_sparse_derivatives_hs071(backend, Ipopt.Optimizer())
            _test_sparse_derivatives_hs071(backend,
                                           AmplNLWriter.Optimizer(Ipopt_jll.amplexe))
        end
    end
end

@testset "cache" begin
    @variables x
    @parameters a = 1.0
    @named sys = OptimizationSystem((x - a)^2, [x], [a];)

    prob = OptimizationProblem(sys, [x => 0.0], []; grad = true, hess = true)
    cache = init(prob, Ipopt.Optimizer(); print_level = 0)
    sol = solve!(cache)
    @test sol.u ≈ [1.0] # ≈ [1]

    cache = OptimizationMOI.reinit!(cache; p = [2.0])
    sol = solve!(cache)
    @test sol.u ≈ [2.0]  # ≈ [2]

    prob = OptimizationProblem(sys, [x => 0.0], []; grad = false, hess = false)
    cache = init(prob, HiGHS.Optimizer())
    sol = solve!(cache)
    @test sol.u≈[1.0] rtol=1e-3 # ≈ [1]

    cache = OptimizationMOI.reinit!(cache; p = [2.0])
    sol = solve!(cache)
    @test sol.u≈[2.0] rtol=1e-3 # ≈ [2]
end

@testset "MOI" begin
    @parameters c = 0.0
    @variables x[1:2]=[0.0, 0.0] [bounds = (c, Inf)]
    @parameters a = 3.0
    @parameters b = 4.0
    @parameters d = 2.0
    @named sys = OptimizationSystem(a * x[1]^2 + b * x[2]^2 + d * x[1] * x[2] + 5 * x[1] +
                                    x[2], [x...], [a, b, c, d];
                                    constraints = [
                                        x[1] + 2 * x[2] ~ 1.0,
                                    ])
    prob = OptimizationProblem(sys, [x[1] => 2.0, x[2] => 0.0], []; grad = true,
                               hess = true)
    sol = solve(prob, HiGHS.Optimizer())
    sol.u
end

@testset "tutorial" begin
    rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    _p = [1.0, 1.0]

    cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])

    optprob = OptimizationFunction(rosenbrock, Optimization.AutoModelingToolkit(),
                                   cons = cons)
    prob = OptimizationProblem(optprob, x0, _p, lcons = [1.0, 0.5], ucons = [1.0, 0.5])
    sol = solve(prob, AmplNLWriter.Optimizer(Ipopt_jll.amplexe))
end
