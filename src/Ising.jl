module Ising

    using JLD, HDF5
    using MonteCarloObservable
    include("json_parameters.jl")

    type parameters
        L::Int
        beta::Float64
        addition_prob::Float64
        seed::Int
        r::MersenneTwister
        algorithm::String

        parameters() = new()
    end


    @inbounds function wolff_sweep!(p::parameters, spins::Array{Float64, 1})
        spins_flipped = 0
        L = p.L

        to_flip_r = Int64[]
        to_flip_c = Int64[]

        flip_attempted = fill(0, p.L^2)
        rows = shuffle(1:p.L)
        cols = shuffle(0:p.L - 1)

        for r in rows
            for c in cols
                initial_idx = r + c * p.L
                if flip_attempted[initial_idx] == 1 continue end

                initial_spin = spins[initial_idx]

                push!(to_flip_r, r)
                push!(to_flip_c, c)

                while length(to_flip_r) != 0
                    r, c = to_flip_r[1], to_flip_c[1]
                    deleteat!(to_flip_r, 1)
                    deleteat!(to_flip_c, 1)

                    if spins[r + c * p.L] == initial_spin
                        spins[r + c * p.L] *= -1

                        for s in [(mod1(r + 1, L) + c * L, (mod1(r + 1, L), c))
                                  , (mod1(r - 1 + L, L) + c * L, (mod1(r - 1 + L, L), c))
                                  , (r + mod(c + 1, L) * L, (r, mod(c + 1, L)))
                                  , (r + mod(c - 1 + L, L) * L, (r, mod(c - 1 + L, L)))]

                            flip_attempted[s[1]] = 1

                            if spins[s[1]] == initial_spin
                                if rand() < p.addition_prob
                                    push!(to_flip_r, s[2][1])
                                    push!(to_flip_c, s[2][2])
                                end
                            end
                        end
                    end
                end
            end
        end
    end


    @inbounds function ising_sweep!(p::parameters, spins::Array{Float64, 1})
        spins_flipped = 0
        L = p.L

        to_flip_r = Int64[]
        to_flip_c = Int64[]

        rows = shuffle(1:p.L)

        for r in rows
            cols = shuffle(0:p.L - 1)
            for c in cols
                initial_idx = r + c * p.L

                dE = spins[initial_idx] * (spins[mod1(r + 1, L) + c * L] + spins[mod1(r - 1 + L, L) + c * L] + spins[r + mod(c + 1, L) * L] + spins[r + mod(c - 1 + L, L) * L])
                if rand() < exp(-p.beta * dE)
                    spins[initial_idx] *= -1
                end
            end
        end
    end


    @inbounds function energy(p::parameters, spins::Array{Float64, 1})
        energy = 0.
        L = p.L
        for r in 1:p.L
            for c in 0:p.L - 1
                initial_idx = r + c * p.L

                for s in [(mod1(r + 1, L) + c * L, (mod1(r + 1, L), c))
                          , (mod1(r - 1 + L, L) + c * L, (mod1(r - 1 + L, L), c))
                          , (r + mod(c + 1, L) * L, (r, mod(c + 1, L)))
                          , (r + mod(c - 1 + L, L) * L, (r, mod(c - 1 + L, L)))]
                    energy -= 0.5 * spins[s[1]]
                end
            end
        end
        return energy
    end

    function run()
        prefix = convert(String, ARGS[1])
        idx = parse(Int, ARGS[2])

        input_file = prefix * ".task" * string(idx) * ".in.json"
        output_file = prefix * ".task" * string(idx) * ".out.h5"
        params = load_parameters(input_file)

        p = parameters()
        p.algorithm = params["ALGORITHM"]
        p.seed = params["SEED"]
        p.r = srand(p.seed)
        p.L = params["L"]
        p.beta = 1. / params["T"]
        measurements = params["MEASUREMENTS"]
        thermalizations = params["THERMALIZATIONS"]
        p.addition_prob = 1.0 - exp(-2. * p.beta)

        spins = rand([-1., 1.], p.L^2)

        m_series = monte_carlo_observable{Float64}("Magnetization", [1])
        m2_series = monte_carlo_observable{Float64}("Magnetization_2", [1])
        m4_series = monte_carlo_observable{Float64}("Magnetization_4", [1])
        e_series = monte_carlo_observable{Float64}("Energy", [1])
        e2_series = monte_carlo_observable{Float64}("Energy_2", [1])

        configurations = Array{Int, 3}(128, p.L, p.L)

        if p.algorithm == "single"
            @time ising_sweep!(p, spins)
        else
            @time wolff_sweep!(p, spins)
        end

        for t in 1:thermalizations
            if p.algorithm == "single_"
                ising_sweep!(p, spins)
            else
                wolff_sweep!(p, spins)
            end
        end

        for sweep in 1:measurements
            if p.algorithm == "single_"
                ising_sweep!(p, spins)
            else
                wolff_sweep!(p, spins)
            end
            configurations[mod1(sweep, 128), :] = spins

            if mod(sweep, 128) == 0
                batch = Int(sweep / 128)
                # println("Writing\t",  "simulation/results/configurations/$(batch)")
                h5write(output_file, "simulation/results/configurations/$(batch)", configurations)
            end

            push!(m_series, abs(sum(spins)))
            push!(m2_series, abs(sum(spins))^2)
            push!(m4_series, abs(sum(spins))^4)
            push!(e_series, energy(p, spins))
            push!(e2_series, energy(p, spins)^2)
        end
        print("Dumping")
        h = h5open(output_file, "r+")
        write(h, m_series)
        write(h, m2_series)
        write(h, m4_series)
        write(h, e_series)
        write(h, e2_series)
        close(h)
    end

    run()
end
