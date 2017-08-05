# Ising

A simulation of the Ising model in Julia. You can choose between single spin flip or Wolff updates. Results are recorded using the [MonteCarloObservable.jl](https://www.github.com/pebroecker/MonteCarloObservable.jl) module and written to HDF5 files. Optionally, the code can save spin configurations for use in Machine Learning applications. 

Input files are in JSON format, see [sample_jobs.jl](https://github.com/pebroecker/Ising.jl/blob/master/test/sample_jobs.jl) for an example of how to generate such scripts. In the job directory, run the program using

```
julia <path to>/Ising.jl <prefix> <task index>
```

