# Surrogate MATSim

This repository provides tools and pipelines for working with MATSim transport simulations and surrogate modeling. It is organized into two main modules:

## Project Structure

```
matsim/
  berlin/
    extract_berlin_linkstats.sh
    generate_configs.sh
    run_matsim_berlin_parallel.sh
    readme.txt
  idf/
    clean_networks.sh
    delete_iterations.sh
    extract_idf_assets.sh
    run_matsim_parallel.sh
    readme.txt
  network/
    apply_policy_to_network.py
    compress-network.py
    extract_higher_order_roads.py
    generate_matsim_configs.py
    plot_matsim_network.py
    sample_policy_ids.py
    readme.txt
surrogate/
  berlin/
    my_data.py
    my_model.py
    my_plot.py
    pipeline_berlin.ipynb
  idf/
    idf_pipeline.py
    my_data.py
    my_model.py
    my_plot.py
    readme.txt
LICENSE.md
```

## Modules

### 1. `matsim/`
Contains scripts and utilities for running and processing MATSim simulations.

- **berlin/**: Scripts for Berlin-specific MATSim runs and data extraction.
- **idf/**: Scripts for Île-de-France (IDF) MATSim runs and data processing.
- **network/**: Python utilities for manipulating and visualizing MATSim network files, generating capacity-reducing policies, and applying these policies to the network.

### 2. `surrogate/`
Contains code for surrogate modeling and analysis of MATSim outputs.

- **berlin/**: Data handling, modeling, and plotting for Berlin simulations.
- **idf/**: Data handling, modeling, and plotting for IDF.

## Getting Started

1. **MATSim Simulations**  
   - Use the shell scripts in `matsim/berlin/` and `matsim/idf/` to run or process MATSim simulations.
   - Network manipulation, policy generation, and visualization tools are in `matsim/network/`.

2. **Surrogate Modeling**  
   - Use the Python scripts and notebooks in `surrogate/berlin/` and `surrogate/idf/` for data analysis, model training, and plotting.

## Requirements

- Python 3.9 (for surrogate modeling)
- tensorflow 2.20.0 (for surrogate modeling)
- tensorflow-gnn 1.0.3 (for suorragete modeling)
- Python 3.10 (for IDF MATSim pipeline)
- JAVA 17 (for MATSim simulation)
- Bash (for running shell scripts)
- MATSim (Java-based, required for simulation runs)

## Usage

- Run MATSim simulations using the provided shell scripts.
- Generate and apply capacity-reducing policies to networks using the tools in `matsim/network/`.
- Process and analyze simulation outputs using the Python scripts and notebooks in the `surrogate` module.
  
## Berlin MATSim Simulation Scenario


The Berlin MATSim simulation in this repository is based on the scenario from the one percent Berlin [matsim-episim project](https://github.com/matsim-org/matsim-episim?tab=readme-ov-file).

## IDF MATSim Simulation Scenario

The IDF MATSim simulation uses the pipeline from the [eqasim-org/ile-de-france project](https://github.com/eqasim-org/ile-de-france), which can generate the synthetic population and provides the MATSim simulator for the Île-de-France region.

## License

See `LICENSE.md` for details.
