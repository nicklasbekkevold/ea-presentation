# EA presentation

Code repository for the presentation on evolutionary algorithms "What can we learn from evolution?" for Capgemini August 25th 2023.

The algorithm showcased here is a variation on the simple genetic algorithm (SGA), whit the following parameters:

| Parameter          | Value                 |
| ------------------ | --------------------- |
| Representation     | Bit-strings           |
| Recombination      | 1-Point crossover     |
| Mutation           | Bit flip              |
| Parent selection   | Tournament selection* |
| Survival selection | Generational          |

*Traditionally, roulette wheel has been used by the SGA. Tournament selection was chosen here instead due to its simplicity and speed.

## Installation

To install the project dependencies, run the following at the terminal:

```bash
pip install -r requirements.txt
```

## Run the optimizer

The main script is located (conveniently) in [main.py](main.py).

To start it, run the following at the terminal:

```bash
python main.py
```

## Unit tests

The unit tests are located in the `tests/`-folder.

To execute them, run the following at the terminal:

```bash
python -m pytest tests/
```

## Results

The results of the optimization become available after each run in the `results/`-folder.

Each run contains the following:

```bash
📂results
┣ 📁1
┣ 📁2
┣ ...
┣ 📁3
┣ ┣ 📈ga.png        # generational plot of the GA optimization progress 
┣ ┣ 🧾metrics.csv   # population metrics for each generation (generation, baseline, best, average, entropy)
┣ ┣ 🐍parameters.py # parameters used to obtain the respective results
┣ ┣ 📄solution.txt  # best solution found at termination 
```
