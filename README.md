# Visualizing Semantic Space Exploration in GSGP

This project provides an interactive web interface to visualize the learning dynamics of **Geometric Semantic Genetic Programming (GSGP)**. It integrates CUDA-accelerated training using **GSGP-CUDA** and visualizes the high-dimensional semantic space using dimensionality reduction techniques like **t-SNE**, **UMAP**, and **PaCMAP**.

---

## 🌐 Web Interface Overview

The web tool allows dynamic and detailed exploration of the semantic evolution during training.

## 🚀 Live Demo

👉 **[Launch the Interactive Web Interface](https://treelab-projects.github.io/Visualizing-Semantic-Space-Exploration-in-GSGP/)** 👈

## 🛠️ How to Use It

1. **Upload your CSV**:
   - Go to **"CSV File" > "Customize Input File"** to select active columns.
   - Essential columns: `Component 1`, `Component 2`, `Generation`.
   - Click **"Select CSV File"** to load your data.
   - 👉 Alternatively, you can use the example file provided in the repository: [`output_file.csv`](output_file.csv)


2. **Visualization Controls** (top-left menu):
   - **Color Palette**: Choose from Viridis, Magma, Cividis, Blue-Red, Purple-Pink, or Electromagnetic Spectrum.
   - **Table Selection Mode**: Toggle row selection to filter generations.
   - **Hover Effect**: Enable/disable interactive tooltips.
   - **Change Scale**: Switch between size, training, and test-based scaling.
   - **Hide Symbolic Expression**: Simplify plot labels by removing symbolic expressions.

3. **Column Configuration**:
   - First six columns are required.
   - Remaining columns are optional and disabled by default.
   - Use **"Toggle All"** and **"Confirm"** to apply changes.

---

## 📊 About the Data

The CSV file must contain the following semantic data:

| Column | Description |
|--------|-------------|
| `Component 1` | X-axis coordinate in 2D semantic space |
| `Component 2` | Y-axis coordinate in 2D semantic space |
| `Generation` | Generation number (`-1` to `-5` for special cases) |
| `IsBest` | Boolean indicator if the individual is best of its generation |
| Others | Optional metadata columns |

Special generation values:
- `-1`: Target value  
- `-2`: Initial population  
- `-3`: Random tree  
- `-5`: Best individual

---
## <a name='Visualizations'></a>Visualizations

### 2D T-SNE Visualization of Semantic Space with GSM

This visualization performs a complete incremental projection of the 2D semantic space, showing the change between semantics during evolution with a gradual shift between points.

![t-sne](results/tsne/population_64/gsm/general_plot/concrete_new_plots/train_820.png?raw=true "T-sne's result on Concrete")

### 2D Evolution with Trajectory Tracking of Best Individual with GSM

This visualization performs a complete incremental projection of the 2D semantic space. It shows the change between semantics during evolution with a gradual change between points. It also plots the search for the best individual over the semantic space.

![t-sne](results/tsne/population_64/gsm/trace/concrete_new_plots/train_820.png?raw=true "T-sne's result result with trace on Concrete")

### 2D Evolution with Path Tracking of the Best Individual Indicating the Random Trees at Each Mutation Event with GSM

This visualization performs a complete incremental projection of the 2D semantic space. It shows semantic change during evolution, with a gradual shift between points. It also plots the search for the best individual in the semantic space, indicating the random trees that were present at the mutation event.

![t-sne](results/tsne/population_64/gsm/trace_rts/concrete_new_plots/train_820.png?raw=true "T-sne's result with trace with random trees on Concrete")


## 🎞️ Semantic Evolution Demo

### 3D T-SNE Visualization of Semantic Space with Incremental Rotation View with GSM

This visualization performs a complete incremental projection of the 3D semantic space. It shows the change between semantics during evolution, revealing the behavior of semantic search.

![Semantic Evolution t-sne](assets/visual_semantic_3D_incremental_rotation.gif?raw=true "T-sne's result with random trees on Concrete with GSM")

### 3D Evolution with Temporal Trajectory Tracking with GSM

This comprehensive visualization combines the 3D semantic space. It shows the complete semantic evolution and also illustrates the learning trajectory of the best individual.


![Semantic Evolution t-sne](assets/visual_semantic_3D_incremental_with_trace.gif?raw=true "T-sne's result with trace with random trees on Concrete with GSM")

### 3D T-SNE Visualization of Semantic Space with Incremental Rotation View with GSM-OMS

This visualization performs a complete incremental projection of the 3D semantic space. It shows the change between semantics during evolution, revealing the behavior of semantic search with GSM with OMS.

![Semantic Evolution t-sne](assets/visual_semantic_3D_incremental_rotation_oms.gif?raw=true "T-sne's result with random trees on Concrete with GSM-OMS")

### 3D T-SNE Visualization of Semantic Space with Incremental Rotation - SLIM-GSGP

**Repository:** [SLIM - DALabNOVA](https://github.com/DALabNOVA/slim)

This visualization presents an incremental three-dimensional projection of the semantic space during SLIM-GSGP (Semantic Learning algorithm with Inflate and deflate Mutations) evolution. The visualization reveals how semantic search evolves throughout the evolutionary process, showing the transformations in semantic space that characterize this geometric semantic genetic programming algorithm.

![Semantic Evolution t-sne](assets/slim_gsgp_visual_semantic_3D_incremental_rotation.gif?raw=true "T-sne's result on Concrete with SLIM-GSGP")

### 3D T-SNE Visualization of GPLearn Learning Process in Semantic Space

This visualization performs a complete incremental projection of the 3D semantic space captured during GPLEARN evolution. Although GPLEARN performs syntactic search by default, we stored the semantic information of each new population across generations to observe and analyze the learning behavior and semantic evolution patterns of the GPLEARN algorithm.

**Repository:** [gplearn - Trevor Stephens](https://github.com/trevorstephens/gplearn)

![Semantic Evolution t-sne](assets/gp_learn_visual_semantic_3D_incremental_rotation.gif?raw=true "T-sne's result on Concrete with GPLEARN")

## ⚙️ GSGP-CUDA Backend

This project relies on [GSGP-CUDA](git@gitlab.com:Jmmc9122/gsgpcuda.git) for efficient GSGP training on the GPU.

### Compilation

```bash
nvcc -std=c++11 -O0 GsgpCuda.cu -o GsgpCuda.x -lcublas
```

---

## 🧪 Generating Your Own Data

If you want to train a model and visualize the semantic evolution using our web interface, you can use the following Python script:

```python
import gs
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
import random
import numpy as np
import statistics as stats

# Load your dataset (example path provided)
path = ""
df = pd.read_csv(path, header=None, sep='\s+')

# Define rows and columns
nrow = len(df.index)
nvar = df.shape[1]

# Separate features and target
X = df.iloc[0:nrow, 0:nvar-1]
y = df.iloc[:nrow, nvar-1]

# Initialize GSGP-CUDA regressor
est = gs.GSGPCudaRegressor(
    g=200,
    pop_size=64,
    max_len=1024,
    func_ratio=0.5,
    variable_ratio=0.5,
    max_rand_constant=10,
    sigmoid=1,
    error_function=0,
    oms=0,
    normalize=0,
    do_min_max=0,
    protected_division=1,
    visualization=0
)

# Split the dataset
n = random.randint(0, 9000)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=n)

# Train and evaluate
est.train_and_evaluate_model(X_train, y_train, X_test, y_test)
```
🔹 This script will:

   * Train a GSGP model using your dataset.

   * Automatically generate two CSV files (for train and test) after the evolution process.

   * Launch the interactive web interface so you can load and explore the generated data.

📁 The output CSVs will be compatible with our visualization tool, and ready to upload directly.
---

## Researchers 🧑‍🔬👩‍🔬 
- *Dr. Leonardo Trujillo Reyes* <br />
 leonardo.trujillo.ttl@gmail.com<br />
https://orcid.org/0000-0003-1812-5736

- *PhD. Student Joel L. Nation* <br />
joel.nation19@tectijuana.edu.mx <br />
https://orcid.org/0000-0001-6409-9597

- *PhD. Student Jose Manuel Muñoz Contreras* <br />
jose.munoz17@tectijuana.edu.mx <br />
https://orcid.org/0009-0007-7134-8708

- *Ing. Diana Sarai Hernandez Fierro* <br />
l20211976@tectijuana.edu.mx <br />


