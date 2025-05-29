# Visualizing Semantic Space Exploration in GSGP

This project provides an interactive web interface to visualize the learning dynamics of **Geometric Semantic Genetic Programming (GSGP)**. It integrates CUDA-accelerated training using **GSGP-CUDA** and visualizes the high-dimensional semantic space using dimensionality reduction techniques like **t-SNE**, **UMAP**, and **PaCMAP**.

---

## 🌐 Web Interface Overview

The web tool allows dynamic and detailed exploration of the semantic evolution during training. To use it:

1. **Upload your CSV**:
   - Go to **"CSV File" > "Customize Input File"** to select active columns.
   - Essential columns: `Component 1`, `Component 2`, `Generation`.
   - Click **"Select CSV File"** to load your data.

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
The following images are visualizations on concrete dataset: 

![t-sne](results/tsne/population_64/gsm/general_plot/concrete_new_plots/train_820.png?raw=true "T-sne's result on Concrete")

![t-sne](results/tsne/population_64/gsm/trace/concrete_new_plots/train_820.png?raw=true "T-sne's result result with trace on Concrete")

![t-sne](results/tsne/population_64/gsm/trace_rts/concrete_new_plots/train_820.png?raw=true "T-sne's result with trace with random trees on Concrete")


## 🎞️ Semantic Evolution Demo

(results/demo/semantic_evolution.gif)

![Semantic Evolution t-sne](results/tsne/population_64/gsm/semantic_evolution.gif)

## ⚙️ GSGP-CUDA Backend

This project relies on [GSGP-CUDA](git@gitlab.com:Jmmc9122/gsgpcuda.git) for efficient GSGP training on the GPU.

### Compilation

```bash
nvcc -std=c++11 -O0 GsgpCuda.cu -o GsgpCuda.x -lcublas
