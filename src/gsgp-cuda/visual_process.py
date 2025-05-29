import os
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib as mpl
mpl.use('Agg')  # Use 'Agg' backend which is non-interactive
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def count_consecutive_labels(df, label_column='Etiqueta'):
    """
    Counts how many consecutive rows have the same label in a DataFrame.
    
    This function analyzes consecutive sequences of identical labels in a DataFrame
    and returns the size of each consecutive block.
    
    Args:
        df (pd.DataFrame): DataFrame containing a column with labels
        label_column (str): Name of the column containing the labels (default: 'Etiqueta')
        
    Returns:
        list: List of tuples (label, count) with the size of each consecutive block
        
    Raises:
        ValueError: If the specified label column doesn't exist in the DataFrame
        
    Example:
        >>> df = pd.DataFrame({'Etiqueta': ['A', 'A', 'B', 'B', 'B', 'A']})
        >>> count_consecutive_labels(df)
        [('A', 2), ('B', 3), ('A', 1)]
    """
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' does not exist in the DataFrame")
    
    # List to store results (label, count)
    blocks = []
    
    # If DataFrame is empty, return empty list
    if df.empty:
        return blocks
    
    # Initialize variables for tracking
    current_label = df[label_column].iloc[0]
    current_count = 1
    
    # Iterate through DataFrame starting from second row
    for i in range(1, len(df)):
        label = df[label_column].iloc[i]
        
        # If label is the same as previous, increment counter
        if label == current_label:
            current_count += 1
        else:
            # Save previous block and start a new one
            blocks.append((current_label, current_count))
            current_label = label
            current_count = 1
    
    # Add the last block
    blocks.append((current_label, current_count))
    
    return blocks

def get_latest_folder(directory):
    """
    Gets the most recently created folder in a given directory.
    
    This function searches through all entries in the specified directory,
    filters only folders, and returns the one with the most recent creation time.
    
    Args:
        directory (str): Path to the directory to search in
        
    Returns:
        str or None: Name of the most recently created folder, or None if no folders exist
        
    Example:
        >>> latest = get_latest_folder('/path/to/directory')
        >>> print(latest)  # 'experiment_2024_01_15'
    """
    # Get all entries in the directory
    entries = os.listdir(directory)
    
    # Filter only folders
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    
    # If no folders exist, return None
    if not folders:
        return None
    
    # Get the most recent folder based on creation date
    latest_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(directory, f)))
    
    return latest_folder

def process_and_visualize(X, y, df_trace, d_fit, method='tsne', output_prefix=''):
    """
    Processes and visualizes data using dimensionality reduction techniques (t-SNE or UMAP).
    
    This function applies dimensionality reduction to high-dimensional data and creates
    comprehensive visualizations showing the evolution of genetic algorithm populations,
    including target values, initial populations, best individuals, and evolutionary traces.
    
    Args:
        X (pd.DataFrame or np.array): Input feature data for dimensionality reduction
        y (pd.Series or np.array): Labels corresponding to each data point
        df_trace (pd.DataFrame): DataFrame containing trace evolution data
        d_fit (pd.DataFrame): DataFrame containing fitness training data
        method (str): Dimensionality reduction method - 'tsne' or 'umap' (default: 'tsne')
        output_prefix (str): Prefix for output file names (default: '')
        
    Returns:
        None: Function saves results to CSV files and generates PDF plots
        
    Notes:
        - Creates scatter plots with different colors and sizes for different label types
        - Special handling for labels: -1 (Target Value), -2 (Initial Population), 
          -3 (Random Trees), -5 (Best Individual)
        - Positive labels (10, 20, 30, etc.) represent different generations
        - Draws evolution trace connecting key points in the optimization process
        
    File Outputs:
        - {method}_with_trace.csv: Combined data with trace coordinates
        - {method}_plot.pdf: Visualization plot
    """
    # Disable specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, learning_rate=200, init='random')
        method_name = 'T-SNE'
    else:  # UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15,  
                        min_dist=0.1, spread=1.0, metric='euclidean', init='random')
        method_name = 'UMAP'
    
    X_reduced = reducer.fit_transform(X)
    
    # Create DataFrame with transformed data and labels
    df_reduced = pd.DataFrame({'Componente 1': X_reduced[:, 0], 
                              'Componente 2': X_reduced[:, 1], 
                              'Etiqueta': y})
    
    # Clean labels by removing commas
    df_reduced['Etiqueta'] = df_reduced['Etiqueta'].str.replace(',','')
    
    # Count consecutive label blocks to find common block size
    block_sizes = count_consecutive_labels(df_reduced)
    tamaños = [tamaño for _, tamaño in block_sizes]

    if tamaños:
        tamaño_mas_comun = Counter(tamaños).most_common(1)[0][0]
        print(f"The most common block size is: {tamaño_mas_comun} rows")
    else:
        print("No blocks found")
    
    # Define labels for plotting - genetic algorithm evolution stages
    labels_plot = ['-1', '-2', '-3', '-5', '10', '20', '30', '40', '50', '60', '70', '80', 
                   '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200']
    
    # Select rows with desired labels
    df_plot = df_reduced[df_reduced['Etiqueta'].isin(labels_plot)]
    
    # Handle special case for '-5' label (Best Individual) - keep only the last occurrence
    df_plot_with_minus5 = df_plot[df_plot['Etiqueta'] == '-5']
    if len(df_plot_with_minus5) > 0:
        df_plot = df_plot[(df_plot['Etiqueta'] != '-5') | (df_plot.index == df_plot_with_minus5.index[-1])]
    
    # Process labels for filtering
    colum_label = df_reduced['Etiqueta']
    # Remove initialization labels and last element
    etiquetas_a_eliminar = ['-1', '-2', '-3', colum_label.iloc[-1]]
    colum_label = colum_label[~colum_label.isin(etiquetas_a_eliminar)]
    colum_label = colum_label.reset_index(drop=True)
    
    # Filter labels for trace visualization (every 10th block)
    colum_label_filtrada = colum_label[::tamaño_mas_comun*10]
    ultimo_elemento = colum_label.iloc[-1]
    # Use concat instead of deprecated append
    colum_label_filtrada = pd.concat([colum_label_filtrada, pd.Series([ultimo_elemento])], ignore_index=True)
    
    # Create DataFrames for plotting
    df_plot_com = df_reduced[df_reduced['Etiqueta'].isin(labels_plot)]
    
    # Handle '-5' label for complete plot data
    df_plot_com_with_minus5 = df_plot_com[df_plot_com['Etiqueta'] == '-5']
    if len(df_plot_com_with_minus5) > 0:
        df_plot_com = df_plot_com[(df_plot_com['Etiqueta'] != '-5') | 
                                 (df_plot_com.index == df_plot_com_with_minus5.index[-1])]
    
    df_plot = df_reduced[df_reduced['Etiqueta'].isin(colum_label_filtrada)]
    
    # Get key evolution points from trace data
    indices_traces = df_trace.iloc[::10, 2]
    indices_seleccionados = df_trace.iloc[::10].index.tolist()
    indices_seleccionados.append(int(df_trace.iloc[-1,2]))
    
    etiquetas_int = [int(etiqueta) for etiqueta in indices_traces]
    etiquetas_int.append(int(df_trace.iloc[-1,2]))
    
    # Configure colors for special genetic algorithm components
    color_map = {
        -1: 'green',   # Target Value
        -5: 'red',     # Best Individual
        -2: 'blue',    # Initial Population
        -3: 'purple'   # Random Trees
    }
    
    # Generate gradient colors for remaining labels
    etiquetas_sin_color_fijo = [lbl for lbl in labels_plot if int(lbl) not in color_map]
    grad_color = sns.color_palette('Blues', len(labels_plot) - len(color_map))
    grad_color_index = 0
    
    # Create color palette for all labels
    label_to_index = {label: index for index, label in enumerate(labels_plot)}
    unique_palette = sns.color_palette('Set1', n_colors=len(label_to_index))
    label_colors = [unique_palette[label_to_index[label]] for label in labels_plot]
    
    # Create the main visualization plot
    fig, ax = plt.subplots(figsize=(12, 7))
    gen = 0
    n_lbl = labels_plot[4:]  # Start from position 4 (after -1, -2, -3, -5)
    
    # Plot points for each label category
    for index, lbl in enumerate(labels_plot):
        # Default plotting parameters
        z = 1  # z-order for layering
        size = 50  # marker size
        linewidth = 2 if lbl in ['-1', '-5'] else 0.5  # border width
        border = 'black' if lbl in ['-1', '-5'] else 'w'  # border color
        
        # Special formatting for key genetic algorithm components
        if lbl == '-1':  # Target Value
            size = 150
            border = 'white'
            linewidth = 3
            z = 10
            lbl_txt = 'Target Value'
        elif lbl == '-5':  # Best Individual
            size = 350
            border = 'white'
            linewidth = 3
            z = 9
            lbl_txt = 'Best Individual'
        
        # Select data subset for current label
        subset_df = df_plot_com['Etiqueta'] == lbl
        lbl_int = int(lbl) 
        
        # Assign colors
        if lbl_int in color_map:
            color = color_map[lbl_int]
        else:
            color = grad_color[grad_color_index % len(grad_color)]
            grad_color_index += 1

        # Convert numeric labels to descriptive names
        lbl = 'Target Value' if lbl == '-1' else lbl
        lbl = 'Initial Population' if lbl == '-2' else lbl
        lbl = 'Random Trees' if lbl == '-3' else lbl
        lbl = 'Best Individual' if lbl == '-5' else lbl
        
        # Handle positive generation labels (10, 20, 30, etc.)
        if lbl.isdigit() and int(lbl) >= 10:
            gen += 10
            lbl = 'Generation ' + str(gen)

        # Create scatter plot with slight jitter for better visualization
        ax.scatter(df_plot_com.loc[subset_df, 'Componente 1'] + np.random.uniform(-0.1, 0.1, subset_df.sum()), 
                  df_plot_com.loc[subset_df, 'Componente 2'], 
                  color=color, alpha=0.7, edgecolor=border, linewidth=linewidth, 
                  label=lbl, s=size, zorder=z)
    
    # Process trace elements to show evolution path
    elementos = []    
    for inx, lb in enumerate(colum_label_filtrada):
        contador = 0
        indx = 0
        cnt = 0        
        subt_df = df_plot['Etiqueta'] == lb 
        data_fil = df_plot[subt_df]
        puntos_x = []  
        puntos_y = []
        for indice in data_fil.index:
            contador += 1
            if contador == etiquetas_int[inx]:
                indx = indice+1           
                elemento1 = data_fil.loc[indx, 'Componente 1']
                elemento2 = data_fil.loc[indx, 'Componente 2']
                puntos_x.append(elemento1)
                puntos_y.append(elemento2)
                elementos.append([puntos_x, puntos_y]) 
    
    # Extract coordinates for evolution trace
    x = [coord[0][0] for coord in elementos]
    y = [coord[1][0] for coord in elementos]
    
    # Create and save combined DataFrame with trace coordinates
    df_combined = df_plot_com.copy()
    df_combined['Evolution 2'] = np.nan
    df_combined['Evolution 1'] = np.nan

    # Add trace coordinates to combined DataFrame
    for i in range(len(x)):
        if i < len(df_combined):
            df_combined.loc[i, 'Evolution 2'] = x[i]
            df_combined.loc[i, 'Evolution 1'] = y[i]

    # Save combined data to CSV
    df_combined.to_csv(f"{output_prefix}{method.lower()}_with_trace.csv", index=False)

    # Draw the evolution trace on the plot
    plt.scatter(x, y, color='black', alpha=0.8, marker='x', label='Trace', linestyle='solid')

    # Add individual trace points
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y[i]], color='black', linestyle='solid', alpha=0.5)
    
    # Connect trace points with lines
    ax.plot(x, y, color='black', alpha=0.8, marker='x', linestyle='solid')
    
    # Configure plot labels and appearance
    ax.set_xlabel(f'Component 1 {method_name}', fontsize=22, fontweight='bold')
    ax.set_ylabel(f'Component 2 {method_name}', fontsize=22, fontweight='bold')
    
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')               
    
    # Adjust plot position to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    
    # Add legend outside the plot area
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16, prop={'weight': 'bold'})                        
    
    # Save plot and clean up
    plt.savefig(f"{output_prefix}{method.lower()}_plot.pdf")
    plt.close('all')

def get_current_process():
    """
    Main processing function that orchestrates the entire dimensionality reduction analysis.
    
    This function:
    1. Finds the most recent experiment folder in the current directory
    2. Loads semantic, fitness, and trace data from CSV files
    3. Processes data using both t-SNE and UMAP dimensionality reduction
    4. Generates comprehensive visualizations and saves results
    
    Expected file structure:
    - *visual_semantic.csv: High-dimensional semantic data with labels
    - *_fitnestrain.csv: Fitness training data
    - *_trace.csv: Evolution trace data
    
    Returns:
        None: Function creates output directory and saves all results there
        
    Output:
        Creates 'dimension_reduction_results' directory containing:
        - t-SNE and UMAP visualizations (PDF)
        - Processed data files (CSV)
        - Evolution trace coordinates
        
    Notes:
        - Automatically detects and processes the most recently created folder
        - Handles multiple file formats and data cleaning
        - Removes corrupted trace data entries (marked with ***)
    """
    # Get current working directory and find latest experiment folder
    directory = os.getcwd()
    latest = get_latest_folder(directory)

    if latest:
        creation_time = os.path.getctime(os.path.join(directory, latest))
        n_d = directory + '/'+ latest
        entries = os.listdir(n_d)
        
        # Load semantic visualization data
        archivos_filtrados = [nombre for nombre in entries if "visual_semantic.csv" in nombre]
        for elemento in archivos_filtrados:
            print(f"Found semantic data file: {elemento}")
        
        element = n_d + '/' + elemento
        df = pd.read_csv(element, header=None, sep='\s+')
        nrow = len(df.index)
        nvar = df.shape[1]
        # Separate features (X) and labels (y)
        X = df.iloc[0:nrow, 0:nvar-1]
        y = df.iloc[:nrow, nvar-1]
        
        # Load fitness training data
        archivos_filtrados = [nombre for nombre in entries if "_fitnestrain.csv" in nombre]
        for elemento in archivos_filtrados:
            print(f"Found fitness training file: {elemento}")
        
        element = n_d + '/' + elemento
        df_fit = pd.read_csv(element, header=None, sep=',',index_col=0)
        print("Fitness training data loaded successfully.")
        
        # Load trace evolution data
        archivos_filtrados = [nombre for nombre in entries if "_trace.csv" in nombre]
        for elemento in archivos_filtrados:
            print(f"Found trace file: {elemento}")
        
        element = n_d + '/' + elemento
        df_trace = pd.read_csv(element, header=None,sep='\s+')
        # Remove corrupted entries marked with ***
        df_trace = df_trace[~df_trace.apply(lambda row: row.astype(str).str.contains('\*\*\*').any(), axis=1)]
        df_trace = df_trace.reset_index(drop=True)
        
        # Create results directory
        results_dir = os.path.join(n_d, "dimension_reduction_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Execute t-SNE analysis
        print("Running t-SNE analysis...")
        process_and_visualize(X, y, df_trace, df_fit ,method='tsne', output_prefix=os.path.join(results_dir, "tsne_"))
        
        # Execute UMAP analysis
        print("Running UMAP analysis...")
        process_and_visualize(X, y, df_trace, df_fit,method='umap', output_prefix=os.path.join(results_dir, "umap_"))
        
        print("Analysis completed! Results saved in:", results_dir)
    else:
        print("No folders found in the specified directory.")