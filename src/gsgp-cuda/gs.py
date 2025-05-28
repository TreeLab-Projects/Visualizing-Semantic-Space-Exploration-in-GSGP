from sklearn.base import BaseEstimator
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import time
import datetime
import webbrowser
import re
from sympy import (sympify, symbols, preorder_traversal, Function, sqrt, 
                   SympifyError)
import visual_process as red

# Global directory reference
this_dir = os.path.dirname(os.path.realpath(__file__))

class GSGPCudaRegressor(BaseEstimator):
    """
    Geometric Semantic Genetic Programming (GSGP) CUDA-accelerated Regressor.
    
    This class implements a scikit-learn compatible wrapper for a CUDA-accelerated
    Geometric Semantic Genetic Programming algorithm. GSGP is an evolutionary computation
    technique that evolves mathematical expressions for regression tasks using geometric
    semantic operators in the semantic space.
    
    The class interfaces with an external CUDA executable (GsgpCuda.x) to perform
    the actual genetic programming evolution and provides methods for model training,
    prediction, and symbolic expression analysis.
    
    Attributes:
        g (int): Number of generations for evolution
        pop_size (int): Population size for genetic algorithm
        max_len (int): Maximum length of individual expressions
        func_ratio (float): Ratio of functions vs terminals in expressions
        variable_ratio (float): Ratio of variables vs constants in terminals
        max_rand_constant (int): Maximum value for random constants
        sigmoid (int): Whether to use sigmoid transformation (0/1)
        error_function (int): Type of error function to use
        oms (int): Optimal mutation step parameter
        normalize (int): Whether to normalize data (0/1)
        do_min_max (int): Whether to apply min-max scaling (0/1)
        protected_division (int): Whether to use protected division (0/1)
        visualization (int): Whether to generate visualizations (0/1)
    """
    
    def __init__(self, g=1024, pop_size=1024, max_len=10, func_ratio=0.5,
                 variable_ratio=0.5, max_rand_constant=10, sigmoid=0,
                 error_function=0, oms=0, normalize=0, do_min_max=0,
                 protected_division=0, visualization=0):
        """
        Initialize the GSGP CUDA Regressor.
        
        Args:
            g (int): Number of generations (default: 1024)
            pop_size (int): Population size (default: 1024)
            max_len (int): Maximum individual length (default: 10)
            func_ratio (float): Function to terminal ratio (default: 0.5)
            variable_ratio (float): Variable to constant ratio (default: 0.5)
            max_rand_constant (int): Maximum random constant value (default: 10)
            sigmoid (int): Use sigmoid transformation flag (default: 0)
            error_function (int): Error function type (default: 0)
            oms (int): Optimal mutation step parameter (default: 0)
            normalize (int): Data normalization flag (default: 0)
            do_min_max (int): Min-max scaling flag (default: 0)
            protected_division (int): Protected division flag (default: 0)
            visualization (int): Visualization generation flag (default: 0)
        """
        # Store all parameters
        self.g = g
        self.pop_size = pop_size
        self.max_len = max_len
        self.func_ratio = func_ratio
        self.variable_ratio = variable_ratio
        self.max_rand_constant = max_rand_constant
        self.sigmoid = sigmoid
        self.error_function = error_function
        self.oms = oms
        self.normalize = normalize
        self.do_min_max = do_min_max
        self.protected_division = protected_division
        self.visualization = visualization
        
        # Setup file and directory names
        self.exe_name = 'GsgpCuda.x'
        self.name_run1 = str(np.random.randint(2**15-1))
        self.log_path = os.path.join(this_dir, self.name_run1) + '/'
        
        # Initialize model storage variables
        self.symbolic_model_simplified = ''
        self.symbolic_model_raw = ''
        self.nvar = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Create configuration text for the external CUDA program
        config_text = self._create_configuration_text()
        
        # Setup directories and files
        self._setup_environment()
        self._create_configuration_file(config_text)

    def _create_configuration_text(self):
        """Create configuration text for the CUDA executable."""
        return '''numberGenerations={}
populationSize={}
maxIndividualLength={}
functionRatio={}
variableRatio={}
maxRandomConstant={}
sigmoid={}
errorFunction={}
oms={}
normalize={}
do_min_max={}
protected_division={}
visualization={}
logPath={}
'''.format(self.g, self.pop_size, self.max_len, self.func_ratio, self.variable_ratio,
          self.max_rand_constant, self.sigmoid, self.error_function,
          self.oms, self.normalize, self.do_min_max, self.protected_division,
          self.visualization, self.log_path)

    def _setup_environment(self):
        """Setup directories and clean up old CSV files."""
        
        # Create log directory
        os.makedirs(self.log_path, exist_ok=True)

        # Clean up old CSV files from the main directory
        for item in os.listdir(this_dir):
            if item.endswith(".csv"):
                try:
                    os.remove(os.path.join(this_dir, item))
                except OSError:
                    pass  # File might be in use, skip it

    def _create_configuration_file(self, config_text):
        """Create the configuration INI file for the CUDA executable."""
        self.name_ini = os.path.join(self.log_path, f"{self.name_run1}_configuration.ini")
        with open(self.name_ini, "w") as f:
            f.write(config_text)
        time.sleep(1)  # Ensure file is written before proceeding

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        """
        Train the model and evaluate it on test data with visualization.
        
        This method trains the GSGP model on training data, evaluates it on test data,
        generates dimensionality reduction visualizations, and opens the results in a browser.
        
        Args:
            X_train (array-like): Training feature data
            y_train (array-like): Training target values
            X_test (array-like): Test feature data  
            y_test (array-like): Test target values
            
        Note:
            This method also triggers visualization generation and opens results in browser.
            The HTML path is currently hardcoded and should be made configurable.
        """
        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Save training data
        train_data = pd.DataFrame(self.X_train)
        train_data['target'] = self.y_train
        train_data.to_csv(os.path.join(this_dir, "train.csv"), 
                         header=None, index=None, sep='\t')
        
        # Save test data
        test_data = pd.DataFrame(self.X_test)
        test_data['target'] = self.y_test
        test_data.to_csv(os.path.join(this_dir, "test.csv"), 
                        header=None, index=None, sep='\t')
        
        time.sleep(1)
        
        # Execute CUDA program for training and testing
        subprocess.call(' '.join([
            os.path.join(this_dir, self.exe_name),
            '-train_file', os.path.join(this_dir, 'train.csv'),
            '-test_file', os.path.join(this_dir, 'test.csv'),
            '-output_model', self.name_run1,
            '-log_path', self.name_ini
        ]), shell=True, cwd=this_dir)
        
        time.sleep(1)

        # Generate dimensionality reduction visualizations
        red.get_current_process()

        # TODO: Make this path configurable instead of hardcoded
        # Obtener directorio actual
        directory = os.getcwd()
        entries = os.listdir(directory)

        # Buscar el archivo HTML
        html_file_path = None
        for entry in entries:
            if entry.startswith(self.name_run1) and entry.endswith('.html'):
                html_file_path = os.path.join(directory, entry)
                break

        # Ahora tienes la ruta en html_file_path
        if html_file_path:
            print(f"Archivo encontrado en: {html_file_path}")
        else:
            print("No se encontró el archivo HTML")
                # html_file_path = '/media/turing/Respaldo/graficas/index.html'
        abs_path = os.path.abspath(html_file_path)
        
        # Open results in browser
        if os.path.exists(abs_path):
            webbrowser.open(f'file://{abs_path}')
        else:
            print(f"Warning: Visualization file not found at {abs_path}")
        
        time.sleep(1)

    def fit(self, X_train, y_train, sample_weight=None):
        """
        Fit the GSGP model to training data.
        
        This method implements the scikit-learn interface for model training.
        It saves the training data to CSV format and calls the external CUDA
        executable to perform the genetic programming evolution.
        
        Args:
            X_train (array-like): Training feature data of shape (n_samples, n_features)
            y_train (array-like): Training target values of shape (n_samples,)
            sample_weight (array-like, optional): Sample weights (not currently used)
            
        Returns:
            self: Returns the fitted estimator instance
        """
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'Training started at: {current_time}')
        
        # Store training data
        self.X_train = X_train
        self.y_train = y_train
        
        # Prepare and save training data
        train_data = pd.DataFrame(self.X_train)
        train_data['target'] = self.y_train
        train_data.to_csv(os.path.join(this_dir, "train.csv"), 
                         header=None, index=None, sep='\t')
        
        # Store number of variables for later use
        self.nvar = train_data.shape[1] - 1
        
        time.sleep(1)
        
        # Execute CUDA program for training
        subprocess.call(' '.join([
            os.path.join(this_dir, self.exe_name),
            '-train_file', os.path.join(this_dir, 'train.csv'),
            '-output_model', self.name_run1,
            '-log_path', self.name_ini
        ]), shell=True, cwd=this_dir)
        
        time.sleep(1)
        return self

    def predict(self, X_test):
        """
        Make predictions on test data using the fitted model.
        
        This method implements the scikit-learn interface for prediction.
        It saves the test data and calls the external CUDA executable
        to generate predictions using the trained model.
        
        Args:
            X_test (array-like): Test feature data of shape (n_samples, n_features)
            
        Returns:
            list: Predicted values for the test data
            
        Raises:
            FileNotFoundError: If the prediction output file is not found
        """
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'Prediction started at: {current_time}')
        
        # Store test data
        self.X_test = X_test
        test_data = pd.DataFrame(self.X_test)
        test_data.to_csv(os.path.join(this_dir, "unseen_data.csv"), 
                        header=None, index=None, sep='\t')
        
        time.sleep(1)

        # Execute CUDA program for prediction
        subprocess.call(' '.join([
            os.path.join(this_dir, self.exe_name),
            '-model', self.name_run1,
            '-input_data', os.path.join(this_dir, 'unseen_data.csv'),
            '-prediction_output', f'{self.name_run1}_prediction.csv',
            '-log_path', self.name_ini
        ]), shell=True, cwd=this_dir)

        time.sleep(1)

        # Read predictions from output file
        prediction_file = os.path.join(self.log_path, f"{self.name_run1}_prediction.csv")
        y_pred = []
        
        try:
            with open(prediction_file, 'r') as f:
                for line in f:
                    y_pred.append(float(line.strip()))
        except FileNotFoundError:
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
        except ValueError as e:
            raise ValueError(f"Error parsing prediction values: {e}")

        return y_pred

    def check_valid_expression(self, expr_str):
        """
        Check if a string represents a valid symbolic expression.
        
        Args:
            expr_str (str): String representation of mathematical expression
            
        Returns:
            bool: True if expression is valid, False otherwise
        """
        try:
            sympify(expr_str)
            return True
        except SympifyError:
            return False

    def best_individual(self, model_type):
        """
        Retrieve the best evolved individual (symbolic expression).
        
        Args:
            model_type (int): Type of model to return (0 for raw, 1 for simplified)
            
        Returns:
            str or sympy.Expr: The best individual expression
        """
        if model_type == 0:
            return self.symbolic_model_raw
        else:
            return self.symbolic_model_simplified

    def get_model(self):
        """
        Get the best evolved model.
        
        Returns:
            str or sympy.Expr: The simplified symbolic model
        """
        return self.best_individual(1)

    def get_n_nodes(self, model_type):
        """
        Calculate the number of nodes in the evolved symbolic expression.
        
        This method processes the model expression file generated by the CUDA executable,
        handles protected division operations, and counts the total nodes in the expression tree.
        
        Args:
            model_type (int): Type of model analysis (0 for raw expression, 1 for simplified)
            
        Returns:
            int: Total number of nodes in the symbolic expression
            
        Note:
            Protected division is a special operation that prevents division by zero
            by using the formula: x / sqrt(1 + y^2) when y = 0, otherwise x / y
        """
        
        class ProtectedDivision(Function):
            """Custom SymPy function for protected division operations."""
            
            @classmethod
            def eval(cls, x, y):
                """Evaluate protected division: x/y with protection against division by zero."""
                if y.is_Number and y.is_zero:
                    return x / sqrt(1 + (y**2))
                else:
                    return x / y

        def convert_protected_division(expr):
            """Convert protected_division function calls to standard division."""
            pattern = re.compile(r'protected_division\((.+?),\s*(.+?)\)')
            while True:
                match = pattern.search(expr)
                if match is None:
                    break
                numer, denom = match.groups()
                # Clean up quotes and apply conversion
                numer = numer.replace("'", "")
                denom = denom.replace("'", "")
                expr = expr[:match.start()] + f"({numer})/({denom})" + expr[match.end():]
            return expr
        
        # Read model expression from file
        model_file = os.path.join(self.log_path, f"{self.name_run1}_ModelExpression.csv")
        
        try:
            with open(model_file) as f:
                contents = f.read()
        except FileNotFoundError:
            print(f"Warning: Model file not found: {model_file}")
            return 0

        # Split expression by colons (sub-expressions)
        sub_expressions = contents.split(":")[:-1]
        
        # Generate variable names for the expression
        var_names = [f"X_{i}" for i in range(self.nvar)]
        print(f"Variables in model: {var_names}")
        
        # Create symbol mapping for SymPy
        symbol_dict = {s: symbols(s) for s in var_names}
        protected_dict = {'protected_division': ProtectedDivision, **symbol_dict}
        
        total_nodes = 0
        combined_expression = ""
        
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'Model processing started at: {current_time} (Type: {model_type})')
        
        if model_type == 0:
            # Process raw expressions without simplification
            for i, sub_exp in enumerate(sub_expressions):
                try:
                    # Parse expression with protected division support
                    expr = sympify(sub_exp, protected_dict, evaluate=False)
                    converted_expr = convert_protected_division(str(expr))
                    
                    # Add to combined expression
                    if i == 0:
                        combined_expression = converted_expr
                    else:
                        combined_expression += '+' + converted_expr
                        
                except Exception as e:
                    print(f"Warning: Error processing sub-expression {i}: {e}")
                    continue
            
            # Store raw model
            self.symbolic_model_raw = combined_expression if combined_expression else "0"
            
            # Count nodes in combined expression
            try:
                final_expr = sympify(combined_expression, evaluate=False)
                for _ in preorder_traversal(final_expr):
                    total_nodes += 1
            except:
                print("Warning: Could not count nodes for raw expression")
                total_nodes = 0
                
        else:
            # Process and simplify expressions  
            for i, sub_exp in enumerate(sub_expressions):
                try:
                    # Parse and evaluate expression
                    expr = sympify(sub_exp, protected_dict, evaluate=True)
                    converted_expr = convert_protected_division(str(expr))
                    simplified_expr = sympify(converted_expr, evaluate=True)
                    
                    # Add to combined expression
                    if i == 0:
                        combined_expression = str(simplified_expr)
                    else:
                        combined_expression += '+' + str(simplified_expr)
                        
                except Exception as e:
                    print(f"Warning: Error processing sub-expression {i}: {e}")
                    continue
            
            # Simplify the combined expression
            try:
                final_expr = sympify(combined_expression, evaluate=True)
                self.symbolic_model_simplified = final_expr
                
                # Count nodes in simplified expression
                for _ in preorder_traversal(final_expr):
                    total_nodes += 1
                    
                print(f"Model complexity (nodes): {total_nodes}")
                
            except Exception as e:
                print(f"Warning: Error in final simplification: {e}")
                self.symbolic_model_simplified = combined_expression
                total_nodes = 0

        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'Model processing completed at: {current_time}')
        
        return total_nodes

    def get_model_complexity(self):
        """
        Get the complexity (number of nodes) of the simplified model.
        
        Returns:
            int: Number of nodes in the simplified symbolic expression
        """
        return self.get_n_nodes(1)

    def get_raw_model_complexity(self):
        """
        Get the complexity (number of nodes) of the raw model.
        
        Returns:
            int: Number of nodes in the raw symbolic expression
        """
        return self.get_n_nodes(0)

    def __str__(self):
        """String representation of the regressor."""
        return (f"GSGPCudaRegressor(g={self.g}, pop_size={self.pop_size}, "
                f"max_len={self.max_len}, visualization={self.visualization})")

    def __repr__(self):
        """Detailed string representation of the regressor."""
        return self.__str__()