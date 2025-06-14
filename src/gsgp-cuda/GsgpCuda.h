/*<one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2020  José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.*/

//! \file   GsgpCuda.h
//! \brief  File containing the definition of the modules (kernels) used to create the population of individuals, evaluate them, the search operator and read data
//! \author Jose Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
//! \date   created on 25/01/2020

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <ctime>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <dirent.h>
#include <vector>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <time.h>
#include <stack>
#include <limits>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// #include </usr/local/cuda-11.7/cuda-samples/Common/helper_cuda.h>
// #include </usr/local/cuda-11.7/cuda-samples/Common/helper_functions.h>
using namespace std;
extern "C"

/*!
* \fn       string currentDateTime()
* \brief    function to capture the date and time of the host, this allows to define the exact moment of each GSGP-CUDA run,
            this allows us to name the output files by date and time.
* \return   char: return date and time from the host computer
* \date     25/01/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
const std::string currentDateTime();

/*!
* \fn       void cudaErrorCheck(const char* functionName)
* \brief    This function catches the error detected by the compiler and prints the user-friendly error message.
* \param    char funtionName: pointer to the name of the kernel executed to verify if there was an error 
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
void cudaErrorCheck(const char* functionName);


/*!
* \brief    Structure used to store the parameters of the configuration.ini file and these are used to initialize the algorithm parameters  
* \param    int numberGenerations: number of generations of the GSGP algorithm 
* \param    int populationSize: number of individuals in the population
* \param    int maxIndividualLength:  variable that stores the length (number of genes) of an individual
* \param    float functionRatio : probability of selecting a function (otherwise a terminal).
* \param    float variableRatio : probability of selecting a variable (otherwise a constant) when a terminal gene has been chosen.
* \param    int maxRandomConstant: max number for ephemeral random constants
* \param    char logPath[100]: name of the output files
* \date     9/11/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
typedef struct cfg_{
  int numberGenerations; 
  int populationSize;       
  int maxIndividualLength;             
  float functionRatio;     
  float variableRatio;      
  int maxRandomConstant;
  int sigmoid;
  int errorFunction;
  int oms; 
  int normalize;
  int do_min_max;
  int protected_division;
  int visualization;
  char logPath[5000];        
}cfg;

/// struct variable containing the values of the parameters specified in the configuration.ini file
cfg config;
/// variable containing the numbers of rows (instances) of the training dataset
int nrow;
/// variable containing the numbers of rows (instances) of the test dataset
int nrowTest;
/// variable containing the numbers of columns (excluding the target) of the training dataset
int nvar;

/*!
* \brief    Structure that represents the tuple used to store information on the individuals involved in each generation and is used for the reconstruction of the best individual.
* \param    int initializePopulationParent: variable containing the index of the parent (mutation)
* \param    int firstParent: variable containing the index of the first random tree (mutation)
* \param    int secondParent: variable containing the index of the second random tree (mutation)
* \param    int newIndividual: variable containing the index of newly created individual
* \param    float mutStep: variable containing the mutation step of semantic mutation
* \date     25/01/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
 typedef struct entry_{
   int firstParent;   /*!< variable containing the index of the first random tree for the  mutation operation */
   int secondParent;  /*!< variable containing the index of the second random tree for the  mutation operation */
   int number;        /*!< variable containing the index of the parent (mutation) or the index of the random tree (crossover)*/
   int event;
   int mark;          /*!< variable used to reconstruct the optimal solution. 1 means that a particular tree is involved in the construction of the optimal solution, 0 means that the particular tree can be ignored.*/
   int newIndividual; /*!< variable containing the index of the newly created individual */
   float mutStep;     /*!< variable containing the mutation step of the semantic mutation */
}entry;

/// variable that stores the length (number of genes) of an individual
int individualLength = 0;

/// Variable that configures the mesh for launching a kernel with maximum resources.
int gridSize;

/// Variable that configures the mesh for launching a kernel with minimum resources.
int minGridSize;

/// Variable that store the number of thread for execution configuration for a kernel in the GPU
int blockSize; 

/// Variable that stores the size of the memory for the individuals in the GPU
int sizeMemIndividuals; 

/// Variable that stores the size of the memory for the random trees normalize parameters
int sizeMemNormalize; 

/// Variable that stores the size of the memory for the individuals in the GPU
int sizeMemPopulation; 

/// Variable that stores the size in bytes of the memory for the initial population to store random numbers
int twoSizeMemPopulation; 

/// Variable storing twice the initial population of individuals to generate random positions
long int twoSizePopulation; 

/// Variable that stores the size in bytes of semantics for the entire population with training data
long int sizeMemSemanticTrain;

/// Variable that stores the size in bytes of semantics for the entire population with test data
long int sizeMemSemanticTest; 

/// Variable that stores the size in bytes of semantics for the entire population with training data
long int sizeMemDataTrain;

/// Variable that stores the size in bytes of semantics for the entire population with test data
long int sizeMemDataTest = sizeof(float)*(nrowTest*nvar);

/// Variable that stores the size in bytes of semantics for the entire population with test data
long int sizeElementsSemanticTest = (config.populationSize*nrowTest);

/// Variables that store the execution configuration for a kernel in the GPU
int gridSizeTest;

/// Variables that store the execution configuration for a kernel in the GPU
int minGridSizeTest;

/// Variables that store the execution configuration for a kernel in the GPU
int blockSizeTest;

/// Variable that stores training data elements
long int sizeElementsSemanticTrain ; 

/// Variable that stores the size in bytes of the structure to store the survival record
long int vectorTracesMem ; 

/// Variable contains the vectors of pointers to store the population and space allocation in the GPU
float *dInitialPopulation;

/// Variable contains the vectors of pointers to store the random trees and space allocation in the GPU
float *dRandomTrees;

/// Variable contains the vectors of pointers to store the population and space allocation in the CPU
float *hInitialPopulation;

/// Variable contains the vectors of pointers to store the random trees and space allocation in the CPU
float *hRandomTrees;

/// This block contains the vectors of pointers to store the structure to keep track of mutation and survival and space allocation in the GPU
entry  *vectorTraces;

/// Variable for the input data train and assignment in the GPU
float *uDataTrain;

/// Variable for the input target train values ​​and assignment in the GPU
float *uDataTrainTarget;

/// Variable for the input data test and assignment in the GPU
float *uDataTest;

/// Variable for the input target test values ​​and assignment in the GPU
float *uDataTestTarget;

/// pointers of vectors of test fitness values at generation g and assignment in the GPU
float *uFitTest;

/// pointers of vectors of train fitness values at generation g and assignment in the GPU
float *uFit;

/// pointer of vectors that contain the semantics of an individual in the population, calculated with the training set and test in generation g and its allocation in GPU
float *uSemanticTrainCases;

/// pointer of vectors that contain the semantics of an individual in the population, calculated with the training set and test in generation g and its allocation in GPU
float *uSemanticRandomTrees;

/// pointer of vectors that contain the semantics of an individual in the population, calculated with the testing set and test in generation g and its allocation in GPU
float *uSemanticTestRandomTrees;

/// pointer of vectors that contain the semantics of an individual in the population, calculated with the testing set and test in generation g and its allocation in GPU
float *uSemanticTestCases;

/// auxiliary pointer vectors for the interpreter and calculate the semantics for the populations and assignment in the GPU
float *uStackInd;

/// auxiliary pointer vectors for the interpreter and calculate the semantics for the populations and assignment in the GPU
int *uPushGenes;

/// this section makes use of the isamin de cublas function to determine the position of the best individual
int result;

/// this section makes use of the isamin de cublas function to determine the position of the best individual
int incx1=1;

/// this section makes use of the isamin de cublas function to determine the position of the best individual
int indexBestIndividual;

/// vector of pointers to save random positions of random trees and allocation in GPU
float *indexRandomTrees;

/// vector of pointers to save the mutation step of the semantic mutation and allocation in GPU
float *mutationStep;

/// this variable makes use of the isamin de cublas function to determine the position of the best individual of the new population
int resultBestOffspring;

/// this variable makes use of the isamin de cublas function to determine the position of the best individual of the new population
int incxBestOffspring=1;

/// this variable makes use of the isamin de cublas function to determine the position of the best individual of the new population
int indexBestOffspring;

/// this variable makes use of the isamin de cublas function to determine the position of the worst individual of the new population
int resultWorst;

/// this variable makes use of the isamin de cublas function to determine the position of the worst individual of the new population
int incxWorst=1;

/// this variable makes use of the isamin de cublas function to determine the position of the worst individual of the new population
int indexWorstOffspring;

/// temporal Variables to perform the movement of pointers in survival
float *tempSemantic;

/// temporal Variables to perform the movement of pointers in survival
float *tempFitnes;

/// temporal Variables to perform the movement of pointers in survival
float *tempSemanticTest;

/// temporal Variables to perform the movement of pointers in survival
float *tempFitnesTest;

/// vectors that contain the semantics of an individual in the population, calculated in the training set in the g + 1 generation and its allocation in GPU
float *uSemanticTrainCasesNew;

/// vectors that contain the semantics of an individual in the population, calculated in the training set in the g + 1 generation and its allocation in GPU
float *uFitNew;

/// vectors that contain the semantics of an individual in the population, calculated in the test set in the g + 1 generation and its allocation in GPU
float *uSemanticTestCasesNew;

/// vectors that contain the semantics of an individual in the population, calculated in the test set in the g + 1 generation and its allocation in GPU
float *uFitTestNew;

/// Variable contains the vectors of pointers to store the random trees normalize paramaters host
float *hNormalizeData;

/// Variable contains the vectors of pointers to store the random trees normalize paramaters device
float *dNormalizeData;



float *uDifferenceRandomTrees;
float *uDifferenceRtPow;
float *inverse;
float *oms;
float targetMean = 0.0;
float fitR2= 0.0;

std::vector <std::string> sample;
std::vector <std::string> randomTrees;
vector< int> initPop;
vector< int> sizerandomTree;

/*!
* \fn       __global__ void init(unsigned int seed, curandState_t* states)
* \brief    This kernel is used to initialize the random states to generate random numbers with a different pseudo sequence in each thread
* \param    int seed: used to generate a random number for each core  
* \param    curandState_t states: pointer to store a random state for each thread
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
__global__ void init(unsigned int seed, curandState_t* states);

/*!
* \fn       __device__ int push(float val,int *pushGenes, float *stackInd)
* \brief    push() function is used to insert an element at the top of the stack. The element is added to the stack container and the size of the stack is increased by 1.
* \param    float val: variable that stores a value resulting from a valid operation in the interpreter
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   int: auxiliary pointer that stores the positions of individuals + 1
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h 
*/
__device__ int push(float val,int *pushGenes, float *stackInd);

/*!
* \fn       __device__ float pop(int *pushGenes, float *stackInd)
* \brief    pop() function is used to remove an element from the top of the stack(newest element in the stack). The element is removed to the stack container and the size of the stack is decreased by 1.
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   float: returns the stackInd without the value positioned in pushGenes [tid]
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
__device__ float pop(int *pushGenes, float *stackInd);

/*!
* \fn       __device__ bool isEmpty(int *pushGenes, unsigned int sizeMaxDepthIndividual)     
* \brief    Check if a stack is empty
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \return   bool - true if the stack is empty, false otherwise
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     gsgpMalloc.h
*/
__device__ bool isEmpty(int *pushGenes, unsigned int sizeMaxDepthIndividual);

/*!
* \fn       __device__ void clearStack(int *pushGenes, unsigned int sizeMaxDepthIndividual, float *stackInd)
* \brief    remove all elements from the stack so that in the next evaluations there are no previous values of other individuals
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the evaluation of each individual
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h 
*/
__device__ void clearStack(int *pushGenes, unsigned int sizeMaxDepthIndividual, float *stackInd);

/*!
* \fn       __global__ void initializePopulation(float* dInitialPopulation, int nvar, int sizeMaxDepthIndividual, curandState_t* states, int maxRandomConstant)
* \brief    The initializePopulation kernel creates the population of programs T and the set of random trees R uses by the GSM kernel, based on the desired population
            size and maximun program length. The individuals are representd using a linear genome, composed of valid terminals (inputs to the program) and functions 
            (basic elements with which programs can be built).
* \param    float *dInitialPopulation: vector pointers to store the individuals of the initial population
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \param    curandState_t *states: random status pointer to generate random numbers for each thread
* \param    int maxRandomConstant: variable containing the maximum number to generate ephemeral constants
* \param    int funtion: variable containing the number of functions
* \param    float functionRatio: probability of selecting a function
* \param    float terminalRatio: probability of selecting a terminal
* \return   void
* \date     09/11/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__global__ void initializePopulation(float* dInitialPopulation, int nvar, int sizeMaxDepthIndividual, curandState_t* states, int maxRandomConstant, int functions, float functionRatio, float terminalRatio);

/*!
* \fn       __global__ void computeSemantics(float *inputPopulation, float *outSemantic, unsigned int sizeMaxDepthIndividual, float *data, int nrow, int nvar, int *pushGenes, float *stackInd)  
* \brief    The ComputeSemantics kernel is an interpreter, that decodes each individual and evaluates it over all fitness cases,
            producing as output the semantic vector of each individual. The chromosome is interpreted linearly, using an auxiliary LIFO stack D that stores 
            terminals from the chromosome and the output from valid operations.
* \param    float *inputPopulation: vector pointers to store the individuals of the population
* \param    float *outSemantic: vector pointers to store the semantics of each individual in the population
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \param    float *data: pointer vector containing training or test data
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h 
*/
__global__ void computeSemantics(float *inputPopulation, float *outSemantic, unsigned int sizeMaxDepthIndividual, float *data,
 int nrow, int nvar, int *pushGenes, float *stackInd, int protected_division);

/*!
* \fn       __global__ void computeError(float *semantics, float *targetValues, float *fit, int nrow)
* \brief    The computeError kernel computes the RMSE between each row of the semantic matrix ST,m×n and the target vector t, computing the
            fitness of each individual in the population.
* \param    float *semantics: vector of pointers that contains the semantics of the individuals of the initial population 
* \param    float *targetValues: pointer containing the target values of train or test
* \param    float *fit: vector that will store the error of each individual in the population
* \param    int nrow: variable containing the number of rows (instances) of the training and test dataset
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     gsgpMalloc.h
*/
__global__ void computeError(float *semantics, float *targetValues, float *fit, int nrow);

/*!
* \fn       __device__ float sigmoid(float n)
* \brief    auxiliary function for the geometric semantic mutation operation
* \param    float n: semantic value of a random tree
* \return   float n: value resulting from the function
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h  
*/
__device__ float sigmoid(float n);

/*!
* \fn       __global__ void initializeIndexRandomTrees(int sizePopulation, float *indexRandomTrees, curandState_t* states);
* \brief    this kernel generates random indexes for random trees that are used in the mutation operator to select two random trees.
* \param    int sizePopulation: this variable contains the number of individuals that the population has
* \param    float *indexRandomTrees: this pointer stores the indexes randomly for mutation
* \param    curandState_t* states: random status pointer to generate random numbers for each thread
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h
*/
__global__ void initializeIndexRandomTrees(int sizePopulation, float *indexRandomTrees, curandState_t* states);

/*!
* \fn       __global__ void geometricSemanticMutation(float *initialPopulationSemantics, float *randomTreesSemantics, float *newSemanticsOffsprings, int sizePopulation, int nrow, int tElements, int generation, float *indexRandomTrees, entry_ *x)
* \brief    The GSM operator is basically a vector addition operation, that can be performed independently for each semantic element STi,j.
            However, it is necessary to select the semantics of two random trees R u and R v , and a random mutation step ms.
* \param    float *initialPopulationSemantics: this vector of pointers contains the semantics of the initial population
* \param    float *randomTreesSemantics: this vector of pointers contains the semantics of the random trees
* \param    float *newSemanticsOffsprings: this vector of pointers will store the semantics of the new offspring
* \param    int sizePopulation: this variable contains the number of individuals that the population has
* \param    int nrow: variable containing the numbers of rows (instances) of the training dataset
* \param    int tElements: variables containing the total number of semantic elements
* \param    int generation: number of generation
* \param    float *indexRandomTrees: this pointer stores the indexes randomly for mutation
* \param    struc *x: variable used to store training and test instances 
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h 
*/
__global__ void geometricSemanticMutation(float *initialPopulationSemantics, float *randomTreesSemantics, float *newSemanticsOffsprings, int sizePopulation,
  int nrow, int tElements, int generation, float *indexRandomTrees, entry_ *y, int index, float *mutationStep, int sigmoid, int normalize);

/*!
* \fn       __host__ void saveTrace(entry *structSurvivor, int generation) 
* \brief    Function that stores the information related to the evolutionary cycle and stores the indices of the individuals that were used in each generation to create new offspring,
            to later perform the reconstruction of the optimal solution in the trace.txt file
* \param    struc *structSurvivor: pointer that stores information of the best individual throughout the generations
* \param    int generation: number of generations
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h  
*/
__host__ void saveTrace(std::string path, entry *x, int generation, int populationSize);

/*!
* \fn        __host__ void readInpuData(char *train_file, char *test_file, float *dataTrain, float *dataTest, float *dataTrainTarget,float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest)
* \brief    function that reads data from training and test files, also reads target values to store them in pointer vectors.
* \param    char *train_file: name of the file with training instances 
* \param    char *test_file: name of the file with test instances
* \param    float *dataTrain: vector pointers to store training data
* \param    float *dataTest: vector pointers to store test data
* \param    float *dataTrainTarget: vector pointers to store training target data
* \param    float *dataTestTarget: vector pointers to store test target data
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int nrowTest: variable containing the number of rows (instances) of the test dataset
* \param    int nvarTest: variable containing the number of columns (excluding the target) of the test dataset
* \return   void: 
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.h 
*/
__host__ void readInpuData(char *trainFile, char *testFile, float *dataTrain, float *dataTest, float *dataTrainTarget,
 float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest);

/*!
* \fn        __host__ void readConfigFile(string path,cfg *config)
* \brief     Function that reads the configuration file where the parameters are found to initialize the algorithm.
* \param     cfg *config: pointer to the struct containing the variables needed to run the program
* \return    void
* \date      01/25/2020
* \author    José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file      GsgpCuda.h
*/
__host__ void readConfigFile(string path,cfg *config);

/*!
* \fn        __host__ void countInputFile(std::string fileName, int &rows, int &cols)
* \brief     function that reads rows and colums of files to train and test
* \param     std::string fileName: This variable store the name of file data train or test
* \param     int rows: This variable store the number of rows of file data
* \param     int nvar: This variable store the number of colums of file data
* \return    void
* \date      05/10/2021
* \author    José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file      GsgpCuda.cpp
*/
void countInputFile(std::string fileName, int &rows, int &cols);

/*!
* \fn       __host__ void saveIndividuals(std::string path, float *Individuals, std::string namePopulation ,int maxDepth, int sizePopulation);
* \brief   This function stores the initial population and the auxiliary population of random trees.
* \param    std::string path: This vector pointers to store the individuals of the initial population.
* \param    float *Individuals: vector pointers to store the semantics of each individual in the population.
* \param    std::string namePopulation: This variable stores the name of the population.
* \param    int maxDepth: This variable thar stores maximum depth for individuals
* \param    int sizePopulation: This variable contains the number of individuals that exist in the population.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__host__ void saveIndividuals(std::string path, float *Individuals, std::string namePopulation ,int maxDepth, int sizePopulation);

/*!
* \fn       __host__ void test(float *individuals, int sizeMaxDepth, int sizePopulation);
* \brief    Function that tests the individuals of the population.
* \param    float *individuals: This vector pointers to store the individuals of the population.
* \param    int sizeMaxDepth: This variable stores the maximum depth of the individuals.
* \param    int sizePopulation: This variable contains the number of individuals that exist in the population.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
__host__ void test(float *individuals, int sizeMaxDepth, int sizePopulation);

/*!
* \fn       bool IsPathExist(const std::string &s)
* \brief    function to check if exist a directory path.
* \param    string &s: name of path to check if exist.
* \return   void
* \date     27/02/2021
* \author   Luis Armando Cardenas Florido, José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     CudaGP.cpp
*/
__host__  bool IsPathExist(const std::string &s);

/*!
* \fn       void checkDirectoryPath(string dirPath)
* \brief    function to check if exist a directory path, if not, create the directory path
* \param    string dirPath: name of path
* \return   void
* \date     27/02/2021
* \author   Luis Armando Cardenas Florido, José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__host__ void checkDirectoryPath(std::string dirPath);

/*!
* \fn       static void list_dir(std::string path, std::string nameFile, int useMultipleFiles, std::vector<string> &files)
* \brief    function for get directories files
* \param    string path: path where the algorithm read files needed to work
* \param    string name: nombre de los archivos a buscar en la ruta
* \param    int useMultipleFiles: variable to exit file search
* \param    vector &files: variable to store the names of the found files.
* \return   void
* \date     10/02/2021
* \author   Luis Armando Cardenas Florido, José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__host__ static void list_dir(std::string path, std::string nameFile, int useMultipleFiles, std::vector<string> &files);

/*!
* \fn       __host__ void markTracesGeneration(entry *vectorTraces, int populationSize, int generationSize ,int bestIndividual)
* \brief    This function that implements the marking procedure used to store the structure of the optimal solution
* \param    entry *vectorTraces: variable used to store the information needed to evaluate the optimal individual on newly provided unseen data.
* \param    int populationSize: Number of individuals in the population
* \param    int generationSize: number of generations
* \param    int bestIndividual: index of best individual of population
* \return   void
* \date     10/02/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__host__ void markTracesGeneration(entry *vectorTraces, int populationSize, int generationSize ,int bestIndividual);

/*!
* \fn       __host__ void saveTraceComplete(std::string path, entry *structSurvivor, int generation, int populationSize)
* \brief    Function that stores the information related to the evolutionary cycle and stores the indices of the individuals that were used in each generation to create new offspring,
            to later perform the reconstruction of the optimal solution in the trace.txt file
* \param    string path: path where the algorithm output files are stored.
* \param    struc *structSurvivor: pointer that stores information of the best individual throughout the generations
* \param    int generation: number of generations
* \param    int populationSize: Number of individuals in the population
* \return   void
* \date     08/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/
__host__ void saveTraceComplete(std::string path, entry *structSurvivor, int generation, int populationSize);

/*!
* \fn       __host__ void saveTrace(entry *structSurvivor, int generation) 
* \brief    Function that stores the information related to the evolutionary cycle and stores the indices of the individuals that were used in each generation to create new offspring,
            to later perform the reconstruction of the optimal solution in the trace.txt file
* \param    string path: path where the algorithm output files are stored.
* \param    struc *structSurvivor: pointer that stores information of the best individual throughout the generations
* \param    int generation: number of generations
* \param    int populationSize: : Number of individuals in the population
* \return   void
* \date     08/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/
__host__ void saveTrace(std::string name, std::string path, entry *structSurvivor, int generation, int populationSize);

/*!
* \fn        __host__ void readInpuTestData(char *train_file, char *test_file, float *dataTrain, float *dataTest, float *dataTrainTarget,float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest)
* \brief    This function that reads data from test file, also reads target values to store them in pointer vectors.
* \param    char *test_file: name of the file with test instances
* \param    float *dataTest: vector pointers to store test data
* \param    float *dataTestTarget: vector pointers to store test target data
* \param    int nrowTest: variable containing the number of rows (instances) of the test dataset
* \param    int nvarTest: variable containing the number of columns (excluding the target) of the test dataset
* \return   void: 
* \date     8/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__host__ void readInpuTestData( char *test_file, float *dataTest, float *dataTestTarget, int nrowTest, int nvarTest );

/*!
* \fn      __host__ void readPopulation( float *initialPopulation, float *randomTrees, int sizePopulation, int depth, std::string log, std::string name, std::string nameR)
* \brief    This function that read the information related to the initial population from file initialPopulation.csv
* \param    float *initialPopulation: This vector pointers to store the individuals of the initial population.
* \param    float *randomTrees: vector of pointers storing random trees
* \param    int sizePopulation: Number of individuals in the population
* \param    int depth: This variable thar stores maximum depth for individuals
* \param    std::string log: path where the algorithm output files are stored.
* \param    std::string name: name of file the initial population
* \param    std::string nameR: name of file the random trees
* \return   void: 
* \date     8/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__host__ void readPopulation( float *initialPopulation, float *randomTrees, int sizePopulation, int depth, std::string log, std::string name, std::string nameR);

/*!
* \fn       __host__ void evaluate_data(std::string path, int generations, float *initialPopulation, float *randomTrees, std::ofstream& OUT, std::string log, int nrow, int numIndi, int nvarTest);
* \brief    This function that evaluates the best model stored in trace.txt over newly provided unseen data
* \param    std::string path: trace file name
* \param    int generations: number of generations.
* \param    float *initialPopulation: vector of pointers storing the initial population
* \param    float *randomTrees: vector of pointers storing random trees
* \param    std::ofstream& OUT: file where the result of the evaluation of the best model with each fitness will be written.
* \param    std::string log: path where the algorithm output files are stored.
* \param    int sizePopulation: number of individuals in the population
* \param    const int depth: ariable that stores maximum depth for individuals.
* \param    int nrow: This variable contains the number of fitness cases.
* \param    int nvarTest:This variable contains the number of features of problem.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__host__ void evaluate_data(std::string path, int generations, float *initialPopulation, float *randomTrees, std::ofstream& OUT, std::string log, int nrow, int nvarTest);