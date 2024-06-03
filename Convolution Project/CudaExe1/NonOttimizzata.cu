/*
// Codice NON Ottimizzato
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <fstream>

#define COLS 7680  //7680
#define ROWS 4320  //4320
#define KERNEL_SIZE 3
#define OUTPUT_ROWS (ROWS - KERNEL_SIZE + 1)
#define OUTPUT_COLS (COLS - KERNEL_SIZE + 1)

void CPU_convolutionRows(int, int);
void CPU_convolution(int);
int* CPU_inputMatrix = new int[ROWS * COLS];
int* CPU_kernel = new int[KERNEL_SIZE * KERNEL_SIZE];
int* CPU_outputMatrix = new int[(ROWS - KERNEL_SIZE + 1) * (COLS - KERNEL_SIZE + 1)];
void init_matrix(int*, int, int);
void stampaMatrix(int*, int, int);

__device__ __shared__ int* d_input, * d_output, * d_kernel;
//__constant__  int d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void convolution2DKernel(int* input, int* output, int* d_kernel)
{
    int IDrow = blockIdx.x * blockDim.x + threadIdx.x;
    int IDcol = blockIdx.y * blockDim.y + threadIdx.y;

    if ((IDcol < OUTPUT_COLS) && (IDrow < OUTPUT_ROWS)) // Questo if va utilizzato quando utilizzi la griglia Bidimensionale
    {
        int result = 0;
        for (int i = 0; i < KERNEL_SIZE; i++)
        {
            //int passo = (i != 0) ? (COLS - KERNEL_SIZE) * i : 0;
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
                //int indice = IDrow + passo; //int indice = (IDrow + i) * COLS + IDcol + j; // questo indice va utilizzato quando utilizzi la griglia Bidimensionale
                int indice = (IDrow + i) * COLS + IDcol + j; // questo indice va utilizzato quando utilizzi la griglia Bidimensionale
                result += input[indice] * d_kernel[i * KERNEL_SIZE + j];
            }
        }
        output[IDrow * OUTPUT_COLS + IDcol] = result; // Questo accesso va utilizzanto quando utilizzi la griglia Bidimensionale
        //printf("Scritto la out[%d]: %d\n", IDrow * OUTPUT_COLS + IDcol, output[IDrow]);
    }
}

int main()
{
    int* input = new int[ROWS * COLS];
    int* kernel = new int[KERNEL_SIZE * KERNEL_SIZE];
    int* output = new int[OUTPUT_ROWS * OUTPUT_COLS];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    init_matrix(input, ROWS, COLS);
    init_matrix(kernel, KERNEL_SIZE, KERNEL_SIZE);



    //printf("array:input \n");
    //stampaMatrix(input, ROWS, COLS);
    //printf("\narray:kernel \n");
    //stampaMatrix(kernel, KERNEL_SIZE, KERNEL_SIZE);


    //int* d_kernel;


    cudaMalloc((void**)&d_input, ROWS * COLS * sizeof(int));
    cudaMalloc((void**)&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int));
    cudaMalloc((void**)&d_output, OUTPUT_COLS * OUTPUT_ROWS * sizeof(int));
    cudaMemcpy(d_input, input, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int));

    dim3 blockSize(32, 32);
    //dim3 blockSize(1024, 1);

    int gridSizeX = (OUTPUT_COLS / blockSize.x) != 0 ? std::ceil(((float)OUTPUT_COLS / (float)blockSize.x)) : 1;
    int gridSizeY = (OUTPUT_ROWS / blockSize.y) != 0 ? std::ceil(((float)OUTPUT_ROWS / (float)blockSize.y)) : 1;


    //int gridSizeOnlyXAxis = std::ceil((OUTPUT_COLS * OUTPUT_ROWS) / blockSize.x + 0.5); //int gridSizeX = (OUTPUT_COLS / blockSize.x) != 0 ? std::ceil((OUTPUT_COLS / blockSize.x + 0.5) ) : 1;
    //int gridSizeYOnlyOne = 1;                                                                 //int gridSizeY = (OUTPUT_ROWS / blockSize.y) != 0 ? std::ceil((OUTPUT_ROWS / blockSize.y + 0.5) ) : 1;

    //int gridSizeX = 1;
    //int gridSizeY = 1;

    dim3 gridSize(gridSizeX, gridSizeY);

    printf("Griglia -> gridSizeX: %d , gridSizeY: %d\n", gridSize.x, gridSize.y);
    //printf("outputMatrixSize: %dx%d\n", OUTPUT_ROWS, OUTPUT_COLS);

    cudaEventRecord(start, 0);
    convolution2DKernel << <gridSize, blockSize >> > (d_input, d_output, d_kernel);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(output, d_output, OUTPUT_ROWS * OUTPUT_COLS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    //printf("\nTempo impiegato %f\n", elapsedTime);
    std::string s = "8k_Block(" + std::to_string(blockSize.x) + "," + std::to_string(blockSize.y) + ").txt";
    std::ofstream fout(s, std::ios::app);
    fout << elapsedTime << "\n";
    fout.close();

    printf("OUTPUT:\n");
    //stampaMatrix(output, OUTPUT_ROWS, OUTPUT_COLS);

    // Fine parte GPU

    // IMplementa la parte per CPU e poi fai il confronto


    return 0;
}

void init_matrix(int* m, int rows, int cols)
{
    int valore = 1;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            m[cols * i + j] = valore;
            valore++;
        }
    }
}

void stampaMatrix(int* outputMatrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            std::cout << outputMatrix[i * cols + j] << "\t";
        std::cout << std::endl;
    }
}

void CPU_convolutionRows(int startRow, int endRow) //DWORD WINAPI convolutionRows(LPVOID lpParam)//(int startRow, int endRow)
{
    //auto params = reinterpret_cast<std::pair<int, int>*>(lpParam);
    //int startRow = params->first;
    //int endRow = params->second;
    int outputColSize = COLS - KERNEL_SIZE + 1; // 1080 - 2 + 1 --> 1079
    int value;

    for (int row = startRow; row < endRow; row++)  // row � [0,1919]
    {
        for (int col = 0; col < outputColSize; col++) // col � [0,1079]
        {
            value = 0;
            for (int kRow = 0; kRow < KERNEL_SIZE; kRow++) // kRow € [0,1]
            {
                for (int kCol = 0; kCol < KERNEL_SIZE; kCol++) // kCol € [0,1]
                {
                    value += CPU_inputMatrix[(row + kRow) * COLS + (col + kCol)] * CPU_kernel[kRow * KERNEL_SIZE + kCol];
                }
            }
            CPU_outputMatrix[row * outputColSize + col] = value;
        }
    }
}

void CPU_convolution(int N_THREAD)
{
    int rowsPerThread = ROWS / N_THREAD;
    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREAD; ++i)
    {
        int startRow = i * rowsPerThread;

        int endRow = (i == N_THREAD - 1) ? (ROWS - KERNEL_SIZE + 1) : (startRow + rowsPerThread);

        auto params = new std::pair<int, int>(startRow, endRow);
        //threadParams.push_back(*params);

        // Creazione del thread sospeso
        //HANDLE hThread = CreateThread(nullptr, 0, convolutionRows, params, CREATE_SUSPENDED, nullptr);
        //threadHandles.push_back(hThread);

        threads.emplace_back(CPU_convolutionRows, startRow, endRow);
    }

    //begin = steady_clock::now(); // ***** TIMER START
    for (auto& thread : threads)
    {
        thread.join();
    }
    //end = steady_clock::now();
}
*/