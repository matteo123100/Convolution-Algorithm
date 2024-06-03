// Codice Ottimizzato Senza Affinity
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <fstream>
#include <chrono>
#include <string>

using namespace std::chrono;

const int KERNEL_SIZE = 3;
const int COLS = 4096;
const int ROWS = 2160;

int** inputMatrix = new int* [ROWS];
int** outputMatrix = new int* [ROWS - KERNEL_SIZE + 1];


// Funzione per eseguire la convoluzione di una porzione delle righe di una matrice
void convolutionRows(const int kernel[KERNEL_SIZE][KERNEL_SIZE], int startRow, int endRow)
{
    // Applica la convoluzione sulle righe specificate

    int outputRowSize = ROWS - KERNEL_SIZE + 1; // 1920 - 2 + 1 --> 1919
    int outputColSize = COLS - KERNEL_SIZE + 1; // 1080 - 2 + 1 --> 1079

    for (int row = startRow; row < endRow; row++)  // row � [0,1919]
    {
        for (int col = 0; col < outputColSize; col++) // col � [0,1079]
        {
            int value = 0;
            for (int kRow = 0; kRow < KERNEL_SIZE; kRow++) // kRow € [0,1]
            {
                for (int kCol = 0; kCol < KERNEL_SIZE; kCol++) // kCol € [0,1]
                {
                    value += inputMatrix[row + kRow][col + kCol] * kernel[kRow][kCol];
                }
            }
            //std::cout << "rows: " << row << "\tcols: " << col << "\n";
            outputMatrix[row][col] = value;
        }
    }
}

// Funzione principale per la convoluzione utilizzando thread
void convolution(const int kernel[KERNEL_SIZE][KERNEL_SIZE], int N_THREAD)
{
    // Calcola il numero di righe da elaborare per ogni thread
    int rowsPerThread = ROWS / N_THREAD; // 4 / 2 = 2

    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREAD; ++i) // 0 , 1
    {
        int startRow = i * rowsPerThread; // i:0 -> startRow = 0                                    | i:1 -> startRow = 1 * 2 = 2
        // i:0 -> endRow = startRow + rowsPerThread = 0 + 2 = 2   | i:1 -> endRow = 4 - 2 + 1 = 3

        int endRow = (i == N_THREAD - 1) ? (ROWS - KERNEL_SIZE + 1) : (startRow + rowsPerThread);

        //std::cout << "startRow: " << startRow << " endRow: " << endRow << std::endl;
        threads.emplace_back(convolutionRows, std::cref(kernel), startRow, endRow);
    }

    // Attendere la fine di tutti i thread
    for (auto& thread : threads) {
        thread.join();
    }
}

void stampaMatrix() {
    for (int i = 0; i < ROWS - KERNEL_SIZE + 1; i++) {
        for (int j = 0; j < COLS - KERNEL_SIZE + 1; j++)
            std::cout << outputMatrix[i][j] << "\t";
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[])
{
    for (int i = 0; i < ROWS; ++i)
    {
        inputMatrix[i] = new int[COLS]; // Alloca un array di interi per ogni riga
        if (inputMatrix[i] == nullptr)
        {
            std::cout << i << "dimensione insufficiente\n";
            return 0;
        }
    }

    int colOutputSize = COLS - KERNEL_SIZE + 1;
    for (int i = 0; i < colOutputSize; ++i)
    {
        outputMatrix[i] = new int[colOutputSize]; // Alloca un array di interi per ogni riga
        if (outputMatrix[i] == nullptr)
        {
            std::cout << i << "dimensione insufficiente\n";
            return 0;
        }
    }

    int N_THREAD = 8;
    if (argc > 1)
        N_THREAD = std::stoi(argv[1]);
    else {
        std::cout << "Inserire il numero di thread: ";
        std::cin >> N_THREAD;
    }

    /*
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 9); // Valori compresi tra 0 e 9
    */

    // ---------------------------------  INPUT MATRIX INITIALIZATION ---------------------------------
    int initialValue = 1;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            inputMatrix[i][j] = initialValue; //distribution(gen);
            initialValue++;
            //std::cout << inputMatrix[i][j] << " ";
        }
        //std::cout << std::endl;
    }
    // ------------------------------------------------------------------------------------------------

    // --------------------------------- KERNEL MATRIX INITIALIZATION ---------------------------------
    initialValue = 1;
    int kernel[KERNEL_SIZE][KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; ++i)
    {
        for (int j = 0; j < KERNEL_SIZE; ++j)
        {
            kernel[i][j] = initialValue; // distribution(gen);
            initialValue++;
            //std::cout << kernel[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // ------------------------------------------------------------------------------------------------


    auto begin = high_resolution_clock::now(); // ***** TIMER START

    convolution(kernel, N_THREAD);

    auto end = high_resolution_clock::now();   // ***** TIMER STOP

    //stampaMatrix();

    // ----------------------------------------- WRITE DATA ON FILE -------------------------------------

    auto elapsed_ms = duration_cast<microseconds>(end - begin);
    std::cout << "Tempo di esecuzione: " << elapsed_ms.count() << " micros" << std::endl;

    std::string s = "4k_" + std::to_string(N_THREAD) + ".txt";
    std::ofstream fout(s, std::ios::app);
    fout << elapsed_ms.count() << "\n";
    fout.close();
    // ---------------------------------------------------------------------------------------------------
    return 0;
}