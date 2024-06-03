// Codice CPU NON ottimizzato
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <fstream>
#include <chrono>
const int KERNEL_SIZE = 3;
const int COLS = 7680;
const int ROWS = 4320;


using namespace std::chrono;

void convolutionRows(const std::vector<std::vector<int>>& inputMatrix, const std::vector<std::vector<int>>& kernelMatrix, std::vector<std::vector<int>>& outputMatrix, int startRow, int endRow)
{
    int numRows = inputMatrix.size();
    int numCols = inputMatrix[0].size();
    int kernelSize = kernelMatrix.size();

    int outputColSize = numCols - kernelSize + 1; // 1080 - 2 + 1 --> 1079
    // std::cout<<"dentro la convolutionROws\n";

    // system("pause");

    for (int row = startRow; row < endRow; row++)  // row � [0,1919]
    {
        for (int col = 0; col < outputColSize; col++) // col � [0,1079]
        {
            int value = 0;
            for (int kRow = 0; kRow < kernelSize; kRow++) // kRow € [0,1]
            {
                for (int kCol = 0; kCol < kernelSize; kCol++) // kCol € [0,1]
                {
                    value += inputMatrix[row + kRow][col + kCol] * kernelMatrix[kRow][kCol];
                }
            }
            //std::cout << "rows: " << row << "\tcols: " << col << "\n";
            outputMatrix[row][col] = value;
        }
    }
}

// Funzione principale per la convoluzione utilizzando N_THREAD thread
void convolution(const std::vector<std::vector<int>>& inputMatrix, const std::vector<std::vector<int>>& kernelMatrix, std::vector<std::vector<int>>& outputMatrix, int N_THREAD) {


    // Alloca la matrice di output con la stessa dimensione della matrice di input
    outputMatrix.resize(inputMatrix.size() - kernelMatrix.size() + 1, std::vector<int>(inputMatrix[0].size() - kernelMatrix.size() + 1)); //

    // Calcola il numero di righe da elaborare per ogni thread
    int rowsPerThread = ROWS / N_THREAD;
    //std::cout <<"rows per thread:"<< rowsPerThread<<std::endl;
    // Creazione e avvio dei thread per eseguire la convoluzione
    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREAD; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == N_THREAD - 1) ? (ROWS - KERNEL_SIZE + 1) : (startRow + rowsPerThread);
        //std::cout << "startRow: " << startRow << " endRow: " << endRow << std::endl;
        //system("pause");
        threads.emplace_back(convolutionRows, std::ref(inputMatrix), std::ref(kernelMatrix), std::ref(outputMatrix), startRow, endRow);
    }

    // Attendere la fine di tutti i thread
    for (auto& thread : threads) {
        thread.join();
    }
}

void stampaMatrix(std::vector<std::vector<int>>& outputMatrix) {
    for (int i = 0; i < outputMatrix.size(); i++) {
        for (int j = 0; j < outputMatrix[0].size(); j++)
            std::cout << outputMatrix[i][j] << "\t";
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[])
{
    // Esempio di matrice di input
    int N_THREAD;
    if (argc > 1)
        N_THREAD = std::stoi(argv[1]);
    else {
        std::cout << "Inserire il numero di thread: ";
        std::cin >> N_THREAD;
    }



    // Inizializzazione del generatore di numeri casuali
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 9); // Valori compresi tra 0 e 9

    // Dichiarazione e inizializzazione della matrice con valori casuali
    std::vector<std::vector<int>> inputMatrix(ROWS, std::vector<int>(COLS));
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            inputMatrix[i][j] = 1;// distribution(gen);
        }
    }

    // Dichiarazione e inizializzazione della matrice del kernel con valori casuali
    std::vector<std::vector<int>> kernelMatrix(KERNEL_SIZE, std::vector<int>(KERNEL_SIZE));
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            kernelMatrix[i][j] = 1; // distribution(gen);
        }
    }

    // Matrice di output
    std::vector<std::vector<int>> outputMatrix;

    // Esegui la convoluzione
    auto begin = high_resolution_clock::now();
    convolution(inputMatrix, kernelMatrix, outputMatrix, N_THREAD);
    auto end = std::chrono::high_resolution_clock::now();

    //stampaMatrix(outputMatrix);



    auto elapsed_ms = duration_cast<microseconds>(end - begin);
   // std::cout << "Tempo di esecuzione: " << elapsed_ms.count() << " ns" << std::endl;
    std::string s = "8k_" + std::to_string(N_THREAD) + ".txt";
    std::cout << std::endl << elapsed_ms.count() << std::endl;
    std::ofstream fout(s, std::ios::app);
    fout << elapsed_ms.count() << "\n";
    fout.close();


    return 0;
}