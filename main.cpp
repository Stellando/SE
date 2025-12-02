#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include "algorithm.h"
using namespace std;

int main(int argc, char *argv[]) {

    // Search Economics 演算法參數設定
    int dimension = 1000;      // OneMax 問題維度 (二進位字串長度)
    int numSearchers = 4;     // 搜尋者數量 (論文建議值 n)
    int numRegions = 4;       // 區域數量 (論文建議值 h)
    int maxIterations = 1000;  // 最大迭代次數
    double minVal = 0.0;      // OneMax 的值域下界
    double maxVal = 1.0;      // OneMax 的值域上界 
    int funcNum = 1;          // 函數編號 (OneMax)
    int runtime = 1;         // 執行次數

    // 用來儲存所有回合的平均收斂數據
    vector<double> averageConvergence(maxIterations, 0.0);

    cout << "=== Search Economics Algorithm for OneMax Problem ===" << endl;
    cout << "Initializing parameters:" << endl;
    cout << "Problem Dimension: " << dimension << endl;
    cout << "Number of Searchers (n): " << numSearchers << endl;
    cout << "Number of Regions (h): " << numRegions << endl;
    cout << "Max Iterations: " << maxIterations << endl;
    cout << "Runtime (Independent Runs): " << runtime << endl;
    cout << "Value Range: [" << minVal << ", " << maxVal << "]" << endl;
    cout << "Target: Find binary string with maximum number of 1s" << endl;
    cout << "Optimal solution: " << dimension << " ones (fitness = " << dimension << ")" << endl;
    cout << "======================================================" << endl << endl;

    // 執行多次 SE 演算法
    for (int r = 0; r < runtime; r++) {
        cout << "\n[Run " << (r + 1) << "/" << runtime << "]" << endl;
        
        algorithm alg;
        alg.RunALG(dimension, numSearchers, maxIterations, (int)maxVal, funcNum);
        
        // 取得這次執行的歷史記錄
        vector<double> history = alg.get_fitness_history();
        
        // 累加到平均值（history 應該已經填滿到 maxIterations）
        for (int i = 0; i < maxIterations && i < history.size(); i++) {
            averageConvergence[i] += history[i];
        }
        
        // 如果 history 長度不足（理論上不應該發生），用最後一個值填充
        if (history.size() < maxIterations && history.size() > 0) {
            double lastFitness = history.back();
            for (int i = history.size(); i < maxIterations; i++) {
                averageConvergence[i] += lastFitness;
            }
        }
        
        // 顯示這次執行的結果
        int idx;
        double bestFitness = alg.get_best_fitness(idx);
        cout << "Run " << (r + 1) << " best fitness: " << bestFitness << "/" << dimension << endl;
    }
    
    // 計算平均值
    for (int i = 0; i < maxIterations; i++) {
        averageConvergence[i] /= runtime;
    }
    
    // 寫入 TXT 檔案
    string filename = "results" + to_string(dimension) + "bits.txt";
    ofstream outFile(filename);
    
    if (outFile.is_open()) {
        for (int i = 0; i < maxIterations; i++) {
            outFile << (i + 1) << " " << fixed << setprecision(4) << averageConvergence[i] << endl;
        }
        outFile.close();
        cout << "\n=== Results Summary ===" << endl;
        cout << "Successfully wrote average convergence data to " << filename << endl;
        cout << "Total runs: " << runtime << endl;
        cout << "Final average best fitness: " << averageConvergence[maxIterations - 1] << "/" << dimension << endl;
        cout << "Average success rate: " << (averageConvergence[maxIterations - 1] / dimension * 100) << "%" << endl;
    } else {
        cout << "Error: Unable to write to file " << filename << endl;
    }

    system("pause");
    return 0;
}