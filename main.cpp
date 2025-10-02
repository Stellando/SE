#include <iostream>
#include <vector>
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


    cout << "=== Search Economics Algorithm for OneMax Problem ===" << endl;
    cout << "Initializing parameters:" << endl;
    cout << "Problem Dimension: " << dimension << endl;
    cout << "Number of Searchers (n): " << numSearchers << endl;
    cout << "Number of Regions (h): " << numRegions << endl;
    cout << "Max Iterations: " << maxIterations << endl;
    cout << "Value Range: [" << minVal << ", " << maxVal << "]" << endl;
    cout << "Target: Find binary string with maximum number of 1s" << endl;
    cout << "Optimal solution: " << dimension << " ones (fitness = " << dimension << ")" << endl;
    cout << "======================================================" << endl << endl;

    // 執行 Search Economics 演算法
    algorithm alg;
    alg.RunALG(dimension, numSearchers, maxIterations, (int)maxVal, funcNum);
    
    // 取得結果
    int idx;
    double bestFitness = alg.get_best_fitness(idx);
    vector<double> bestPosition = alg.get_best_position();
    
    cout << endl << "=== Final Results ===" << endl;
    cout << "Best fitness: " << bestFitness << "/" << dimension << endl;
    cout << "Success rate: " << (bestFitness / dimension * 100) << "%" << endl;
    
    cout << "Best solution (binary string): ";
    for (double val : bestPosition) {
        cout << (int)val;
    }
    cout << endl;
    
    // 驗證結果
    int ones_count = 0;
    for (double val : bestPosition) {
        if ((int)val == 1) ones_count++;
    }
    cout << "Verification - Ones count: " << ones_count << endl;
    
    if (bestFitness == dimension) {
        cout << "🎉 SUCCESS: Found optimal solution!" << endl;
    } else {
        cout << "📊 Partial solution found. " << (dimension - bestFitness) << " bits away from optimal." << endl;
    }

    system("pause");
    return 0;
}