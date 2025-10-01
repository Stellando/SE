#include <iostream>
#include <vector>
#include "algorithm.h"
#include "functions.h"
using namespace std;

int main(int argc, char *argv[]) {

    // OneMax 問題的參數設定
    int D = 100;        // OneMax 問題維度 (二進位字串長度)
    int NP = 4;        // 搜尋者數量 (論文建議值)
    int G = 500;       // 最大迭代次數
    double pb = 0.05;  // 保留參數 (SE 中未使用)
    double c = 0.1;    // 保留參數 (SE 中未使用)
    int maxVal = 1;    // OneMax 的值域是 [0,1]
    int func_num = 1;  // 函數編號 (OneMax)

    cout << "=== Search Economics Algorithm for OneMax Problem ===" << endl;
    cout << "Initializing parameters:" << endl;
    cout << "Problem Dimension (D): " << D << endl;
    cout << "Number of Searchers (NP): " << NP << endl;
    cout << "Max Iterations (G): " << G << endl;
    cout << "Target: Find binary string with maximum number of 1s" << endl;
    cout << "Optimal solution: " << D << " ones (fitness = " << D << ")" << endl;
    cout << "======================================================" << endl << endl;

    // 執行 Search Economics 演算法
    algorithm alg;
    alg.RunALG(D, NP, G, pb, c, maxVal, func_num);
    
    // 取得結果
    int idx;
    double bestFitness = alg.get_best_fitness(idx);
    vector<double> bestPosition = alg.get_best_position();
    
    cout << endl << "=== Final Results ===" << endl;
    cout << "Best fitness: " << bestFitness << "/" << D << endl;
    cout << "Success rate: " << (bestFitness / D * 100) << "%" << endl;
    
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
    
    if (bestFitness == D) {
        cout << "🎉 SUCCESS: Found optimal solution!" << endl;
    } else {
        cout << "📊 Partial solution found. " << (D - bestFitness) << " bits away from optimal." << endl;
    }

    system("pause");
    return 0;
}