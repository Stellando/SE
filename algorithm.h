#include <vector>
#include <functional>
#include <random>
#include <ctime>
#include "se_structures.h"

#ifndef ALGORITHM_H
#define ALGORITHM_H

using namespace std;

// Search Economics 演算法主類別
class SearchEconomicsAlgorithm 
{
public:
    // 建構函數
    SearchEconomicsAlgorithm();
    SearchEconomicsAlgorithm(const SEParameters& params);
    
    // 主要執行函數
    void RunSE(int dimension, int numSearchers, int numRegions, int maxIterations, 
               double minVal, double maxVal, int funcNum);
    
    // 取得結果
    double getBestFitness() const;
    vector<double> getBestPosition() const;
    Solution getGlobalBest() const;
    SEStatistics getStatistics() const;
    vector<double> getFitnessHistory() const;
    
private:
    // SE 核心組件
    SEParameters parameters;
    vector<Region> regions;
    vector<Searcher> searchers;
    SEStatistics statistics;
    mt19937 randomGenerator;
    int functionNumber;
    vector<double> fitnessHistory;
    
    // === 三大主要階段 ===
    
    // 1. Resource Arrangement (RA) - 資源配置
    void resourceArrangement();
    void initializeRegions();
    void initializeSearchers();
    void distributeSearchersToRegions();
    
    // 2. Vision Search (VS) - 視覺搜尋
    void visionSearch();
    void transition();      // 轉換：產生暫時候選解 v_ijk
    void evaluation();      // 評估：計算期望值 e_ij
    void determination();   // 決策：選擇下一個投資區域
    
    // 3. Marketing Research (MR) - 市場研究
    void marketingResearch();
    void updateGoods();     // 更新goods和區域資訊
    void accumulateStats(); // 累積統計資料 ta, tb
    void feedbackToVS();    // 回饋給Vision Search
    
    // === 輔助函數 ===
    
    // 初始化相關
    Solution generateRandomSolution(int dimension);
    Solution generateRandomSolutionForRegion(int dimension, int regionIdx);
    void evaluateSolution(Solution& solution);
    
    // 交叉突變操作
    Solution crossover(const Solution& parent1, const Solution& parent2);
    Solution mutation(const Solution& solution);
    
    // 期望值計算 (核心公式)
    double calculateExpectedValue(int searcherIdx, int regionIdx);
    double calculateF1(const Region& region);  // f1(M_j) = tb_j / ta_j
    double calculateF2(int searcherIdx, int regionIdx);  // f2(V_ij) 暫候選平均
    double calculateF3(int regionIdx);         // f3(ρ_j) 區域相對權重
    
    // 區域管理
    void updateRegionBest(int regionIdx);
    void updateRegionStatistics();
    int selectRegionByExpectedValue(int searcherIdx);
    
    // 統計與記錄
    void updateGlobalStatistics();
    void recordCurrentIteration();
    bool shouldTerminate();
    
    // 工具函數
    double evaluateFunction(const vector<double>& position, int funcNum);
    void printProgress(int iteration);
    void printFinalResults();
};

//原有的介面 (向後相容)
class algorithm 
{
public:
    void RunALG(int dimension, int numSearchers, int maxIterations, int maxVal, int funcNum);
    double get_best_fitness(int& best_idx) const;
    vector<double> get_best_position() const;
    vector<double> get_fitness_history() const;
    
private:
    SearchEconomicsAlgorithm seAlgorithm;
    Solution bestSolution;
    double bestFitness;
};

#endif // ALGORITHM_H