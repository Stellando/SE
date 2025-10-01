#include <ctime>
#include <cstdlib>
#include "algorithm.h"
#include "functions.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>

using namespace std;

// ==================== SearchEconomicsAlgorithm 實作 ====================

SearchEconomicsAlgorithm::SearchEconomicsAlgorithm() {
    random_device rd;
    randomGenerator.seed(rd());
}

SearchEconomicsAlgorithm::SearchEconomicsAlgorithm(const SEParameters& params) 
    : parameters(params) {
    random_device rd;
    randomGenerator.seed(rd());
}

void SearchEconomicsAlgorithm::RunSE(int dimension, int numSearchers, int numRegions, 
                                     int maxIterations, double minVal, double maxVal, int funcNum) {
    // 設定參數
    parameters.dimension = dimension;
    parameters.numSearchers = numSearchers;
    parameters.numRegions = numRegions;
    parameters.maxIterations = maxIterations;
    parameters.minValue = minVal;
    parameters.maxValue = maxVal;
    parameters.isBinaryProblem = true; // 專注於 OneMax
    functionNumber = funcNum;
    
    cout << "Starting Search Economics Algorithm for OneMax problem..." << endl;
    cout << "Dimension: " << dimension << ", Searchers: " << numSearchers 
         << ", Regions: " << numRegions << ", Max Iterations: " << maxIterations << endl;
    
    // 初始化
    regions.clear();
    searchers.clear();
    statistics = SEStatistics();
    
    // Resource Arrangement (RA)
    resourceArrangement();
    
    // 主要迭代循環
    for (int iteration = 0; iteration < maxIterations; iteration++) {
        // Vision Search (VS)
        visionSearch();
        
        // Marketing Research (MR)
        marketingResearch();
        
        // 更新統計資料
        updateGlobalStatistics();
        
        if (iteration % 100 == 0) {
            printProgress(iteration);
        }
        
        // 檢查終止條件
        if (shouldTerminate()) {
            cout << "Early termination at iteration " << iteration << endl;
            break;
        }
    }
    
    printFinalResults();
}

// Resource Arrangement (RA) 實作
void SearchEconomicsAlgorithm::resourceArrangement() {
    cout << "Resource Arrangement phase..." << endl;
    
    // 初始化區域
    initializeRegions();
    
    // 初始化搜尋者
    initializeSearchers();
    
    // 分配搜尋者到區域
    distributeSearchersToRegions();
}

void SearchEconomicsAlgorithm::initializeRegions() {
    regions.resize(parameters.numRegions);
    
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j] = Region(j, parameters.goodsPerRegion);
        
        // 為每個區域產生 w 個隨機 goods
        for (int k = 0; k < parameters.goodsPerRegion; k++) {
            Solution good = generateRandomSolution(parameters.dimension);
            evaluateSolution(good);
            regions[j].addGood(good);
        }
        
        // 更新區域最佳解
        regions[j].updateRegionBest();
    }
}

void SearchEconomicsAlgorithm::initializeSearchers() {
    searchers.resize(parameters.numSearchers);
    
    for (int i = 0; i < parameters.numSearchers; i++) {
        searchers[i] = Searcher(i, parameters.dimension, parameters.numRegions);
        searchers[i].investment = generateRandomSolution(parameters.dimension);
        evaluateSolution(searchers[i].investment);
        
        // 初始化暫時候選解容器
        searchers[i].temporaryCandidates.resize(parameters.numRegions);
        for (int j = 0; j < parameters.numRegions; j++) {
            searchers[i].temporaryCandidates[j].resize(parameters.goodsPerRegion);
        }
    }
}

void SearchEconomicsAlgorithm::distributeSearchersToRegions() {
    // 照順序分配 searcher 到 region
    for (int i = 0; i < parameters.numSearchers; i++) {
        searchers[i].currentRegion = i % parameters.numRegions;
    }
    
    cout << "Searchers distributed to regions:" << endl;
    for (int i = 0; i < parameters.numSearchers; i++) {
        cout << "Searcher " << i << " -> Region " << searchers[i].currentRegion << endl;
    }
}

// Vision Search (VS) 實作
void SearchEconomicsAlgorithm::visionSearch() {
    // Transition
    transition();
    
    // Evaluation
    evaluation();
    
    // Determination
    determination();
}

void SearchEconomicsAlgorithm::transition() {
    // 對每個 searcher 與每個 region 的每個 good 做 crossover/mutation
    for (int i = 0; i < parameters.numSearchers; i++) {
        for (int j = 0; j < parameters.numRegions; j++) {
            for (int k = 0; k < parameters.goodsPerRegion; k++) {
                if (k < regions[j].goods.size()) {
                    // v_ijk = si ⊗ mjk (crossover + mutation)
                    Solution crossoverResult = crossover(searchers[i].investment, regions[j].goods[k]);
                    Solution mutationResult = mutation(crossoverResult);
                    evaluateSolution(mutationResult);
                    
                    searchers[i].temporaryCandidates[j][k] = mutationResult;
                    statistics.totalEvaluations++;
                }
            }
        }
    }
}

void SearchEconomicsAlgorithm::evaluation() {
    // 計算每個 searcher 對每個 region 的期望值 e_ij
    for (int i = 0; i < parameters.numSearchers; i++) {
        for (int j = 0; j < parameters.numRegions; j++) {
            searchers[i].expectedValues[j] = calculateExpectedValue(i, j);
        }
    }
}

void SearchEconomicsAlgorithm::determination() {
    for (int i = 0; i < parameters.numSearchers; i++) {
        vector<Solution> tournamentCandidates;
        
        // 加入自己區域的候選解 v_ii
        int currentRegion = searchers[i].currentRegion;
        for (int k = 0; k < parameters.goodsPerRegion; k++) {
            if (k < searchers[i].temporaryCandidates[currentRegion].size()) {
                tournamentCandidates.push_back(searchers[i].temporaryCandidates[currentRegion][k]);
            }
        }
        
        // 隨機選擇其他區域的候選解參與 tournament
        uniform_real_distribution<double> prob(0.0, 1.0);
        for (int j = 0; j < parameters.numRegions; j++) {
            if (j != currentRegion && prob(randomGenerator) < parameters.randomSelectionRate) {
                for (int k = 0; k < parameters.goodsPerRegion; k++) {
                    if (k < searchers[i].temporaryCandidates[j].size()) {
                        tournamentCandidates.push_back(searchers[i].temporaryCandidates[j][k]);
                    }
                }
            }
        }
        
        // Tournament selection：選擇最好的候選解
        if (!tournamentCandidates.empty()) {
            Solution best = tournamentCandidates[0];
            for (const Solution& candidate : tournamentCandidates) {
                if (candidate.fitness > best.fitness) {
                    best = candidate;
                }
            }
            
            // 如果找到更好的解，更新投資
            if (best.fitness > searchers[i].investment.fitness) {
                searchers[i].investment = best;
                
                // 更新當前區域
                searchers[i].currentRegion = best.getRegionByOnesCount(parameters.numRegions);
            }
        }
    }
}

// Marketing Research (MR) 實作
void SearchEconomicsAlgorithm::marketingResearch() {
    updateGoods();
    accumulateStats();
    feedbackToVS();
}

void SearchEconomicsAlgorithm::updateGoods() {
    // 更新各區域的 goods，替換較差的
    for (int i = 0; i < parameters.numSearchers; i++) {
        for (int j = 0; j < parameters.numRegions; j++) {
            for (int k = 0; k < parameters.goodsPerRegion; k++) {
                if (k < searchers[i].temporaryCandidates[j].size()) {
                    Solution& candidate = searchers[i].temporaryCandidates[j][k];
                    
                    // 檢查是否屬於這個區域
                    if (regions[j].belongsToRegion(candidate, parameters.numRegions)) {
                        regions[j].addGood(candidate);
                    }
                }
            }
        }
    }
    
    // 更新所有區域的最佳解
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j].updateRegionBest();
    }
}

void SearchEconomicsAlgorithm::accumulateStats() {
    // 重設所有區域的 tb (未被搜尋次數)
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j].tb++;
    }
    
    // 更新被搜尋區域的統計
    for (int i = 0; i < parameters.numSearchers; i++) {
        int targetRegion = searchers[i].currentRegion;
        if (targetRegion >= 0 && targetRegion < parameters.numRegions) {
            regions[targetRegion].ta++;     // 增加被投資次數
            regions[targetRegion].tb = 1;   // 重設未被搜尋次數
        }
    }
    
    // 更新 f1 值
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j].calculateF1();
    }
}

void SearchEconomicsAlgorithm::feedbackToVS() {
    // 更新區域統計以回饋給下一輪的 Vision Search
    updateRegionStatistics();
}

// 期望值計算函數
double SearchEconomicsAlgorithm::calculateExpectedValue(int searcherIdx, int regionIdx) {
    double f1 = calculateF1(regions[regionIdx]);
    double f2 = calculateF2(searcherIdx, regionIdx);
    double f3 = calculateF3(regionIdx);
    
    return f1 * f2 * f3;
}

double SearchEconomicsAlgorithm::calculateF1(const Region& region) {
    // f1(M_j) = tb_j / ta_j
    return static_cast<double>(region.tb) / region.ta;
}

double SearchEconomicsAlgorithm::calculateF2(int searcherIdx, int regionIdx) {
    // f2(V_ij) = (1/w) * Σf(v_ijk) - 暫時候選解的平均適應度
    double sum = 0.0;
    int count = 0;
    
    for (int k = 0; k < parameters.goodsPerRegion; k++) {
        if (k < searchers[searcherIdx].temporaryCandidates[regionIdx].size()) {
            sum += searchers[searcherIdx].temporaryCandidates[regionIdx][k].fitness;
            count++;
        }
    }
    
    return (count > 0) ? (sum / count) : 0.0;
}

double SearchEconomicsAlgorithm::calculateF3(int regionIdx) {
    // f3(ρ_j) = f(rb_j) / Σf(rb_k) - 區域最佳解的相對權重
    double totalBestFitness = 0.0;
    for (int k = 0; k < parameters.numRegions; k++) {
        totalBestFitness += regions[k].regionBest.fitness;
    }
    
    if (totalBestFitness > 0) {
        return regions[regionIdx].regionBest.fitness / totalBestFitness;
    }
    return 1.0 / parameters.numRegions; // 均等權重
}

// 輔助函數實作
Solution SearchEconomicsAlgorithm::generateRandomSolution(int dimension) {
    Solution solution(dimension, true); // OneMax 是二進位問題
    
    uniform_int_distribution<int> bitDist(0, 1);
    for (int i = 0; i < dimension; i++) {
        solution.binaryPosition[i] = bitDist(randomGenerator);
    }
    
    return solution;
}

void SearchEconomicsAlgorithm::evaluateSolution(Solution& solution) {
    if (solution.isBinary) {
        // OneMax: 適應度 = 1 的個數
        solution.fitness = solution.countOnes();
    } else {
        // 其他函數 (暫不實作)
        solution.fitness = 0.0;
    }
}

Solution SearchEconomicsAlgorithm::crossover(const Solution& parent1, const Solution& parent2) {
    Solution offspring(parent1.dimension, parent1.isBinary);
    
    if (parent1.isBinary && parent2.isBinary) {
        // 單點交叉
        uniform_int_distribution<int> crossoverPoint(1, parent1.dimension - 1);
        int point = crossoverPoint(randomGenerator);
        
        for (int i = 0; i < parent1.dimension; i++) {
            if (i < point) {
                offspring.binaryPosition[i] = parent1.binaryPosition[i];
            } else {
                offspring.binaryPosition[i] = parent2.binaryPosition[i];
            }
        }
    }
    
    return offspring;
}

Solution SearchEconomicsAlgorithm::mutation(const Solution& solution) {
    Solution mutated = solution;
    
    if (solution.isBinary) {
        // Bit-flip mutation
        uniform_real_distribution<double> mutProb(0.0, 1.0);
        for (int i = 0; i < solution.dimension; i++) {
            if (mutProb(randomGenerator) < parameters.mutationRate) {
                mutated.binaryPosition[i] = 1 - mutated.binaryPosition[i]; // flip bit
            }
        }
    }
    
    return mutated;
}

// 統計相關函數
void SearchEconomicsAlgorithm::updateGlobalStatistics() {
    // 找到所有 searcher 中的最佳解
    Solution currentBest;
    bool foundBest = false;
    
    for (int i = 0; i < parameters.numSearchers; i++) {
        if (!foundBest || searchers[i].investment.fitness > currentBest.fitness) {
            currentBest = searchers[i].investment;
            foundBest = true;
        }
    }
    
    // 檢查各區域的最佳解
    for (int j = 0; j < parameters.numRegions; j++) {
        if (!foundBest || regions[j].regionBest.fitness > currentBest.fitness) {
            currentBest = regions[j].regionBest;
            foundBest = true;
        }
    }
    
    if (foundBest) {
        statistics.updateGlobalBest(currentBest);
    }
    
    statistics.recordIteration();
}

void SearchEconomicsAlgorithm::updateRegionStatistics() {
    statistics.regionSearchCounts.resize(parameters.numRegions, 0);
    statistics.regionBestFitness.resize(parameters.numRegions, 0.0);
    
    for (int j = 0; j < parameters.numRegions; j++) {
        statistics.regionSearchCounts[j] = regions[j].ta;
        statistics.regionBestFitness[j] = regions[j].regionBest.fitness;
    }
}

bool SearchEconomicsAlgorithm::shouldTerminate() {
    // 如果找到最優解 (所有位元都是1)，提早結束
    return (statistics.globalBest.fitness >= parameters.dimension);
}

void SearchEconomicsAlgorithm::printProgress(int iteration) {
    cout << "Iteration " << iteration 
         << " | Best fitness: " << statistics.globalBest.fitness 
         << "/" << parameters.dimension;
    
    if (statistics.globalBest.isBinary) {
        cout << " | Ones: " << statistics.globalBest.countOnes();
    }
    cout << endl;
}

void SearchEconomicsAlgorithm::printFinalResults() {
    cout << "\n=== Search Economics Algorithm Results ===" << endl;
    cout << "Best fitness: " << statistics.globalBest.fitness << "/" << parameters.dimension << endl;
    cout << "Total evaluations: " << statistics.totalEvaluations << endl;
    cout << "Final iterations: " << statistics.currentIteration << endl;
    
    if (statistics.globalBest.isBinary) {
        cout << "Best solution: ";
        for (int bit : statistics.globalBest.binaryPosition) {
            cout << bit;
        }
        cout << endl;
    }
    
    cout << "\nRegion Statistics:" << endl;
    for (int j = 0; j < parameters.numRegions; j++) {
        cout << "Region " << j << ": searched " << regions[j].ta 
             << " times, best fitness " << regions[j].regionBest.fitness << endl;
    }
}

// 公開介面函數
double SearchEconomicsAlgorithm::getBestFitness() const {
    return statistics.globalBest.fitness;
}

vector<double> SearchEconomicsAlgorithm::getBestPosition() const {
    vector<double> position;
    if (statistics.globalBest.isBinary) {
        for (int bit : statistics.globalBest.binaryPosition) {
            position.push_back(static_cast<double>(bit));
        }
    } else {
        position = statistics.globalBest.position;
    }
    return position;
}

Solution SearchEconomicsAlgorithm::getGlobalBest() const {
    return statistics.globalBest;
}

SEStatistics SearchEconomicsAlgorithm::getStatistics() const {
    return statistics;
}

// ==================== 向後相容的 algorithm 類別 ====================

void algorithm::RunALG(int D, int NP, int G, double pb, int c, int maxVal, int fun_num) {
    // 將參數映射到 SE 算法
    // D: dimension, NP: numSearchers, G: maxIterations
    // 設定 SE 專用參數
    int numRegions = 4;  // 論文建議值
    
    cout << "Running SE Algorithm with parameters:" << endl;
    cout << "D=" << D << ", numSearchers=" << NP << ", numRegions=" << numRegions 
         << ", maxIterations=" << G << endl;
    
    seAlgorithm.RunSE(D, NP, numRegions, G, -maxVal, maxVal, fun_num);
    
    // 儲存結果供後續查詢
    bestSolution = seAlgorithm.getGlobalBest();
    bestFitness = seAlgorithm.getBestFitness();
}

double algorithm::get_best_fitness(int& best_idx) const {
    best_idx = 0; // OneMax 只有一個最佳解
    return bestFitness;
}

vector<double> algorithm::get_best_position() const {
    return seAlgorithm.getBestPosition();
}