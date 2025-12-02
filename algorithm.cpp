
#include <ctime>
#include <cstdlib>
#include "algorithm.h"
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
    fitnessHistory.clear();
    
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
        
        // 記錄當前代的最佳 fitness
        fitnessHistory.push_back(statistics.globalBest.fitness);
        
        //每10回合輸出結果
        if (iteration % 10 == 0) {
            printProgress(iteration);
        }
        
        // 檢查終止條件
        if (shouldTerminate()) {
            cout << "Early termination at iteration " << (iteration + 1) << endl;
            // 將剩餘的 iteration 都填入當前最佳 fitness
            for (int remaining = iteration + 1; remaining < maxIterations; remaining++) {
                fitnessHistory.push_back(statistics.globalBest.fitness);
            }
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
        
        // 為每個區域產生 w 個隨機 goods，前兩個bit根據區域決定
        // 區域0: 00, 區域1: 01, 區域2: 10, 區域3: 11
        for (int k = 0; k < parameters.goodsPerRegion; k++) {
            Solution good = generateRandomSolutionForRegion(parameters.dimension, j);
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
        // 先分配區域，再生成對應區域的初始投資
        int assignedRegion = i % parameters.numRegions;
        searchers[i].currentRegion = assignedRegion;
        searchers[i].investment = generateRandomSolutionForRegion(parameters.dimension, assignedRegion);
        evaluateSolution(searchers[i].investment);
        
        // 初始化暫時候選解容器
        searchers[i].temporaryCandidates.resize(parameters.numRegions);
        for (int j = 0; j < parameters.numRegions; j++) {
            searchers[i].temporaryCandidates[j].resize(parameters.goodsPerRegion);
        }
    }
}

void SearchEconomicsAlgorithm::distributeSearchersToRegions() {
    // 區域分配已在 initializeSearchers() 中完成
    // 原作者版本：初始化 ta 為實際分配到該區域的 Searcher 數量
    vector<int> searcherCountPerRegion(parameters.numRegions, 0);
    
    for (int i = 0; i < parameters.numSearchers; i++) {
        searcherCountPerRegion[searchers[i].currentRegion]++;
    }
    
    // 設定每個區域的初始 ta 和 tb
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j].ta = searcherCountPerRegion[j];
        regions[j].tb = 1;
    }
    
    cout << "Searchers distributed to regions based on initial investment pattern:" << endl;
    for (int i = 0; i < parameters.numSearchers; i++) {
        cout << "Searcher " << i << " -> Region " << searchers[i].currentRegion 
             << " (prefix: " << searchers[i].investment.binaryPosition[0] 
             << searchers[i].investment.binaryPosition[1] << ")" << endl;
    }
    
    cout << "Initial ta values per region: ";
    for (int j = 0; j < parameters.numRegions; j++) {
        cout << "R" << j << "=" << regions[j].ta << " ";
    }
    cout << endl;
}

// Vision Search (VS) 實作
void SearchEconomicsAlgorithm::visionSearch() {
    // Transition
    transition();
    
    // Evaluation
    evaluation();
    
    // 在 Determination 之前，先記錄當前搜尋統計（統計這一代在哪些區域搜尋）
    // 原作者版本：在改變區域之前先統計
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j].tb++;
    }
    
    for (int i = 0; i < parameters.numSearchers; i++) {
        int currentRegion = searchers[i].currentRegion;
        if (currentRegion >= 0 && currentRegion < parameters.numRegions) {
            regions[currentRegion].ta++;
            regions[currentRegion].tb = 1;
        }
    }
    
    // 更新 f1 值
    for (int j = 0; j < parameters.numRegions; j++) {
        regions[j].calculateF1();
    }
    
    // Determination：決定下一代要去哪個區域
    determination();
}

void SearchEconomicsAlgorithm::transition() {
    // 對每個 searcher 與每個 region 的每個 good 做 crossover/mutation
    // 原作者版本：crossover 產生的兩個子代都保留
    for (int i = 0; i < parameters.numSearchers; i++) {
        for (int j = 0; j < parameters.numRegions; j++) {
            int candidateIdx = 0;
            for (int k = 0; k < regions[j].goods.size() && candidateIdx < parameters.goodsPerRegion; k++) {
                // v_ijk = si ⊗ mjk (crossover + mutation)
                Solution offspring1(parameters.dimension, true);
                Solution offspring2(parameters.dimension, true);
                
                // 單點交叉，產生兩個子代
                uniform_int_distribution<int> crossoverPoint(1, parameters.dimension - 1);
                int point = crossoverPoint(randomGenerator);
                
                for (int b = 0; b < parameters.dimension; b++) {
                    if (b < point) {
                        offspring1.binaryPosition[b] = searchers[i].investment.binaryPosition[b];
                        offspring2.binaryPosition[b] = regions[j].goods[k].binaryPosition[b];
                    } else {
                        offspring1.binaryPosition[b] = regions[j].goods[k].binaryPosition[b];
                        offspring2.binaryPosition[b] = searchers[i].investment.binaryPosition[b];
                    }
                }
                
                // 第一個子代 mutation 後加入
                Solution mutated1 = mutation(offspring1);
                evaluateSolution(mutated1);
                if (candidateIdx < parameters.goodsPerRegion) {
                    searchers[i].temporaryCandidates[j][candidateIdx] = mutated1;
                    statistics.totalEvaluations++;
                    candidateIdx++;
                }
                
                // 第二個子代 mutation 後加入
                if (candidateIdx < parameters.goodsPerRegion) {
                    Solution mutated2 = mutation(offspring2);
                    evaluateSolution(mutated2);
                    searchers[i].temporaryCandidates[j][candidateIdx] = mutated2;
                    statistics.totalEvaluations++;
                    candidateIdx++;
                }
            }
        }
    }
}

void SearchEconomicsAlgorithm::evaluation() {
    // 計算每個 searcher 對每個 region 的期望值 e_ij
    // 原作者版本：在計算期望值時即時更新解（Sequential update）
    for (int i = 0; i < parameters.numSearchers; i++) {
        for (int j = 0; j < parameters.numRegions; j++) {
            // 先檢查並更新 temporaryCandidates
            for (int k = 0; k < parameters.goodsPerRegion; k++) {
                if (k < searchers[i].temporaryCandidates[j].size()) {
                    Solution& candidate = searchers[i].temporaryCandidates[j][k];
                    
                    // 如果候選解比當前 searcher 的投資好，立即更新
                    if (candidate.fitness > searchers[i].investment.fitness) {
                        searchers[i].investment = candidate;
                        // 更新當前區域基於新投資的前兩個bit
                        int newRegion = (candidate.binaryPosition[0] << 1) | candidate.binaryPosition[1];
                        searchers[i].currentRegion = newRegion;
                    }
                    
                    // 如果候選解屬於這個區域且比區域內的 goods 好，立即更新 goods
                    if (regions[j].belongsToRegion(candidate, parameters.numRegions)) {
                        regions[j].addGood(candidate);
                    }
                }
            }
            
            // 計算期望值
            searchers[i].expectedValues[j] = calculateExpectedValue(i, j);
        }
        
        // 更新該 searcher 所影響的區域最佳解
        for (int j = 0; j < parameters.numRegions; j++) {
            regions[j].updateRegionBest();
        }
    }
}

void SearchEconomicsAlgorithm::determination() {
    // 原作者版本：Vision Selection 是「區域選擇」而非「解選擇」
    // 基於期望值進行 Tournament，選出目標區域
    for (int i = 0; i < parameters.numSearchers; i++) {
        // Tournament: 選擇期望值最高的區域
        int selectedRegion = 0;
        double maxExpectedValue = searchers[i].expectedValues[0];
        
        for (int j = 1; j < parameters.numRegions; j++) {
            if (searchers[i].expectedValues[j] > maxExpectedValue) {
                maxExpectedValue = searchers[i].expectedValues[j];
                selectedRegion = j;
            }
        }
        
        // 更新 searcher 的目標區域
        searchers[i].currentRegion = selectedRegion;
    }
}

// Marketing Research (MR) 實作
void SearchEconomicsAlgorithm::marketingResearch() {
    updateGoods();
    // 作者代碼的特殊重置邏輯
    for (int j = 0; j < parameters.numRegions; j++) {
        if (regions[j].tb > 1) {
            regions[j].ta = 1.0;
        }
    }
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
    // 統計邏輯已移到 visionSearch 中（在 determination 之前執行）
    // 這裡保留空函數以維持結構
}

void SearchEconomicsAlgorithm::feedbackToVS() {
    // 更新區域統計以回饋給下一輪的 Vision Search
    updateRegionStatistics();
}

// 期望值計算函數（原作者版本）
double SearchEconomicsAlgorithm::calculateExpectedValue(int searcherIdx, int regionIdx) {
    // 1. T_j (原作者邏輯: ta / tb)
    // 注意：這是 exploitation 策略，鼓勵搜索"熱門"區域
    double Tj = (regions[regionIdx].tb > 0) ? 
                static_cast<double>(regions[regionIdx].ta) / regions[regionIdx].tb : 
                static_cast<double>(regions[regionIdx].ta);
    
    // 2. M_j (原作者邏輯: 區域最佳解 / 所有區域所有樣本的 fitness 總和)
    double totalSampleFitness = 0.0;
    for (const auto& r : regions) {
        for (const auto& g : r.goods) {
            totalSampleFitness += g.fitness;
        }
    }
    
    double Mj = 0.0;
    if (totalSampleFitness > 0) {
        Mj = regions[regionIdx].regionBest.fitness / totalSampleFitness;
    }
    
    // 原作者沒有使用 f2（候選解平均值）
    // 直接返回 Tj * Mj
    return Tj * Mj;
}

double SearchEconomicsAlgorithm::calculateF1(const Region& region) {
    // f1(M_j) = tb_j / ta_j
    //return static_cast<double>(region.tb) / region.ta;
    //實作CODE和論文相反
    return static_cast<double>(region.ta) / region.tb;
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

// 為特定區域生成隨機解（前兩個bit固定）
Solution SearchEconomicsAlgorithm::generateRandomSolutionForRegion(int dimension, int regionIdx) {
    Solution solution(dimension, true);
    
    // 設定前兩個bit根據區域
    // 區域0: 00, 區域1: 01, 區域2: 10, 區域3: 11
    solution.binaryPosition[0] = (regionIdx >> 1) & 1;  // 第一個bit
    solution.binaryPosition[1] = regionIdx & 1;         // 第二個bit
    
    // 隨機生成剩餘的bit
    uniform_int_distribution<int> bitDist(0, 1);
    for (int i = 2; i < dimension; i++) {
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
    Solution offspring1(parent1.dimension, parent1.isBinary);
    Solution offspring2(parent1.dimension, parent1.isBinary);
    
    if (parent1.isBinary && parent2.isBinary) {
        // 單點交叉，產生兩個子代
        uniform_int_distribution<int> crossoverPoint(1, parent1.dimension - 1);
        int point = crossoverPoint(randomGenerator);
        
        // 產生第一個子代 AB：前半部分來自parent1，後半部分來自parent2
        for (int i = 0; i < parent1.dimension; i++) {
            if (i < point) {
                offspring1.binaryPosition[i] = parent1.binaryPosition[i];
                offspring2.binaryPosition[i] = parent2.binaryPosition[i];
            } else {
                offspring1.binaryPosition[i] = parent2.binaryPosition[i];
                offspring2.binaryPosition[i] = parent1.binaryPosition[i];
            }
        }
        
        // 評估兩個子代的適應度
        evaluateSolution(offspring1);
        evaluateSolution(offspring2);
        
        // 選擇表現較好的子代
        if (offspring1.fitness >= offspring2.fitness) {
            return offspring1;
        } else {
            return offspring2;
        }
    }
    
    return offspring1;
}

Solution SearchEconomicsAlgorithm::mutation(const Solution& solution) {
    Solution mutated = solution;
    
    if (solution.isBinary) {
        // 原作者版本：Exactly One Bit Flip
        // 隨機選擇一個 bit 位置進行翻轉
        uniform_int_distribution<int> bitPos(0, solution.dimension - 1);
        int m = bitPos(randomGenerator);
        mutated.binaryPosition[m] = 1 - mutated.binaryPosition[m]; // flip bit
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

vector<double> SearchEconomicsAlgorithm::getFitnessHistory() const {
    return fitnessHistory;
}

// ==================== 從原本的RUNALG做修改 ====================

void algorithm::RunALG(int dimension, int numSearchers, int maxIterations, int maxVal, int funcNum) {
    // 將參數映射到 SE 算法
    // dimension: 問題維度, numSearchers: 搜尋者數量, maxIterations: 最大迭代次數
    // 設定 SE 專用參數
    int numRegions = 4;  // 論文建議值
    
    cout << "Running SE Algorithm with parameters:" << endl;
    cout << "dimension=" << dimension << ", numSearchers=" << numSearchers << ", numRegions=" << numRegions 
         << ", maxIterations=" << maxIterations << endl;
    
    seAlgorithm.RunSE(dimension, numSearchers, numRegions, maxIterations, -maxVal, maxVal, funcNum);
    
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

vector<double> algorithm::get_fitness_history() const {
    return seAlgorithm.getFitnessHistory();
}
