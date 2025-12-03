#ifndef SE_STRUCTURES_H
#define SE_STRUCTURES_H

#include <vector>
#include <random>

// Search Economics 演算法的核心資料結構

// 單一解的結構 (候選解/投資/goods)
// 對應論文符號: si (投資), mjk (goods), vijk (暫時候選解)
struct Solution {
    std::vector<double> position;    // 解的位置向量 (連續問題)
    std::vector<int> binaryPosition; // 二進位位置向量 (OneMax問題)
    double fitness;                  // 適應度值
    int dimension;                   // 維度
    bool isBinary;                   // 是否為二進位問題
    
    Solution() : fitness(0.0), dimension(0), isBinary(false) {}
    Solution(int dim, bool binary = false) : dimension(dim), fitness(0.0), isBinary(binary) {
        if (binary) {
            binaryPosition.resize(dim);
        } else {
            position.resize(dim);
        }
    }
    
    // 複製建構函數
    Solution(const Solution& other) 
        : position(other.position), binaryPosition(other.binaryPosition), 
          fitness(other.fitness), dimension(other.dimension), isBinary(other.isBinary) {}
    
    // OneMax 專用：計算1的個數
    int countOnes() const {
        if (!isBinary) return 0;
        int count = 0;
        for (int bit : binaryPosition) {
            if (bit == 1) count++;
        }
        return count;
    }
    
    // 判斷屬於哪個區域 (基於1的個數)
    int getRegionByOnesCount(int numRegions) const {
        if (!isBinary) return 0;
        int ones = countOnes();
        int regionSize = (dimension + numRegions - 1) / numRegions; // 向上取整
        int regionId = ones / regionSize;
        return std::min(regionId, numRegions - 1); // 確保不超過範圍
    }
};

// 區域 (Region) 結構
// 對應論文符號: rj (區域), mjk (goods), rbj (區域最佳解), taj/tbj (統計資料)
struct Region {
    int regionId;                           // 區域ID (j)
    std::vector<Solution> goods;            // 區域內的goods (mjk: 第 j 區域的第 k 個候選解樣本)
    Solution regionBest;                    // 區域最佳解 (rbj: 第 j 區域當前的 best good)
    int maxGoods;                          // 最大goods數量 (w: 每區域有 w 個 goods)
    
    // SE 統計資料
    int ta;                                // taj: 第 j 區域被「投資／搜尋」的次數（初值 1，搜尋一次加 1）
    int tb;                                // tbj: 第 j 區域「未被搜尋」的持續次數（初值 1，每次未被搜尋加 1；若被搜尋則重設為 1）
    int totalSearchCount;                  // 累積總搜尋次數（用於統計輸出，不會被重置）
    
    // 期望值計算相關
    double f1_value;                       // f1(M_j) = tb_j / ta_j
    double f3_value;                       // f3(ρ_j) 區域best相對權重
    
    Region() : regionId(-1), maxGoods(0), ta(0), tb(1), totalSearchCount(0), f1_value(1.0), f3_value(0.0) {}
    Region(int id, int max_goods) 
        : regionId(id), maxGoods(max_goods), ta(0), tb(1), totalSearchCount(0), f1_value(1.0), f3_value(0.0) {
        goods.reserve(max_goods);
    }
    
    // 更新區域最佳解
    void updateRegionBest();
    
    // 新增goods到區域 (替換最差的如果超過容量)
    void addGood(const Solution& good);
    
    // 計算f1值
    void calculateF1() { f1_value = static_cast<double>(tb) / ta; }
    
    // OneMax 專用：檢查解是否屬於此區域
    bool belongsToRegion(const Solution& solution, int numRegions) const {
        if (!solution.isBinary) return false;
        return solution.getRegionByOnesCount(numRegions) == regionId;
    }
};

// 搜尋者 (Searcher) 結構
// 對應論文符號: si (投資), vijk (暫時候選解), eij (期望值)
struct Searcher {
    int searcherId;                        // 搜尋者ID (i)
    Solution investment;                   // 當前投資 (si: searcher i 擁有的投資/當前解)
    int currentRegion;                     // 當前所在區域
    
    // 與各區域的暫時候選解
    std::vector<std::vector<Solution>> temporaryCandidates;  // vijk: 由 searcher i 的投資 si 與區域 j 的 goods mjk 做 crossover/mutation 得到的暫時候選解
    std::vector<double> expectedValues;   // eij: searcher i 在區域 j 的期望值（用以比較不同區域）
    
    Searcher() : searcherId(-1), currentRegion(-1) {}
    Searcher(int id, int dimension, int numRegions) 
        : searcherId(id), investment(dimension), currentRegion(-1) {
        temporaryCandidates.resize(numRegions);
        expectedValues.resize(numRegions, 0.0);
    }
    
    // 計算對特定區域的期望值
    double calculateExpectedValue(int regionIndex, const Region& region);
    
    // 選擇下一個投資區域
    int selectNextRegion(std::mt19937& gen);
};

// SE 演算法主要參數結構
struct SEParameters {
    int numSearchers;          // n: searchers數量
    int numRegions;           // h: regions數量  
    int goodsPerRegion;       // w: 每區域goods數量
    int maxStoredGoods;       // k: 每區域儲存的goods上限
    int dimension;            // 問題維度
    int maxIterations;        // 最大迭代次數
    double minValue;          // 解空間下界
    double maxValue;          // 解空間上界
    bool isBinaryProblem;     // 是否為二進位問題 (OneMax)
    
    // Crossover/Mutation 參數
    double crossoverRate;     // 交叉率
    double mutationRate;      // 突變率
    double mutationStrength;  // 突變強度
    
    // Tournament 參數
    int tournamentSize;       // tournament 中參與比較的候選解數量
    double randomSelectionRate; // 隨機選擇其他區域 v_ij 的機率
    
    SEParameters() : 
        numSearchers(4), numRegions(4), goodsPerRegion(2), maxStoredGoods(5),
        dimension(20), maxIterations(1000), minValue(-30.0), maxValue(30.0),
        isBinaryProblem(false), crossoverRate(0.8), mutationRate(0.01), 
        mutationStrength(0.1), tournamentSize(3), randomSelectionRate(0.3) {}
};

// SE 統計資訊結構
struct SEStatistics {
    std::vector<double> bestFitnessHistory;    // 歷代最佳適應度
    std::vector<Solution> bestSolutionHistory; // 歷代最佳解
    Solution globalBest;                       // 全域最佳解
    int totalEvaluations;                      // 總評估次數
    int currentIteration;                      // 當前迭代次數
    
    // 區域統計
    std::vector<int> regionSearchCounts;       // 各區域被搜尋次數統計
    std::vector<double> regionBestFitness;     // 各區域最佳適應度
    
    SEStatistics() : totalEvaluations(0), currentIteration(0) {}
    
    void updateGlobalBest(const Solution& candidate) {
        if (candidate.fitness > globalBest.fitness) {
            globalBest = candidate;
        }
    }
    
    void recordIteration() {
        bestFitnessHistory.push_back(globalBest.fitness);
        bestSolutionHistory.push_back(globalBest);
        currentIteration++;
    }
};

#endif // SE_STRUCTURES_H