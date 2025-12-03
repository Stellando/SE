#include "se_structures.h"
#include <algorithm>
#include <limits>

// Region 類別的方法實作
void Region::updateRegionBest() {
    if (goods.empty()) return;
    
    regionBest = goods[0];
    for (const Solution& good : goods) {
        if (good.fitness > regionBest.fitness) {
            regionBest = good;
        }
    }
}

void Region::addGood(const Solution& good) {
    // 限制：最多保留 maxGoods 個 goods (k=goodsPerRegion=2)
    if (goods.size() < maxGoods) {
        // 空間足夠，直接加入
        goods.push_back(good);
    } else {
        // 空間滿了，需要替換策略
        // 策略：替換最差的 good（如果新 good 更好）
        int worstIdx = 0;
        double worstFitness = goods[0].fitness;
        
        for (int i = 1; i < goods.size(); i++) {
            if (goods[i].fitness < worstFitness) {
                worstFitness = goods[i].fitness;
                worstIdx = i;
            }
        }
        
        // 如果新的 good 比最差的還好，就替換最差的
        if (good.fitness > worstFitness) {
            goods[worstIdx] = good;
        }
        // 注意：如果新 good 不如最差的，則不加入（保持原有 goods）
    }
}

// Searcher 類別的方法實作
double Searcher::calculateExpectedValue(int regionIndex, const Region& region) {
    // 這個方法在 SearchEconomicsAlgorithm 中實作
    // 這裡只是保留介面
    return expectedValues[regionIndex];
}

int Searcher::selectNextRegion(std::mt19937& gen) {
    // 基於期望值選擇區域
    if (expectedValues.empty()) return 0;
    
    // 找到最高期望值的區域
    int bestRegion = 0;
    double bestValue = expectedValues[0];
    
    for (int i = 1; i < expectedValues.size(); i++) {
        if (expectedValues[i] > bestValue) {
            bestValue = expectedValues[i];
            bestRegion = i;
        }
    }
    
    return bestRegion;
}