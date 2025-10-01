#include <vector>
#include <random>

// OneMax 問題不需要 cec14 函數
// void cec14_test_func(double *x, double *f, int nx, int mx, int func_num); // 移除

// 保留基本數學函數供未來使用
double ackley(const std::vector<double>& x);
double sphere_func (const std::vector<double>& x);
double rastrigin(const std::vector<double>& x);
double rosenbrock(const std::vector<double>& x);
double griewank(const std::vector<double>& x);
double schwefel_func(const std::vector<double>& x);
double zakharov_func(const std::vector<double>& x);
double michalewicz_func(const std::vector<double>& x);

std::vector<double> generateRandomIndividual(int D, double min, double max, std::mt19937& gen);

// OneMax 專用函數 (但在 SE 演算法中直接實作)
// int onemax(const std::vector<int>& binary_string);
double F11(const std::vector<double>& x);
double F12(const std::vector<double>& x);
double F13(const std::vector<double>& x);
