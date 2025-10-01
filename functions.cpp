#include "functions.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#define M_PI 3.14159265358979323846
using namespace std;

// 簡化版 functions.cpp - 專注於 OneMax 問題
// 只保留必要的函數以避免編譯錯誤

double ackley(const vector<double>& x) {
    const double a = 20.0, b = 0.2, c = 2 * M_PI;
    int D = x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for (double val : x) {
        sum1 += val * val;
        sum2 += cos(c * val);
    }
    return -a * exp(-b * sqrt(sum1 / D)) - exp(sum2 / D) + a + exp(1.0);
}

double sphere_func(const vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}

double rastrigin(const vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val - 10.0 * cos(2 * M_PI * val) + 10.0;
    }
    return sum;
}

double rosenbrock(const vector<double>& x) {
    double sum = 0.0;
    int D = x.size();
    for (int i = 0; i < D - 1; ++i) {
        double term1 = 100 * pow((x[i + 1] - x[i] * x[i]), 2);
        double term2 = pow((x[i] - 1), 2);
        sum += term1 + term2;
    }
    return sum;
}

double griewank(const vector<double>& x) {
    double sum = 0.0, product = 1.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] * x[i]) / 4000.0;
        product *= cos(x[i] / sqrt(i + 1));
    }
    return sum - product + 1.0;
}

double schwefel_func(const vector<double>& x) {
    int nx = x.size();
    double tmp, sum = 0.0;

    for (int i = 0; i < nx; i++) {
        double zi = x[i];
        
        if (zi > 500) {
            tmp = (500 - fmod(zi, 500));
            sum += (500 - tmp) * sin(sqrt(fabs(tmp))) - (zi - 500) * (zi - 500) / 10000.0 / nx;
        }
        else if (zi < -500) {
            tmp = fmod(fabs(zi), 500) - 500;
            sum += (-500 + tmp) * sin(sqrt(fabs(tmp))) - (zi + 500) * (zi + 500) / 10000.0 / nx;
        }
        else {
            sum += zi * sin(sqrt(fabs(zi)));
        }
    }
    return 418.9829 * nx - sum;
}

double zakharov_func(const vector<double>& x) {
    int nx = x.size();
    double sum1 = 0.0, sum2 = 0.0;

    for (int i = 0; i < nx; i++) {
        sum1 += x[i] * x[i];
        sum2 += 0.5 * (i + 1) * x[i];
    }
    return sum1 + sum2 * sum2 + sum2 * sum2 * sum2;
}

double michalewicz_func(const vector<double>& x) {
    int nx = x.size();
    double sum = 0.0;
    for (int i = 0; i < nx; i++) {
        sum += sin(x[i]) * pow(sin((i + 1) * x[i] * x[i] / M_PI), 20);
    }
    return -sum;
}

vector<double> generateRandomIndividual(int D, double min, double max, mt19937& gen) {
    vector<double> individual(D);
    uniform_real_distribution<> dis(min, max);
    for (int j = 0; j < D; ++j) {
        individual[j] = dis(gen);
    }
    return individual;
}