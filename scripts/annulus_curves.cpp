#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <stdlib.h>
#include <math.h>
#include <array>
#include <vector>
#include <omp.h>
#include <fstream>
#include <cmath>

using namespace std;

// Function to generate a random number between 0 and 1
mt19937 gen(random_device{}());
random_device rd;
uniform_real_distribution<double> unif(0.0, 1.0);


// Binary Entropy function
double h2(double p) {
    if (p == 0 || p == 1) {
        return 0;
    }
    return -p * log(p) - (1 - p) * log(1 - p);
}


// intersection check
bool check_intersection(double x1, double y1, double x2, double y2, double minor_radius) {
    double t_star = 0.5 * (x1*x2 - 2*x2*x2 - 2*y2*y2) / (x1*x1 + y1*y1 - x1*x2 - y2*y2 + x2*x2 + y2*y2);
    double M = sqrt(t_star*t_star*(x1*x1 + y1*y1) - t_star*(1-t_star) * (x1*x2 + y1*y2) + (1-t_star)*(1-t_star) * (x2*x2 + y2*y2));
    return M < minor_radius;
}


// Rayleigh Fading
double rayleigh(double r, double r0, double eta) {
    return exp(-pow(r / r0, eta));
}


double local_entropy(double x, double y, double r0, double eta, int L, double radius) {
    uniform_real_distribution<double> radial_unif(radius,1.0);
    double avg = 0;
    for (int l = 0; l < L; l++) {
        double new_r = sqrt(radial_unif(gen));
        double new_theta = 6.283185 * unif(gen);
        double pointx = new_r * cos(new_theta);
        double pointy = new_r * sin(new_theta);

        if (check_intersection(x, y, pointx, pointy, radius)) {
            continue; // Skip if the points intersect
        }

        double dist = sqrt(pow(x - pointx, 2) + pow(y - pointy, 2));
        avg += h2(rayleigh(dist, r0, eta))/L;
    }
    return avg;
}


double entropy_graph(const int N, double r0, double eta, int L, double minor_radius) {
    // Prepare variable to count average degree
    double avg = 0;
    double C = (double) 0.5 * (N * (N - 1));

    #pragma omp parallel reduction(+:avg)
    {
        std::mt19937 thread_gen(omp_get_thread_num());
        std::uniform_real_distribution<double> thread_unif(0.0, 1.0);
        std::uniform_real_distribution<double> radial_unif(minor_radius, 1.0);

        #pragma omp for
        for (int i = 0; i < L; i++) {
        // Create point set
        vector<vector<double>> points;
        vector<vector<double>> distances;
        vector<vector<double>> edges;

        for (int j = 0; j < N; j++) {
            vector<double> point(2);
            double radius = sqrt(radial_unif(gen));
            double theta = 6.283185 * unif(gen);
            point[0] = radius * cos(theta);
            point[1] = radius * sin(theta);
            points.push_back(point);
        }

        // Calculate distances and edges
        for (int j = 0; j < N; j++) {
            vector<double> distance(N);
            vector<double> edge(N);
            for (int k = j; k < N; k++) {
                if (check_intersection(points[j][0], points[j][1], points[k][0], points[k][1], minor_radius)) {
                    edge[k] = 0;
                    continue;
                }
                double dist = sqrt(pow(points[j][0] - points[k][0], 2) + pow(points[j][1] - points[k][1], 2));
                distance[k] = dist;
                edge[k] = unif(gen) < h2(rayleigh(dist, r0, eta)) ? 1 : 0;
            }
            distances.push_back(distance);
            edges.push_back(edge);
        }

        // Calculate average degree
        double sum = 0;
        for (int j = 0; j < N; j++) {
            double degree = 0;
            for (int k = j; k < N; k++) {
                degree += edges[j][k];
            }
            sum += degree;
        }
        avg += sum / C;
        }
    }

    avg /= L; // Normalize by the number of iterations
    return avg;
}


// Main function
int main(int argc, char *argv[]){

    if (argc != 8) {
        cout << "Usage: " << argv[0] << " <N> <eta> <L> or <N> <eta> <L> <r0min> <r0max> <step> <minor_radius>" << endl;
        return 1;
    }

    int N = atoi(argv[1]); // Number of points
    double eta = atof(argv[2]); // Path loss exponent
    int L = atoi(argv[3]); // Number of iterations
    double r0min = atof(argv[4]); // Minimum r0
    double r0max = atof(argv[5]); // Maximum r0
    double step = atof(argv[6]); // Step size
    double minor_radius = atof(argv[7]); // Maximum angle
    

    cout << "Number of points: " << N << endl;
    cout << "Path loss exponent: " << eta << endl;
    cout << "Number of iterations: " << L << endl;
    cout << "Minimum r0: " << r0min << endl;
    cout << "Maximum r0: " << r0max << endl;
    cout << "Step size: " << step << endl;
    cout << "Entropy values:" << endl;

    // Create a unique file name to write to
    string filename = "../data/entropy_values_annulus_" + to_string(minor_radius) + "_" + to_string(N) + "_" + to_string((int) eta) + ".csv";
    cout << "Writing to file: " << filename << endl;

    // Open the file for writing
    ofstream file(filename);
    file << "r0,entropy" << endl; // Write header
    for (double r0 = r0min; r0 <= r0max; r0 += step) {
        cout << "r0: " << r0 << ", Progress: " << (r0/r0max)*100.0 << "%" << endl;

        // Calculate entropy
        double entropy = entropy_graph(N, r0, eta, L, minor_radius);
    
        // Write the entropy value to the file
        file << r0 << "," << entropy << endl;
    }

    file.close();

    return 0;
}