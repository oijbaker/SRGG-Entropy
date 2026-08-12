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



// Rayleigh Fading
double rayleigh(double r, double r0, double eta) {
    return exp(-pow(r / r0, eta));
}


double local_entropy(double x, double y, double r0, double eta, int L) {
    double avg = 0;
    for (int l = 0; l < L; l++) {
        double pointx = unif(gen);
        double pointy = unif(gen);

        double dist = sqrt(pow(x - pointx, 2) + pow(y - pointy, 2));
        avg += h2(rayleigh(dist, r0, eta))/L;
    }
    return avg;
}

double entropy_graph(const int N, double r0, double eta, int L) {
    // Prepare variable to count average degree
    double avg = 0;
    double C = (double)0.5 * (N * (N - 1));

    #pragma omp parallel reduction(+:avg)
    {
        std::mt19937 thread_gen(omp_get_thread_num());
        std::uniform_real_distribution<double> thread_unif(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < L; i++) {
        // Create point set
        vector<vector<double>> points;
        vector<vector<double>> distances;
        vector<vector<double>> edges;

        for (int j = 0; j < N; j++) {
            vector<double> point(2);
            point[0] = unif(gen);
            point[1] = unif(gen);
            points.push_back(point);
        }

        // Calculate distances and edges
        for (int j = 0; j < N; j++) {
            vector<double> distance(N);
            vector<double> edge(N);
            for (int k = j; k < N; k++) {
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

    if (argc != 5) {
        cout << "Usage: " << argv[0] << " <eta> <L> <r0> <step>" << endl;
        return 1;
    }

    double eta = atof(argv[1]); // Path loss exponent
    int L = atoi(argv[2]); // Number of iterations
    double r0 = atof(argv[3]); // Connection range
    double step = atof(argv[4]); // Step size
    

    cout << "Path loss exponent: " << eta << endl;
    cout << "Number of iterations: " << L << endl;
    cout << "Connection range" << r0 << endl;
    cout << "Step size: " << step << endl;

    // Create a unique file name to write to
    string filename = "../data/mass_values_square_" + to_string(r0) + "_" + to_string((int) eta) + ".csv";
    cout << "Writing to file: " << filename << endl;

    // Open the file for writing
    ofstream file(filename);
    file << "x,y,mass" << endl; // Write header
    for (double x = 0; x <= 1.0; x += step) {
        cout << "x: " << x << ", Progress: " << x * 100.0 << "%" << endl;
        for (double y = 0; y <= 1.0; y += step) {
            double mass = local_entropy(x, y, r0, eta, L);
            file << x << "," << y << "," << mass << endl;
        };
    }
    file.close();

    return 0;
}