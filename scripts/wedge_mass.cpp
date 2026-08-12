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


// Rayleigh Fading
double rayleigh(double r, double r0, double eta) {
    return exp(-pow(r / r0, eta));
}


double local_entropy(double x, double y, double r0, double eta, int L, double max_angle) {
    double avg = 0;
    for (int l = 0; l < L; l++) {
        double pointr= sqrt(unif(gen));
        double pointtheta = max_angle * unif(gen);

        double pointx = pointr * cos(pointtheta);
        double pointy = pointr * sin(pointtheta);

        double dist = sqrt(pow(x - pointx, 2) + pow(y - pointy, 2));
        avg += h2(rayleigh(dist, r0, eta))/L;
    }
    return avg;
}


double entropy_graph(const int N, double r0, double eta, int L, double max_angle) {
    // Prepare variable to count average degree
    double avg = 0;
    double C = (double) 0.5 * (N * (N - 1));

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
            double radius = sqrt(unif(gen));
            double theta = max_angle * unif(gen);
            point[0] = radius * cos(theta);
            point[1] = radius * sin(theta);
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

int main(int argc, char *argv[]) {

    if (argc != 7) {
        cout << "Usage: " << argv[0] << "<eta> <L> <r0> <step_r> <step_theta> <max_angle>" << endl;  
        return 1; 
    }

    double eta = atof(argv[1]); // Path loss exponent
    int L = atoi(argv[2]); // Number of iterations
    double r0 = atof(argv[3]); // Minimum r0
    double stepr = atof(argv[4]); // Step size
    double steptheta = atof(argv[5]); // Step size
    double max_angle = atof(argv[6]); // Maximum angle

    cout << "Path loss exponent: " << eta << endl;
    cout << "Number of iterations: " << L << endl;
    cout << "r0: " << r0 << endl;
    cout << "Radius step size: " << stepr << endl;
    cout << "Angle step size: " << steptheta << endl;
    cout << "Entropy values:" << endl;

    // Create a unique file name to write to
    string filename = "../data/mass_values_wedge_" + to_string(max_angle) + "_" + to_string(r0) + "_" + to_string((int) eta) + ".csv";
    cout << "Writing to file: " << filename << endl;

    // Open the file for writing
    ofstream file(filename);
    file << "r,theta,mass" << endl; // Write header
    for (double r = 0; r <= 1; r += stepr) {
        cout << "r: " << r << ", Progress: " << r * 100.0 << "%" << endl;
        for (double theta = 0; theta <= max_angle; theta += steptheta) {
            double mass = local_entropy(r * cos(theta), r * sin(theta), r0, eta, L, max_angle);
            file << r << "," << theta << "," << mass << endl;
        };
    }

    file.close();

    return 0;    
}