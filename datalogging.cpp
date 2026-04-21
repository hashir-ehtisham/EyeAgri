#include "functions_init.h" // include the main header file where all the functions are initialized
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

void datalog(string log_data[31]
) {
	// enter path to csv here
	string filepath = "C:/Users/shaha/UNI/Uni Work/Fundamentals of Programming/Lab Tasks/FoPSemProject/rice_crop_patchwise_log_v2.csv";
	ofstream file(filepath, ios::app);
	if (file.is_open()) {
		for (int i = 0; i < 32; i++) {
			file << log_data[i] << ",";
		}
		file << "\n";
		file.close();
	}
	else {
		cout << "The file opening was either denied permission or failed.\n";

	}

}
