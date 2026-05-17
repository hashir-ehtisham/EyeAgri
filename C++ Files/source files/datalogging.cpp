#include "functions_init.h" // include the main header file where all the functions are initialized
#include <string>
#include <iostream>
#include <fstream>
#include <QString>
using namespace std;
void datalog(QString log_data[29]
) {
	// enter path to csv here
    string filepath = "C:/Users/shaha/Downloads/final_crop_log_xiq7z_p9.csv";
	ofstream file(filepath, ios::app);
	if (file.is_open()) {
        for (int i = 0; i < 29; i++) {
            file << log_data[i].toStdString() << ",";
		}
        file << "Healthy \n";
	}
	else {
		cout << "The file opening was either denied permission or failed.\n";

	}
    file.close();
}
