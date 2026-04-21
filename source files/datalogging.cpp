#include "functions_init.h" // include the main header file where all the functions are initialized
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

void datalog(string log_date,
	int crop_size,
	int PatchID,
	int PatchSize,
	string crop_type,
	int num_of_patches,
	string crop_start_date,
	int crop_duration,
	float water_usage,
	string irrigation_method,
	string water_frequency,
	string fertilizer_type,
	float fertilizer_qunatity,
	string N_P_K_ratio,
	string apply_method,
	string apply_time,
	float soil_moisture_level,
	string soil_type,
	float soil_pH,
	float soil_temp,
	string drainage_condition,
	float air_temp,
	float humidity,
	bool rainfall,
	int rain_duration,
	float sunlight_hours,
	string wind_condition,
	float plant_height,
	string leaf_color
) {
	// enter path to csv here
	string filepath = "C:\Users\shaha\UNI\Uni Work\Fundamentals of Programming\Lab Tasks\FoPSemProject\rice_crop_patchwise_log_v2.csv";  
	ofstream file(filepath, ios::app);
	if (file.is_open()) {
		file << crop_size << ","
			<< PatchID << ","
			<< PatchSize << ","
			<< crop_type << ","
			<< num_of_patches << ","
			<< crop_start_date << ","
			<< crop_duration << ","
			<< water_usage << ","
			<< irrigation_method << ","
			<< water_frequency << ","
			<< fertilizer_type << ","
			<< fertilizer_qunatity << ","
			<< N_P_K_ratio << ","
			<< apply_method << ","
			<< apply_time << ","
			<< soil_moisture_level << ","
			<< soil_type << ","
			<< soil_pH << ","
			<< soil_temp << ","
			<< drainage_condition << ","
			<< air_temp << ","
			<< humidity << ","
			<< rainfall << ","
			<< rain_duration << ","
			<< sunlight_hours << ","
			<< wind_condition << ","
			<< plant_height << ","
			<< leaf_color << "\n";
		file.close();
	}
	else {
		cout << "The file opening was either denied permission or failed.\n";

	}

}
