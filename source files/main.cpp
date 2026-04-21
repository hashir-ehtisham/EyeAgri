#include <iostream>
#include "functions_init.h"

using namespace std;

int main() {
    //Sample array for checking logging

    std::string rowData[31] = {
    "21/04/2026",
    "500",                  // crop_sizew
    "101",                  // PatchID
    "50",                   // PatchSize
    "Wheat",                // crop_type
    "10",                   // num_of_patches
    "2026-04-21",           // crop_start_date
    "120",                  // crop_duration
    "15.5",                 // water_usage
    "Drip",                 // irrigation_method
    "Daily",                // water_frequency
    "Organic",              // fertilizer_type
    "5.2",                  // fertilizer_qunatity
    "10:10:10",             // N_P_K_ratio
    "Spray",                // apply_method
    "Morning",              // apply_time
    "45.5",                 // soil_moisture_level
    "Loam",                 // soil_type
    "6.8",                  // soil_pH
    "22.4",                 // soil_temp
    "Good",                 // drainage_condition
    "26.5",                 // air_temp
    "60.0",                 // humidity
    "No",                // rainfall
    "0",                    // rain_duration
    "8.5",                  // sunlight_hours
    "Calm",                 // wind_condition
    "14.2",                 // plant_height
    "Dark Green",            // leaf_color
    "Diseased",
    "Bacterial Blight"
    };

    datalog(rowData);
}