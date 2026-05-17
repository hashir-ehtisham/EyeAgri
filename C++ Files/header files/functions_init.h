#pragma once
#include <QString>
#include <QTableWidget>
using namespace std;

void datalog(QString data_array [29]
	);

void fileupdate(QString file_cache[1000][30]);
void fileRead(QString file_cache[1000][30]);
void update_file_cache(QString file_cache[1000][30], QString data_array[29]);
void update_file_cache(QString file_cache[1000][30], QString Change_Val, int col_index, int row_index);
void populate_table(QTableWidget *Table, QString main_file_cache[1000][30]);