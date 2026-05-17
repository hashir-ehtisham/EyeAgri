#include <iostream>
#include <string>
#include <fstream>
#include <QString>
#include "functions_init.h"
#include <QTableWidget>
using namespace std;
 int index = 0;
string filepath = "C:/Users/shaha/Downloads/csvtocheck.csv";
void fileRead(QString file_cache[1000][30]){
    QString line;
    string element ="";

    std::ifstream file(filepath);

    if(file.is_open()){
        while(index < 1000 && file.peek() != EOF){
            for(int i = 0;i<30;i++){
                if (i == 29){
                    getline(file,element,'\n');
                    file_cache[index][i] = QString::fromStdString(element);
                }else{
                    getline(file,element,',');
                    file_cache[index][i] = QString::fromStdString(element);
                }
            }
            index++;
        }
        file.close();
    }
    else{
        std::cout << "The file opening was either denied permission or failed to open.\n";

    }
}

void fileupdate(QString file_cache[1000][30]){
    std::ofstream file_out(filepath);
    if(file_out.is_open()){
        for (int i = 0; i< index; i++){
            for( int j = 0;j < 30; j++){
                if ( j == 29){
                    file_out << file_cache[i][j].toStdString() << "\n";
                }
                else{
                    file_out << file_cache[i][j].toStdString() << ",";
                }
                }
        }
        file_out.close();
    }
    else{
        cout << "File could not be written to\n";
    }

}

void update_file_cache(QString file_cache[1000][30], QString data_array[29]){
    for(int i = 0; i <29;i++){
        file_cache[index][i] = data_array[i];
    }
    file_cache[index][29] = "healthy";
    index++;
    fileupdate(file_cache);

}
void update_file_cache(QString file_cache[1000][30], QString Change_Val, int col_index, int row_index){
    if(row_index < index && col_index < 30){
        file_cache[row_index][col_index] = Change_Val;
    }
    else{
        if(row_index >= index){
            cout << "Invalid Row index.\n";
        }
        if(col_index >29){
            cout << "Invalid Column index\n";
        }

    }
}

void populate_table(QTableWidget *Table, QString main_file_cache[1000][30]){
    if(!Table) return;
    Table->blockSignals(true);


    Table->setRowCount(index);
    Table->setColumnCount(30);
    for (int i = 0; i < index; ++i) {
        for (int j = 0; j < 30; ++j) {
            QTableWidgetItem *item = Table->item(i, j);
            if (!item) {
                item = new QTableWidgetItem();
                Table->setItem(i, j, item);
            }
            item->setText(main_file_cache[i][j]);
        }
    }
    Table->blockSignals(false);
}