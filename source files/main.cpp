#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QStackedWidget>
#include <QGraphicsOpacityEffect>
#include <QPropertyAnimation>
#include <QLineEdit>
#include <QtWebEngineWidgets/QWebEngineView>
#include <QWebEngineProfile>
#include <QWebEngineDownloadRequest>
#include <QStandardPaths>
#include <QComboBox>
#include <QHeaderView>
#include "functions_init.h"
#include <QPixmap>

int main(int argc, char *argv[]) {


    qputenv("QTWEBENGINE_CHROMIUM_FLAGS",
            "--ignore-gpu-blocklist "
            "--enable-gpu-rasterization "
            "--enable-zero-copy "
            "--num-raster-threads=4 "
            "--allow-file-access-from-files "
            "--enable-experimental-web-platform-features");

    QApplication app(argc, argv);

    QStackedWidget *stack = new QStackedWidget();
    QString main_file_cache[1000][30];
    fileRead(main_file_cache);
    // ================= MAIN PAGE =================
    QWidget *mainPage = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout(mainPage);
    layout->setContentsMargins(40, 25, 40, 25);
    layout->setSpacing(0);

    QHBoxLayout *header = new QHBoxLayout();
    header->setContentsMargins(0, 0, 0, 0);

    QVBoxLayout *logoBlock = new QVBoxLayout();
    logoBlock->setSpacing(0);
    logoBlock->setContentsMargins(0, 0, 0, 0);

    QLabel *logo = new QLabel("EyeAgri");
    QPixmap *image = new QPixmap("C:/Users/shaha/Downloads/fopui.zip/fopui/images/Logo landscape.png");
    logo->setPixmap(image->scaled(300, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    logo->setStyleSheet("background: transparent;");

    logo->setAlignment(Qt::AlignLeft);


    logoBlock->setAlignment(Qt::AlignLeft);
    logoBlock->addWidget(logo);


    header->addLayout(logoBlock);


    layout->addLayout(header);
    layout->addStretch(1);

    QLabel *title = new QLabel("EyeAgri - AI based Agriculture Management");
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet(
        "font-size: 62px;"
        "font-weight: 700;"
        "color: #3b2a1a;"
        "font-family: 'Georgia';"
        );
    title->setWordWrap(true);

    QLabel *subtitle = new QLabel("Smarter farming through AI-powered analysis and chat assistance");
    subtitle->setAlignment(Qt::AlignCenter);
    subtitle->setStyleSheet("color:#6b4f2a; font-size:32px;");


    QPushButton *btn1 = new QPushButton("🌿  Image Analysis");
    QPushButton *btn2 = new QPushButton("💬  Chat with EyeAgri");
    QPushButton *btn3 = new QPushButton("📊  Manage Data");
    btn1->setObjectName("mainBtn");
    btn2->setObjectName("mainBtn");
    btn3->setObjectName("mainBtn");
    btn1->setFixedSize(280, 52);
    btn2->setFixedSize(280, 52);
    btn3->setFixedSize(280, 52);

    QVBoxLayout *btnLayout = new QVBoxLayout();
    btnLayout->setSpacing(30);
    btnLayout->addWidget(btn3, 0, Qt::AlignCenter);
    btnLayout->addWidget(btn1, 0, Qt::AlignCenter);
    btnLayout->addWidget(btn2, 0, Qt::AlignCenter);

    layout->addWidget(title);
    layout->addSpacing(40);
    layout->addWidget(subtitle);
    layout->addSpacing(80);
    layout->addLayout(btnLayout);
    layout->addStretch(1);

    // ================= PAGE 1 =================
    QWidget *page1 = new QWidget();
    QVBoxLayout *p1 = new QVBoxLayout(page1);
    QHBoxLayout *topBar1 = new QHBoxLayout();
    QPushButton *back1 = new QPushButton("← Back to Home");
    back1->setObjectName("backBtn");
    topBar1->addWidget(back1);
    topBar1->addStretch();
    topBar1->setContentsMargins(40, 25, 40, 10);
    QLabel *p1Title = new QLabel("Image Analysis");
    p1Title->setAlignment(Qt::AlignCenter);
    p1Title->setStyleSheet("font-family: 'Georgia'; font-size:34px; font-weight:700; color:#1e361b;");
    QLabel *p1Desc = new QLabel("Upload crop images and detect diseases using AI.");
    p1Desc->setAlignment(Qt::AlignCenter);
    p1Desc->setStyleSheet("font-family: 'Segoe UI'; font-size:16px; color:#556b52;");
    QWebEngineProfile *web_prfl = QWebEngineProfile::defaultProfile();
    QWebEngineView *yolo = new QWebEngineView(page1);

    yolo->setUrl(QUrl("https://hashirehtisham-yolo-eyeagri.hf.space"));
    yolo->setMinimumHeight(600);
    yolo->setObjectName("embeddedWebView");
    QObject::connect(web_prfl, &QWebEngineProfile::downloadRequested,[](QWebEngineDownloadRequest *download) {
        QString file_path =QStandardPaths::writableLocation(QStandardPaths::DownloadLocation) ;
        download->setDownloadDirectory(file_path);
        download->setDownloadFileName(download->suggestedFileName());
        download->accept();
    });
    p1->setContentsMargins(0,0,0,0);
    p1->setSpacing(0);
    p1->addLayout(topBar1);
    p1->addWidget(p1Title);
    p1->addWidget(p1Desc);
    p1->addWidget(yolo);
    p1->addStretch();

    // ================= PAGE 2 =================
    QWidget *page2 = new QWidget();
    QVBoxLayout *p2 = new QVBoxLayout(page2);
    QHBoxLayout *topBar2 = new QHBoxLayout();
    QPushButton *back2 = new QPushButton("← Back to Home");

    back2->setObjectName("backBtn");
    p2->setContentsMargins(0, 0, 0, 0);
    p2->setSpacing(0);
    topBar2->addWidget(back2);
    topBar2->addStretch();
    topBar2->setContentsMargins(40, 25, 40, 10);
    QLabel *p2Title = new QLabel("Chat with EyeAgri");
    p2Title->setAlignment(Qt::AlignCenter);
    p2Title->setStyleSheet("font-family: 'Georgia'; font-size:34px; font-weight:700; color:#1e361b;");
    QLabel *p2Desc = new QLabel("Ask questions and get smart farming advice.");
    p2Desc->setAlignment(Qt::AlignCenter);
    p2Desc->setStyleSheet("font-family: 'Segoe UI'; font-size:16px; color:#556b52;");
    QWebEngineView *Eye_Agri_chat = new QWebEngineView(page2);
    Eye_Agri_chat->setMinimumHeight(600);
    Eye_Agri_chat->setObjectName("embeddedWebView");
    Eye_Agri_chat->setUrl(QUrl("http://127.0.0.1:7860"));
    p2->addLayout(topBar2);
    p2->addWidget(p2Title);
    p2->addWidget(p2Desc);
    p2->addWidget(Eye_Agri_chat);
    p2->addStretch();

    // ================= PAGE 3 =================
    QWidget *page3 = new QWidget();
    QVBoxLayout *p3 = new QVBoxLayout(page3);
    QHBoxLayout *topBar3 = new QHBoxLayout();
    QPushButton *back3 = new QPushButton("← Back to Home");
    QPushButton *ReLog = new QPushButton("Refresh datalogging");
    ReLog->setObjectName("backBtn");
    back3->setObjectName("backBtn");
    topBar3->addWidget(back3);
    topBar3->addStretch();
    topBar3->addWidget(ReLog);
    QLabel *p3Title = new QLabel("Data Logging");
    p3Title->setAlignment(Qt::AlignCenter);
    p3Title->setStyleSheet("font-size:34px; font-weight:700; color:#3b2a1a;");
    QLabel *p3Desc = new QLabel("Track and log crop health data over time.");

    QVBoxLayout *inputsystem = new QVBoxLayout();
    QString data_array[31] = {};
    int data_index = 0;
    QString inputLabels[29] = {
        "Log Date (DD/MM/YYYY)",
        "Total Crops",
        "Patch ID",
        "Patch Size",
        "Crop Type",
        "Number of Patches",
        "Crop Start Date (YYYY-MM-DD)",
        "Crop Duration (Days)",
        "Water Usage (Liters)",
        "Irrigation Method",
        "Watering Frequency",
        "Fertilizer Type",
        "Fertilizer Quantity (kg)",
        "N-P-K Ratio",
        "Application Method",
        "Application Time",
        "Soil Moisture Level (%)",
        "Soil Type",
        "Soil pH",
        "Soil Temperature (°C)",
        "Drainage Condition",
        "Air Temperature (°C)",
        "Humidity (%)",
        "Rainfall (Yes/No)",
        "Rain Duration (Hours)",
        "Sunlight Hours",
        "Wind Condition",
        "Plant Height (cm)",
        "Leaf Color"
    };
    QLabel *Prompt = new QLabel(inputLabels[0]);
    QLineEdit *user_input = new QLineEdit();
    QPushButton *next_data_btn = new QPushButton("Next");
    Prompt->setObjectName("prompt");
    next_data_btn->setObjectName("data_page_button");
    next_data_btn->setFixedSize(280,52);




    inputsystem-> addWidget(Prompt);
    inputsystem-> addWidget(user_input);
    inputsystem-> addWidget(next_data_btn);



    QWidget *tablecontainer = new QWidget();
    tablecontainer->setFixedSize(1000,600);
    QTableWidget *data_table = new QTableWidget(tablecontainer);
    data_table->setGeometry(0, 0, 1000, 600);
    tablecontainer->setObjectName("data_table");

    QHBoxLayout *Bottom_bar = new QHBoxLayout();
    QPushButton *view_data = new QPushButton("View File Data");
    view_data->setObjectName("data_page_button");
    QPushButton *refresh = new QPushButton("Refresh Table Data");
    refresh->setObjectName("data_page_button");
    Bottom_bar->addWidget(view_data);
    Bottom_bar->addStretch();
    Bottom_bar->addWidget(refresh);

    QObject::connect(next_data_btn, &QPushButton::clicked, [=,&data_index,&data_array,&inputLabels,&main_file_cache]() mutable {
        if (data_index < 29){
            data_array[data_index] = user_input->text();
            data_index++;
        }
        if (data_index < 29){
            Prompt->setText(inputLabels[data_index]);
            user_input->clear();
            user_input->setFocus();
        }
        else{
            if (data_index == 29){
                Prompt->setText("All Data collected. Storing now.");
                datalog(data_array);
                update_file_cache(main_file_cache,data_array);
                user_input->setEnabled(false);
                next_data_btn->setEnabled(false);
                data_index ++;
            }
        }

    } );
    QObject::connect(ReLog, &QPushButton::clicked, [=,&data_index,&inputLabels]() mutable{
        data_index = 0;
        user_input->clear();
        user_input->setEnabled(true);
        next_data_btn->setEnabled(true);
        Prompt->setText(inputLabels[0]);
    });

    QObject::connect(view_data,&QPushButton::clicked, [=, &data_table, &main_file_cache]() mutable {
        populate_table(data_table,main_file_cache);
    });
    QObject::connect(refresh,&QPushButton::clicked, [=, &data_table, &main_file_cache]() mutable {
        populate_table(data_table,main_file_cache);
    });
    QObject::connect(data_table, &QTableWidget::cellChanged, [=, &main_file_cache](int row, int column) {
        QTableWidgetItem *item = data_table->item(row, column);
        if(item){
            update_file_cache(main_file_cache, item->text(), column, row);
            fileupdate(main_file_cache);
        }
    });
    p3Desc->setAlignment(Qt::AlignCenter);
    p3Desc->setStyleSheet("font-size:16px; color:#6b4f2a;");
    p3->addLayout(topBar3);
    p3->addStretch();
    p3->addWidget(p3Title);
    p3->addWidget(p3Desc);
    p3->addLayout(inputsystem);
    p3->addWidget(tablecontainer);
    p3->addStretch();
    p3->addLayout(Bottom_bar);
    p3->addStretch();


    stack->addWidget(mainPage);
    stack->addWidget(page1);
    stack->addWidget(page2);
    stack->addWidget(page3);
    QObject::connect(btn1, &QPushButton::clicked, [=]() {
        stack->setCurrentIndex(1);
    });
    QObject::connect(btn2, &QPushButton::clicked, [=]() {
        stack->setCurrentIndex(2);
    });
    QObject::connect(btn3, &QPushButton::clicked, [=]() {
        stack->setCurrentIndex(3);
    });

    QObject::connect(back1, &QPushButton::clicked, [=]() {

        stack->setCurrentIndex(0);
    });
    QObject::connect(back2, &QPushButton::clicked, [=]() {

        stack->setCurrentIndex(0);
    });
    QObject::connect(back3, &QPushButton::clicked, [=]() {
        stack->setCurrentIndex(0);
    });

    app.setStyleSheet(R"(
    QWidget {
        font-family: 'Georgia', 'Segoe UI', serif;
        background-color: #eae8e1;
        color: #1e361b;
    }

    QPushButton#mainBtn {
        background-color: #1e361b;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: 600;
        border: none;
        border-radius: 26px;
        padding: 10px 24px;
    }

    QPushButton#mainBtn:hover {
        background-color: #2a4b26;
    }

    QPushButton#mainBtn:pressed {
        background-color: #132411;
    }

    QPushButton#data_page_button {
        background-color: #ffffff;
        color: #1e361b;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid #d1cfc7;
        border-radius: 18px;
        padding: 6px 16px;
        min-height: 36px;
        max-width: 180px;
    }

    QPushButton#data_page_button:hover {
        background-color: #f5f4f0;
        border-color: #1e361b;
    }

    QPushButton#data_page_button:pressed {
        background-color: #e2e0d8;
    }

    QPushButton#data_page_button:disabled {
        background-color: #d8d6ce;
        color: #a1a098;
        border-color: #d8d6ce;
    }

    QPushButton#backBtn {
        background-color: #ffffff;
        color: #1e361b;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid #d1cfc7;
        border-radius: 18px;
        padding: 6px 16px;
        min-height: 36px;
    }

    QPushButton#backBtn:hover {
        background-color: #f5f4f0;
        border-color: #1e361b;
    }

    QPushButton#backBtn:pressed {
        background-color: #e2e0d8;
    }

    QLineEdit {
        background-color: #ffffff;
        border: 1px solid #d1cfc7;
        border-radius: 24px;
        padding: 12px 24px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        color: #1e361b;
        min-height: 48px;
    }

    QLineEdit:focus {
        border: 1px solid #1e361b;
    }

    QLineEdit:disabled {
        background-color: #f5f4f0;
        color: #a1a098;
    }

    QTableWidget#data_table {
        background-color: #ffffff;
        border: 1px solid #ffffff;
        gridline-color: #f2f0ea;
        border-radius: 24px;
        color: #1e361b;
        font-family: 'Segoe UI', sans-serif;
    }

    QTableWidget#data_table::item {
        padding: 12px;
    }

    QTableWidget#data_table::item:selected {
        background-color: #b6cca1;
        color: #1e361b;
    }

    QHeaderView::section {
        background-color: #1e361b;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        padding: 10px;
        border: none;
    }

QLabel#prompt {
        font-family: 'Segoe UI', sans-serif;
        font-size: 15px;
        font-weight: 600;
        color: #1e361b;
        background-color: transparent;
        margin-bottom: 6px;
        padding-left: 4px;
)");

    stack->resize(2560, 1600);
    stack->show();
    return app.exec();
    ;
}