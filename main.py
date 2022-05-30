from cgi import test
from re import X
from tkinter import Y
from PyQt5 import QtCore, QtGui, QtWidgets


# app = FastAPI(debug=True)

# @app.get('/')
# def home():
#     return {'text':'Khanh cute
# if __name__ == '__main__':
#     uvicorn.run(app)

#  phần xử lí
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 
import sys
import pickle
import numpy as np
#For training
def train() -> None:
    with open('DoAnDataMining/MFG10YearTerminationData_2.csv') as f:
        df = pd.read_csv(f)
    df_filtered = df.replace('unknown',np.nan)
    df_filtered.dropna(inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    dataset = df_filtered.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in dataset.columns[ [i == object for i in dataset.dtypes] ]:
        dataset.loc[:,col] = le.fit_transform(dataset[col])
    dataset = dataset[['age', 'length_of_service', 'population_category', 'hierarchy_job', 'department_service_Customer', 'STATUS']]
 
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
 
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[], remainder='passthrough' )
    x = np.array(ct.fit_transform(x))
 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[], remainder='passthrough' )
    x = np.array(ct.fit_transform(x))
 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
 
#train test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators =10, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    R= classifier.fit(x_train,y_train)
 
    
#Save Model As Pickle File 
    with open('R.pkl','wb') as m:
        pickle.dump(R,m)
    test(x_test,y_test)
 
#Test accuracy of the model 
def test(X_test,Y_test):
    with open('R.pkl','rb') as mod: 
        p=pickle.load(mod)
    pre=p.predict(X_test)
    print (accuracy_score(Y_test,pre)) #Prints the accuracy of the model
 
def find_data_file(filename):
    if getattr(sys, "frozen", False): # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
# The application is not frozen.
        datadir = os.path.dirname( __file__)
    return os.path.join(datadir, filename)
 
def check_input(data) ->int :
    df=pd.DataFrame(data=data,index=[0])
    with open(find_data_file('R.pkl'),'rb') as model:
        p=pickle.load(model)
    op=p.predict(df)
    return op


from PyQt5 import QtCore, QtGui, QtWidgets

# from f import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 524)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 741, 51))
        font = QtGui.QFont()
        font.setPointSize(32)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 120, 170, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setKerning(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 170, 170, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setKerning(False)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 220, 170, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setKerning(False)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 270, 170, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setKerning(False)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(20, 70, 170, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setKerning(False)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_6 = QtWidgets.QLabel(self.centralwidget)
        # self.label_6.setGeometry(QtCore.QRect(30, 200, 111, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # self.label_6.setFont(font)
        # self.label_6.setObjectName("label_6")
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setKerning(False)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(210, 120, 550, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(210, 170, 550, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(210, 220, 550, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_1.setGeometry(QtCore.QRect(210, 70, 550, 31))
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(210, 270, 550, 31))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 360, 251, 71))
        # self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_7.setGeometry(QtCore.QRect(150, 230, 611, 31))
        # self.lineEdit_7.setObjectName("lineEdit_7")
        # self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_8.setGeometry(QtCore.QRect(150, 260, 611, 31))
        # self.lineEdit_8.setObjectName("lineEdit_8")
        # self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_9.setGeometry(QtCore.QRect(150, 290, 611, 31))
        # self.lineEdit_9.setObjectName("lineEdit_9")
        # self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_10.setGeometry(QtCore.QRect(150, 320, 611, 31))
        # self.lineEdit_10.setObjectName("lineEdit_10")
        # self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_10.setGeometry(QtCore.QRect(150, 350, 611, 31))
        # self.lineEdit_10.setObjectName("lineEdit_10")

        font = QtGui.QFont()
        font.setPointSize(32)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(490,360, 251, 71))
        font = QtGui.QFont()
        font.setPointSize(32)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        # self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        # self.lineEdit_6.setGeometry(QtCore.QRect(150, 200, 611, 31))
        # self.lineEdit_6.setObjectName("lineEdit_6")
        # self.label_7 = QtWidgets.QLabel(self.centralwidget)
        # self.label_7.setGeometry(QtCore.QRect(30, 230, 111, 31))
        # font = QtGui.QFont()
        # font.setPointSize(14)
        # font.setKerning(False)
        # self.label_7.setFont(font)
        # self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
 
        self.pushButton.clicked.connect(self.Crun)
        self.pushButton_2.clicked.connect(self.Clr)
        # train()
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Phần mềm dự đoán"))
        self.label.setText(_translate("MainWindow", "DỰ ĐOÁN KHẢ NĂNG CHẤM DỨT HỢP ĐỒNG CỦA NHÂN VIÊN"))
        self.label_1.setText(_translate("MainWindow", "age"))
        self.label_2.setText(_translate("MainWindow", "length_of_service"))
        self.label_3.setText(_translate("MainWindow", "population_category"))
        self.label_4.setText(_translate("MainWindow", "hierarchy_job"))
        self.label_5.setText(_translate("MainWindow", "depart_service_Cus"))
        self.pushButton.setText(_translate("MainWindow", "RUN"))
        self.pushButton_2.setText(_translate("MainWindow", "DELETE"))
    def Clr(self) -> None:
        self.lineEdit_1.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()
        self.lineEdit_5.clear()
    def Crun(self) -> None:
        my_dict =   {"age":float(self.lineEdit_1.text()), "length_of_service":float(self.lineEdit_2.text()), "population_category":float(self.lineEdit_3.text())
        , "hierarchy_job":float(self.lineEdit_4.text()), "department_service_Customer":float(self.lineEdit_5.text())} 
        t=str('Nhân Viên')
        print(my_dict)
    
        output = check_input(my_dict)
        print(output)  
 
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        a = ""
        if output == 0:
            a="KHÔNG CÓ KHẢ NĂNG"
            msg.setInformativeText(" {} {}  chấm dứt hợp đồng".format(t,str(a)))
            
        elif output ==1:
            a="CÓ KHẢ NĂNG"
            msg.setInformativeText(" {} {}  chấm dứt hợp đồng".format(t,str(a)))
        msg.setWindowTitle("Kết quả")
        msg.exec_() 
    
    # from sklearn.metrics import accuracy_score

if __name__ == '__main__':
        train()        
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())