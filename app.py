from flask import Flask
import pandas as pd
import numpy as np 
import psycopg2
import psycopg2.extras as extras
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve, classification_report
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import string 
import math
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

from openpyxl import *
from tkinter import *

from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTEN
from pprint import pprint
from flask import Flask, render_template
from flask import request

# postgres://xyz:XgD0wCoT3xceevDxhjsUGUf2No51seaS@dpg-chh461e7avjbbjtocq30-a.singapore-postgres.render.com/xyz_6o3t

# conn = psycopg2.connect( dbname='epilepsy',host='localhost',user='project',password='Password',port='5432')
conn = psycopg2.connect( dbname='xyz_6o3t',host='dpg-chh461e7avjbbjtocq30-a.singapore-postgres.render.com',user='xyz',password='XgD0wCoT3xceevDxhjsUGUf2No51seaS',port='5432')
curr= conn.cursor()
table='epilepsydata'

df= pd.read_sql("select * from \"epilepsydata\"", conn)
# print(df)


#functions
def clean_data(df):
  
  df.columns= df.columns.str.strip().str.lower()
  
  #remove a. 
  alphabet = list(string.ascii_lowercase)
  columns = list(df)
  for col in columns:
    for i, j in enumerate(df[col]):
      if type(j)==type('aaaa'):
        if len(j)<3:
          continue
        if j[0] in alphabet and j[1:3]==". ":
          df[col][i]=j[3:]
  #adding no to null value      
  for i, j in enumerate(df['wandering_headbanging_observation']):
      if df['wandering_headbanging_observation'][i]!='No' and df['wandering_headbanging_observation'][i]!='Yes':
         df['wandering_headbanging_observation'][i]='No'
            
  #removing uncertain value
  f = df.loc[df['final_diagnosis'] == 'Uncertain'].index
  df.drop(f, inplace = True)
  
  df['patientage'] = df['patientage'].astype(float)
  #convert age from months to years
  for i,age in enumerate(df['patientage']):
    if type(age)==type('aa'):
        s=""
        s1=""
        print(age)
        j=0
        for alpha in age:
          if alpha >='0' and alpha<='9':
            if(alpha=='0' and j!=0):
                s+=alpha
          elif alpha >='a' and alpha<='z':
            s1+=alpha
          j+=1
        s1=s1.lower()
          
        if s1=='month' or s1=='months':
          df['patientage'][i]=float(s)/12
        else:
           df['patientage'][i]=float(s)
  
  

  return df



def pre(df):

    y = df.final_diagnosis
    x = df.drop('final_diagnosis',axis=1)


    sampler = SMOTEN(random_state=10)
    x_train, y_train = sampler.fit_resample(x, y)
    return x_train, y_train


def one_hot1(df):

    new_cols = dict()
    old_cols = []

    for col in df:
        unique = set()
        for entry in df[col]:
            unique.add(entry)
        if len(unique) > 2 and type(list(unique)[0]) != type(1.0):
            old_cols.append(col)
            for category in unique:
                col_name=col+" "+category
                new_cols[col_name] = []
            
                for entry in df[col]:
                    if entry == category:
                        new_cols[col_name].append('Yes')
                    else:
                        new_cols[col_name].append('No')
        
    df.drop(old_cols, axis = 1,inplace=True)
    for col_name in new_cols.keys():
        df[col_name] = new_cols[col_name]
    
   
    return df

numeric_mappings = {}
def mapping(x_train, y_train):
    columns = list(x_train)
    columns.append('final_diagnosis')
    for col in columns:
        rowval=set()
        if col=='final_diagnosis':
            for j in y_train:
                rowval.add(j)
        else:
            for j in x_train[col]:
                rowval.add(j)
        
        enumval=[k for k in range(len(rowval))]
        if col=='final_diagnosis':
            y_train.replace(list(rowval),enumval,inplace =True)
        else:    
            x_train[col].replace(list(rowval),enumval,inplace =True)
        
        
        
        ind = 0
        for rowvals in rowval:
            numeric_mappings[rowvals] = enumval[ind]
            ind+=1
    return x_train,y_train

def verdict(x_test, x_train, y_train):
    clf = LogisticRegression()
    # print(x_train)
    # print(y_train)
    # print(x_test)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    diagnosis=''
    if numeric_mappings['Epileptic Seizure']==1:
        if prediction==1:
            diagnosis='Epileptic Seizure'
        else:
            diagnosis='Non epileptic spell'
    else:
        if prediction==1:
            diagnosis='Non epileptic spell'
        else:
            diagnosis='Epileptic Seizure'

    return diagnosis



def read_user_data():
    columns = list(doctor_ui)

    def insert():
        row = []

        for field_value in field_entry_boxes:
            if type(field_value) == Entry:
                row.append(field_value.get())
            else:
                row.append(field_value[1].get())
    
        if 'Select' in row:
            heading.config( text = 'Please select a value for each dropdown' )
        elif row[-1]=='Uncertain':
            #type casting patientage to float

            row[1]=float(row[1])
            
            x_train, y_train = pre(doctor_ui)
            x_train.loc[len(x_train.index)] = row[:-1]
            
            x_train = one_hot1(x_train)
            x_train, y_train = mapping(x_train, y_train)
            dui2 = x_train.iloc[-1:]
            x_train.drop(x_train.tail(1).index,inplace=True)
            
            heading.config(text = verdict(dui2, x_train, y_train))
            
        else:
            query = 'INSERT INTO %s("%s") VALUES (%s)' % (table, '","'.join(columns), ','.join(['%s'] * len(row)))
            curr.execute(query, row)
            conn.commit()
            heading.config(text = 'Submitted')
        
   
        
    print("comes here")
    root = Tk()
    root.configure()
    root.title("Symptoms Form")
    root.geometry("1800x800")

    heading = Label(root, text="Form")

    labels = []
    field_entry_boxes = []
    col = 0
    row = 0
    run = 0
    for i, column in enumerate(columns):
        labels.append(Label(root, text=column))
        options = set()

        clicked = StringVar()
        clicked.set( "Select" )
        
        is_number = 0
        
        for value in doctor_ui[column]:
            if type(value) == type(0):
                is_number = 1
                break
            options.add(value)

        if is_number == 0:
            if column=='caseno' or column=='patientage':
                field_entry_boxes.append(Entry(root))
                field_entry_boxes[i].grid(row=row, column=col+1, ipadx="100")
            else:
                if column=='final_diagnosis':
                    options.add('Uncertain')
                drop = OptionMenu( root , clicked , *options )
                field_entry_boxes.append([drop, clicked])
                field_entry_boxes[i][0].grid(row=row, column=col+1, ipadx="100")
        else:
            field_entry_boxes.append(Entry(root))
            field_entry_boxes[i].grid(row=row, column=col+1, ipadx="100")
        labels[i].grid(row=row, column=col)
        if(col == 0):
            col = 2
        else:
            col = 0
        run += 1
        run %= 2
        if (run == 0):
            row += 1

    heading.grid(row=row+2, column=4)
    submit = Button(root, text="Submit", command=insert)
    submit.grid(row=row+2, column=1)
    root.mainloop()
    
    return 


doctor_ui = df.copy()
doctor_ui=clean_data(doctor_ui)

app = Flask(__name__)

# @app.route('/retrive',)
# def retrive():
    
    
def scan_db(caseno):
    curr.execute("SELECT * FROM epilepsydata WHERE caseno=%s", (caseno,))
    data=curr.fetchall()
    if(len(data)==0):
        return False,data 
    else:
        return True,data   
    


@app.route('/',methods =["GET", "POST"])
def hello_world():
   
    if request.method == "POST":
        caseno = request.form.get("caseno")
        #if caseno exits in db 
		#flag var
        flag,data=scan_db(caseno)
        print(caseno)
        # print(flag)
        if flag==True:
            # print(caseno)
            curr.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'epilepsydata'")
            columns=curr.fetchall()
            for i in range(len(columns)):
                columns[i]=str(columns[i])
                columns[i]=columns[i].replace('(', '').replace(')', '').replace("'", '').replace('_',' ').replace(',','')


            mapping = dict(zip(list(columns), data[0]))
            return render_template('home.html', my_dict=mapping)
            # return mapping

        
        columns = list(doctor_ui)
        read_user_data()
        for column in columns:
            for value in doctor_ui[column]:
                if value == 'Select':
                    return
    
    
    return render_template('home.html',my_dict=dict())
    


if __name__ == '__main__':
    app.run()
    conn.close()
    curr.close()
