#!/usr/bin/env python
# coding: utf-8

# # Healthcare- EDA and Prediction

# In[62]:


import os


# In[63]:


os.getcwd()


# In[64]:


os.chdir("C:/Users/Apollo/OneDrive - Apollo Hospitals Enterprise Ltd/Documents/Python Scripts/archive")


# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


for dirname, _, filenames in os.walk("C:/Users/Apollo/OneDrive - Apollo Hospitals Enterprise Ltd/Documents/Python Scripts"):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# In[67]:


Data_Dictionary = pd.read_excel("Data_Dictionary.xlsx")
First_Health_Camp_Attended = pd.read_csv("First_Health_Camp_Attended.csv")
Health_Camp_Detail = pd.read_csv("Health_Camp_Detail.csv")
Patient_Profile = pd.read_csv("Patient_Profile.csv")
Second_Health_Camp_Attended = pd.read_csv("Second_Health_Camp_Attended.csv")
Third_Health_Camp_Attended = pd.read_csv("Third_Health_Camp_Attended.csv")
Train = pd.read_csv("Train.csv")
Test = pd.read_csv("test.csv")


# In[68]:


Data_Dictionary.head()


# In[69]:


First_Health_Camp_Attended.head()


# In[70]:


Health_Camp_Detail.head()


# In[71]:


Patient_Profile.head()


# In[72]:


Second_Health_Camp_Attended.head()


# In[73]:


Third_Health_Camp_Attended.head()


# In[74]:


Train.head()


# In[75]:


Patient_Profile.info()


# In[76]:


Patient_Profile.describe()


# In[77]:


numeric = ("int16","int32","int64","float16","float32","float64")
numeric_col_patients = Patient_Profile.select_dtypes(include=numeric)


# In[78]:


numeric_col_patients.info()


# In[79]:


numeric_col_patients.isna().sum()


# In[80]:


merged_details=pd.merge(right=Patient_Profile,left=First_Health_Camp_Attended,on="Patient_ID")
merged_details.info()


# In[81]:


merged_details.describe()


# In[82]:


merged_details.head()


# In[83]:


merged_details=merged_details.merge(Health_Camp_Detail,on=
                                   "Health_Camp_ID")
merged_details.head()


# In[84]:


merged_details=merged_details.merge(Second_Health_Camp_Attended,on="Patient_ID")
merged_details.head()


# In[85]:


merged_details=merged_details.merge(Third_Health_Camp_Attended,on="Patient_ID")
merged_details.head()


# In[86]:


merged_details.describe()


# In[87]:


numeric_details=merged_details.select_dtypes(include=numeric)
numeric_details.head()


# In[88]:


merged_details.columns


# In[89]:


merged_details.info


# In[90]:


merged_details.isna().sum()


# In[91]:


merged_details.describe()


# In[92]:


#percentage of missing values in each column

missing_value = (merged_details.isna().sum().sort_values(ascending=False)/len(merged_details))*100
missing_value


# In[93]:


#Plotting a visual of all missing values.

plot = plt.figure(figsize=(5.5,1))
missing_value[missing_value!=0].plot(kind="barh")


# In[94]:


merged_details=merged_details.drop(["Unnamed: 4"],axis=1)
merged_details.head()


# In[95]:


merged_details.columns


# In[96]:


imp_cols = ['Patient_ID', 'Health_Camp_ID_x', 'Donation', 'Health_Score',
       'Income', 'Education_Score', 'Age',
       'Camp_Start_Date', 'Camp_End_Date','Health Score','Number_of_stall_visited']
merged_details[imp_cols]


# In[97]:


merged_details['healthscore'] = merged_details["Health_Score"]+merged_details['Health_Score']


# In[98]:


merged_details.head()


# In[136]:


merged_details.drop(["Health_Score","Health Score"], axis=1,inplace=True)


# In[137]:


merged_details.isna().sum().sort_values(ascending=False)


# In[138]:


merged_details["City_Type"].mode()


# In[139]:


merged_details["Employer_Category"].mode()


# In[140]:


merged_details.City_Type.fillna("H",inplace=True)
merged_details.Employer_Category.fillna("Technology",inplace=True)


# In[141]:


merged_details["City_Type"].mode()


# In[142]:


merged_details.isna().sum().sort_values(ascending=False)


# In[143]:


Col_Pred = ['Employer_Category','Number_of_stall_visited','City_Type','Age','Income','Donation','Education_Score','healthscore']


# In[144]:


Imp_details = merged_details[Col_Pred]
Imp_details.head()


# In[147]:


Imp_details.describe()


# In[148]:


Imp_details.info()


# In[161]:


for col in Imp_details.columns:
    if Imp_details[col].dtype=='int64':
        plt.pie(Imp_details[col].value_counts(),labels=Imp_details[col].unique())
        plt.title("Piechart for {}".format(col))
        plt.show()
    else:
        plt.hist(Imp_details[col].value_counts())
        plt.title("Histogram for {}".format(col))
        plt.show();


# In[162]:


Imp_details.head()


# In[174]:


plt.pie(Imp_details['Employer_Category'].value_counts(),labels=Imp_details["Employer_Category"].unique())
plt.show()


# In[197]:


plt.scatter(x='Employer_Category',y="healthscore",data=Imp_details,s=20)
plt.figure(figsize=(50,30))


# In[199]:


plt.scatter(x=Imp_details['Income'],y="healthscore",data=Imp_details,s=20)
plt.figure(figsize=(50,30))


# In[207]:


#Moving to ML Prediction


Dummy_data = pd.get_dummies(merged_details,prefix=None,prefix_sep='',drop_first=False)
Dummy_data.info()


# In[214]:


Dummy_data["healthscore"]


# In[222]:


x= Dummy_data.iloc[:,:-1]
y=Dummy_data.healthscore
x.shape


# In[217]:


from sklearn.model_selection import train_test_split


# In[227]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.3,random_state=23)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model = model.fit(x_train,y_train)
rsq = model.score(x_train,y_train)
rsq


# In[229]:


rsq=model.score(x_test,y_test)
rsq


# In[ ]:




