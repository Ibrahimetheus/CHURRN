#!/usr/bin/env python
# coding: utf-8

# # Step 1: Importing packages

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# # Step 2: Load Dataset

# In[6]:


df = pd.read_csv("7 churn.csv")


# In[7]:


df.head(4)


# In[8]:


df['Contract'].value_counts()


# # Step 3: Data Preprocessing
# 
# Perform data preprocessing tasks such as handling missing values, encoding categorical variables, and feature scaling:

# In[9]:


df.shape #rows and columns in the data 


# In[10]:


df.isnull().sum() #null values


# # Keeping important columns
# ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'Contract', 'TotalCharges', 'Churn']

# In[11]:


# Define the columns to keep feature selection
columns_to_keep = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'Contract', 'TotalCharges', 'Churn']
# Select only the specified columns
df = df[columns_to_keep]


# In[12]:


df.head()


# In[ ]:





# # Encode binary variables (e.g., Yes/No columns)
# binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
# 

# In[13]:


#use label encoder
from sklearn.preprocessing import LabelEncoder
# Initialize the LabelEncoder
label_encoder = LabelEncoder()
# List of columns to label encode
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract', 'Churn']
# Apply label encoding to each column
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[14]:


df


# # Alternative Techniques

# In[16]:


# ALternative technique
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'Churn']
df[binary_columns] = df[binary_columns].replace({'Yes':1,'No':0})


# In[17]:


df[['MultipleLines','Contract']] = df[['MultipleLines','Contract']].replace({'Yes':1,'No':0,'No phone service':2,"Month-to-month":1,'One year':2,'Two year':3})


# In[18]:


df['gender'] = df['gender'].replace({'Male':1,'Female':0})


# In[ ]:





# # Split the dataset into training and testing sets

# In[19]:


# Split the dataset into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']


# In[20]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Convert 'TotalCharges' column to float, and handle errors='coerce' to replace non-numeric values with NaN
X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'], errors='coerce')
X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'], errors='coerce')


# In[22]:


# Replace missing values in the 'TotalCharges' column with the mean of the column
X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(), inplace=True)


# In[23]:


X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(), inplace=True)


# # Standardize features (optional but often beneficial for logistic regression)

# In[24]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Logistic regression

# In[25]:


lg = LogisticRegression()
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)


# In[26]:


y_pred


# # Accuracy score

# In[27]:


from sklearn.metrics import accuracy_score


# In[28]:


accuracy_score(y_test,y_pred)


# # save model

# In[29]:


import pickle
pickle.dump(lg,open("7 logistic_model.pkl",'wb'))


# In[30]:


df.head()


# # Classification system for Web Interface
# 

# In[31]:


def prediction(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge):
    data = {
    'gender': [gender],
    'SeniorCitizen': [Dependents],
    'Partner': [Partner],
    'Dependents': [Phoneservice],
    'tenure': [tenure],
    'PhoneService': [Phoneservice],
    'MultipleLines': [multiline],
    'Contract': [contact],
    'TotalCharges': [totalcharge]
    }
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)


    # Encode the categorical columns
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column]) 
    df = scaler.fit_transform(df)

    result = lg.predict(df).reshape(1,-1)
    return result[0]


# In[32]:


gender = "Female"
Seniorcitizen = "No"
Partner = "Yes"
Dependents = "No"
tenure = 1
Phoneservice="No"
multiline = "No phone service"
contact="Month-to-month"
totalcharge = 29.85
result = prediction(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge)

if result==1:
    print('churn')
else:
    print('not churn')


# In[33]:


df


# In[ ]:





# In[34]:


import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

model = pickle.load(open('C:\\Users\\moizk\\Downloads\\7 logistic_model.pkl', 'rb')) 
df = pd.read_csv('C:\\Users\\moizk\\Downloads\\7 churn.csv') 

st.title("Customer Churn Prediction Using Logistic Regression for Classification")
gender = st.selectbox("Select Gender",options=['Female','Male'])
SeniorCitizen = st.selectbox("Your you a senior citizen?", options=['Yes','No'])
Partner = st.selectbox("Do you have partner?", options=['Yes','No'])
Dependents	 = st.selectbox("Are you dependents on other?", options=['Yes','No'])
tenure = st.text_input("Enter Your tenure?")
PhoneService = st.selectbox("Do have phone service?",options=['Yes','No'])
MultipleLines = st.selectbox("Do you have mutlilines servics?", options=['Yes','No','no phone service'])
Contract = st.selectbox("Your Contracts?",options=['One year','Two year','Month-to_month'])
TotalCharges = st.text_input("Enter your Total charges?")


def prediction(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge):
    data = {
    'gender': [gender],
    'SeniorCitizen': [Dependents],
    'Partner': [Partner],
    'Dependents': [Phoneservice],
    'tenure': [tenure],
    'PhoneService': [Phoneservice],
    'MultipleLines': [multiline],
    'Contract': [contact],
    'TotalCharges': [totalcharge]
    }
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)


    # Encode the categorical columns
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    df = scaler.fit_transform(df)

    result = model.predict(df).reshape(1,-1)
    return result[0]



# Tips for Churn Prevention
churn_tips_data = {
    "Tips for Churn Prevention": [
        "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
        "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
        "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
        "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
        "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
        "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
        "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
        "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
        "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
        "Competitive Analysis: Stay aware of what competitors are offering. Ensure your products, services, and workplace environment remain competitive in the market."
    ]
}

# Tips for Customer Retention (Not Churning)
retention_tips_data = {
    "Tips for Customer Retention": [
        "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
        "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
        "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
        "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
        "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
        "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
        "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
        "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
        "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
        "Show Appreciation: Express gratitude to loyal customers and acknowledge their continued support."
    ]
}

# Create DataFrames
churn_tips_df = pd.DataFrame(churn_tips_data)
retention_tips_df = pd.DataFrame(retention_tips_data)

if st.button("Predict churn or not"):
    result = prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract,TotalCharges)
    if result == 1:
        st.title("Churn")
        st.write("Here are 10 tips for Churn Prevention:")
        st.dataframe(churn_tips_df, height=400,width=600)
    else:
        st.title('Not Churn')
        st.write("Here are 10 tips for Customer Retention (Not Churning):")
        st.dataframe(retention_tips_df, height=400,width=400)


# In[ ]:





# In[ ]:




