# Insight_Analysis_For_Any_CSV_File_using_Python_StreamLit
Not everybody knows Python. So if not python will be not able to analyse the data?. Not so! 
     
Here is automated analysis genie satisfying all the basic analytical needs a beginner wants.Automated Insight Analysis is a composition of Python library StreamLit which was used in developing a website based application. You upload the csv file, we process it for you.

All you need to do is use Visual Basic code editor for compilng and a web browser for viewing the output.

# INSTALATION
        Install necessary libraries using pip install command in the terminal
               EXAMPLE:
                     pip install streamlit
             
# RUNNING PROCESS
         streamlit run appname.py
         
# Functional Requirements

1. Data collection
2. Data processing
3. Training and Testing
4. Modeling
5. Predicting

# SYSTEM COMPONENTS(MODULES)

IMPORT LIBRARIES

In this module, different libraries like pandas, NumPy, matplotlib, and statsmodels are imported, which are useful for data processing, visualization, 
accuracy, and prediction.

DATA PROCESSING

In this module, the data undergoes pre-processing like cleaning the dataset and removing null and unwanted values.

VISUALIZATION

1. Initially, the data undergoes visualization for the data given by the user.
2. Next for the entered Data_set, visualization is done.
3. Next comparing the two columns to plot it  (i.e for the initial and final data in the data set).
4. Then the graph is visualised.

BUILT ARIMA MODEL

1. An autoregressive integrated moving average model, a statistical analysis model that uses time series data to better understand the data set or predict future trends.
2. Here the values in the data set range from 2001 to 2012(Year may vary according to the users dataset).
3. Here the data is fed from 2013 to 2022 manually.
4. Then the model is built and the graph is displayed.

BUILT SARIMA MODEL

1. A seasonal autoregressive integrated moving average model which is like ARIMA but more powerful.
2. We can use statsmodels implementation of SARIMA.
3. Then the model is built and the graph is displayed using a seasonal pattern.
