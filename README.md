# Insight_Analysis_For_Any_CSV_File_using_Python_StreamLit
Not everybody knows Python. So if not python will be not able to analyse the data?. Not so! 
     
Here is automated analysis genie satisfying all the basic analytical needs a beginner wants.Automated Insight Analysis is a composition of Python library StreamLit which was used in developing a website based application. You upload the csv file, we process it for you.

All you need to do is use Visual Basic code editor for compilng and a web browser for viewing the output.

# INSTALLATION
        Install necessary libraries using pip install command in the terminal
               EXAMPLE:
                     pip install streamlit
             
# RUNNING PROCESS
         streamlit run appname.py
         
# Functional Requirements

Data collection
Data processing
Training and Testing
Modeling
Predicting

# SYSTEM COMPONENTS(MODULES)

Import Libraries
In this module, different libraries like pandas, NumPy, matplotlib, and statsmodels are imported, which are useful for data processing,visualization, accuracy, and prediction.

Data Processing
In this module, the data undergoes pre-processing like cleaning the dataset and removing null and unwanted values.

Visualization
Initially, the data undergoes visualization for the crime given by the user.
Next for the entered year, visualization is done for all the crime counts for all the states in India.
Next the crime rate is compared for the years 2001 and 2012 (i.e for the initial and final data in the data set).
Then the graph is plotted against the crime count district-wise for the entered state.
Built ARIMA model
An autoregressive integrated moving average model, a statistical analysis model that uses time series data to better understand the data set or predict future trends.
Here the values in the data set range from 2001 to 2012.
Here the data is fed from 2013 to 2022 manually.
Then the model is built and the graph is displayed.
Built SARIMA model
A seasonal autoregressive integrated moving average model which is like ARIMA but more powerful.
We can use statsmodels implementation of SARIMA.
Then the model is built and the graph is displayed using a seasonal pattern.
