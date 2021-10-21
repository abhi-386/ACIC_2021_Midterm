# ACIC_2021_Midterm

This repository was made for a midterm project in the course 'Applied Concepts in Cyberinfrastructure' during the Fall 2021 Semester at University of Arizona.
The goal of this midterm project was form a team and submit a competitive entry to the MLCAS2021 Crop Yield Prediction Challenge.

Link to competition website: https://eval.ai/web/challenges/challenge-page/1251/overview

In this case the challenge was to create, train, test, and evaluate a Machine Learning model for predicting crop yields based on historical weather data, location, and genotype

Team name: Cyber Crop-Bots

Team members, Sebastian Calleja, Nik Pearce, Hayden Payne, Abhishek Agarwal, and Melanie Grudinschi

The following resources were used in building the model:

Background paper from competition: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252402#sec019  
    
GitHub repository from authors of above paper: https://github.com/tryambakganguly/Yield-Prediction-Temporal-Attention
    
Tutorial on multivariate time series forcasting with LSTM models: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/


Guide to this repository:
- data_prep.py is code written to prepare and format the raw weather, trait, and yield data (see Data folder and dataset information below) so it can be used for the model
- lstm_weather_prep.py is code written to build an LSTM model using a converted supervised time series of weather data, use the model to make predictions of crop yield from a test set of data, and evaluate the predictions based on the criteria of root mean squared error (RMSE).
- lstm_traits_prep.py is code written to build an LSTM model using a converted supervised time series of trait data, use the model to make predictions of crop yield from a test set of data, and evaluate the predictions based on the criteria of root mean squared error (RMSE).
- Final_data.ipynb is a Jupyter Notebook that structures a final data set in csv form from the various npy files as input.
- Data folder 
    - Dataset_Competition.zip was provided by the competition (see Information on datasets below) and contains all raw data used in this code
    - Description.txt describes the datasets used (see also Information on datasets below)
    - avg_performance_record.csv (created from inputs_weather_train.npy) groups all weather data by performance record and averages annual values for weather variables
    - trait_df.csv (created from inputs_others_train.npy and yield_train.npy) organizes the numpy array into a dataframe and adds a column for yield to the rest of the others_train dataset
    - weather_short.csv is a shortened version of input_weather_train.npy that only includes values with performance record '0' for the purpose of easier testing

Information on datasets:
All data was provided by MLCAS2021 Crop Yield Prediction Challenge and was downloaded from the competition website link:
https://eval.ai/web/challenges/challenge-page/1251/overview

The zip folder provided consists of two sub-folders: Training, Test Inputs.

The training dataset comprises 93,028 performance records. The Training folder consists of the following files:
  - inputs_weather_train.npy: For each record, daily weather data - a total of 214 days spanning the crop growing season (defined April 1 through October 31). Daily weather    records were compiled based on the nearest grid point from a gridded 30km product. Each day is represented by the following 7 weather variables:
      - Average Direct Normal Irradiance (ADNI)
      - Average Precipitation (AP)
      - Average Relative Humidity (ARH)
      - Maximum Direct Normal Irradiance (MDNI)
      - Maximum Surface Temperature (MaxSur)
      - Minimum Surface Temperature (MinSur)
      - Average Surface Temperature (AvgSur)
  - inputs_others_train.npy: Maturity Group (MG), Genotype ID, State, Year, and Location for each performance record.
  - yield_train.npy: Yearly crop yield value for each record.

The test dataset comprises 10,337 performance records. The Test Inputs folder consists of the following files:
    - inputs_weather_test.npy: Daily weather data for each performance record for a total of 214 days (time-steps). Each day is represented by 7 weather variables: ADNI, AP, ARH, MDNI, MaxSur, MinSur, AvgSur.
    - inputs_others_test.npy: Maturity Group (MG), Genotype ID, State, Year, and Location for each performance record.

We provide genotype clustering information in clusterID_genotype.npy. The file contains cluster ID for each of the 5839 genotypes. Participants may or may not use this information. We developed a completely connected pedigree for all lines with available parentage information, resulting in the formation of a 5839 x 5839 correlation matrix. From the correlation matrix, we used the K-means algorithm for clustering. 

Please refer to the paper (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252402#sec01) for additional details.
