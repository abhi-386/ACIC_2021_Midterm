#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import argparse
import os
#------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Individual plant temperature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('dir',
    #                     metavar='dir',
    #                     help='Directory containing geoTIFFs')

    parser.add_argument('-w',
                        '--weather_data',
                        help='Numpy array containing the weather data',
                        metavar='weather_data',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-t',
                        '--trait_data',
                        help='Numpy array containing the traits data',
                        metavar='trait_data',
                        type=str,
                        default=None,
                        required=True)
    
    parser.add_argument('-c',
                        '--cluster_data',
                        help='Numpy array containing the cluster ID data',
                        metavar='cluster_data',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-y',
                        '--yield_data',
                        help='Numpy array containing the yield data',
                        metavar='yield_data',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory where resulting csv will be saved',
                        metavar='str',
                        type=str,
                        default='Thermal_Output')

    return parser.parse_args()
#------------------------------------------

def jdtodatestd (jdate):
    fmt = '%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)

def organize_weather(weather):
    perf_records, days, variables = weather.shape
    out_arr = np.column_stack((np.repeat(np.arange(perf_records),days),weather.reshape(perf_records*days,-1)))
    out_df = pd.DataFrame(out_arr)
    out_df = pd.DataFrame(out_arr, columns=['Performance Record','ADNI', 'AP', 'ARH', 'MDNI', 'MaxSur', 'MinSur', 'AvgSur'])
    out_df['Performance Record'] = pd.to_numeric(out_df['Performance Record']).astype(int)

    ## Counts all of the days in Performance Record
    out_df['Day'] = out_df.groupby('Performance Record').cumcount() + 1

    ## Adds Julian day
    out_df['Julian_date'] = out_df.groupby('Performance Record').cumcount() + 91

    ## Only takes a subset of the dataframe and appends to the rest (much quicker)
    subset = out_df[out_df['Performance Record'] == 0]

    date_list = []
    for i, row in subset.iterrows():
        julian = int(row['Julian_date'])
        date = datetime.datetime.strptime(f'{julian}', '%j').date()
        clean_date = date.strftime("%m-%d")
        date_list.append(clean_date)
    
    date_cycle = cycle(date_list)
    out_df['date'] = [next(date_cycle) for cycle in range(len(out_df))]
    return out_df

def organize_traits(traits, train_yield, cluster_IDs):
    trait_df = pd.DataFrame(traits, columns=['Maturity Group', 'Genotype ID', 'State', 'Year', 'Location'])
    trait_df['Year'] = pd.to_numeric(trait_df['Year'])
    trait_df['Genotype ID'] = pd.to_numeric(trait_df['Genotype ID'])
    trait_df['Year'] = trait_df['Year'].astype(int)
    trait_df['Genotype ID'] = trait_df['Genotype ID'].astype(int)
    yield_df = pd.DataFrame(train_yield)
    yield_df.rename(columns = {0:'Yield'}, inplace = True)
    trait_df['Yield'] = yield_df['Yield']
    
    # Add cluster IDs to trait dataframe
    num_genos = len(clusterID)
    cluster_dict = {}
    for i in range(1, num_genos):
      cID = clusterID[i-1]
      cluster_dict.update({i:cID})
    cluster_list = []
    for i in trait_df.index:
      genotype = trait_df.iloc[i]['Genotype ID']
      cluster = cluster_dict.get(genotype)
      cluster_list.append(cluster)
    trait_df['Cluster'] = cluster_list 
    
    return trait_df.reset_index().rename(columns={'index':'Performance Record'})

# def organize_yield(train_yield):
#     yield_df = pd.DataFrame(train_yield)
#     yield_df.rename(columns = {0:'Yield'}, inplace = True)

# def merge_traits():
#     yield_df = organize_yield(train_yield)
#     traits_df = organize_traits(train_yield)
#     traits_df['Yield'] = yield_df['Yield']
#     return trait_df.reset_index().rename(columns={'index':'Performance Record'})

# def expand_and_merge():
#     merge_traits_df = merge_traits()
#     weather_df = organize_weather()
#     expanded_df = pd.concat([merge_traits_df]*214)
#     expanded_df = expanded_df.sort_values(by = 'Performance Record')
#     final_df = weather_df.merge(expanded_df, on = 'Performance Record')
#     return final_df

#----------------------------------

def main():
    args = get_args()

    weather = np.load(args.weather_data)
    cluster_IDs = np.load(args.cluster_data)
    traits = np.load(args.trait_data)
    
    train_yield = np.load(args.yield_data)
    print('** Data Downloaded **')

    out_df = organize_weather(weather)
    trait_df = organize_traits(traits, train_yield, cluster_IDs)
    #yield_df = organize_yield(train_yield)

    #trait_df = merge_traits()

    expanded_df = pd.concat([trait_df]*214)
    expanded_df = expanded_df.sort_values(by = 'Performance Record')
    print('** Data Organized **')

    n = 200000  #chunk row size
    list_df = [expanded_df[i:i+n] for i in range(0, expanded_df.shape[0],n)]

    res = pd.DataFrame() 

    for chunk in list_df:
        res = pd.concat([res, out_df.merge(chunk, on = 'Performance Record')]) 

    print('** Data split into chunks and merged **')

    #final_df = out_df.merge(expanded_df, on = 'Performance Record')
    #out_df.to_csv(os.path.join(args.outdir, 'weather_data.csv'))
    #expanded_df.to_csv(os.path.join(args.outdir, 'other_data.csv'))
    #final_df.to_csv(os.path.join(args.outdir, 'prepared_data.csv'))

    res.to_csv(os.path.join(args.outdir, 'prepared_data.csv'))
    print(f'Done, see outputs in ./{args.outdir}.')

#----------------------------------
if __name__ == '__main__':
    main()
