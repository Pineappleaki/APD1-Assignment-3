# Import the modules needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import scipy.optimize as opt

def refineDataframe(df, years, select_years=False):
    
    # Getting rid of unncessecary columns and rows
    df = df.reset_index(drop = True)
    df = df.iloc[2:]
    df = df.transpose()
    df = df.rename(columns=df.iloc[0])
    df = df.reset_index(drop = True)
    
    df = df.drop([0, 1, 2, 3])
    df = df.reset_index(drop = True)

    
    # Refine the df to chosen years
    refined_df = []
    if select_years:
        for i in range(len(df)):
            if years[0] <= df.iloc[i,0] <= years[1]:
                refined_df.append(df.iloc[i])

        refined_df = pd.DataFrame(refined_df)

        refined_df = refined_df.rename(columns={'Country Name': 'Year'})
        refined_df['Year'] = pd.to_datetime(refined_df['Year'], format = '%Y')
        refined_df = refined_df.set_index('Year')
        
        df = refined_df
    
    return df
    
def selectCountries(df, selected_countries, add = True):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    selected_countries : TYPE
        DESCRIPTION.


    Returns
    -------
    selective_df : TYPE
        DESCRIPTION.

    """

    """
    Refine the data to select specific countries of intrest

    Parameters
    ----------
    df : PANDAS_DATAFRAME
        Dataframe to be modified.
    selected_countries : LIST
        List of countries to be selected.

    Returns
    -------
    selective_df : PANDAS_DATAFRAME
        Dataframe containing only the selected countries.

    """
    
    
    selective_df = [df.iloc[1]]  # initating the selective_df list 
    for i in range(len(df)):
        if (df.iloc[i, 1] in selected_countries) is add:  # Country in list:
            selective_df.append(df.iloc[i])  # Add country to list
        else:
            pass
    
    
    selective_df = pd.DataFrame(selective_df)
    selective_df = selective_df.reset_index(drop = True)
    selective_df = selective_df.drop(0)
    selective_df = selective_df.reset_index(drop = True)
    return selective_df


# Next we need to bring in the relevant data
# Identify file locations
co2_file = './data/co2_emissions.xls'
forest_file = './data/forest_area.xls'
gdp_file = './data/gdp_per_cap.xls'
electric_file = './data/power_consumption.xls'
renewable_file = './data/renewable_energy_use.xls'
renewableout_file = './data/renewable_energy_out.xls'
urban_pop_file = './data/pop_total.xls'

# Load data into pandas df
df1 = pd.read_excel(co2_file)
df2 = pd.read_excel(forest_file)
df3 = pd.read_excel(gdp_file)
df4 = pd.read_excel(electric_file)
df5 = pd.read_excel(renewable_file)
df6 = pd.read_excel(renewableout_file)
df7 = pd.read_excel(urban_pop_file)

data = [df1, df2, df3, df4, df5, df6, df7]
indicator_names = ['CO2', 'Forest Area', 'GDP', 'Electric Consumption', 'Renewable','R_output', 'Pop']
non_countries = ['AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 
                 'EMU', 'TMN', 'TSA', 'TSS', 'UMC'  'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 
                 'IBT', 'IDA', 'IDB', 'IDX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 
                 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 
                 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA']

years = [1960, 2020]
refined_data = []
for df in data:
    df = selectCountries(df, non_countries, add = False)
    df = refineDataframe(df, years, select_years = True)
    refined_data.append(df)

data = refined_data

def snapshotData(data, year, feature_names=[], normalise = True):
    
    if isinstance(data, list) != True:
        print( 'Data must be a list!')
    if isinstance(year, int):
        year = str(year)
    if feature_names == []:
        for i in range(len(data)):
            feature_names.append(f'Feature {i+1}')
    elif feature_names != range(len(data)):
        missing_names = ((len(data)) - len(feature_names))
        for i in range(missing_names):
            feature_names.append(f'NaN_{i+1}')

    snapshot = pd.DataFrame()

    for i in range(len(data)):
        df = data[i]
        if normalise == True:
            normalise_df = (df - df.min()) / (df.max() - df.min())
            df = normalise_df
        temp = df.loc[year]
        temp = temp.transpose()

        col_name = feature_names[i]
        temp = temp.iloc[:,0]

        snapshot[col_name] = temp

    snapshot = snapshot.dropna()
    return snapshot

snap = snapshotData(data, 2010, indicator_names)

def makePlot(df):
    k = 0
    combinations = []

    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i == j or ([j,i] in combinations):
                pass
            else:
                combinations.append([i,j])

    colour = sns.color_palette("hls", len(combinations))

    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i == j or ([j,i] in combinations):
                pass
            else:
                fig, ax = plt.subplots()
                ax.scatter(df.iloc[:,i], df.iloc[:,j], color = colour[k])
                ax.set_xlabel(indicator_names[i])
                ax.set_ylabel(indicator_names[j])
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                k += 1
    fig.tight_layout()
makePlot(snap)