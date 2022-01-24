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


# Clustering

def kMeansCluster(df, chosen_features, expected_clusters):
    
    label_a, label_b = chosen_features
    label_no = []

    
    kmeans = cluster.KMeans(n_clusters = expected_clusters)

    df_cluster = df[chosen_features].copy()
    kmeans.fit(df_cluster)


    # extract labels and cluster centres
    labels = kmeans.labels_
    center = kmeans.cluster_centers_

    # plot using the labels to select colour
    plt.figure(figsize=(5.0,5.0))

    colour = sns.color_palette("hls", expected_clusters)
    for l in range(expected_clusters):     # loop over the different labels
        plt.plot(df_cluster[label_a][labels==l], df_cluster[label_b][labels==l], "o", markersize=3, color=colour[l])
        label_no.append(l)
        
        plt.legend(label_no, loc='center left', bbox_to_anchor=(1, 0.5))
    # show cluster centres
    for ic in range(expected_clusters):
        xc, yc = center[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)
        
        
    if label_a == 'R_output':
        label_a = 'Renewable Energy Output'
    elif label_b == 'R_output':
        label_b = 'Renewable Energy Output'

    plt.xlabel(label_a)
    plt.ylabel(label_b)
    plt.title(f'k-Means Cluster (of {expected_clusters})')
    plt.show()
    
    df["labels"] = labels
    df = df.sort_values(["labels"], ignore_index=False)

    return df

exp_clusters = 4
features = ['GDP','R_output' ]

kmeansdf = kMeansCluster(snap, features, exp_clusters)
kmeansdf

def ACCluster(df, chosen_features, expected_clusters):
    
    label_no = []
    # Agglomerative clustering
    ac = cluster.AgglomerativeClustering(n_clusters=expected_clusters)

    label_a, label_b = chosen_features

    # carry out the fitting
    df_cluster = df[chosen_features].copy()
    ac.fit(df_cluster)

    labels = ac.labels_

    # The clusterer does not return cluster centres, but they are easily computed
    xcen = []
    ycen = []
    for ic in range(expected_clusters):
        xc = np.average(df_cluster[label_a][labels==ic])
        yc = np.average(df_cluster[label_b][labels==ic])
        xcen.append(xc)
        ycen.append(yc)

    # plot using the labels to select colour
    plt.figure(figsize=(5.0,5.0), dpi = 200)

    colour = sns.color_palette("hls", expected_clusters)
    for l in range(expected_clusters):     # loop over the different labels
        plt.plot(df_cluster[label_a][labels==l], df_cluster[label_b][labels==l], "o", markersize=3, color=colour[l])
        label_no.append(l)
        
        plt.legend(label_no, loc='center left', bbox_to_anchor=(1, 0.5))

    # show cluster centres
    for ic in range(expected_clusters):
        plt.plot(xcen[ic], ycen[ic], "dk", markersize=10)

    if label_a == 'R_output':
        label_a = 'Renewable Energy Output'
    elif label_b == 'R_output':
        label_b = 'Renewable Energy Output'

    plt.xlabel(label_a)
    plt.ylabel(label_b)
    plt.title(f'Agglomerative Cluster (of {expected_clusters})')
    plt.grid()
    plt.show()

    df["labels"] = labels
    df = df.sort_values(["labels"], ignore_index=False)
    

    return df

ac_df = ACCluster(snap, features, exp_clusters)

# Using boolean indexing:

df = ac_df

label0 = df[df['labels'] == 0]
label1 = df[df['labels'] == 1]
label2 = df[df['labels'] == 2]
label3 = df[df['labels'] == 3]

label0.sort_values('GDP', ascending=False)


label_sorted = [label0, label1, label2, label3]


#choosing countries at random
import random as rd
rd.seed(1234)
cc = []

for df in label_sorted:
    df.reset_index()
    count = (df.count()[0]+1)
    index_value = rd.randint(0, count)
    print(index_value)
    df = df.iloc[index_value]
    cc.append(df.name)
    
def fitPrep(df, countries):
    chosen_df = pd.DataFrame()
    

    df = df.transpose()
    for i in countries:
        if i in df.index:
            chosen_df[i] = df.loc[i]
    df = chosen_df
    df = df.dropna()
    
    #getting years
    years = []
    for year in df.index:
        years.append(int(str(year)[:4]))
        
    df.index = years
    df.index.names = ['Year']
    
    # Creating errors:
    for country in countries:
        col_name = f'{country[:3]} ($\epsilon$)'
        error_val = np.sqrt(df[country])
        df[col_name] = error_val
        

    return df
        
df = fitPrep(data[5], cc)


# In[297]:



countries = cc
def plotSTS(df, countries):
    """Function to plot single time seriers"""

    fig, ax = plt.subplots(len(countries), sharex = True, dpi = 200)
    colour = sns.color_palette("hls", len(countries))

    for i in range(len(countries)):
        country = countries[i]
        ax[i].plot(df.index, df[country], color = colour[i])
        ax[i].set_ylabel(country)
    ax[i].set_xlabel('Year')
        
    fig.suptitle('Renewable Energy Output')
    return 'Process complete'

plotSTS(df, cc)

# Fitting functions:

def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters"""
    f = scale * np.exp(growth * (t-1990))
    
    return f

def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters"""
    f = a / (1.0 + np.exp(-k * (t - t0)))
    
    return f

def poly_fit(t, a7, a6, a5, a4, a3, a2, a1, a0):
    
    poly = [a7, a6, a5, a4, a4, a3, a2, a1, a0]
    x = t - 1990

    f = poly[0]
    for p in poly[1:]:
        f = f * x + p
    return f

def periodic_exp_growth(t, A1, G1, A2, period, bias):
    x = t - 1990
    f = A1 * np.exp(G1 * x) + A2 * x * np.sin(period * x) + bias
    return f


country = cc[2]
country_err = (f'{country[:3]} ($\epsilon$)')
# fit each method

# Exponential
exp_popt, exp_pcov = opt.curve_fit(exp_growth, df.index, df[country], 
                                  sigma=df[country_err], absolute_sigma=True)
exp_name = f'{country[:3]} ($e$)'
df[exp_name] = exp_growth(df.index, * exp_popt)
# Calculate Chi^2
exp_chi = '$e \chi^2$'
df[exp_chi] = ((df[country] - df[exp_name]) / df[country_err])
exp_rchi = df[exp_chi].mean()
exp_rchi = np.round(exp_rchi, 3)


# Logistic
log_popt, log_pcov = opt.curve_fit(logistics, df.index, df[country], 
                            p0=(1e9, 0.5, 2000.0))
logistic_name = f'{country[:3]} ($log$)'
df[logistic_name] = logistics(df.index, * log_popt)
# Calculate Chi^2
log_chi = '$log \chi^2$'
df[log_chi] = ((df[country] - df[logistic_name]) / df[country_err])
log_rchi = df[log_chi].mean()
log_rchi = np.round(log_rchi, 3)


# Poly
pol_popt, pol_pcov = opt.curve_fit(poly_fit, df.index, df[country], 
                                  sigma=df[country_err], absolute_sigma=True)
poly_name = f'{country[:3]} ($poly$)'
df[poly_name] = poly_fit(df.index, * pol_popt)
# Calculate Chi^2
poly_chi = '$poly \chi^2$'
df[poly_chi] = ((df[country] - df[poly_name]) / df[country_err])
poly_rchi = df[poly_chi].mean()
poly_rchi = np.round(poly_rchi, 3)


# Exp + Sin
exb_popt, exb_pcov = opt.curve_fit(periodic_exp_growth, df.index, df[country], 
                                  sigma=df[country_err], absolute_sigma=True)
exb_name = f'{country[:3]} ($e_p$)'
df[exb_name] = periodic_exp_growth(df.index, * exb_popt)


# Now we need to assess the perfromance, for this we use $\chi^2$


# Plotting:
# Inital

sns.set_palette('hls', 3)

plt.figure(dpi =200)
plt.title(f'{country}: Curve Fitting')
plt.plot(df.index, df[country], label = 'Data', alpha = 0.9, color = 'black')

# Exponential
exp_label = f'$e$ Fit | $\chi^2_r= ${exp_rchi}'
plt.plot(df.index, df[exp_name], label = exp_label, linestyle = '-.')

#plt.legend()
#plt.show()

# Inital
#plt.figure()
#plt.title('Logistic fitting')
#plt.plot(df.index, df[country], label = 'Data')
# Logistic
log_label = f'$log$ Fit | $\chi^2_r= ${log_rchi}'
plt.plot(df.index, df[logistic_name], label = log_label, linestyle = '-.')

#plt.legend()
#plt.show()

# Inital
#plt.figure()
#plt.title('Poly fitting')
#plt.plot(df.index, df[country], label = 'Data')
# Poly
poly_label = f'$poly$ Fit | $\chi^2_r= ${poly_rchi}'
plt.plot(df.index, df[poly_name], label = poly_label, linestyle = '-.')

plt.grid()
plt.xlabel('Year')
plt.ylabel('Renewable Energy Output %')
plt.legend()
plt.show()


# # Making future predictions
# - Predict values for years up to 2025

p_year = 2025
lb_year = df.index.max()

diff = p_year - lb_year

x = np.linspace(lb_year,p_year, diff).astype(int)

y = []
for i in x:
    y.append(poly_fit(i, * pol_popt))

poly_values = pd.DataFrame(y, index = x, columns = [country])

y = []
for i in x:
    y.append(exp_growth(i, * exp_popt))

exp_values = pd.DataFrame(y, index = x, columns = [country])

y = []
for i in x:
    y.append(logistics(i, * log_popt))

log_values = pd.DataFrame(y, index = x, columns = [country])



# Inital

# Anotating points on the graph:
mark = [1,5,7,9] 

sns.set_palette('hls',3)

plt.figure(dpi = 200)
plt.title('Estimated Predictions')
plt.plot(df.index, df[country], label = 'Data', color = 'black')
plt.plot(poly_values.index, exp_values[country], label = '$e$ Prediction',
         linestyle = ':', alpha = 0.5)
plt.plot(poly_values.index, log_values[country], label = '$log$ Prediction',
         linestyle = '-.', alpha = 0.5)
plt.plot(poly_values.index, poly_values[country], '-o', label = '$poly$ Prediction',
         linestyle = '--', markevery = mark)


# Anootating points on the graph:
annotated_years = [2016, 2020, 2022, 2025]
#plt.set_markers(mark)

for year in annotated_years:
    y_loc = poly_values.loc[year]
    annotation = f'{year}: {np.round(y_loc[0],2)}%'
    
    plt.annotate(annotation, # this is the text
                 (year,y_loc), # these are the coordinates to position the label
                 textcoords = "offset points", # how to position the text
                 xytext =(0,1), # distance from text to points (x,y)
                 ha ='right')

plt.grid()
plt.xlabel('Year')
plt.ylabel('Renewable Energy Output %')
plt.legend()
plt.show()
