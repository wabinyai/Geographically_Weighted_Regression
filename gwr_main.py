

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysal as ps
from pysal.lib import weights
from pysal.model import spreg

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = [16,8]

import pandas as pd
import geopandas as gpd

fn=r'/content/DEC_FEB.csv'
data= pd.read_csv(fn)
data.head()

data['pm_ration'] = (data.pm2_5_calibrated_value)/ (data.pm10_calibrated_value)

data.head()

data['pm_ration'].plot()

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Select the columns to be standardized
columns_to_standardize = ['pm2_5_calibrated_value', 'pm10_calibrated_value', 'humidity', 'temperature', 'wind_speed']

# Create a DataFrame with only the selected columns
selected_columns = data[columns_to_standardize]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
standardized_data = scaler.fit_transform(selected_columns)

# Create a DataFrame with standardized data and column names
standardized_df = pd.DataFrame(standardized_data, columns=columns_to_standardize)

# Concatenate the standardized columns with the non-standardized columns
final_df = pd.concat([data[['site_name','wind_direction' ,'site_latitude', 'site_longitude']], standardized_df], axis=1)

data =final_df

# Assuming you have loaded your spatial data into data_geo
geometry = gpd.points_from_xy(data['site_longitude'], data['site_latitude'])
geo_df_kma= gpd.GeoDataFrame(data, geometry=geometry)

geo_df_kma.head()

geo_df = gpd.read_file('https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_UGA_4.json.zip')

jinja=geo_df[(geo_df.NAME_1=='Kampala')|(geo_df.NAME_1=='Wakiso')|(geo_df.NAME_1=='Mukwano')]

jinja =jinja[['NAME_1','NAME_3','NAME_4','geometry']]

jinja.head()

jinja.plot() 
merged = gpd.sjoin(jinja,geo_df_kma, how='left', op='contains', lsuffix='_polygon', rsuffix='_point')

merged.head()

merged.plot(column='pm2_5_calibrated_value', legend=True)

#merged.plot(column='pm_ration', legend=True)
#plt.title('PM sources')

#merged.plot(column='pm2_5_calibrated_value', legend=True)

merged.NAME_3.unique()

merged =merged.dropna(subset='temperature')

merged =merged.dropna(subset='wind_direction')

merged.info()

import pandas as pd
import geopandas as gpd
from pysal.model import spreg
from pysal.lib import weights
import numpy as np

# Fit the GWR model
y = merged['pm2_5_calibrated_value'].values.reshape((-1,1)) # reshape is needed to have column array
y.shape

#X = merged[['humidity', 'temperature', 'wind_speed', 'wind_direction', 'site_latitude', 'site_longitude']].values
X = merged[['wind_speed','temperature', 'humidity','wind_direction']].values
X.shape

v = merged['site_longitude']
u = merged['site_latitude']
coords = list(zip(u,v))

from mgwr.sel_bw import Sel_BW
gwr_selector = Sel_BW(coords, y, X)
gwr_bw = gwr_selector.search()

print('GWR bandwidth =', gwr_bw)

from mgwr.gwr import GWR, MGWR
gwr_results = GWR(coords, y, X, gwr_bw).fit()
gwr_results.summary()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))
merged.plot(color = 'white', edgecolor = 'black', ax = ax)
merged.centroid.plot(ax=ax)
ax.set_title('Map of Kampala ', fontsize=12)
ax.axis()
#plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
plt.show()

# As reference, here is the (average) R2, AIC, and AICc
print('Mean R2 =', gwr_results.R2)
print('AIC =', gwr_results.aic)
print('AICc =', gwr_results.aicc)

# Add R2 to GeoDataframe
merged['gwr_R2'] = gwr_results.localR2

fig, ax = plt.subplots(figsize=(12, 8))
merged.plot(column='gwr_R2', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=5, legend=True, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=ax)
ax.set_title('Local R2', fontsize=12)
ax.axis("off")
#plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
plt.show()

merged['gwr_intercept'] = gwr_results.params[:,0]
merged['gwr_wind_speed']        = gwr_results.params[:,1]
merged['gwr_temperature']        = gwr_results.params[:,2]
merged['gwr_humidity']     = gwr_results.params[:,3]
merged['gwr_wind_direction']     = gwr_results.params[:,4]

merged[["NAME_1", "NAME_3","NAME_4", "gwr_humidity","gwr_wind_speed"]].sort_values(by="gwr_wind_speed", ascending=True)

# Filter t-values: standard alpha = 0.05
gwr_filtered_t = gwr_results.filter_tvals(alpha = 0.05)

pd.DataFrame(gwr_filtered_t)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,10))

merged.plot(column='gwr_wind_speed', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=7, legend=True, legend_kwds={'bbox_to_anchor':(0.10, 0.96)},  ax=axes[0])

merged.plot(column='gwr_wind_speed', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.6)},  ax=axes[1])
merged[gwr_filtered_t[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])


merged.plot(column='gwr_wind_speed', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[2])
#gdf[gwr_filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])

plt.tight_layout()

axes[0].axis("off")
axes[1].axis("off")
axes[2].axis("off")

axes[0].set_title('(March-May) PM2.5: Wind Speed (BW: ' + str(gwr_bw) +'), all coeffs', fontsize=12)
axes[1].set_title('(b) PM2.5: Wind Speed (BW: ' + str(gwr_bw) +'), significant coeffs',     fontsize=12)
axes[2].set_title('(c) PM2.5: Wind Speed (BW: ' + str(gwr_bw) +'), significant coeffs and corr. p-values',     fontsize=12)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,10))

merged.plot(column='gwr_temperature', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=7, legend=True, legend_kwds={'bbox_to_anchor':(0.10, 0.96)},  ax=axes[0])

merged.plot(column='gwr_temperature', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.6)},  ax=axes[1])
merged[gwr_filtered_t[:,2] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])


merged.plot(column='gwr_temperature', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[2])
#gdf[gwr_filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])

plt.tight_layout()

axes[0].axis("off")
axes[1].axis("off")
axes[2].axis("off")

axes[0].set_title('(March-May) PM2.5: Temperature (BW: ' + str(gwr_bw) +'), all coeffs', fontsize=12)
axes[1].set_title('(b) PM2.5: Temperature (BW: ' + str(gwr_bw) +'), significant coeffs',     fontsize=12)
axes[2].set_title('(c) PM2.5: Temperature (BW: ' + str(gwr_bw) +'), significant coeffs and corr. p-values',     fontsize=12)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,10))

merged.plot(column='gwr_humidity', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=5, legend=True, legend_kwds={'bbox_to_anchor':(0.10, 0.96)},  ax=axes[0])

merged.plot(column='gwr_humidity', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.6)},  ax=axes[1])
merged[gwr_filtered_t[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])


merged.plot(column='gwr_humidity', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[2])
#gdf[gwr_filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])

plt.tight_layout()

axes[0].axis("off")
axes[1].axis("off")
axes[2].axis("off")

axes[0].set_title('(March-May) PM2.5: Humidity (BW: ' + str(gwr_bw) +'), all coeffs', fontsize=12)
axes[1].set_title('(b) PM2.5: Humidity (BW: ' + str(gwr_bw) +'), significant coeffs',     fontsize=12)
axes[2].set_title('(c) PM2.5: Humidity (BW: ' + str(gwr_bw) +'), significant coeffs and corr. p-values',     fontsize=12)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,10))

merged.plot(column='gwr_wind_direction', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=7, legend=True, legend_kwds={'bbox_to_anchor':(0.10, 0.96)},  ax=axes[0])

merged.plot(column='gwr_wind_direction', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.6)},  ax=axes[1])
merged[gwr_filtered_t[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])


merged.plot(column='gwr_wind_direction', cmap = 'coolwarm', linewidth=0.05, scheme = 'FisherJenks', k=7, legend=False, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=axes[2])
#merged[gwr_filtered_tc[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])

plt.tight_layout()

axes[0].axis("off")
axes[1].axis("off")
axes[2].axis("off")

axes[0].set_title('(March-May) PM2.5: Wind direction  (BW: ' + str(gwr_bw) +'), all coeffs', fontsize=12)
axes[1].set_title('(b) PM2.5: Wind direction  (BW: ' + str(gwr_bw) +'), significant coeffs',     fontsize=12)
axes[2].set_title('(c) PM2.5: Wind direction  (BW: ' + str(gwr_bw) +'), significant coeffs and corr. p-values',     fontsize=12)
plt.show()



gwr_p_values_stationarity = gwr_results.spatial_variability(gwr_selector, 50)

gwr_p_values_stationarity

LCC, VIF, CN, VDP = gwr_results.local_collinearity()

pd.DataFrame(VIF)

merged['gwr_CN'] = CN

fig, ax = plt.subplots(figsize=(6, 6))
merged.plot(column='gwr_CN', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=5, legend=True, legend_kwds={'bbox_to_anchor':(0.10, 0.96)},  ax=ax)
ax.set_title('Local multicollinearity (CN > 30)?', fontsize=12)
ax.axis("off")
#plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
plt.show()

data1 = data[data.site_name=='Kireka, Kira Municipality']
data1