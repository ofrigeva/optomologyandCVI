import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

#data
DR_Adjusted = pd.read_csv("CountyAnyStageDR_AdjustedPrev.csv")
tableHealth = pd.read_csv("CVI-county-pct-cat-CC-Health.gis.csv")
tableSocioEco = pd.read_csv("CVI-county-pct-cat-CC-Social & Economic.gis.csv")
tableEvironment = pd.read_csv("CVI-county-pct-cat-Environment.gis.csv")
tableInfrastructure = pd.read_csv("CVI-county-pct-cat-Infrastructure.gis.csv")

#merge data into one table:
merged_table = pd.merge(DR_Adjusted, tableHealth, on = "FIPS", how = "inner", suffixes=('', '_health'))
merged_table2 = pd.merge(merged_table, tableSocioEco, on = "FIPS", how = "inner", suffixes=('', '_socio'))
merged_table3 = pd.merge(merged_table2, tableEvironment, on = "FIPS", how = "inner", suffixes=('', '_env'))
merged_table4 = pd.merge(merged_table3, tableInfrastructure, on = "FIPS", how = "inner", suffixes=('', '_infra'))
#print(merged_table.head())
#print(merged_table4.columns)
#print(table1.columns)
#print(table2.columns)
#print(merged_table.columns)

#load the data
df = pd.DataFrame(merged_table4)
df.to_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/mergedTable.csv", index = False)
data = pd.read_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/mergedTable.csv")
print("Script is running...")

#prepare data
X = data.drop(columns=['Temperature-related deaths', 'Disaster-related deaths', 'Air pollution-related deaths', 'Air pollution-related illnesses', 'Infectious Diseases', 'Costs of Climate Disasters', 'Economic & Productivity Losses', 'Transition Risks', 'Social Stressors', 'Transportation Sources', 'Exposures & Risks', 'Pollution Sources', 'Criteria Air Pollutants', 'Land Use', 'Transportation', 'Energy', 'Food, Water, and Waste Management', 'Communications', 'Financial Services', 'Governance'])
X = X.select_dtypes(include=[np.number])  #only keep numerical columns
y = data['Prevalence']

#test different train/test splits and track the best performing model
best_mse = float('inf') #setting the 'best' to a very high number (infinity) so that it is gaurenteed that the actal best is less than the default value
best_split = None
best_model = None

split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

print("\nTesting different train/test splits (with max_leaf_nodes=5):")
for test_size in split_ratios: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = DecisionTreeRegressor(max_leaf_nodes=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test size = {int(test_size*100)}% --> MSE: {mse:.4f}")
    
    if mse < best_mse: #update the lowest mse with the best_mse
        best_mse = mse
        best_split = test_size
        best_model = model
        best_X_train = X_train
        best_X_test = X_test
        best_y_train = y_train
        best_y_test = y_test

print(f"\n✅ Best test split: {int(best_split*100)}% --> MSE: {best_mse:.4f}")

#visualize best model
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=X.columns, filled=True, rounded=True)
plt.title(f"Best Decision Tree (test split = {int(best_split*100)}%)")
plt.show()
