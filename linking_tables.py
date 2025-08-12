import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


#data
tableSocioEco = pd.read_csv("CVI-county-pct-cat-Social-Economic.gis.csv")
tableEvironment = pd.read_csv("CVI-county-pct-cat-Environment.gis.csv")
tableInfrastructure = pd.read_csv("CVI-county-pct-cat-Infrastructure.gis.csv")
DR_Adjusted = pd.read_csv("CountyAnyStageDR_AdjustedPrev.csv")
DR_Crude = pd.read_csv("CountyAnyStageDR_CRUDEPrev.csv")
AMD_Adjusted = pd.read_csv("CountyAnyAMD_AdjustedPrev.csv")
AMD_Crude = pd.read_csv("CountyAnyAMD_CRUDE_Prevalence.csv")
AnyVision_Adjusted = pd.read_csv("CountyAnyVisionLossAdjustedPrevalence.csv")
AnyVision_Crude = pd.read_csv("CrudeVisionLoss.csv")
allInOne = pd.read_csv("CVI-county-pct-allinone.gis.csv")

#merge data into one table:
merged_table1 = pd.merge(tableSocioEco, tableEvironment, on = "FIPS", how = "inner", suffixes=('', '_socio'))
merged_table2 = pd.merge(merged_table1, tableInfrastructure, on = "FIPS", how = "inner", suffixes=('', '_env'))
merged_tableDR_Adjusted = pd.merge(merged_table2, DR_Adjusted, on = "FIPS", how = "inner")
merged_tableAMD_Adjusted = pd.merge(merged_table2, AMD_Adjusted, on = "FIPS", how = "inner")
merged_tableAnyVision_Adjusted = pd.merge(merged_table2, AnyVision_Adjusted, on = "FIPS", how = "inner")
# merged_tableDR_Crude = pd.merge(merged_table2, DR_Crude, on = "FIPS", how = "inner")
# merged_tableAMD_Crude = pd.merge(merged_table2, AMD_Crude, on = "FIPS", how = "inner")
# merged_tableAnyVision_Crude = pd.merge(merged_table2, AnyVision_Crude, on = "FIPS", how = "inner")

#prepare data
dfDR_Adjusted = pd.DataFrame(merged_tableDR_Adjusted)
dfAMD_Adjusted = pd.DataFrame(merged_tableAMD_Adjusted)
dfAnyVision_Adjusted = pd.DataFrame(merged_tableAnyVision_Adjusted)
# dfDR_Crude = pd.DataFrame(merged_tableDR_Crude)
# dfAMD_Crude = pd.DataFrame(merged_tableAMD_Crude)
# dfAnyVision_Crude = pd.DataFrame(merged_tableAnyVision_Crude)
dfDR_Adjusted.to_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/merged_tableDR_Adjusted.csv", index = False)
dataDR_Adjusted = pd.read_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/merged_tableDR_Adjusted.csv")
XDR_Adjusted = dataDR_Adjusted[['Transportation Sources', 'Exposures & Risks', 'Pollution Sources', 'Criteria Air Pollutants', 'Land Use', 'Transportation', 'Energy', 'Food, Water, and Waste Management', 'Communications', 'Financial Services', 'Governance', 'Socioeconomic Stressors', 'Housing Composition & Disability', 'Minority Status & Language', 'Housing Type & Transportation']]
XDR_Adjusted = XDR_Adjusted.select_dtypes(include=[np.number])  #only keep numerical columns
yDR_Adjusted = dataDR_Adjusted['Prevalence']
dfAMD_Adjusted.to_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/merged_tableAMD_Adjusted.csv", index = False)
dataAMD_Adjusted = pd.read_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/merged_tableAMD_Adjusted.csv")
XAMD_Adjusted = dataAMD_Adjusted[['Transportation Sources', 'Exposures & Risks', 'Pollution Sources', 'Criteria Air Pollutants', 'Land Use', 'Transportation', 'Energy', 'Food, Water, and Waste Management', 'Communications', 'Financial Services', 'Governance', 'Socioeconomic Stressors', 'Housing Composition & Disability', 'Minority Status & Language', 'Housing Type & Transportation']]
XAMD_Adjusted = XAMD_Adjusted.select_dtypes(include=[np.number])
yAMD_Adjusted = dataAMD_Adjusted['Prevalence']
dfAnyVision_Adjusted.to_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/merged_tableAnyVision_Adjusted.csv", index = False)
dataAnyVision_Adjusted = pd.read_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/merged_tableAnyVision_Adjusted.csv")
XAnyVision_Adjusted = dataAnyVision_Adjusted[['Transportation Sources', 'Exposures & Risks', 'Pollution Sources', 'Criteria Air Pollutants', 'Land Use', 'Transportation', 'Energy', 'Food, Water, and Waste Management', 'Communications', 'Financial Services', 'Governance', 'Socioeconomic Stressors', 'Housing Composition & Disability', 'Minority Status & Language', 'Housing Type & Transportation']]
XAnyVision_Adjusted = XAnyVision_Adjusted.select_dtypes(include=[np.number])
yAnyVision_Adjusted = dataAnyVision_Adjusted['Prevalence']

#test different train/test splits and track the best performing model
bestMseDR_Adjusted = float('inf') #setting the 'best' to a very high number (infinity) so that it is gaurenteed that the actal best is less than the default value
bestSplitDR_Adjusted = None
bestModelDR_Adjusted = None
bestMseAMD_Adjusted = float('inf') #setting the 'best' to a very high number (infinity) so that it is gaurenteed that the actal best is less than the default value
bestSplitAMD_Adjusted = None
bestModelAMD_Adjusted = None
bestMseAnyVision_Adjusted = float('inf') #setting the 'best' to a very high number (infinity) so that it is gaurenteed that the actal best is less than the default value
bestSplitAnyVision_Adjusted = None
bestModelAnyVision_Adjusted = None
splitRatios = [0.1, 0.2, 0.3, 0.4, 0.5]
print("\nTesting different train/test splits (with max_leaf_nodes=5):")
for testSize in splitRatios: 
    XTrain, XTest, yTrain, yTest = train_test_split(XDR_Adjusted, yDR_Adjusted, test_size = testSize, random_state = 42)
    model = DecisionTreeRegressor(random_state=42, min_samples_leaf=100) #max_leaf_nodes=5, 
    model.fit(XTrain, yTrain)
    yPred = model.predict(XTest)
    mse = mean_squared_error(yTest, yPred)
    print(f"Test size = {int(testSize*100)}% --> MSE: {mse:.4f}")
    if mse < bestMseDR_Adjusted: #update the lowest mse with the best_mse
        bestMseDR_Adjusted = mse
        bestSplitDR_Adjusted = testSize
        bestModelDR_Adjusted = model
        bestXtrain = XTrain
        bestXtest = XTest
        bestYTrain = yTrain
        bestYTest = yTest
print("\nTesting different train/test splits (with max_leaf_nodes=5):")
for testSize in splitRatios: 
    XTrain, XTest, yTrain, yTest = train_test_split(XAMD_Adjusted, yAMD_Adjusted, test_size = testSize, random_state = 42)
    model = DecisionTreeRegressor(random_state=42, min_samples_leaf=100) #max_leaf_nodes=5, 
    model.fit(XTrain, yTrain)
    yPred = model.predict(XTest)
    mse = mean_squared_error(yTest, yPred)
    print(f"Test size = {int(testSize*100)}% --> MSE: {mse:.4f}")  
    if mse < bestMseAMD_Adjusted: #update the lowest mse with the best_mse
        bestMseAMD_Adjusted = mse
        bestSplitAMD_Adjusted = testSize
        bestModelAMD_Adjusted = model
        bestXtrain = XTrain
        bestXtest = XTest
        bestYTrain = yTrain
        bestYTest = yTest
print("\nTesting different train/test splits (with max_leaf_nodes=5):")
for testSize in splitRatios: 
    XTrain, XTest, yTrain, yTest = train_test_split(XAnyVision_Adjusted, yAnyVision_Adjusted, test_size = testSize, random_state = 42)
    model = DecisionTreeRegressor(random_state=42, min_samples_leaf=100) #max_leaf_nodes=5, 
    model.fit(XTrain, yTrain)
    yPred = model.predict(XTest)
    mse = mean_squared_error(yTest, yPred)
    print(f"Test size = {int(testSize*100)}% --> MSE: {mse:.4f}")
    if mse < bestMseAnyVision_Adjusted: #update the lowest mse with the best_mse
        bestMseAnyVision_Adjusted = mse
        bestSplitAnyVision_Adjusted = testSize
        bestModelAnyVision_Adjusted = model
        bestXtrain = XTrain
        bestXtest = XTest
        bestYTrain = yTrain
        bestYTest = yTest

#visualize best model
plt.figure(figsize=(20, 10))
plot_tree(bestModelDR_Adjusted, feature_names = XDR_Adjusted.columns, filled = True, rounded = True, impurity = False)
plt.title(f"DR_Adjusted Decision Tree (test split = {int(bestSplitDR_Adjusted*100)}%)")
plt.savefig("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/DR_Adjusted_Tree.png", dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(20, 10))
plot_tree(bestModelAMD_Adjusted, feature_names = XAMD_Adjusted.columns, filled = True, rounded = True, impurity = False)
plt.title(f"AMD_Adjusted Decision Tree (test split = {int(bestSplitAMD_Adjusted*100)}%)")
plt.savefig("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/AMD_Adjusted_Tree.png", dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(20, 10))
plot_tree(bestModelAnyVision_Adjusted, feature_names = XAnyVision_Adjusted.columns, filled = True, rounded = True, impurity = False)
plt.title(f"AnyVision_Adjusted Decision Tree (test split = {int(bestSplitAnyVision_Adjusted*100)}%)")
plt.savefig("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/AnyVision_Adjusted_Tree.png", dpi=300, bbox_inches='tight')
plt.close()

#code for poisson regression

merged_prevalences1 = pd.merge(DR_Adjusted, allInOne, on = "FIPS", how = "inner", suffixes=('', '_DR'))
merged_prevalences2 = pd.merge(merged_prevalences1, AMD_Adjusted[["FIPS", "Prevalence"]], on = "FIPS", how = "inner", suffixes=('', '_AMD'))
merged_prevalences = pd.merge(merged_prevalences2, AnyVision_Adjusted[["FIPS", "Prevalence"]], on = "FIPS", how = "inner", suffixes=('', '_AnyVision'))

allInOne["SEIsumColumn"] = allInOne["Baseline.SocialEconomic"] + allInOne["Baseline.Environment"] + allInOne["Baseline.Infrastructure"]
SEIpercentile1 = np.percentile(allInOne["SEIsumColumn"], 25)
SEIpercentile2 = np.percentile(allInOne["SEIsumColumn"], 50)
SEIpercentile3 = np.percentile(allInOne["SEIsumColumn"], 75)
SEIpercentile4 = np.percentile(allInOne["SEIsumColumn"], 100)

print(allInOne["SEIsumColumn"])
print(allInOne["Baseline.SocialEconomic"])
allInOne["SEI_quartile"] = pd.qcut(
    allInOne["SEIsumColumn"], 
    q=4, 
    labels=["Q1", "Q2", "Q3", "Q4"]
)

merged_prevalences_amd = pd.merge(
    allInOne[['FIPS', 'SEI_quartile']],
    AMD_Adjusted[['FIPS', 'Prevalence']],
    on='FIPS',
    how='inner'
)

model_amd = smf.glm(
    formula="Prevalence ~ C(SEI_quartile)", 
    data=merged_prevalences_amd, 
    family=sm.families.Poisson()
).fit()

coefs = model_amd.params
conf = model_amd.conf_int()
conf.columns = ['2.5%', '97.5%']
conf['RR'] = np.exp(coefs)
conf['lower'] = np.exp(conf['2.5%'])
conf['upper'] = np.exp(conf['97.5%'])

# Slice off the intercept row
conf_subset = conf[1:]

# Calculate asymmetric error bars: (upper - RR), (RR - lower)
yerr = np.array([
    conf_subset['RR'] - conf_subset['lower'],  # lower error
    conf_subset['upper'] - conf_subset['RR']   # upper error
])

# Plot
conf_subset.plot(
    kind='bar',
    y='RR',
    yerr=yerr,
    capsize=4,
    legend=False
)
plt.title("Rate Ratios by SEI Quartile (AMD)")
plt.ylabel("Rate Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

