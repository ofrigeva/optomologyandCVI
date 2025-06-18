import pandas as pd;

DR_Adjusted = pd.read_csv("CountyAnyStageDR_AdjustedPrev.csv");
tableHealth = pd.read_csv("CVI-county-pct-cat-CC-Health.gis.csv");
tableSocioEco = pd.read_csv("CVI-county-pct-cat-CC-Social & Economic.gis.csv");
tableEvironment = pd.read_csv("CVI-county-pct-cat-Environment.gis.csv");
tableInfrastructure = pd.read_csv("CVI-county-pct-cat-Infrastructure.gis.csv");
merged_table = pd.merge(DR_Adjusted, tableHealth, on = "FIPS", how = "inner", suffixes=('', '_health'));
merged_table2 = pd.merge(merged_table, tableSocioEco, on = "FIPS", how = "inner", suffixes=('', '_socio'));
merged_table3 = pd.merge(merged_table2, tableEvironment, on = "FIPS", how = "inner", suffixes=('', '_env'));
merged_table4 = pd.merge(merged_table3, tableInfrastructure, on = "FIPS", how = "inner", suffixes=('', '_infra'));
#print(merged_table.head());
print(merged_table4.columns);
#print(table1.columns);
#print(table2.columns);
#print(merged_table.columns);

df = pd.DataFrame(merged_table4)
df.to_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/mergedTable.csv", index = False)
data = pd.read_csv("/Users/ofri.geva/Desktop/eyeDiseases&SocioeconomicStatus/mergedTable.csv")
print("Script is running...")
X = data.drop(columns=['Prevalence'])
X = X.select_dtypes(include=[np.number])  #only keep numerical columns
y = data['Prevalence']

# split up data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor = DecisionTreeRegressor(random_state=42) #this tells python to use the same "seed" for randomness every time 
#^(like in R, if you want to produce the same list of random numbers every time you run your code, you would use a seed)
regressor.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
