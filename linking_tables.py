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

