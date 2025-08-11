from data_analysis import DataAnalyzer

# Create an instance
analyzer = DataAnalyzer()

# Then call the methods on the instance
overview = analyzer.get_dataset_overview()
missing_values = analyzer.get_missing_value()
duplicates = analyzer.get_duplicate_rows()
ihtiyac_statistics = analyzer.analyze_ihtiyac_statistics()
sorted_data = analyzer.plot_ihtiyac_distribution()

print(f"High level overview: \n {overview}")
print()
print(f"Number of features: \n {overview.num_features}")
print()
print(f"Number of rows: \n {overview.num_rows}")
print()
print(f"Feature names: \n {overview.feature_names}")
print()
print(f"Missing values: {missing_values}")
print()
print(f"Duplicated values: {duplicates}")
print()
print(f"Ihtiyac statistics: {ihtiyac_statistics}")

corr_results = analyzer.analyze_feature_correlations()

print("Correlations with İhtiyaç Kg:")
for feature, correlation in corr_results.correlation_with_target.items():
    print(f"{feature}: {correlation:.3f}")

cat_results = analyzer.analyze_categorical_contribution()
for feature, score in sorted(cat_results.feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.2f}")
