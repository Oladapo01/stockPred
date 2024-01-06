def calculate_correlation_matrix(all_data):
    # Calculate correlation matrix for the given DataFrame
    corr_matrix = all_data.corr(method='pearson')
    # print(f"Correlation matrix shape: {corr_matrix.shape}")

    return corr_matrix

def top_correlated_stocks(corr_matrix, selected_stocks, top_n=10):

    top_correlation = {}
    # print(f"Correlation matrix: {corr_matrix.columns}")
    for stock in selected_stocks:
        # Get correlations for the current stock and drop its own correlation value
        correlations = corr_matrix.loc[stock].drop(stock)
        # Get top positive and negative correlated stocks
        top_positive = correlations.nlargest(top_n).index.tolist()
        top_negative = correlations.nsmallest(top_n).index.tolist()
        # Add the top correlated stocks to the dictionary
        top_correlation[stock] = (top_positive, top_negative)
    return top_correlation