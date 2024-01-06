def calculate_correlation_matrix(all_data):
    # Calculate correlation matrix for the given DataFrame
    corr_matrix = all_data.corr(method='pearson')

    return corr_matrix

def top_correlated_stocks(correlation_matrix, selected_stocks, top_n=10):
    top_correlation = {}

    for stock in selected_stocks:
        correlation = correlation_matrix[stock].drop(stock)
        top_positive = correlation.nlargest(top_n).index.tolist()
        top_negative = correlation.nsmallest(top_n).index.tolist()
        top_correlation[stock] = (top_positive, top_negative)
    return top_correlation