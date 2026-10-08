from sqlalchemy import create_engine
from stock_chatbot.data_loader import load_clustering_data
from stock_chatbot.clustering_preprocessing import preprocess_data
from stock_chatbot.clustering_model import perform_clustering
from stock_chatbot.clustering_analysis import (
    assign_clusters,
    analyse_clusters,
    assign_cluster_categories,
    label_cluster_categories,
    find_centroid,
    create_result_df,
    get_safest_companies
)

from stock_chatbot.config import get_db_url, setup_logger

#definign logger
logger = setup_logger()

def main():
    # Load data from database using the existing loader
    df = load_clustering_data()
    
    # Process the data
    summary_df, filtered_df, scaled_features = preprocess_data(df)
    
    # Perform clustering
    model, labels = perform_clustering(scaled_features)
    
    # Analyze results
    filtered_df = assign_clusters(filtered_df, labels)
    cluster_summary = analyse_clusters(filtered_df)
    cluster_to_category = assign_cluster_categories(cluster_summary)
    filtered_df = label_cluster_categories(filtered_df, cluster_to_category)
    centroid = find_centroid(filtered_df, scaled_features, cluster_to_category, model)
    
    # Create database connection for saving results
    engine = create_engine(get_db_url(), echo=False)
    
    # Save results
    result_df = create_result_df(summary_df, filtered_df, cluster_to_category, engine)

    # Get safest companies from the filtered DataFrame
    companies = get_safest_companies(filtered_df)
    logger.info(f"Safest companies: {companies}")
    logger.info(f"Total number of safest companies: {len(companies)}")

if __name__ == '__main__':
    main()