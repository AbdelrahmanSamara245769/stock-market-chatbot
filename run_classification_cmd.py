from sqlalchemy import create_engine
from stock_chatbot.data_loader import load_clustering_data
from stock_chatbot.clustering_preprocessing import preprocess_data
from stock_chatbot.clustering_model import perform_clustering
from stock_chatbot.clustering_analysis import (
    assign_clusters,
    analyse_clusters,
    assign_cluster_categories,
    label_cluster_categories,
    get_safest_companies
)
from stock_chatbot.model import score_new_data, get_predictions_summary
from stock_chatbot.config import get_db_url, setup_logger

# defining logger
logger = setup_logger()

def main():
    # --- CLUSTERING PIPELINE TO GET SAFEST COMPANIES ---
    df = load_clustering_data()
    summary_df, filtered_df, scaled_features = preprocess_data(df)
    model, labels = perform_clustering(scaled_features)
    filtered_df = assign_clusters(filtered_df, labels)
    cluster_summary = analyse_clusters(filtered_df)
    cluster_to_category = assign_cluster_categories(cluster_summary)
    filtered_df = label_cluster_categories(filtered_df, cluster_to_category)
    safest_companies = get_safest_companies(filtered_df)
    logger.info(f"\nSafest companies: {safest_companies}")

    # --- CLASSIFICATION PIPELINE FOR SAFEST COMPANIES ---
    model_path = "models/xgb_ctsh_model.pkl"
    encoder_path = "models/label_encoder_ctsh.pkl"
    predictions_df = score_new_data(model_path, encoder_path, safe_companies=safest_companies)
    summary_df = get_predictions_summary(predictions_df)

    # --- PRINT RESULTS ---
    logger.info("\nClassification predictions for safest companies:")
    logger.info(summary_df.to_string(index=False))

    # --- UPLOAD TO SQL SERVER ---
    if not summary_df.empty:
        engine = create_engine(get_db_url(), echo=False)
        summary_df.to_sql('classification_predictions_poetry', engine, if_exists='replace', index=False)
        logger.info("Classification predictions uploaded to SQL server.")
    else:
        logger.info("No classification predictions to upload.")

if __name__ == "__main__":
    main()