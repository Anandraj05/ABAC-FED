import pandas as pd
import logging

pd.set_option('mode.chained_assignment', None)

def engineer_features(data):
    try:

        required_columns = ["User ID", "Resource ID", "User Role", "Resource Sensitivity"]
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise KeyError(f"Missing columns: {', '.join(missing_columns)}")


        data["User_Access_Count"] = data.groupby("User ID")["Resource ID"].transform('count')
        data["Resource_Access_Count"] = data.groupby("Resource ID")["User ID"].transform("count")


        categorical_features = ["User Role", "Resource Sensitivity", "Location", "Device Type", "Connection Security"]
        existing_categorical_features = [col for col in categorical_features if col in data.columns]
        data = pd.get_dummies(data, columns=existing_categorical_features, prefix=existing_categorical_features)

        data = data.fillna(0).infer_objects()

        return data

    except KeyError as e:
        logging.error(f"Error: Column not found : {str(e)}")
        raise

    except ValueError as e:
        logging.error(f"Error during feature engineering: {str(e)}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error during feature engineering: {str(e)}")
        raise