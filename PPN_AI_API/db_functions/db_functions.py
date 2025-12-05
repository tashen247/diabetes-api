import os
from datetime import datetime

import pyodbc
import logging
import json

import logging
import sys
from logging.handlers import RotatingFileHandler
import os

from config import connection_string


# Database connection helper (adjust as needed)
def get_connection():
    return pyodbc.connect(
        connection_string
    )


# Call this at the start of your application to set up logging
#logger = setup_logging()
#SELECT ADS_ID, ADS_SEVERITY FROM AI_DIAGNOSIS_SEVERITIES WHERE ADS_AD_ID = 2
diabetes_severities_to_ads_id = {
    "No Suspect or Negative" : 2,
    "Mild" : 3,
    "moderate" : 4,
    "severe" : 5,
    "proliferative" : 6
}

#SELECT ADS_ID, ADS_SEVERITY FROM AI_DIAGNOSIS_SEVERITIES WHERE ADS_AD_ID = 3
hypertension_severity_to_ads_id = {
    "No Suspect or Negative" : 7,
    "Decreased Retinal Artery Elasticity" : 8,
    "Hypertensive Retinopathy Grade 1 or 2" : 9,
    "Hypertensive Retinopathy Grade 3 or 4" : 10
}

#SELECT ADS_ID, ADS_SEVERITY FROM AI_DIAGNOSIS_SEVERITIES WHERE ADS_AD_ID = 4
macular_degeneration_to_ads_id = {
    "No Suspect or Negative" : 11,
    "Drusen" : 12,
    "AMD Early Or IntermediateStage" : 13,
"AMD Advanced Stage" : 14
}

class DbFunctions:
    def __init__(self, use_live=False):
        """
        Initialize DB connection with environment-based string.
        """
        self.connection_string = (
            "DRIVER={SQL Server};"
            "SERVER=psylocke;"
            "DATABASE=Ophthalmology;"
            "UID=www.eyepath.co.za;"
            "PWD=3y3p@th"
        ) if use_live else (
            "DRIVER={SQL Server};"
            "SERVER=sql-dev-ppn;"
            "DATABASE=Ophthalmology;"
            "Trusted_Connection=yes;"
        )

    # Database connection helper (adjust as needed)
    def get_connection(self):
        return pyodbc.connect(
            rf"{connection_string }"
        )

    @staticmethod
    def log_error_to_db(error_message: str, stack_trace: str, endpoint: str):
        """
        Log an error to the database using the STP_INSERT_AI_ERROR_LOG procedure.
        """
        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute("""
                EXEC STP_INSERT_AI_ERROR_LOG 
                    @Error_Message = ?, 
                    @Stack_Trace = ?, 
                    @End_point = ?
            """, error_message, stack_trace, endpoint)
            conn.commit()
            cursor.close()
            conn.close()
            #logging.info(" Error logged to database.")
        except Exception as e:
            logging.error(f" Failed to log error to database: {e}")

    # Write Results to DB
    # def write_results_to_db(self, referral_id, ai_id, ads_ad_id, ads_id, model_no,confidence_scores
    #                         ):
    #     try:
    #         logging.info(
    #             f"Writing results to database...STP_INSERT_DIAGNOSIS_RESULT  {referral_id}, {ai_id}, {ads_ad_id}, {ads_id}, {ai_id}, {model_no}, {confidence_scores}")
    #         conn = pyodbc.connect(self.connection_string)
    #         cursor = conn.cursor()
    #         # cursor.execute("""
    #         #     EXEC [dbo].[STP_INSERT_DIAGNOSIS_RESULT]
    #         #      @ADR_AR_ID = ?,
    #         #      @AI_ID = ?,
    #         #      @ADS_AD_ID = ?,
    #         #      @ADS_ID = ?,
    #         #      @ADR_AM_ID = ?,
    #         #      @ConfidenceScores = ?
    #         # """,referral_id ,ai_id , ads_ad_id,
    #         #                ads_id, model_no ,json.dumps(confidence_scores))
    #         cursor.execute(
    #             "EXEC [dbo].[STP_INSERT_DIAGNOSIS_RESULT] @ADR_AR_ID = ?, @CON_OCULUS_ID = ?, @ADS_AD_ID = ?, @ADS_ID = ?, @AMD_ID = ?, @CONFIDENCE_SCORES = ?",
    #             referral_id, ai_id, ads_ad_id, ads_id, model_no, json.dumps(confidence_scores)
    #         )
    #
    #         conn.commit()
    #         cursor.close()
    #     except Exception as e:
    #         self.log_error_to_db(f"Error writing results to database: {e}",
    #                             f"write_results_to_db(  {referral_id}, {ai_id}, {ads_ad_id}, {ads_id}, {model_no}, {confidence_scores})",
    #                             "/diagnose")
    #         logging.error(f"Error writing results to database: {e}")

    def get_left_and_right_eye_image_ids(self, adr_ar_id):
        # Database connection setup
        conn = pyodbc.connect(connection_string
                              # 'DRIVER={ODBC Driver 17 for SQL Server};'
                              # 'SERVER=your_server;'
                              # 'DATABASE=your_database;'
                              # 'UID=your_username;'
                              # 'PWD=your_password'
                              )
        cursor = conn.cursor()

        # Query for LEFT EYE IMAGE ID
        cursor.execute("""
            SELECT AI_ID
            FROM AI_IMAGES WITH (NOLOCK)
            WHERE AI_AR_ID = ? AND AI_CON_OCULUS_ID = 130
        """, adr_ar_id)
        left_row = cursor.fetchone()
        left_image_id = left_row[0] if left_row else None

        # Query for RIGHT EYE IMAGE ID
        cursor.execute("""
            SELECT AI_ID
            FROM AI_IMAGES WITH (NOLOCK)
            WHERE AI_AR_ID = ? AND AI_CON_OCULUS_ID = 131
        """, adr_ar_id)
        right_row = cursor.fetchone()
        right_image_id = right_row[0] if right_row else None

        # Cleanup
        cursor.close()
        conn.close()

        return left_image_id, right_image_id

    # Fetch Existing Results
    def get_existing_results(self,image_name):
        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            query = """
                SELECT TOP (1) *
                FROM [Ophthalmology].[dbo].[RETINAL_PATHOLOGY_CLASSIFICATIONS]
                WHERE RPC_IMAGE_LOCATION LIKE ?
            """
            cursor.execute(query, f"%{image_name}%")
            columns = [column[0] for column in cursor.description]
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result:
                return dict(zip(columns, result))
            return None
        except Exception as e:
            logging.error(f"Error retrieving results from database: {e}")
            return None

    def insert_endpoint_logs(self, ar_id, url, name, left_eye_file, right_eye_file, ad_id):
        start_time = datetime.now()
        end_time = datetime.now()  # replace with actual end time
        left_eye_file = left_eye_file
        right_eye_file = right_eye_file
        # url = "http://localhost:8000/api/v1/analyze"
        # name = "AI Fundus Inference"

        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        # Step 4: Call stored procedure
        cursor.execute("""
            EXEC STP_INSERT_AI_ENDPOINT_LOG 
                @AR_ID = ?, 
                @START = ?, 
                @END = ?, 
                @LEFT_EYE_FILE = ?, 
                @RIGHT_EYE_FILE = ?, 
                @URL = ?, 
                @NAME = ?,
                @AD_ID = ?
            """,
                       ar_id, start_time, end_time, left_eye_file, right_eye_file, url, name, ad_id)

        # Step 5: Commit the transaction (if applicable)
        conn.commit()

        # Step 6: Close connections
        cursor.close()
        conn.close()

    # Write Results to DB
    def write_results_to_db(self,referral_id, ai_id, ads_ad_id, ads_id, model_no, confidence_scores):
        try:

            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute("""
                EXEC [dbo].[STP_INSERT_DIAGNOSIS_RESULT] 
                 @ADR_AR_ID = ?, 
                 @CON_OCULUS_ID = ?,                 
                 @ADS_AD_ID = ?,
                 @ADS_ID = ?,                  
                 @AMD_ID = ?, 
                 @CONFIDENCE_SCORES = ? 
            """, referral_id, ai_id, ads_ad_id, ads_id, model_no,json.dumps(confidence_scores))
            conn.commit()
            cursor.close()
        except Exception as e:
            logging.error(f"Error writing results to database: {e}")

    @staticmethod
    def _db_get_g_folder(ai_ar_id: int):
        """Return (g_root_folder, db_image_name); create G if needed. Works with/without GFolder column."""
        if not connection_string:
            return (None, None)
        try:
            import pyodbc, os
            conn = pyodbc.connect(connection_string, autocommit=True)
            cur = conn.cursor()
            cur.execute("EXEC dbo.STP_GET_AI_IMAGE_PATH @AI_AR_ID = ?", ai_ar_id)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if cur.description else []
            cur.close();
            conn.close()
            if not row:
                return (None, None)

            # map row -> dict by column name (case-insensitive)
            r = {cols[i].upper(): row[i] for i in range(len(cols))}

            img_name = r.get("AI_IMAGE_NAME")
            img_loc = r.get("AI_IMAGE_LOCATION")
            g_root = r.get("GFOLDER")

            # If proc doesn't return GFolder, derive it
            if not g_root and img_loc:
                sep = "" if img_loc.endswith(("\\", "/")) else "\\"
                g_root = f"{img_loc}{sep}G"

            if g_root:
                os.makedirs(g_root, exist_ok=True)

            return (g_root, str(img_name) if img_name else None)
        except Exception as e:
            logging.error(f"_db_get_g_folder failed for {ai_ar_id}: {e}")
            return (None, None)           

    def fetch_model_paths_from_db(self):
        """
        Retrieve all active model paths from the AI_MODELS table.
        """
        model_paths = {}
        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute("""
                STP_GET_MODELS
            """)
            for model_type, model_name, model_location in cursor.fetchall():
                key = model_name.replace(".keras", "").lower()
                full_path = os.path.join(model_location, model_name)
                model_paths[key] = full_path
            cursor.close()
            conn.close()
            logging.info("✅ Model paths loaded from database.")
        except Exception as e:
            logging.error(f" Error loading model paths from DB: {e}")
        return model_paths

    def setup_logging(self):
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Clear any existing handlers to prevent duplicate logs
        logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create file handler
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'fundus_analysis.log'),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    # Generic function to fetch ADS_IDs for a given ADS_AD_ID (e.g., 2 = diabetes, 3 = hypertension, etc.)
    def get_severities_for_diagnosis_type(self,ads_ad_id: int) -> dict:
        from diagnosis_classes import connection_string
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        query = """
            SELECT ADS_ID, ADS_SEVERITY
            FROM [Ophthalmology].[dbo].[AI_DIAGNOSIS_SEVERITIES]
            WHERE ADS_AD_ID = ?
        """
        cursor.execute(query, ads_ad_id)
        results = cursor.fetchall()

        # Normalize keys to lowercase for consistent lookup
        severity_map = {row.ADS_SEVERITY.lower(): row.ADS_ID for row in results}

        cursor.close()
        conn.close()
        return severity_map

    #TODO
    # Shortcut wrappers for specific conditions
    # def get_diabetes_severities():
    #     return get_severities_for_diagnosis_type(2)
    #
    # def get_hypertension_severities():
    #     return get_severities_for_diagnosis_type(3)
    #
    # def get_macular_degeneration_severities():
    #     return get_severities_for_diagnosis_type(4)

    # def build_model_paths_from_db(sql_rows):
    #     # If sql_rows is already a dictionary of model paths, just return it
    #     if isinstance(sql_rows, dict) and all(isinstance(v, dict) for v in sql_rows.values()):
    #         return sql_rows
    #
    #     model_paths = {}
    #
    #     # Check if we received a list of rows or a dictionary
    #     if isinstance(sql_rows, list):
    #         for row in sql_rows:
    #             # Assuming row is a tuple with column values in specific order
    #             # Replace index numbers with the actual positions of these columns in your result set
    #             AMD_id = row[0]  # Index for AMD_ID
    #             AMD_short_name = row[1].lower() if row[1] else f"model_{AMD_id}"  # Index for AMD_SHORT_NAME
    #             AMD_model_location = row[2]  # Index for AMD_MODEL_LOCATION
    #
    #             model_paths[AMD_short_name] = {
    #                 "number": AMD_id,
    #                 "path": AMD_model_location
    #             }
    #
    #     return model_paths
    # def build_model_paths_from_db(sql_rows):
    #     model_paths = {}
    #
    #     # Check if sql_rows is already a dictionary of model_id -> model_info
    #     if isinstance(sql_rows, dict):
    #         for row_id, row_data in sql_rows.items():
    #             # Need to get the model short name from somewhere to use as the key
    #             # If available in row_data, use it, otherwise default to the ID
    #
    #             # Check if AMD_SHORT_NAME is part of the row_data dictionary
    #             if isinstance(row_data, dict) and 'AMD_SHORT_NAME' in row_data:
    #                 model_type = row_data['AMD_SHORT_NAME'].lower()
    #             else:
    #                 # Use a default naming convention based on ID
    #                 model_type = f"model_{row_id}"
    #
    #             model_paths[model_type] = {
    #                 "number": row_data.get("number"),
    #                 "path": row_data.get("path")
    #             }
    #     else:
    #         # If sql_rows is a list of row objects
    #         for row in sql_rows:
    #             if isinstance(row, dict):
    #                 # If row is a dictionary containing the required keys
    #                 if 'AMD_SHORT_NAME' in row and 'AMD_ID' in row and 'AMD_MODEL_LOCATION' in row:
    #                     model_type = row['AMD_SHORT_NAME'].lower()
    #                     model_paths[model_type] = {
    #                         "number": row['AMD_ID'],
    #                         "path": row['AMD_MODEL_LOCATION']
    #                     }
    #             elif hasattr(row, '__getitem__') and hasattr(row, '__len__'):
    #                 # If row is a sequence (tuple, list) with indices
    #                 # You'd need to know the exact position of each field in the sequence
    #                 # This is just an example assuming a specific column order
    #                 if len(row) >= 3:  # Make sure row has enough elements
    #                     model_id = row[0]  # Assuming first column is AMD_ID
    #                     model_name = row[1] if row[1] else f"model_{model_id}"  # Second column as AMD_SHORT_NAME
    #                     model_path = row[2]  # Third column as AMD_MODEL_LOCATION
    #
    #                     model_paths[model_name.lower()] = {
    #                         "number": model_id,
    #                         "path": model_path
    #                     }
    #
    #     return model_paths

    # def build_model_paths_from_db(sql_rows):
    #     model_paths = {}
    #     for row in sql_rows:
    #         # model_type = row["AMD_SHORT_NAME"].lower()  # assuming keys like "diabetes", etc.
    #         model_type = row["AMD_SHORT_NAME"].lower()  # assuming keys like "diabetes", etc.
    #         model_paths[model_type] = {
    #             "number": row["AMD_ID"],
    #             "path": row["AMD_MODEL_LOCATION"]
    #         }
    #     return model_paths
    def build_model_paths_from_db(self,sql_rows):
        """
        Builds a dictionary of model paths from database rows
        Expected format from stored procedure:
        Each row contains: AMD_ID, AMD_SHORT_NAME, AMD_MODEL_LOCATION
        """
        model_paths = {}

        # First, check if we have valid data
        if not sql_rows:
            print("Warning: No model data received from database")
            return model_paths

        # Process each row from the stored procedure
        for row in sql_rows:
            try:
                # Determine what type of row object we have
                if isinstance(row, dict):
                    # If row is already a dictionary
                    model_id = row.get('AMD_ID')
                    model_name = row.get('AMD_SHORT_NAME', '').lower()
                    model_path = row.get('AMD_MODEL_LOCATION', '')
                elif hasattr(row, '_fields'):
                    # For named tuples (some DB APIs return these)
                    model_id = getattr(row, 'AMD_ID', None)
                    model_name = getattr(row, 'AMD_SHORT_NAME', '').lower()
                    model_path = getattr(row, 'AMD_MODEL_LOCATION', '')
                else:
                    # For tuple/list-like objects
                    # You need to know the exact column order from your stored procedure
                    # Adjust these indices based on the actual columns in your result set
                    model_id = row[0]  # Assuming AMD_ID is the first column
                    model_name = str(row[1]).lower() if row[1] else ''  # AMD_SHORT_NAME
                    model_path = row[2] if len(row) > 2 else ''  # AMD_MODEL_LOCATION

                # Skip if we don't have a valid model name
                if not model_name:
                    model_name = f"model_{model_id}"

                # Add to our model paths dictionary
                model_paths[model_name] = {
                    "number": model_id,
                    "path": model_path
                }
            except Exception as e:
                print(f"Error processing model row: {e}")
                continue

        return model_paths

    def get_model_paths_from_db(self):
        try:
            with pyodbc.connect(connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("EXEC [dbo].[STP_GET_MODELS]")

                # Debug column information
                print("Cursor description:", cursor.description)
                columns = [column[0] for column in cursor.description]
                print("Columns:", columns)

                rows = cursor.fetchall()
                print(f"Rows count: {len(rows)}")

                if rows:
                    print(f"First row type: {type(rows[0])}")
                    print(f"First row: {rows[0]}")

                    # Check how pyodbc returns individual fields
                    if len(rows[0]) > 0:
                        print(f"First field type: {type(rows[0][0])}")
                        print(f"First field value: {rows[0][0]}")

                model_paths = {}

                for row in rows:
                    try:
                        print(f"Processing row: {row}")

                        # Check if row is already a dictionary
                        if isinstance(row, dict):
                            row_dict = row
                        else:
                            # Create dictionary from row data
                            row_dict = {}
                            for i, column in enumerate(columns):
                                try:
                                    row_dict[column] = row[i]
                                except Exception as e:
                                    print(f"Error processing column {column} at index {i}: {e}")
                                    print(f"Row type: {type(row)}, Row: {row}")

                        print(f"Created row_dict: {row_dict}")

                        # Convert integer fields to strings for keys
                        model_type = str(row_dict["AMD_ID"])

                        model_paths[model_type] = {
                            "number": row_dict["AMD_ID"],
                            "path": os.path.join(row_dict["AMD_MODEL_LOCATION"], row_dict["AMD_MODEL_NAME"])
                        }
                    except Exception as inner_e:
                        print(f"Error processing row: {inner_e}")
                        print(f"Row data: {row}")
                        continue

                return model_paths

        except Exception as e:
            logging.error(f"Error fetching model paths from database: {e}", exc_info=True)
            print(f"Database connection error: {e}")
            return {}
