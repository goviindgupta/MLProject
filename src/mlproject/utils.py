# import os
# import sys 
# from src.mlproject.exception import CustomException 
# from src.mlproject.logger import logging
# import pandas as pd
# import pyodbc
# from dotenv import load_dotenv

# load_dotenv()

# host = os.getenv("host")
# user = os.getenv("user")
# password = os.getenv("password")
# db = os.getenv("db")

# def read_sql_data():
#     logging.info("Reading Sql Server started :")
#     try:
#         # Set up the connection using environment variables
#         conn = pyodbc.connect(
#             f"Driver={{ODBC Driver 17 for SQL Server}};"
#             f"Server={host};"
#             f"Database={db};"
#             f"UID={user};"
#             f"PWD={password};"
#         )
#         logging.info('Connection Established',conn)
        
#         # Example query - Replace with your own SQL query
#         query = "SELECT * FROM students"
#         df = pd.read_sql(query, conn)
#         print(df.head())

#         return df

#     except Exception as ex:
#         raise CustomException(ex)












# import os
# import sys
# import pandas as pd
# import pyodbc
# from dotenv import load_dotenv
# from src.mlproject.exception import CustomException
# from src.mlproject.logger import logging

# # Load environment variables from a .env file
# load_dotenv()

# # Fetch database credentials from environment variables
# host = os.getenv("host")
# user = os.getenv("user")
# password = os.getenv("password")
# db = os.getenv("db")

# def read_sql_data():
#     logging.info("Reading data from SQL Server started.")
#     try:
#         # Set up the connection using environment variables
#         # conn = pyodbc.connect(
#         #     f"Driver={{ODBC Driver 17 for SQL Server}};"
#         #     f"Server={host};"
#         #     f"Database={db};"
#         #     f"UID={user};"
#         #     f"PWD={password};"
#         # ) 
#         server = 'localhost , 1433'
#         database = 'college'
#         username = 'SA'
#         password = 'govind@123'
#         cnxn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)


#         cursor = cnxn.cursor()

#         if cursor:
#             print('good connection')
#         logging.info('Connection established successfully.')

#         # Example query - Replace with your own SQL query
#         # query = "SELECT * FROM students"
#         # df = pd.read_sql(query, cnxn)
#         # print(df.head())

#         # return df

#     except Exception as ex:
#         raise CustomException(ex, sys)


















import os
import sys
import pandas as pd
import pyodbc
from dotenv import load_dotenv
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

# Load environment variables from a .env file
load_dotenv()

# Fetch database credentials from environment variables
server = os.getenv("host")
username = os.getenv("user")
password = os.getenv("password")
database = os.getenv("db")

def read_sql_data():
    logging.info("Reading data from SQL Server started.")
    try:
        # Set up the connection using SQL Server Authentication
        # server = 'localhost,1433'  # Note the correction: comma instead of a space
        # database = 'college'
        # username = 'SA'
        # password = 'govind@123'
        
        # Adjusted connection string for SQL Server Authentication
        cnxn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
        )

        cursor = cnxn.cursor()

        if cursor:
            print('Good connection')
        logging.info('Connection established successfully.')

        # Example query - Replace with your own SQL query
        query = "SELECT * FROM students"
        df = pd.read_sql(query, cnxn)
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex, sys)

# Example function call (You can run this to test the connection)
if __name__ == "__main__":
    read_sql_data()

