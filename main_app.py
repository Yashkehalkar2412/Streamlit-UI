import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import os
from datetime import datetime

# Database connection details
# IMPORTANT: Replace these with your actual MySQL Workbench credentials if they differ
MYSQL_HOST = '194.59.164.10'
MYSQL_USER = 'u758484694_stringpricing2'
MYSQL_PASSWORD = 'sO+9+TAzN6Qx'
MYSQL_DB = 'u758484694_stringpricing2'

# --- Table Names (Adjust if your table names are different) ---
# 'clientDetails' is used to fetch client names for the 'incomes' table's dropdown
clientDetails = 'clientDetails'
# 'Incomes' variable is used for specific column configuration (Client Name dropdown)
# Its value reflects the actual table name in the DB, which is 'incomes' (lowercase) based on previous context.
Incomes = 'incomes' 

# --- Configure Google Generative AI ---
llm_model = None
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}. "
             "Please ensure the GOOGLE_API_KEY environment variable is set correctly. "
             "Natural language query feature will be unavailable.")

import mysql.connector
# ... (other imports) ...

@st.cache_resource(ttl=3600)
def get_db_connection():
    """
    Establishes and returns a database connection, caching it for efficiency.
    Handles potential connection errors and displays them to the user.
    """
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"Database connection error: {err}. Please check your MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DB settings. Also ensure the MySQL server is running and accessible from this environment (e.g., firewall rules).")
        return None

def execute_query(sql_query, params=None):
    """
    Executes a given SQL query and returns the result as a pandas DataFrame (for SELECT) 
    or an operation success message (for INSERT/UPDATE/DELETE).
    Includes logic to automatically reconnect if the database connection has been lost.
    """
    conn = get_db_connection()
    if conn is None:
        return None, "Database connection failed."

    # Allow one retry in case of a lost connection
    for attempt in range(2): 
        try:
            # Check if connection is active; if not, attempt to reconnect
            if not conn.is_connected():
                st.warning("Database connection was lost. Attempting to reconnect...")
                conn.reconnect()
                st.success("Reconnected to the database.")

            # Use a dictionary cursor to get results as dictionaries (column_name: value)
            with conn.cursor(dictionary=True) as cursor:
                if params:
                    cursor.execute(sql_query, params)
                else:
                    cursor.execute(sql_query)
                
                # Commit changes for DML (Data Manipulation Language) queries
                if sql_query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                    conn.commit()
                    return pd.DataFrame(), f"Query executed successfully. {cursor.rowcount} row(s) affected."
                else: # Assume SELECT or DDL (Data Definition Language)
                    result = cursor.fetchall()
                    df = pd.DataFrame(result)
                    return df, None
        except mysql.connector.Error as err:
            # Handle specific MySQL connection errors for retries
            if err.errno in (errorcode.CR_SERVER_LOST, errorcode.CR_CONN_HOST_ERROR):
                st.error(f"Connection lost during query (attempt {attempt + 1}). Retrying...")
                if attempt == 0: # Only retry once
                    continue
                else:
                    return None, str(err)
            else:
                # Return other MySQL errors
                return None, str(err)
        except Exception as e:
            # Catch any other unexpected Python errors
            return None, str(e)
    
    # If all retries fail
    return None, "Failed to execute query after multiple attempts."

def get_table_names(conn):
    """
    Fetches all table names from the connected database.
    """
    if conn is None:
        return []
    try:
        if not conn.is_connected():
            st.warning("Database connection was lost while fetching table names. Attempting to reconnect...")
            conn.reconnect()
            st.success("Reconnected to the database for table names.")

        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [table[0] for table in cursor.fetchall()]
        return tables
    except mysql.connector.Error as err:
        st.error(f"Error fetching table names: {err}")
        return []

def get_table_columns_and_types(conn, table_name):
    """
    Fetches column names, their data types, nullability, and primary key status for a given table.
    Returns a tuple: (list of column info dicts, primary_key_column_name).
    """
    if conn is None or not table_name:
        return [], None
    try:
        if not conn.is_connected():
            st.warning(f"Database connection was lost while fetching columns for {table_name}. Attempting to reconnect...")
            conn.reconnect()
            st.success(f"Reconnected to the database for columns of {table_name}.")

        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"DESCRIBE `{table_name}`")
        columns_info = []
        pk_column_name = None
        for col in cursor.fetchall():
            col_name = col['Field']
            col_type = col['Type']
            is_nullable = col['Null'] == 'YES'
            is_pk = col['Key'] == 'PRI' # 'PRI' in the 'Key' field indicates a primary key

            columns_info.append({
                "name": col_name,
                "type": col_type,
                "null": is_nullable,
                "is_pk": is_pk
            })
            if is_pk:
                pk_column_name = col_name
        return columns_info, pk_column_name
    except mysql.connector.Error as err:
        st.error(f"Error fetching columns for table {table_name}: {err}")
        return [], None

def get_database_schema_for_llm(conn):
    """
    Fetches the schema of all tables in the database and formats it for an LLM.
    """
    schema_description = "Database Schema:\n"
    table_names = get_table_names(conn)
    if not table_names:
        return ""

    for table_name in table_names:
        schema_description += f"- Table `{table_name}`:\n"
        columns_info, pk_column = get_table_columns_and_types(conn, table_name)
        if columns_info:
            for col in columns_info:
                pk_marker = " (PK)" if col['is_pk'] else ""
                null_marker = " NULL" if col['null'] else " NOT NULL"
                schema_description += f"  - Column `{col['name']}`: Type={col['type']}{pk_marker}{null_marker}\n"
        else:
            schema_description += f"  (No columns found for {table_name} or error fetching columns)\n"
    return schema_description

def generate_sql_with_llm(user_question, db_schema):
    """
    Uses the LLM to convert a natural language command/question into an SQL query,
    now including support for DML operations with strong safety instructions.
    """
    if llm_model is None:
        return None, "LLM not configured. Please set GOOGLE_API_KEY."

    # Define the core instructions for the LLM
    prompt_instructions = """You are a helpful assistant that converts natural language commands/questions into MySQL queries.
    Generate only the SQL query, do not include any other text, explanations, or markdown fences.
    
    Database Schema:
    {db_schema}

    Key Guidelines:
    1.  **Always use backticks (`)** around table and column names (e.g., `Client ID`, `incomes`).
    2.  **For SELECT queries (asking for data):**
        * Use aggregate functions (SUM(), COUNT(), AVG()) if an aggregation is requested.
        * Use JOIN clauses if the question involves data from multiple tables.
        * Be precise with column and table names based on the schema.
        * If a specific ID or name is mentioned (e.g., "resource id 2", "Client ID 2", "Ashu"), use it in the WHERE clause.
    3.  **For DELETE queries (removing data):**
        * **CRITICAL SAFETY RULE:** A DELETE query **MUST** always include a specific `WHERE` clause that identifies the row(s) to be deleted.
        * **NEVER** generate a DELETE query without a `WHERE` clause (e.g., `DELETE FROM table;`). If the user's request is too vague or could imply deleting all data, return "INVALID_QUERY".
        * Identify the most appropriate unique identifier or combination of columns for the `WHERE` clause (e.g., ID, Name).
        * Example: "Delete Ashu from Incomes" should translate to `DELETE FROM `incomes` WHERE `Client Name` = 'Ashu';` or `DELETE FROM `incomes` WHERE `Client ID` IN (SELECT `Client ID` FROM `clientDetails` WHERE `Name` = 'Ashu');` (if `incomes` links to `clientDetails` via Client ID and 'Client Name' is in clientDetails). Prioritize direct match in the target table if applicable.
    4.  **For INSERT/UPDATE queries (modifying data):**
        * **CRITICAL SAFETY RULE:** Similar to DELETE, INSERT/UPDATE queries must be very specific. If the request is ambiguous or could lead to incorrect data, return "INVALID_QUERY".
        * For UPDATE, a `WHERE` clause is mandatory.
    5.  **Error Handling:** If you cannot generate a meaningful, safe, or precise SQL query, return the string "INVALID_QUERY".
    
    User command/question: "{user_question}"
    Generated SQL:"""

    try:
        response = llm_model.generate_content(prompt_instructions.format(db_schema=db_schema, user_question=user_question))
        sql_query = response.text.strip()
        
        # Post-processing: Remove markdown fences if the LLM includes them
        if sql_query.startswith("```sql") and sql_query.endswith("```"):
            sql_query = sql_query[len("```sql"):-len("```")].strip()
        elif sql_query.startswith("```") and sql_query.endswith("```"):
            sql_query = sql_query[len("```"):-len("```")].strip()
        
        # Basic validation to ensure it's a SQL query. Now also allows DELETE, INSERT, UPDATE.
        if not sql_query.upper().startswith(("SELECT", "SHOW", "DESCRIBE", "PRAGMA", "DELETE", "INSERT", "UPDATE")):
            return None, "LLM generated an invalid or unsupported query type. Please try rephrasing."
        
        # Additional safety check for DML: ensure it has a WHERE clause if applicable
        if sql_query.upper().startswith(("DELETE", "UPDATE")) and " WHERE " not in sql_query.upper():
            return None, "Unsafe DML query generated (missing WHERE clause). Refused to execute. Please be more specific."
            
        return sql_query, None
    except Exception as e:
        return None, f"Error generating SQL with LLM: {e}. Please try rephrasing your question."

def manage_table_data(conn, selected_table, available_client_names):
    """
    Displays and manages data for a given table using st.data_editor.
    Handles inserts, updates, and deletes for the selected table.
    """
    if not selected_table:
        st.info("Please select a table to manage.")
        return

    st.subheader(f"Edit Data in `{selected_table}`")
    st.info("Click on a cell to edit. Use '+' to add rows. Use the trashcan icon to delete selected rows.")

    # Get detailed column information and the primary key for the selected table
    table_columns_info, pk_column_name = get_table_columns_and_types(conn, selected_table)

    if not table_columns_info:
        st.warning(f"Could not retrieve column information for '{selected_table}'. "
                   "Please ensure the table exists and is accessible.")
        return

    if not pk_column_name:
        st.warning(f"The table '{selected_table}' does not have a primary key defined. "
                   "Editing/deleting existing rows will not work correctly without a primary key. "
                   "You can still add new rows, but managing existing ones will be limited.")

    # Fetch existing data for the selected table
    current_data_df, err_fetch = execute_query(f"SELECT * FROM `{selected_table}`")

    if err_fetch:
        st.error(f"Error fetching data for '{selected_table}': {err_fetch}")
        st.warning(f"Please ensure the table '{selected_table}' exists and is accessible. "
                   "If you've recently created the table, a full Streamlit app restart might be needed (Ctrl+C and rerun).")
        initial_data_df = pd.DataFrame() # Initialize an empty DataFrame on error
    else:
        # Convert date/datetime/timestamp columns to proper datetime objects for st.data_editor
        for col_info in table_columns_info:
            if ('date' in col_info['type'] or 'datetime' in col_info['type'] or 'timestamp' in col_info['type']) and col_info['name'] in current_data_df.columns:
                # Ensure column is not empty before attempting to convert to datetime
                if not current_data_df[col_info['name']].empty:
                    current_data_df[col_info['name']] = pd.to_datetime(current_data_df[col_info['name']], errors='coerce')
        initial_data_df = current_data_df.copy() # Create a copy for comparison and session state

    # Store the original DataFrame in session state. This is crucial for change detection.
    # The key is unique to the selected table to avoid conflicts when switching tables.
    session_key_original_df = f"original_df_{selected_table}"
    st.session_state[session_key_original_df] = initial_data_df.copy() # Always update with fresh data on load/rerun

    # Configure columns for the st.data_editor dynamically based on DB column info
    column_config = {}
    for col_info in table_columns_info:
        col_name = col_info['name']
        col_type = col_info['type']
        
        # Disable editing for primary key columns if a PK is found for the table
        if col_info['is_pk'] and pk_column_name is not None:
             column_config[col_name] = st.column_config.NumberColumn(
                 col_name,
                 disabled=True, # Prevent direct editing of PK
                 help="Primary Key - Not editable directly. Value is usually auto-incremented or system-managed."
             )
        # Special handling for 'Client Name' column in the 'incomes' table
        elif selected_table == Incomes and col_name == 'Client Name':
            column_config[col_name] = st.column_config.SelectboxColumn(
                "Client Name",
                help="Select an existing client from your client details.",
                width="large",
                options=available_client_names, # Options derived from the clientDetails table
                required=not col_info['null'] # Set required based on DB schema nullability
            )
        # Configure Date columns (without time)
        elif 'date' in col_type and 'datetime' not in col_type and 'timestamp' not in col_type:
            column_config[col_name] = st.column_config.DateColumn(
                col_name,
                format="YYYY-MM-DD",
                required=not col_info['null']
            )
        # Configure Datetime/Timestamp columns (with time)
        elif 'datetime' in col_type or 'timestamp' in col_type:
            column_config[col_name] = st.column_config.DatetimeColumn(
                col_name,
                format="YYYY-MM-DD HH:mm:ss", # Format to include time components
                required=not col_info['null']
            )
        # Configure Number columns (integers, decimals, floats)
        elif 'int' in col_type or 'decimal' in col_type or 'float' in col_type or 'double' in col_type:
            column_config[col_name] = st.column_config.NumberColumn(
                col_name,
                format="%d" if 'int' in col_type else "%.2f", # Basic formatting for integers vs. floats
                required=not col_info['null']
            )
        # Default configuration for other types (e.g., text, varchar)
        else:
            column_config[col_name] = st.column_config.TextColumn(
                col_name,
                required=not col_info['null']
            )

    # Display the data editor with the dynamically configured columns
    edited_data_df = st.data_editor(
        st.session_state[session_key_original_df], # The data to display and edit
        column_config=column_config, # Apply the dynamic column configuration
        num_rows="dynamic", # Allow users to add/delete rows directly in the editor
        key=f"data_editor_{selected_table}" # Unique key for this specific table's data editor
    )

    # Button to trigger saving changes to the database
    if st.button(f"Save Changes to `{selected_table}`", key=f"save_changes_btn_{selected_table}"):
        with st.spinner("Saving changes..."):
            # Retrieve the changes detected by st.data_editor from its session state
            current_editor_state = st.session_state[f"data_editor_{selected_table}"]
            added_rows_list = current_editor_state['added_rows']
            edited_rows_dict = current_editor_state['edited_rows']
            deleted_rows_indices = current_editor_state['deleted_rows']

            # --- Handle Deleted Rows ---
            if deleted_rows_indices:
                if not pk_column_name:
                    st.warning(f"Cannot delete rows from '{selected_table}' as no primary key is defined. "
                               "Skipping deletion of selected rows.")
                else:
                    st.info(f"Detected {len(deleted_rows_indices)} row(s) to delete.")
                    for df_index in deleted_rows_indices:
                        # Ensure the index exists in the original DataFrame before accessing PK
                        if df_index < len(st.session_state[session_key_original_df]):
                            pk_value_to_delete = st.session_state[session_key_original_df].loc[df_index, pk_column_name]
                            delete_query = f"DELETE FROM `{selected_table}` WHERE `{pk_column_name}` = %s;"
                            _, msg = execute_query(delete_query, (pk_value_to_delete,))
                            if msg:
                                st.text(f"Deleted row with {pk_column_name}={pk_value_to_delete}: {msg}")
                            else:
                                st.error(f"Failed to delete row with {pk_column_name}={pk_value_to_delete}")
                        else:
                            st.warning(f"Skipping deletion for row at internal index {df_index}: index out of bounds for original data. This row might have been a new, unsaved row.")

            # --- Handle Added Rows ---
            if added_rows_list:
                st.info(f"Detected {len(added_rows_list)} new row(s) to add.")
                for row_dict in added_rows_list:
                    cols = []
                    vals = []
                    
                    for col, val in row_dict.items():
                        # Skip primary key columns for insertion if they are auto-incrementing
                        col_info = next((item for item in table_columns_info if item["name"] == col), None)
                        if col_info and col_info['is_pk']:
                            continue # Skip PK as it's usually auto-generated

                        # Handle empty strings/None values: if column is non-nullable and value is empty, warn.
                        if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                            if col_info and not col_info['null']:
                                st.warning(f"Column '{col}' is non-nullable but received empty/null value for a new row. This row might fail insertion if no default value is defined in the database.")
                                # We still add it as None; MySQL will then throw an error if no default is defined.
                                vals.append(None) 
                                cols.append(f"`{col}`")
                            else:
                                continue # Skip for nullable empty values, effectively inserting NULL
                        else:
                            cols.append(f"`{col}`")
                            if isinstance(val, (datetime, pd.Timestamp)):
                                vals.append(val.strftime('%Y-%m-%d %H:%M:%S')) # Format datetime for MySQL
                            else:
                                vals.append(val)

                    if cols: # Only proceed if there are columns to insert
                        placeholders = ", ".join(["%s"] * len(cols))
                        insert_query = f"INSERT INTO `{selected_table}` ({', '.join(cols)}) VALUES ({placeholders});"
                        _, msg = execute_query(insert_query, tuple(vals))
                        if msg:
                            st.text(f"New row insert: {msg}")
                        else:
                            st.error(f"Failed to insert row: {row_dict}")
                    else:
                        st.warning(f"Skipped inserting an empty row (no valid columns to insert): {row_dict}")


            # --- Handle Edited Rows ---
            if edited_rows_dict:
                if not pk_column_name:
                    st.warning(f"Cannot update rows in '{selected_table}' as no primary key is defined. "
                               "Skipping update of edited rows.")
                else:
                    st.info(f"Detected {len(edited_rows_dict)} edited row(s) to update.")
                    for index_in_df, changes in edited_rows_dict.items():
                        # Ensure the index exists in the original DataFrame before accessing PK
                        if index_in_df < len(st.session_state[session_key_original_df]):
                            pk_value = st.session_state[session_key_original_df].loc[index_in_df, pk_column_name]
                            
                            set_clauses = []
                            update_params = []

                            for col, val in changes.items():
                                # Exclude PK column from update clause as it's typically not updated
                                if col == pk_column_name:
                                    continue

                                set_clauses.append(f"`{col}` = %s")
                                if isinstance(val, (datetime, pd.Timestamp)):
                                    update_params.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                                elif pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                                    # Handle NaNs/empty strings for updates: set to NULL or warn if non-nullable
                                    col_info = next((item for item in table_columns_info if item["name"] == col), None)
                                    if col_info and not col_info['null']:
                                        st.warning(f"Attempted to set non-nullable column '{col}' to NULL/empty during update for row {pk_value}. This might cause a DB error.")
                                        update_params.append(None) # Pass None, let DB handle error based on schema
                                    else:
                                        update_params.append(None) # Set to None for nullable columns
                                else:
                                    update_params.append(val)
                            
                            if set_clauses: # Only proceed if there are actual changes
                                update_params.append(pk_value) # Add PK value for WHERE clause
                                update_query = (
                                    f"UPDATE `{selected_table}` SET "
                                    f"{', '.join(set_clauses)} "
                                    f"WHERE `{pk_column_name}` = %s;"
                                )
                                _, msg = execute_query(update_query, tuple(update_params))
                                if msg:
                                    st.text(f"Row with {pk_column_name}={pk_value} updated: {msg}")
                                else:
                                    st.error(f"Failed to update row with {pk_column_name}={pk_value}")
                            else:
                                st.warning(f"No effective changes detected for row with {pk_column_name}={pk_value}, skipping update.")
                        else:
                            st.warning(f"Skipping update for row at internal index {index_in_df}: index out of bounds for original data. This row might have been already deleted or a new unsaved row.")

            # Re-fetch data to reflect latest changes from DB and update session state
            # This ensures the data editor displays the most current state of the database
            reloaded_df, _ = execute_query(f"SELECT * FROM `{selected_table}`")
            if reloaded_df is not None and not reloaded_df.empty:
                for col_info in table_columns_info:
                    if ('date' in col_info['type'] or 'datetime' in col_info['type'] or 'timestamp' in col_info['type']) and col_info['name'] in reloaded_df.columns:
                        reloaded_df[col_info['name']] = pd.to_datetime(reloaded_df[col_info['name']], errors='coerce')
            st.session_state[session_key_original_df] = reloaded_df if reloaded_df is not None else pd.DataFrame()
            
            st.success("All changes processed. Table reloaded from database.")
            st.rerun() # Rerun Streamlit app to refresh the data editor with the latest state
    

# --- Streamlit Application Layout ---
st.set_page_config(layout="wide", page_title="Database Data Management Dashboard")
st.title("Database Data Management Dashboard")

conn = get_db_connection() # Attempt to establish a database connection

if conn:
    # --- Natural Language Query Section ---
    st.header("Ask Your Database a Question")
    st.markdown("---")
    
    if llm_model: # Only show the search bar if LLM is configured
        user_question = st.text_input(
            "Type your question here (e.g., 'What is the total amount in incomes table?', "
            "'Give me the address of client id 2', "
            "'What is the amount and project name for Client ID 2?', "
            "'Delete Ashu from Incomes')", # Added example for delete
            value=st.session_state.get('user_question_input', ''), # Keep input value
            key="nl_query_input"
        )
        
        # Initialize session state for generated_sql, is_dml, and query_result
        if 'generated_sql' not in st.session_state:
            st.session_state.generated_sql = None
        if 'is_dml' not in st.session_state:
            st.session_state.is_dml = False
        if 'query_result_df' not in st.session_state:
            st.session_state.query_result_df = None
        if 'query_result_msg' not in st.session_state:
            st.session_state.query_result_msg = None
        if 'query_error_msg' not in st.session_state:
            st.session_state.query_error_msg = None

        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            if st.button("Generate SQL / Get Answer", key="nl_query_button"):
                if user_question:
                    st.session_state.user_question_input = user_question # Store current input
                    with st.spinner("Thinking... Generating SQL..."):
                        db_schema = get_database_schema_for_llm(conn)
                        if not db_schema:
                            st.error("Could not retrieve database schema to assist with query generation. "
                                     "Please ensure your database has tables and connection details are correct.")
                            st.session_state.generated_sql = None
                            st.session_state.is_dml = False
                            st.session_state.query_result_df = None
                            st.session_state.query_result_msg = None
                            st.session_state.query_error_msg = "Could not retrieve database schema."
                        else:
                            generated_sql, llm_error = generate_sql_with_llm(user_question, db_schema)

                            if llm_error:
                                st.error(f"Failed to generate SQL: {llm_error}")
                                st.session_state.generated_sql = None
                                st.session_state.is_dml = False
                                st.session_state.query_result_df = None
                                st.session_state.query_result_msg = None
                                st.session_state.query_error_msg = f"Failed to generate SQL: {llm_error}"
                            elif generated_sql == "INVALID_QUERY":
                                st.warning("I couldn't generate a meaningful or safe SQL query for your request. Please try rephrasing or be more specific.")
                                st.session_state.generated_sql = None
                                st.session_state.is_dml = False
                                st.session_state.query_result_df = None
                                st.session_state.query_result_msg = None
                                st.session_state.query_error_msg = "Invalid or unsafe query generated."
                            else:
                                st.session_state.generated_sql = generated_sql
                                st.session_state.is_dml = generated_sql.upper().startswith(("DELETE", "INSERT", "UPDATE"))
                                st.session_state.query_result_df = None # Clear previous results
                                st.session_state.query_result_msg = None
                                st.session_state.query_error_msg = None
                else:
                    st.warning("Please enter a question or command.")
                    st.session_state.generated_sql = None
                    st.session_state.is_dml = False
                    st.session_state.query_result_df = None
                    st.session_state.query_result_msg = None
                    st.session_state.query_error_msg = None

        with col2:
            # Add a clear button
            if st.button("Clear Query/Result", key="clear_query_button"):
                st.session_state.generated_sql = None
                st.session_state.is_dml = False
                st.session_state.query_result_df = None
                st.session_state.query_result_msg = None
                st.session_state.query_error_msg = None
                st.session_state.user_question_input = "" # Clear the input box as well
                st.rerun() # Rerun to clear display

        # Always display the generated SQL if it exists in session state
        if st.session_state.generated_sql:
            st.subheader("Generated SQL Query:")
            st.code(st.session_state.generated_sql, language="sql")
            
            # Display confirmation button only if SQL was generated
            if st.session_state.is_dml:
                st.warning("This is a Data Manipulation (DELETE/INSERT/UPDATE) query. "
                           "Executing this will modify your database data.")
                confirm_execute = st.button("Confirm and Execute DANGER Query", key="confirm_dml_execution_button")
            else:
                confirm_execute = st.button("Execute Query", key="execute_dql_execution_button")
            
            if confirm_execute:
                with st.spinner("Executing query..."):
                    st.subheader("Query Result:")
                    try:
                        result_df, query_error = execute_query(st.session_state.generated_sql)
                        if query_error:
                            st.error(f"Error executing query: {query_error}")
                            st.session_state.query_error_msg = f"Error executing query: {query_error}"
                            st.session_state.query_result_df = None
                            st.session_state.query_result_msg = None
                        elif isinstance(result_df, pd.DataFrame): # Check if it's a DataFrame (for SELECT)
                            if not result_df.empty:
                                st.dataframe(result_df)
                                st.session_state.query_result_df = result_df
                                st.session_state.query_result_msg = None
                                st.session_state.query_error_msg = None
                            else:
                                st.info("No results found for your query.")
                                st.session_state.query_result_df = pd.DataFrame()
                                st.session_state.query_result_msg = "No results found for your query."
                                st.session_state.query_error_msg = None
                        else: # Else, it's a success message from DML
                            st.success(result_df) # result_df will contain the success message for DML
                            st.session_state.query_result_msg = result_df
                            st.session_state.query_result_df = None
                            st.session_state.query_error_msg = None
                    except Exception as e:
                        st.error(f"An unexpected error occurred during query execution: {e}")
                        st.session_state.query_error_msg = f"An unexpected error occurred: {e}"
                        st.session_state.query_result_df = None
                        st.session_state.query_result_msg = None
                # No st.rerun() here, allow results to persist
        
        # Always display the stored result/error messages if they exist
        if st.session_state.query_result_df is not None:
            st.subheader("Query Result:")
            if not st.session_state.query_result_df.empty:
                st.dataframe(st.session_state.query_result_df)
            else:
                st.info("No results found for your query.")
        elif st.session_state.query_result_msg:
            st.subheader("Query Result:")
            st.success(st.session_state.query_result_msg)
        elif st.session_state.query_error_msg:
            st.subheader("Query Result:")
            st.error(st.session_state.query_error_msg)


    else:
        st.warning("Natural language query feature is unavailable because Google Generative AI could not be configured. "
                   "Please set the `GOOGLE_API_KEY` environment variable.")

    st.markdown("---") # Separator after the search section

    # --- Data Management Section (Existing Functionality) ---
    # Fetch all table names available in the database to populate the selection dropdown
    all_table_names = get_table_names(conn)
    if not all_table_names:
        st.warning("No tables found in the database. Please ensure your database has tables and connection details are correct.")
    
    # Fetch Client Names from the 'clientDetails' table. This list will be used
    # to populate the 'Client Name' dropdown specifically in the 'incomes' table management.
    client_names_query = f"SELECT `Name` FROM `{clientDetails}`" 
    client_details_df, err_client = execute_query(client_names_query)
    if err_client:
        st.error(f"Error fetching client details for suggestions: {err_client}")
        available_client_names = [] # Initialize as empty list on error
    else:
        available_client_names = client_details_df['Name'].unique().tolist() if not client_details_df.empty else []
        if not available_client_names:
            st.warning(f"No client names found in `{clientDetails}` table. Please ensure data exists if you plan to use 'Client Name' fields in the '{Incomes}' table.")

    st.header("Manage Table Entries")
    st.markdown("---") # Visual separator
    
    # Dropdown to allow the user to select which database table they want to manage
    selected_table_to_manage = st.selectbox(
        "Select a Table to Manage:", 
        [""] + all_table_names, # Add an empty option as the default / initial state
        key="main_table_selection" # Unique key for this selectbox
    )

    # If a table is selected, call the `manage_table_data` function to display and handle its data
    if selected_table_to_manage:
        manage_table_data(conn, selected_table_to_manage, available_client_names)
    else:
        st.info("Please select a table from the dropdown above to start managing its data. "
                "The selected table's data will appear below, enabling you to add, edit, and delete entries.")

else:
    # Display error if the database connection could not be established
    st.error("Database connection is not established. Please check your configuration and ensure the MySQL server is running.")