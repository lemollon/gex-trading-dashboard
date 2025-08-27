# test_databricks.py
"""
Streamlit app to test Databricks connection and data loading
Run with: streamlit run test_databricks.py
"""

import streamlit as st
import pandas as pd
from databricks import sql
import json
from datetime import datetime
import traceback

st.set_page_config(
    page_title="GEX Databricks Tester",
    page_icon="🔧", 
    layout="wide"
)

st.title("🔧 GEX Databricks Connection Tester")

# Connection test function
@st.cache_resource
def get_databricks_connection():
    """Create and test Databricks connection"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        return connection
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")
        return None

# Test queries
def test_basic_query(connection):
    """Test basic SQL functionality"""
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1 as test_value")
        result = cursor.fetchone()
        cursor.close()
        return result[0] == 1
    except Exception as e:
        st.error(f"Basic query failed: {str(e)}")
        return False

def test_tables_exist(connection):
    """Check if your GEX tables exist"""
    tables_to_check = [
        "gex_trading.scan_results",
        "gex_trading.gamma_profiles", 
        "gex_trading.trading_setups"
    ]
    
    results = {}
    cursor = connection.cursor()
    
    for table in tables_to_check:
        try:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            results[table] = {"exists": True, "columns": len(columns)}
        except Exception as e:
            results[table] = {"exists": False, "error": str(e)}
    
    cursor.close()
    return results

def fetch_recent_data(connection):
    """Fetch your actual GEX scan data"""
    try:
        cursor = connection.cursor()
        
        # Try your existing scan results table
        query = """
        SELECT *
        FROM gex_trading.scan_results 
        ORDER BY scan_timestamp DESC 
        LIMIT 10
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        cursor.close()
        
        if results:
            df = pd.DataFrame(results, columns=columns)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

# Main testing interface
def main():
    st.sidebar.markdown("## Test Steps")
    st.sidebar.markdown("""
    1. **Connection** - Test basic connectivity
    2. **Tables** - Verify GEX tables exist  
    3. **Data** - Load recent scan results
    4. **Display** - Show your actual data
    """)
    
    # Step 1: Test connection
    st.header("1️⃣ Connection Test")
    
    connection = get_databricks_connection()
    
    if connection:
        st.success("✅ Successfully connected to Databricks!")
        
        # Step 2: Test basic query
        st.header("2️⃣ Basic Query Test")
        if test_basic_query(connection):
            st.success("✅ Basic SQL queries working!")
        else:
            st.error("❌ Basic SQL queries failed")
            return
        
        # Step 3: Check tables
        st.header("3️⃣ Table Structure Test")
        table_results = test_tables_exist(connection)
        
        for table, info in table_results.items():
            if info["exists"]:
                st.success(f"✅ {table} exists ({info['columns']} columns)")
            else:
                st.error(f"❌ {table} missing: {info['error']}")
        
        # Step 4: Fetch actual data
        st.header("4️⃣ Data Loading Test")
        
        with st.spinner("Loading recent GEX data..."):
            df = fetch_recent_data(connection)
        
        if not df.empty:
            st.success(f"✅ Loaded {len(df)} records!")
            
            # Display the data
            st.subheader("Recent Scan Results")
            st.dataframe(df, use_container_width=True)
            
            # Show data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Records Found", len(df))
                st.metric("Columns", len(df.columns))
            
            with col2:
                if 'scan_timestamp' in df.columns:
                    latest_scan = df['scan_timestamp'].max()
                    st.metric("Latest Scan", latest_scan)
                
                if 'symbol' in df.columns:
                    unique_symbols = df['symbol'].nunique()
                    st.metric("Unique Symbols", unique_symbols)
            
            # Show sample symbols if available
            if 'symbol' in df.columns:
                st.subheader("Sample Symbols Found")
                symbols = df['symbol'].unique()[:10]
                st.write(", ".join(symbols))
        
        else:
            st.warning("⚠️ No data found. Your tables exist but may be empty.")
            
            # Show alternative query to debug
            st.subheader("Debug: Try Manual Query")
            
            debug_query = st.text_area(
                "Test your own query:", 
                value="SHOW TABLES FROM gex_trading",
                height=100
            )
            
            if st.button("Run Debug Query"):
                try:
                    cursor = connection.cursor()
                    cursor.execute(debug_query)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    if results:
                        debug_df = pd.DataFrame(results, columns=columns)
                        st.dataframe(debug_df)
                    else:
                        st.info("Query returned no results")
                        
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")
        
        connection.close()
    
    else:
        st.error("❌ Cannot proceed - fix connection first")
        
        # Show expected secrets format
        st.subheader("Expected .streamlit/secrets.toml format:")
        st.code("""
[databricks]
server_hostname = "your-workspace.cloud.databricks.com"
http_path = "/sql/1.0/warehouses/your-warehouse-id" 
access_token = "your-databricks-token"
        """)

if __name__ == "__main__":
    main()
