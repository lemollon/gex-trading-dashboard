"""
Databricks Connection Troubleshooting Script
Run this to test your connection step by step
"""

import streamlit as st
import time

st.set_page_config(page_title="Databricks Connection Test", layout="wide")

st.title("🔧 Databricks Connection Troubleshooter")

# Step 1: Check if secrets exist
st.markdown("## Step 1: Checking Secrets Configuration")

if "databricks" in st.secrets:
    st.success("✅ Databricks secrets found!")
    
    # Check individual keys
    required_keys = ["server_hostname", "http_path", "access_token"]
    missing_keys = []
    
    for key in required_keys:
        if key in st.secrets["databricks"]:
            st.write(f"✅ {key}: Found")
        else:
            st.write(f"❌ {key}: Missing")
            missing_keys.append(key)
    
    if missing_keys:
        st.error(f"Missing required keys: {missing_keys}")
        st.stop()
else:
    st.error("❌ No Databricks configuration found in secrets!")
    st.markdown("""
    ### Add to `.streamlit/secrets.toml`:
    ```toml
    [databricks]
    server_hostname = "your-workspace.cloud.databricks.com"
    http_path = "/sql/1.0/warehouses/your-warehouse-id"
    access_token = "your-token"
    ```
    """)
    st.stop()

# Step 2: Try importing the connector
st.markdown("## Step 2: Testing Databricks SQL Connector")

try:
    from databricks import sql
    st.success("✅ databricks-sql-connector is installed")
except ImportError as e:
    st.error(f"❌ Missing databricks-sql-connector: {e}")
    st.markdown("Run: `pip install databricks-sql-connector`")
    st.stop()

# Step 3: Test the connection
st.markdown("## Step 3: Testing Connection")

if st.button("Test Connection"):
    try:
        with st.spinner("Connecting to Databricks..."):
            start_time = time.time()
            
            # Create connection with timeout
            connection = sql.connect(
                server_hostname=st.secrets["databricks"]["server_hostname"],
                http_path=st.secrets["databricks"]["http_path"],
                access_token=st.secrets["databricks"]["access_token"],
                _socket_timeout=10  # 10 second timeout
            )
            
            connection_time = time.time() - start_time
            st.success(f"✅ Connected successfully in {connection_time:.2f} seconds!")
            
            # Step 4: Test a simple query
            st.markdown("## Step 4: Testing Simple Query")
            
            cursor = connection.cursor()
            
            # Test query 1: Simple SELECT
            try:
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                st.success(f"✅ Simple query works: {result}")
            except Exception as e:
                st.error(f"❌ Simple query failed: {e}")
            
            # Test query 2: Check if your table exists
            st.markdown("## Step 5: Checking Your Table")
            
            try:
                # First check what catalog we're in
                cursor.execute("SELECT current_catalog()")
                catalog = cursor.fetchone()[0]
                st.write(f"Current catalog: {catalog}")
                
                # Check what schemas exist
                cursor.execute("SHOW SCHEMAS")
                schemas = cursor.fetchall()
                st.write("Available schemas:", [s[0] for s in schemas])
                
                # Try to check your specific table
                test_query = """
                SELECT COUNT(*) as row_count 
                FROM quant_projects.gex_trading.gex_pipeline_results
                """
                
                cursor.execute(test_query)
                count = cursor.fetchone()[0]
                st.success(f"✅ Table found with {count} rows")
                
                # Get sample data
                if count > 0:
                    st.markdown("### Sample Data")
                    cursor.execute("""
                    SELECT symbol, confidence_score, net_gex 
                    FROM quant_projects.gex_trading.gex_pipeline_results 
                    LIMIT 5
                    """)
                    
                    sample = cursor.fetchall()
                    for row in sample:
                        st.write(f"Symbol: {row[0]}, Confidence: {row[1]}, GEX: {row[2]}")
                
            except Exception as e:
                st.error(f"❌ Table query failed: {e}")
                st.markdown("""
                Possible issues:
                1. Table doesn't exist
                2. Wrong catalog/schema name
                3. No permissions
                4. Table name is different
                """)
            
            # Clean up
            cursor.close()
            connection.close()
            
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        st.markdown("""
        Common issues:
        1. Wrong server hostname
        2. Wrong HTTP path
        3. Invalid access token
        4. Warehouse is stopped
        5. Network/firewall issues
        """)

# Step 6: Alternative connection method
st.markdown("## Alternative: Direct Table Path Test")

if st.button("Test Alternative Query"):
    try:
        from databricks import sql
        
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        
        cursor = connection.cursor()
        
        # Try different table paths
        table_paths = [
            "quant_projects.gex_trading.gex_pipeline_results",
            "gex_trading.gex_pipeline_results",
            "default.gex_pipeline_results",
            "gex_pipeline_results"
        ]
        
        for path in table_paths:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {path}")
                count = cursor.fetchone()[0]
                st.success(f"✅ Found table at: {path} with {count} rows")
                break
            except:
                st.write(f"❌ Not found at: {path}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        st.error(f"Failed: {e}")

st.markdown("---")
st.markdown("### Debug Info")
st.write("Python version:", st.__version__)
st.write("Current working directory:", st.secrets.get("cwd", "Not set"))
