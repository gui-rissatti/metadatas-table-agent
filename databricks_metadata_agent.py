import os
import pandas as pd
import json
import time
from typing import Annotated, TypedDict, List, Dict, Any
from dotenv import load_dotenv
import traceback

# Load env vars
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- configuration ---
if "GOOGLE_API_KEY" not in os.environ:
    pass

LLM_MODEL = "gemini-2.5-flash" 
BATCH_SIZE = 10 # Process 10 tables per LLM call

# --- 1. Graph State ---

class AgentState(TypedDict):
    metadata_file_path: str
    pending_tables: List[str]
    current_batch: List[str]
    current_batch_schemas: Dict[str, Dict[str, Any]]
    generated_batch_descriptions: str # JSON string of list of TableDocumentation
    completed_updates: List[Dict[str, str]] # Track updates

# --- 2. Tool Definitions ---

@tool
def parse_metadata_csv(file_path: str) -> List[str]:
    """
    Reads the metadata CSV. Filters for tables where 'description' is missing OR any 'column_description' is missing.
    Returns a list of unique fully qualified table names (catalog_path) that need documentation.
    """
    print(f"--- [Tool] Parsing CSV: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        df['description'] = df['description'].fillna('')
        df['column_description'] = df['column_description'].fillna('')
        
        missing_table_desc = df[df['description'].str.strip() == '']['catalog_path'].unique()
        missing_col_desc = df[df['column_description'].str.strip() == '']['catalog_path'].unique()
        
        pending = list(set(missing_table_desc) | set(missing_col_desc))
        
        print(f"--- [Tool] Found {len(pending)} tables needing documentation (table or column level) ---")
        return pending
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

# --- 3. Nodes ---

def loader_node(state: AgentState):
    print("--- [Node] Loader ---")
    file_path = state.get("metadata_file_path")
    pending_tables = parse_metadata_csv.invoke(file_path)
    print(f"--- [Node] Loader - Found {len(pending_tables)} tables to process ---")
    return {"pending_tables": pending_tables, "completed_updates": []}

def dispatcher_node(state: AgentState):
    print("--- [Node] Dispatcher ---")
    pending = state.get("pending_tables", [])
    if not pending:
        return {"current_batch": []}
    
    # Take a batch
    batch_tables = pending[:BATCH_SIZE]
    remaining = pending[BATCH_SIZE:]
    
    file_path = state.get("metadata_file_path")
    df = pd.read_csv(file_path)
    
    batch_schemas = {}
    
    for table in batch_tables:
        schema_df = df[df['catalog_path'] == table]
        schema = {}
        for _, row in schema_df.iterrows():
            col_name = row['column_name']
            col_type = row['column_type']
            col_desc = row.get('column_description', '')
            if pd.isna(col_desc): col_desc = ""
            
            schema[col_name] = {
                "type": col_type,
                "description": str(col_desc)
            }
        batch_schemas[table] = schema

    print(f"--- [Node] Dispatcher - Prepared batch of {len(batch_tables)} tables ---")

    return {
        "current_batch": batch_tables,
        "pending_tables": remaining,
        "current_batch_schemas": batch_schemas
    }

class TableDocs(BaseModel):
    table_name: str
    table_description: str = Field(description="The description of the table.")
    column_descriptions: Dict[str, str] = Field(description="A dictionary mapping column names to their descriptions.")

class BatchTableDocs(BaseModel):
    tables: List[TableDocs]

def generation_node(state: AgentState):
    print("--- [Node] Generation (Batch) ---")
    batch_tables = state["current_batch"]
    batch_schemas = state["current_batch_schemas"]
    
    if not batch_tables:
        return {"generated_batch_descriptions": json.dumps([])}

    # Simple rate limiting logic
    time.sleep(4) # 15 RPM limit = 1 req every 4s. With batching, this is fine.
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: GOOGLE_API_KEY not found in environment variables.")
        return {"generated_batch_descriptions": json.dumps([])}
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=api_key, max_retries=10)
    structured_llm = llm.with_structured_output(BatchTableDocs)
    
    # Prepare prompt with all schemas
    schemas_str = json.dumps(batch_schemas, indent=2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Databricks data steward. You must generate documentation for multiple tables and their columns in Brazilian Portuguese."),
        ("user", """
        You will receive a JSON dictionary where keys are table names and values are their schemas (columns and existing descriptions).
        
        Input Schemas:
        {schemas}
        
        Task:
        For EACH table in the input:
        1. Generate a professional, concise description for the TABLE in Brazilian Portuguese.
        2. Generate a concise description for EACH COLUMN in Brazilian Portuguese.
           - If a column already has a good description, you can keep it or refine it.
           - If it is missing (empty), you MUST generate one based on the column name and type.
           - Ensure output maps correctly to the provided table names.
           
        Return ONLY the JSON structure matching the BatchTableDocs schema (a list of table documentations).
        """)
    ])
    
    try:
        chain = prompt | structured_llm
        result = chain.invoke({"schemas": schemas_str})
        
        #desc_data = result.dict()
        desc_data = result.model_dump()  # For compatibility with Pydantic v2
        desc_str = json.dumps(desc_data)
        
    except Exception as e:
        print(f"LLM Batch Generation Error: {e}")
        desc_str = json.dumps({"tables": []})
    
    return {"generated_batch_descriptions": desc_str}

def commit_node(state: AgentState):
    print("--- [Node] Commit (Accumulate Batch) ---")
    desc_json = state["generated_batch_descriptions"]
    
    try:
        data = json.loads(desc_json)
        tables_data = data.get("tables", [])
        
        print(f"Committing {len(tables_data)} results from batch...")
        
        new_updates = []
        for item in tables_data:
            # item is dict corresponding to TableDocs model
            # Re-serialize for storage structure as before or adapt save_node
            # Let's store individual updates as before to minimize save_node changes
            # We need table name, table desc, col descs
            
            # The structure we used before in 'documentation' field was {table_desc, column_descriptions}
            # item has {table_name, table_description, column_descriptions}
            
            doc_obj = {
                "table_description": item["table_description"],
                "column_descriptions": item["column_descriptions"]
            }
            
            new_updates.append({
                "catalog_path": item["table_name"],
                "documentation": json.dumps(doc_obj)
            })
            
        existing_updates = state.get("completed_updates", [])
        return {"completed_updates": existing_updates + new_updates}
        
    except Exception as e:
        print(f"Error parsing batch results: {e}")
        return {}

def save_node(state: AgentState):
    print("--- [Node] Save (Output CSV & XLSX) ---")
    updates = state.get("completed_updates", [])
    file_path = state.get("metadata_file_path")
    
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, "updated_metadata.csv")
    output_xlsx = os.path.join(output_dir, "updated_metadata.xlsx")
    
    print(f"Saving {len(updates)} table updates to {output_csv} and {output_xlsx}...")
    
    try:
        df = pd.read_csv(file_path)
        
        table_desc_map = {}
        col_desc_map = {}
        
        for item in updates:
            t_name = item["catalog_path"]
            try:
                doc = json.loads(item["documentation"])
                t_desc = doc.get("table_description", "")
                c_descs = doc.get("column_descriptions", {})
                
                table_desc_map[t_name] = t_desc
                for col, desc in c_descs.items():
                    col_desc_map[(t_name, col)] = desc
            except:
                continue
        
        def update_table_desc(row):
            if row['catalog_path'] in table_desc_map:
                new_desc = table_desc_map[row['catalog_path']]
                if new_desc: return new_desc
            return row['description']

        df['description'] = df.apply(update_table_desc, axis=1)
        
        def update_col_desc(row):
            key = (row['catalog_path'], row['column_name'])
            if key in col_desc_map:
                return col_desc_map[key]
            return row['column_description']
            
        df['column_description'] = df.apply(update_col_desc, axis=1)
        
        df.to_csv(output_csv, index=False, sep=';')
        print("CSV saved.")
        
        df.to_excel(output_xlsx, index=False)
        print("Excel saved.")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error saving files: {e}")
        
    return {}

# --- 4. Edge Logic ---

def should_process_next(state: AgentState):
    if state.get("current_batch"):
        return "generation_node"
    return "save_node"

# --- 5. Graph Definition ---

workflow = StateGraph(AgentState)

workflow.add_node("loader_node", loader_node)
workflow.add_node("dispatcher_node", dispatcher_node)
workflow.add_node("generation_node", generation_node)
workflow.add_node("commit_node", commit_node)
workflow.add_node("save_node", save_node)

workflow.set_entry_point("loader_node")

workflow.add_edge("loader_node", "dispatcher_node")
workflow.add_conditional_edges(
    "dispatcher_node",
    should_process_next,
    {
        "generation_node": "generation_node",
        "save_node": "save_node"
    }
)
workflow.add_edge("generation_node", "commit_node")
workflow.add_edge("commit_node", "dispatcher_node")
workflow.add_edge("save_node", END)

app = workflow.compile()

# --- Execution ---

if __name__ == "__main__":
    csv_path = r"c:\Users\guilh\OneDrive\Documentos\PROFISSIONAL\PROJETOS\agents\tables_metadata\data\New_Query_2025_12_14_11_05pm (3).csv"
    
    print(f"Starting Databricks Metadata Agent (Batch Size: {BATCH_SIZE})...")
    
    initial_state = {
        "metadata_file_path": csv_path,
        "pending_tables": [],
        "current_batch": [],
        "current_batch_schemas": {},
        "generated_batch_descriptions": "",
        "completed_updates": []
    }
    
    try:
        # Increase recursion limit to handle many batches
        for output in app.stream(initial_state, config={"recursion_limit": 1000000}):
            pass 
        print("Agent workflow completed.")
    except Exception as e:
        print(f"An error occurred: {e}")
