import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import time
import pprint

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

file_path = "../AI_FinanceAgent/public/TaxInvoiceWB1242503BV94995.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()
full_text = "\n".join([doc.page_content for doc in docs])

system_prompt = """ 
You are an expert financial analyst capable of reading and analyzing invoice documents in PDF format. Your task is to extract the following key information from these invoices based on **tag names** or **terms** within the document:
    1. Invoice Number (or Invoice No.)
    2. Invoice Date (or Date)
    3. Vendor Name (or Business Name, Supplier Name)
    4. Amount (Total Invoice Value, or Total Amount)
    5. Due Date (or Payment Due Date)

Here are the steps you will follow:

1. **Step: "Plan"**
   - In this step, you will read the PDF text and determine if the content is in a **tabular format** (e.g., a table) or a **regular unstructured text format**.
   - Identify the structure of the document and decide how to extract the required fields based on the format.

2. **Step: "Action"**
   - Extract the required information (Invoice Number, Invoice Date, Vendor Name, Amount, Due Date) by locating the **tag names** or **terms** in the text:
     - **Invoice Number**: Look for terms like "Invoice No.", "Invoice Number", or similar identifiers.  
       - **Example**: "Invoice No.: AbC-A001924001" or "Invoice Number: UP24-A001"
     - **Invoice Date**: Look for terms like "Date", "Invoice Date", or similar.  
       - **Example**: "Invoice Date: 04/01/2025" or "Date: 01-Apr-2025"
     - **Vendor Name**: Look for terms like "Business Name", "Supplier Name", "Vendor", or similar.  
       - **Example**: "Vendor: Tulsi Travel" or "Supplier Name: redBus India"
     - **Amount**: Look for terms like "Total Invoice Value", "Amount", "Total Amount", or similar.  
       - **Example**: "Total Amount: 4,195.80" or "Invoice Value: 3,996.00"
     - **Due Date**: Look for terms like "Due Date", "Payment Due Date", or similar.  
       - **Example**: "Due Date: 04/15/2025" or "Payment Due Date: 15-Apr-2025"
   - If any information is missing, fill in the field with `"N/A"`.

3. **Step: "Observe"**
   - After extracting the data, **validate** that each field is logically correct:
     - **Invoice Number**: Ensure it contains both numbers and letters (e.g., "RUP24-A001").
     - **Invoice Date**: Ensure it's in a valid date format (e.g., "04/01/2025").
     - **Vendor Name**: Ensure it's a valid string (non-empty).
     - **Amount**: Ensure it's a valid numerical value (e.g., "4,195.80").
     - **Due Date**: Ensure it's in a valid date format if available.
   
4. **Step: "Output"**
   - Provide the final extracted data in a structured **JSON format** as output.

**Rules:**
- You shouldnot be stuck in a loop of observe and action. If you have done the action and observe step once, you can directly move to output.
- Always output the extracted data in **JSON format**.
- If a field cannot be extracted or is unclear, place `"N/A"` in the output for that field.
- Always validate the logical correctness of the extracted data. 
- For tabular data, be sure to correctly match the required fields to their respective values, even if they are not in the same row or column.
- When extracting data, focus on **tag names** or **terms** (like "Invoice No.", "Due Date", "Vendor Name") to easily locate the relevant information.

**Output JSON Format:**
{
    "step": "string",  # Describes the current step (Plan, Action, Observe, Output)
    "content": "string"  # The JSON-formatted content with the extracted information
}
"""

messages = [
    { "role": "system", "content": system_prompt }
]
messages.append({ "role": "user", "content": full_text })
# messages.append({ "role": "user", "content": json.dumps({
#     "step": "Plan",
#     "content": "The document appears to be a tax invoice in a semi-structured text format. It contains key-value pairs and a table-like structure for describing charges. I will extract the required fields by identifying the relevant tag names."
# }) })
# messages.append({ "role": "user", "content": json.dumps({
#     "step": "Action",
#     "content": {
#         "Invoice Number": "WB1242503BV94995",
#         "Invoice Date": "04-Mar-2025",
#         "Vendor Name": "InterGlobe Aviation Limited",
#         "Amount": "4,648.00",
#         "Due Date": "N/A"
#     }
# }) })
# messages.append({ "role": "user", "content": json.dumps({
#     "step": "Observe",
#     "content": {
#         "Invoice Number": "WB1242503BV94995",
#         "Invoice Date": "04-Mar-2025",
#         "Vendor Name": "InterGlobe Aviation Limited",
#         "Amount": "4,648.00",
#         "Due Date": "N/A"
#     }
# }) })

attempt=4
while attempt>0:
    attempt-=1
    response = client.chat.completions.create(
        model='gemini-2.0-flash',
        response_format={"type": "json_object"},
        messages=messages,
    )
    print(response.choices[0].message.content)
    parsed_output = json.loads(response.choices[0].message.content)
    messages.append({ 'role': 'assistant', 'content': json.dumps(parsed_output) })
    if 'step' in parsed_output and parsed_output['step'] == 'output':
        break
