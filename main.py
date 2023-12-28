import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import openai

# Set up OpenAI API key
openai.api_key = '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'

# Load input Excel sheet
input_excel_path = 'Genoshi Intern Test - Input Excel Sheet.xlsx'
df = pd.read_excel(input_excel_path)

# Connect to MongoDB and retrieve data
# (You might need to install pymongo library: pip install pymongo)
from pymongo import MongoClient

mongo_uri = 'mongodb+srv://intern:JeUDstYbGTSczN4r@interntest.i7decv0.mongodb.net/'
client = MongoClient(mongo_uri)
db = client.intern
papers_collection = db.papers

# Function to query MongoDB for data
def query_mongodb(row, column):
    result = papers_collection.find_one({'row': row, 'column': column})
    return result['data'] if result else None

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to generate text using GPT-2 model
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Iterate through each cell in the input Excel sheet
for index, row in df.iterrows():
    for column, cell_value in row.items():
        if pd.notna(cell_value):
            # Query MongoDB for data
            data_from_corpus = query_mongodb(row['Field'], column)

            if data_from_corpus:
                # Combine existing cell value with data from the corpus
                prompt = f"{row['Field']} {column} {cell_value}"
                generated_data = generate_text(prompt)

                # Update the cell value with the generated data
                df.at[index, column] = generated_data

# Save the output Excel file
output_excel_path = 'output_generated.xlsx'
df.to_excel(output_excel_path, index=False)