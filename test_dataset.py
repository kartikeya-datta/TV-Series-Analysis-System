import pandas as pd
import json
import re

data_path ="/Users/kartikeyadatta/Documents/Projects/Naruto/code/data/jutsus.json"


try:
    with open(data_path, 'r', encoding='utf-8') as file:
        json.load(file)
    print("✅ JSON is valid!")
except json.JSONDecodeError as e:
    print("❌ JSON Error:", e)
