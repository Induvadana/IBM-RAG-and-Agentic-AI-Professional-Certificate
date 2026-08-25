from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in Santa Monica, Mar de Cortez serves as a sun-drenched, casual taqueria "
    "specializing in Baja-style seafood. With a 4.2/5 rating, it captures the salt-air "
    "energy of the coast through its signature beer-battered snapper tacos and zesty "
    "octopus ceviche, making it a premier spot for open-air dining near the pier. "
    "Price range: $$."
)

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a Data Extraction Assistant specialized in structuring unstructured restaurant descriptions into a well defined JSON 
    format. Your task is to extract key attributes from restaurant descriptions and organize them consistently.Always return valid JSON
    with exact schema shown in the example.For the price range field, instruct the LLM to convert dollar signs (e.g., \\\\\\
    $$) into an integer (1,2,3) representing the number of dollar symbols.Do not output the dollar symbols.
    """
    
    base_user_prompt = f"""
    Task:
    Extract and structure the following restaurant description into a JSON object with these fields:
    - name: restaurant name (string)
    - location: location/neighborhood (string)
    - type: restaurant type (string)
    - food_style: cuisine/food style (string)
    - rating: numeric rating (float or null)
    - price_range: number of dollar signs as integer(1,2,3) or null
    - signatures: list of signature dishes (array of strings)
    - vibe: atmosphere/vibe description (string)
    - environment: environment description (string)
    - shortcomings: list of negative aspects or shortcomings (array of strings)

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    
    """
    return base_system_msg, base_user_prompt

# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt, params=None):
	#system_msg: the system message given to the LLM
    #prompt_txt: the user prompt
    
    model_id = "ibm/granite-4-h-small"

    project_id="skills-network"

    credentials = Credentials(
                    url = "https://us-south.ml.cloud.ibm.com"
                    )

    ### 1.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id
    )

    ### 1.2: Define the messages
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    ### 1.3: Get the final response output and return it
    response = model.chat(messages=messages)
    output_text = response["choices"][0]["message"]["content"]
    
    return output_text

def JSON_auto_repair_prompts(response, error_message):
    system_msg = "You repair malformed JSON and output JSON only."
    prompt_txt = (
        "The following JSON is invalid for the restaurant schema.\n\n"
        f"Validation error:\n{error_message}\n\n"
        f"Broken JSON:\n{response}\n\n"
        "Return corrected JSON with keys:\n"
        "id, name, city, cuisine, ambience, rating, price_range, signature_dishes, description\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- rating must be number or null.\n"
        "- signature_dishes must be an array of strings.\n"
        "- Keep meaning, only fix schema/types."
    )
    return system_msg, prompt_txt

def new_data_entry_process(paragraph, itemId):	
    system_msg, prompt_txt = restaurant_data_structure_prompt_generation(paragraph)
    raw_text = llm_model(system_msg, prompt_txt)
    raw_text = _safe_json_text(raw_text)
    max_retries = 3
    last_error = None
    for _ in range(max_retries):
        try:
            record = Restaurant.model_validate_json(raw_text).model_dump()
            record["id"] = int(itemId)
            return record
        except ValidationError as ve:
            last_error = str(ve)
            rep_sys, rep_prompt = JSON_auto_repair_prompts(raw_text, last_error)
            raw_text = _safe_json_text(llm_model(rep_sys, rep_prompt))
        except Exception as e:
            last_error = str(e)
            break
    raise ValueError(f"Failed to structure new entry after retries. Last error: {last_error}")

def load_data(filepath: str = FILEPATH) -> list:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data: list, filepath: str = FILEPATH) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")
        
        if choice == '1':
            print("\n--- Current Listings ---")
            list_restaurants(data)
			
        elif choice == '2':
            item_id = int(input("Enter id: ").strip())
            view_restaurant(data, item_id)
			
        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1
                data = add_restaurant(data)
                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                data = update_restaurant(data)

            elif choice == '5': # DELETE DATA
                data = delete_restaurant(data)

            elif choice == "6":
                save_data(data, FILEPATH)
                print("Saved and exiting.")
                break
            else:
                print("Invalid input.")

def get_next_id(data: list) -> int:
    if not data:
        return 1
    return max(int(item.get("id", 0)) for item in data) + 1


def list_restaurants(data: list) -> None:
    if not data:
        print("No restaurant records found.")
        return
    for item in data:
        print(
            f"[{item.get('id')}] {item.get('name')} | "
            f"{item.get('city', 'N/A')} | rating={item.get('rating', 'N/A')}"
        )


def view_restaurant(data: list, item_id: int) -> None:
    record = next((x for x in data if int(x.get("id", -1)) == item_id), None)
    if not record:
        print("Record not found.")
        return
    print(json.dumps(record, indent=2, ensure_ascii=False))


def add_restaurant(data: list) -> list:
    paragraph = input("Paste new restaurant paragraph: ").strip()
    if not paragraph:
        print("No paragraph provided.")
        return data
    item_id = get_next_id(data)
    record = new_data_entry_process(paragraph, item_id)
    data.append(record)
    print(f"Added restaurant id={item_id}: {record.get('name')}")
    return data


def update_restaurant(data: list) -> list:
    item_id = int(input("Enter id to update: ").strip())
    idx = next((i for i, x in enumerate(data) if int(x.get("id", -1)) == item_id), None)
    if idx is None:
        print("Record not found.")
        return data
    paragraph = input("Paste updated restaurant paragraph: ").strip()
    if not paragraph:
        print("No paragraph provided.")
        return data
    record = new_data_entry_process(paragraph, item_id)
    data[idx] = record
    print(f"Updated restaurant id={item_id}: {record.get('name')}")
    return data


def delete_restaurant(data: list) -> list:
    item_id = int(input("Enter id to delete: ").strip())
    new_data = [x for x in data if int(x.get("id", -1)) != item_id]
    if len(new_data) == len(data):
        print("Record not found.")
        return data
    print(f"Deleted restaurant id={item_id}")
    return new_data

# RUN THE UI
if __name__ == "__main__":
    manage_restaurants(FILEPATH, BACKUP_PATH)

class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())
		
if __name__ == "__main__":
    unittest.main() # Unit Test
