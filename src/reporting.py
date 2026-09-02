# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Javadian. All rights reserved.

import json
import os


# Function to append/update JSON file with new dictionary
def append_to_json(file_path, new_data):
    # Initialize data structure as an empty dictionary if file doesn't exist
    data = {}

    # If the file exists, load its content into `data`
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

    # Use the "name" entry from `new_data` as the key
    key = new_data["name"]

    # Append new_data to the existing data, or create a new entry if it doesn't exist
    # If the key already exists, this will overwrite the existing data for that key
    data[key] = new_data

    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
