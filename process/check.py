
import json
def check_consistency(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    for timestamp, values in data.items():
        pow_rx_tx = values['pow_rx_tx']

        # Check if all lists have the same 2nd and 3rd elements
        second_element = pow_rx_tx[0][1]
        third_element = pow_rx_tx[0][2]

        for entry in pow_rx_tx:
            if entry[1] != second_element or entry[2] != third_element:
                return f"Inconsistent data found at timestamp {timestamp}"

    return "All pow_rx_tx entries have consistent 2nd and 3rd elements."


# Check the data
a = check_consistency("../processed_data/data.json")
print(a)