import json

# Function to convert the original JSON
def convert_json(original):
    converted = {}
    bsNames = [name.capitalize() for name in original["facsNames"]]
    for index, entry in enumerate(original["weightMat"]):
        converted[str(index)] = {
            "Audio2Face": {
                "Body": {},
                "Facial": {
                    "Names": bsNames,
                    "Weights": entry,
                },
                "FrameTiming": {
                    "FPS": 30,
                    "Index": index
                }
            }
        }
    return converted

# Read the original JSON from a file
with open('C:/Users/ARNO/Desktop/audio2pose/datasets/sample/actual-speech1.json', 'r') as file:
    original_json = json.load(file)

# Convert the JSON
converted_json = convert_json(original_json)

# Write the converted JSON to a new file
with open('C:/Users/ARNO/Desktop/audio2pose/datasets/sample/sgc-actual-speech1.json', 'w') as outfile:
    json.dump(converted_json, outfile, indent=4)

# Optionally print the converted JSON
#print(json.dumps(converted_json, indent=4))