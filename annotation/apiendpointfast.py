from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import os
import json

app = FastAPI()

# Directory where uploaded JSON files will be saved
UPLOAD_FOLDER = os.path.expanduser('~/Desktop/apitest/app/uploads')
print(UPLOAD_FOLDER)

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def save_json_files(json_data: List[Dict]) -> None:
    try:
        # Iterate through each item in the received JSON data
        for idx, item in enumerate(json_data):
            # Check if 'coco' key exists
            if 'coco' in item:
                coco_file_location = os.path.join(UPLOAD_FOLDER, f"coco_{item['project_title']}.json")
                print(f"Saving coco file {coco_file_location}")
                with open(coco_file_location, "w", encoding="utf-8") as file:
                    json.dump(item['coco'], file, indent=4)

            # Check if 'json' key exists
            if 'json' in item:
                json_file_location = os.path.join(UPLOAD_FOLDER, f"json_{item['project_title']}.json")
                print(f"Saving json file {json_file_location}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                with open(json_file_location, "w", encoding="utf-8") as file:
                    json.dump(item['json'], file, indent=4)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save JSON data: {e}")

# API Endpoint to receive multiple JSON files
@app.post("/upload")
async def upload_multiple_json(json_data: List[Dict]):
    # Call the function to save JSON files
    save_json_files(json_data)
    # Send confirmation message after successful processing
    print(f"Processed JSON data and saved to disk.")

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "JSON files processed and saved successfully",
            "note": "Entries containing 'coco' and 'json' saved to separate files."
        }
    )

# Get to the root server
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI server. Use /upload to send multiple JSON files."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)