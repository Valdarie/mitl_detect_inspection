import json
import os
import time
import zipfile

import requests
from label_studio_sdk import Client

LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_KEY = "e95c4de73a049134538dee46704dd245c63e359e"
PROJECT_ID = 2

HPC_SERVER_URL = "http://192.168.79.119:8000/upload"


def export_ls_annotations(url: str, api_key: str, project_ids: list, formats: list):
    exports = []

    # connect to LabelStudio
    ls = Client(url=url, api_key=api_key)
    ls.check_connection()

    # export from all projects
    for i in project_ids:
        try:
            project = ls.get_project(i)
            details = project.get_params()
            export = {"project_title": details["title"]}
            export_id = project.export_snapshot_create(title="Export Snapshot")["id"]

            # wait until snapshot is ready
            while project.export_snapshot_status(export_id).is_in_progress():
                time.sleep(1.0)

            # download snapshot file for all export formats
            for f in formats:
                status, fname = project.export_snapshot_download(export_id, export_type=f)
                assert status == 200
                assert fname is not None

                export.update({f: fname})

            exports.append(export)
        except Exception as e:
            print(e)

    return exports


def upload_annotations(exports: list, url: str):
    annotation_data = []

    for export in exports:
        annotation = {
            "project_title": export["project_title"],
            "coco": json.loads(zipfile.ZipFile(export["COCO"], 'r').read("result.json")),
            "json": json.load(open(export["JSON"], "r"))
        }

        # TODO format filepath???
        # annotation["coco"] = format_coco_annotation()
        # annotation["json"] = format_json_annotation()
        annotation_data.append(annotation)

        os.remove(export["COCO"])
        os.remove(export["JSON"])

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    try:
        response = requests.post(url, data=json.dumps(annotation_data), headers=headers)
        if response.status_code == 200:
            print(f"Projects exported and uploaded: {', '.join([annotation['project_title'] for annotation in annotation_data])}")

        print(f"Server response: {response.json()}")
    except TimeoutError as e:
        print(e)


def run():
    # get project ids from users
    user_input = input("Please enter project ids to be exported seperated by \",\" (refer to labelstudio url):")
    project_ids = [int(i) for i in user_input.strip().split(",")]

    annotation_exports = export_ls_annotations(
        url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_KEY,
        project_ids=project_ids,
        formats=["COCO", "JSON"]
    )
    upload_annotations(
        exports=annotation_exports,
        url=HPC_SERVER_URL
    )


if __name__ == "__main__":
    run()
