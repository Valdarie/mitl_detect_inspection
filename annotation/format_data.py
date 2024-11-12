import os
import shutil

# change accordingly
src_folder = "Raw Cassettes"
dest_folder = "cassette_data_formatted"

# recreate destination folder if exists
if os.path.exists(dest_folder):
    shutil.rmtree(dest_folder)

os.makedirs(dest_folder)

for i in os.scandir(src_folder):
    # skip wonky folder
    if i.name == "Wonky":
        continue

    for j in os.scandir(i):
        # reformat sides
        match j.name:
            case "Front":
                side = "F"
            case "Back":
                side = "B"
            case _:
                direction = "_"

        for k in os.scandir(j):
            # reformat direction
            match k.name:
                case "1E":
                    direction = "E"
                case "1N":
                    direction = "N"
                case "1S":
                    direction = "S"
                case "1W":
                    direction = "W"
                case _:
                    direction = "_"

            for idx, f in enumerate(os.listdir(k)):
                # get new name and save to dest folder
                new_name = f"{i.name}{side}{direction}0{idx}{f[-4:]}"

                src_path = os.path.join(k, f)
                dest_path = os.path.join(dest_folder, new_name)

                shutil.copy(src_path, dest_path)
