import utils
from pathlib import Path

root = "/home/shubhamch/project/EdgeBTS/VISTA/Videos/"

# Output Videos From TileClipper
datasets = ["AIConditions/removedTileMp4", "DETRAC/removedTileMp4", "AINormal/removedTileMp4", "OurRec/removedTileMp4"]

# Aggregating removedTile videos of TileClipper
for i in range(len(datasets)):
    for data in Path(root + datasets[i]).iterdir():
        print(f"Aggregating {str(data)}")
        utils.aggregateSingleVideoSegments(str(data), str(data.name))
        print(f"Done on {str(data)}")


###### Recalibration Videos #################################################################
videos = root + "TileClipper_Without_Recalibration/When_calibrated_at_noon/AIConditions/removedTileMp4"

for data in Path(videos).iterdir():
    print(f"Aggregating {str(data)}")
    utils.aggregateSingleVideoSegments(str(data), "TileClipper_Without_Recalibration/When_calibrated_at_noon/" + str(data.name))
    print(f"Done on {str(data)}")


