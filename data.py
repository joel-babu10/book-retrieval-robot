
from roboflow import Roboflow
rf = Roboflow(api_key="WIBWPyhosWpzyNgf9IRx")
project = rf.workspace("harald-varner-xv5u7").project("book-spine-instance-segmentation")
dataset = project.version(1).download("yolov8-obb")