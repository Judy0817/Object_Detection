used yolo8

from yolo8 default it will return these values
1. class
2. cofidence
3. box


To maintain the same ID for the same vehicle across frames, we can use an object tracking algorithm such as SORT (Simple Online and Realtime Tracking) or Deep SORT.
git clone https://github.com/abewley/sort.git
cd sort
pip install -r requirements.txt
