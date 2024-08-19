from ultralytics import YOLO


# Load YOLO Model best.pt Path
Load_Model_Path = 'TOP&BOTTOM_Detection.pt'

model = YOLO(Load_Model_Path)

# Test Clothing Image Path
Predict_Images_Path = 'test_Dir'

# Predict Clothing Image Path
Top_Bottom_Combination = 'Top_Bottom_Combination'

Insta_Images_Path = 'Insta_images'

# One Image Path
# One_Image_Path = '/Users/seunghunjang/Desktop/WOOTD_2/test_Combination/test_23.png'

# Save Predict Result Path
Predict_Result_Path = 'Predict_Result'

# Predict Model
result_Combination = model.predict(source=Predict_Images_Path, device='mps')
# result_Combination = model.predict(source=One_Image_Path, device='mps')
