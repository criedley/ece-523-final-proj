from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import pandas as pd

df = DeepFace.find(
    img_path = "img1.png", 
    db_path = "C:\python-projects\ece-523-final-proj\database",
    )

print(df.head())