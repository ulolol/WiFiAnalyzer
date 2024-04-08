import pandas as pd
import folium
import torch
from tkinter import Tk, Button
from folium.plugins import MarkerCluster
from torch import nn
import numpy as np
from predictor import WifiSignalStrengthModel


#Load the saved model
model = WifiSignalStrengthModel()
model.load_state_dict(torch.load('wifi_signal_strength_model.pth'))
model.eval()  #Set the model to evaluation mode

#Load my WiFi data
data = pd.read_csv('your_data.csv')
data[['Latitude', 'Longitude']] = data['GPS Coordinates'].str.split(',', expand=True).astype(float)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour

#Normalize signal strength?? Check and only do if needed and prepare features for prediction

#Function to generate the map
def generate_map():
    wifi_map = folium.Map(location=[data.iloc[0]['Latitude'], data.iloc[0]['Longitude']], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(wifi_map)
    
    for index, row in data.iterrows():
        features = torch.tensor([[row['Hour'], row['Latitude'], row['Longitude']]], dtype=torch.float32)
        prediction = model(features).item()
        
        #Marker for actual signal strength
        folium.Marker(location=(row['Latitude'], row['Longitude']),
                      popup=f"Actual Signal Strength: {row['Signal Strength']}\nPredicted: {prediction}",
                      icon=folium.Icon(color='blue', icon='wifi', prefix='fa')).add_to(marker_cluster)
    
    wifi_map.save('wifi_signal_map.html')

#Function to load and display the map in a browser
def show_map():
    generate_map()
    import webbrowser
    webbrowser.open('wifi_signal_map.html', new=2)

#GUI
root = Tk()
root.title("Wi-Fi Signal Strength Map")

map_button = Button(root, text="Show Wi-Fi Signal Map", command=show_map)
map_button.pack()

root.mainloop()