#!/bin/bash

#Specify the wireless interface and output file
interface="wlan0"
output_file="wifi_strength.csv"

#Check if the script is run as root
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

#Check if the wireless interface exists
if ! iwconfig $interface &> /dev/null
then
    echo "The interface $interface does not exist."
    exit
fi

#Write the header to the CSV file
echo "Timestamp,Signal Strength,GPS Coordinates" > $output_file

#Ask for initial GPS coordinates
read -p "Enter initial GPS coordinates (format: latitude,longitude): " gps_coordinates

#Start the loop
while true; do
    #Get the signal strength
    strength=$(iwconfig $interface | grep 'Signal level' | awk '{print $4}' | cut -d'=' -f2)

    #Check if the command was successful
    if [ $? -eq 0 ]
    then
        #Get the current timestamp
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")

        #Write the data to the CSV file
        echo "$timestamp,$strength,$gps_coordinates" >> $output_file

        #Print the data to the console
        echo "Timestamp: $timestamp, Wi-Fi Signal Strength: $strength dBm, GPS Coordinates: $gps_coordinates"
    else
        echo "An error occurred. Could not get the signal strength."
        exit
    fi

    #Ask for new GPS coordinates every 10 secs
    if (( SECONDS % 10 == 0 ))
    then
        read -p "Enter new GPS coordinates (or press Enter to keep the current ones): " new_coordinates
        if [[ ! -z "$new_coordinates" ]]
        then
            gps_coordinates=$new_coordinates
        fi
    fi

    #Wait for 1 second
    sleep 1
done
