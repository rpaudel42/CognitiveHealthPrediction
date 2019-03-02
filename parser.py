import pandas as pd
import csv, sys
from datetime import datetime

class SensorParser():

    def __init__(self):
        print "\n\nStarting Sensor Parsing ------ "

    def read_files(self, filename):
        patient_action ={}
        with open(filename) as CASAS:
            data_file = csv.DictReader(CASAS)
            try:
                i = 0
                for row in data_file:
                    patient_action[i] = row
                    i += 1
            except csv.Error as e:
                sys.exit('Error in line %d: %s' % (data_file.line_num, e))
        return patient_action

    def parse_sensor(self, id):
        with open('Data/'+id+'.txt', 'r') as ro, open('Data/new'+id+'.txt', 'a') as rw:
            for line in ro.readlines():
                ls = line.replace("\t", " ").replace("="," ")
                rw.write(ls)

        df = pd.read_csv('Data/new'+id+'.txt', sep=" ", header=None, names=["date", "time", "sensor", "sensor_val", "action", "action_val"])
        #if df["action"]
        df.to_csv("Data/"+id+".csv")

    def get_part_of_day(self, hour):
        if hour >= 5 and hour < 8:
            return "early morning"
        elif hour >= 8 and hour <= 11:
            return "morning"
        elif hour == 12:
            return "noon"
        elif hour >= 13 and hour < 17 :
            return "afternoon"
        elif hour >= 17 and hour < 19:
            return "early evening"
        elif hour >= 19 and hour < 21:
            return "evening"
        elif hour >= 21 and hour <= 24:
            return "night"
        elif hour >= 0 and hour <= 4:
            return "night"


    def generate_features(self, id, writer):
        resident = self.read_files("Data/"+id+".csv")
        action_begin = False
        activity = ""
        time_to_next_activity = 0
        for row in resident:
            #print resident[row]
            if resident[row]["action"] and resident[row]["action_val"] == 'begin':
                action_begin = True
                activity = resident[row]["action"]

            if action_begin:
                try:
                    hour = datetime.strptime(resident[row]["time"], '%H:%M:%S.%f').hour
                except:
                    hour = datetime.strptime(resident[row]["time"]+'.000000', '%H:%M:%S.%f').hour

                part_of_day = self.get_part_of_day(hour)

                weekday = datetime.strptime(resident[row]["date"], '%Y-%m-%d').weekday()
                writer.writerow({'id': id,
                             'date': resident[row]["date"], 'time': resident[row]["time"],
                             'sensor': resident[row]["sensor"], 'sensor_val': resident[row]["sensor_val"],
                             'action': activity, 'action_val': resident[row]["action_val"],
                             'day_of_week': weekday, 'part_of_day':part_of_day})

            if resident[row]["action"] and resident[row]["action_val"] == 'end':
                action_begin = False

    def start_parsing(self):
        for i in range(10):
            self.parse_sensor(str(i + 1))
            # parse_sensor("3")
        fieldnames = ["id", "date", "time", "sensor", "sensor_val", "action", "action_val", "day_of_week",
                      "part_of_day"]
        fw = open('Data/transit.csv', 'w')
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(10):
            print "\nPatient [" + str(i) + "] successfully Parsed -----"
            self.generate_features(str(i + 1), writer)
            # generate_features("3")