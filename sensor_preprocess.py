import csv, sys
import datetime
from datetime import datetime
from collections import namedtuple
import statistics

action_mean = {}
action_sd = {}

class SensorPreprocessor():
    def __init__(self):
        print "\n\nStarting Sensor Data Preprocessing ------ "

    def get_age(self, id):
        id = int(id)
        age = 0
        gender = 0
        if id == 1:
            age = 91
            gender = 2
        if id == 2:
            age = 82
            gender = 2
        if id == 3:
            age = 83
            gender = 1
        if id == 4:
            age = 82
            gender = 1
        if id == 5:
            age = 90
            gender = 1
        if id == 6:
            age = 91
            gender = 1
        if id == 7:
            age = 73
            gender = 1
        if id == 8:
            age = 90
            gender = 2
        if id == 9:
            age = 89
            gender = 2
        if id == 10:
            age = 84
            gender = 1
        print age, gender
        return age, gender


    def collect_unique_action(self, patient_action):
        actions = []
        for i in patient_action:
            if patient_action[i]["action"] not in actions:
                actions.append(patient_action[i]["action"])
        return actions


    def find_mean_sd_per_action(self, patient_action, actions):
        durations = {}
        for i in patient_action:
            if patient_action[i]["action"] not in durations:
                durations[patient_action[i]["action"]] = []
            durations[patient_action[i]["action"]].append(float(patient_action[i]["duration"]))

        for a in actions:
            action_mean[a] = statistics.mean(durations[a])
            action_sd[a] = statistics.stdev(durations[a])


    def bucketize_duration(self, duration, action):
        if duration < (action_mean[action] - 2 * action_sd[action]):
            val = "low"
        elif (action_mean[action] - 2 * action_sd[action]) <= duration <= (action_mean[action] - action_sd[action]):
            val = "mid-low"
        elif (action_mean[action] - action_sd[action]) <= duration <= (action_mean[action] + action_sd[action]):
            val = "mid"
        elif  (action_mean[action] + action_sd[action]) <= duration <= (action_mean[action] + 2 * action_sd[action]):
            val = "mid-high"
        elif duration > (action_mean[action] + 2 * action_sd[action]):
            val = "high"
        return  val


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

    def split_date_action(self, patient_action):
        #"id", "date", "time", "sensor", "sensor_val", "action", "action_val", "day_of_week", "part_of_day"

        fieldnames = ['id', 'month', 'action', 'followed_by', 'preceed_by', 'hr_of_day', 'duration', 'time_to_next_activity',
                      'start_pt', 'end_pt', 'start_time', 'end_time', 'day_of_week',
                      'motion_sensor_activated', 'light_sensor_count', 'light_on_count', 'light_off_count', 'is_weekend', 'mci']
        fw = open('Data/finalsensor.csv', 'w')
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        action =  patient_action[0]["action"]
        #id = patient_action[]
        start_point_of_day = patient_action[0]["part_of_day"]
        start_time = patient_action[0]["date"]+" "+patient_action[0]["time"]
        followed_by = ''
        #preceed_by =''
        action_start_time = patient_action[0]["time"]
        #print patient_action[0]
        i = 1
        is_cont_action = False
        motion_sensor_count = 0
        light_sensor_count = 0
        light_on_count = 0
        light_off_count = 0
        time_to_next_activity = 0
        last_activity_time = ""
        for row in patient_action:
            if row > 0:
                new_action = patient_action[row]["action"]
                new_point_of_day = patient_action[row]["part_of_day"]
                if action == new_action:# and point_of_day == new_point_of_day:
                    a = 1
                    if patient_action[row-1]["sensor"][0:1] == "M" and patient_action[row-1]["sensor_val"] == "ON":
                        motion_sensor_count += 1
                    if patient_action[row-1]["sensor"][0:2] == "LS" and int(patient_action[row-1]["sensor_val"]) >0:
                        light_sensor_count += 1
                    if patient_action[row-1]["sensor"][0:1] == "L" and patient_action[row-1]["sensor_val"] == "ON":
                        light_on_count += 1
                    if patient_action[row-1]["sensor"][0:1] == "L" and patient_action[row-1]["sensor_val"] == "OFF":
                        light_off_count += 1
                    if patient_action[row]["action_val"] == "end":
                        last_activity_time = patient_action[row]["date"]+" "+patient_action[row]['time']
                else:

                    FMT = '%Y-%m-%d %H:%M:%S.%f'
                    #print patient_action[row]["date"] + " " + start_time
                    if patient_action[row]["time"] != "NA":
                        end_time = patient_action[row-1]["date"]+" "+patient_action[row-1]["time"]
                        duration = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
                        min = duration.total_seconds()/60
                        #min = bucketize_duration(min, patient_action[row-1]["act"])
                        #print "last :- " + last_activity_time
                        #print "This :- " + end_time
                        time_to_next_activity = datetime.strptime(patient_action[row]["date"]+" "+patient_action[row]["time"], FMT) - datetime.strptime(last_activity_time, FMT)
                        time = time_to_next_activity.total_seconds() / 60
                        action = patient_action[row]["action"]
                        end_point_of_day = patient_action[row-1]["part_of_day"]
                        i += 1
                        #print "Total: " + str(i)

                        if patient_action[row-1]['day_of_week'] in ["5","6"]:
                            wknd = True
                        else:
                            wknd = False

                        if patient_action[row-1]["id"] == "1" or patient_action[row-1]["id"] == "6" or patient_action[row-1]["id"] == "7" or patient_action[row-1]["id"] == "8" or patient_action[row-1]["id"] == "9":
                            mci = '0'
                        else:
                            mci = '1'

                        #Add count 1 if sensor acitivity on same row
                        '''if patient_action[row]["sensor"][0:1] == "M" and patient_action[row]["sensor_val"] == "ON":
                            motion_sensor_count += 1
                        if patient_action[row]["sensor"][0:2] == "LS" and int(patient_action[row]["sensor_val"]) > 0:
                            light_sensor_count += 1
                        if patient_action[row]["sensor"][0:1] == "L" and patient_action[row]["sensor_val"] == "ON":
                            light_on_count += 1
                        if patient_action[row]["sensor"][0:1] == "L" and patient_action[row]["sensor_val"] == "OFF":
                            light_off_count += 1'''

                        writer.writerow({'id': patient_action[row-1]["id"], 'month': patient_action[row-1]["date"][5:7], 'action' : patient_action[row-1]["action"],
                                     'followed_by' : followed_by,'preceed_by' : patient_action[row]["action"], 'hr_of_day' : action_start_time[0:2] , 'duration': min,
                                     'start_pt' : start_point_of_day, 'end_pt': end_point_of_day, 'start_time' : action_start_time,
                                     'time_to_next_activity':time, 'end_time' : patient_action[row-1]["time"], 'day_of_week':patient_action[row-1]['day_of_week'], 'motion_sensor_activated': motion_sensor_count,
                                     'light_sensor_count':light_sensor_count, 'light_on_count':light_on_count, 'light_off_count':light_off_count,'is_weekend' : wknd,
                                     'mci' : mci})
                        if patient_action[row-1]["part_of_day"] != new_point_of_day and patient_action[row-1]["action"] == new_action:
                            is_cont_action = True
                        else:
                            is_cont_action = False
                        followed_by = patient_action[row-1]["action"]
                        start_point_of_day = patient_action[row-1]["part_of_day"]
                        start_time = patient_action[row]["date"] + " " + patient_action[row]["time"]
                        action_start_time = patient_action[row]["time"]
                        motion_sensor_count = 0
                        light_sensor_count = 0
                        light_on_count = 0
                        light_off_count = 0

    def start_preprocessing(self):
        p_dict = self.read_files("Data/transit.csv")
        self.split_date_action(p_dict)

        '''new_dict = read_files("finalsensor.csv")
        actions = collect_unique_action(new_dict)
        find_mean_sd_per_action(new_dict, actions)
        split_date_action(p_dict)

        print action_sd
        print "---"
        print action_mean
        print bucketize_duration(2, "Toilet")'''


