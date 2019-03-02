import csv, sys
import datetime
from datetime import datetime
from collections import namedtuple
import statistics

action_mean = {}
action_sd = {}


def get_age(id):
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


def collect_unique_action(patient_action):
    actions = []
    for i in patient_action:
        if patient_action[i]["action"] not in actions:
            actions.append(patient_action[i]["action"])
    return actions


def find_mean_sd_per_action(patient_action, actions):
    durations = {}
    for i in patient_action:
        if patient_action[i]["action"] not in durations:
            durations[patient_action[i]["action"]] = []
        durations[patient_action[i]["action"]].append(float(patient_action[i]["duration"]))

    for a in actions:
        action_mean[a] = statistics.mean(durations[a])
        action_sd[a] = statistics.stdev(durations[a])


def bucketize_duration(duration, action):
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


def read_files(filename):
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

def split_date_action(patient_action):
    fieldnames = ['id', 'age', 'gender', 'month', 'action', 'followed_by','preceed_by', 'hr_of_day', 'duration', 'start_pt', 'end_pt','start_time','end_time', 'is_weekend', 'mci']
    fw = open('preceedby.csv', 'w')
    writer = csv.DictWriter(fw, fieldnames=fieldnames)
    writer.writeheader()

    action =  patient_action[0]["act"]
    #id = patient_action[]
    start_point_of_day = patient_action[0]["pt"]
    start_time = patient_action[0]["date"]+" "+patient_action[0]["time"]
    followed_by = ''
    #preceed_by =''
    action_start_time = patient_action[0]["time"]
    #print patient_action[0]
    i = 1
    is_cont_action = False
    for row in patient_action:
        if row > 0 :
            new_action = patient_action[row]["act"]
            new_point_of_day = patient_action[row]["pt"]
            if action == new_action:# and point_of_day == new_point_of_day:
                a = 1
            else:
                FMT = '%Y-%m-%d %H:%M:%S'
                #print patient_action[row]["date"] + " " + start_time
                if patient_action[row]["time"] != "NA":
                    end_time = patient_action[row-1]["date"]+" "+patient_action[row-1]["time"]
                    duration = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
                    min = duration.total_seconds()/60
                    #min = bucketize_duration(min, patient_action[row-1]["act"])
                    #print min
                    action = patient_action[row]["act"]
                    end_point_of_day = patient_action[row-1]["pt"]
                    i += 1
                    #print "Total: " + str(i)
                    age, gender = get_age(patient_action[row-1]["id"])
                    if patient_action[row-1]["mci"] == '1' or patient_action[row-1]["mci"] == '3':
                        mci = '0'
                    else:
                        mci = '1'
                    writer.writerow({'id': patient_action[row-1]["id"], 'age': age, 'gender': gender, 'month': patient_action[row-1]["date"][5:7], 'action' : patient_action[row-1]["act"],
                                 'followed_by' : followed_by,'preceed_by' : patient_action[row]["act"], 'hr_of_day' : action_start_time[0:2] , 'duration': min, 'start_pt' : start_point_of_day, 'end_pt': end_point_of_day,
                                 'start_time' : action_start_time, 'end_time' : patient_action[row-1]["time"], 'is_weekend' : patient_action[row-1]["wknd"],
                                 'mci' : mci})
                    if patient_action[row-1]["pt"] != new_point_of_day and patient_action[row-1]["act"] == new_action:
                        is_cont_action = True
                    else:
                        is_cont_action = False
                    followed_by = patient_action[row-1]["act"]
                    start_point_of_day = patient_action[row-1]["pt"]
                    start_time = patient_action[row]["date"] + " " + patient_action[row]["time"]
                    action_start_time = patient_action[row]["time"]

def casas_parser(argv):
    p_dict = read_files("MyData.csv")
    split_date_action(p_dict)

    '''new_dict = read_files("final.csv")
    actions = collect_unique_action(new_dict)
    find_mean_sd_per_action(new_dict, actions)
    split_date_action(p_dict)

    print action_sd
    print "---"
    print action_mean
    print bucketize_duration(2, "Toilet")'''


if __name__ == '__main__':
    casas_parser(sys.argv[1:])
