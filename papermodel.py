import csv, sys
import pandas as pd
import statistics
from sklearn.preprocessing import LabelEncoder

action_mean = {}
action_sd = {}

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
    data_file = pd.read_csv(filename)
    encoded_data = data_file.apply(LabelEncoder().fit_transform)
    return encoded_data


def split_date_action(patient_action):
    mean = statistics.mean(patient_action["id"])
    print mean
    fieldnames = ['id', 'ic', 'ic_sq', 'id-date', 'id-pt','date', 'action', 'is_weekend', 'pt','mci']
    fw = open('papermodel.csv', 'w')
    writer = csv.DictWriter(fw, fieldnames=fieldnames)
    writer.writeheader()

    for index, row in patient_action.iterrows():
        #print row['id'], row['pt']
        ic = float(row['id'])-mean
        #print ic
        writer.writerow({'id': row['id'], 'ic': ic, 'ic_sq': ic*ic, 'id-date': row['id']*row['date'],'id-pt' : row['id']*row['pt'],
                         'date': row['date'], 'action': row['act'], 'is_weekend': row['wknd'],
                         'pt' : row['pt'], 'mci': row['mci']})
def casas_parser(argv):
    p_dict = read_files("Ultimate.csv")
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
