import numpy as np
import pandas as pd
import torch

from helpers.dataloader import Dataloader
# from nn_architecture.models import TtsGenerator

if __name__=='__main__':
    """Take the trial samples of a study's csv-file and average over the participant and the condition"""

    kw_timestep = 'Time'
    kw_condition = 'Condition'

    # load file
    # file = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\generated_samples\sd_len100_train20_500ep.csv'
    file = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganTrialERP_len100.csv'
    df = pd.read_csv(file)
    data = Dataloader(file).get_data(shuffle=False)

    # get ParticipantIDs
    participant_ids = df['ParticipantID'].unique()

    # draw all samples from one participant into one vector
    participant_samples = np.zeros((participant_ids.shape[0]*2, data.shape[1]+1))
    for i, id in enumerate(participant_ids):
        # id = int(id)
        # get all samples from one participant and of one condition in one vector each
        samples_participant = data[df['ParticipantID'] == id]
        samples_cond0 = samples_participant[samples_participant[:, 0] == 0]
        samples_cond1 = samples_participant[samples_participant[:, 0] != 0]
        # add samples
        participant_samples[2*i, 0] = id
        participant_samples[2*i, 1:] = samples_cond0.mean(axis=0)
        participant_samples[2*i+1, 0] = id
        participant_samples[2*i+1, 1:] = samples_cond1.mean(axis=0)

        print(f'Participant {id} done')

    # save dataframe with columns ParticipantID, Condition, Time...,
    # create a column name for each time step
    time_steps = np.arange(0, data.shape[1]-1)
    time_steps = [kw_timestep + str(i) for i in time_steps]
    df = pd.DataFrame(participant_samples, columns=['ParticipantID', kw_condition]+time_steps)
    df.to_csv(r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganAverageERP_len100_OwnAvg.csv', index=False)
    print('Done!')



