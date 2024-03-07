###############################################
## IMPORT MODULES                            ##
###############################################
import numpy as np
import random as rnd
from tqdm import tqdm

np.random.seed(1251)
rnd.seed(1251)

###############################################
## IMPORT MODULES                            ##
###############################################

#We will use the same training, test, and validation participants as Williams, Weinhardt et al., 2023
training_participants = [1, 2, 11, 12, 20, 23, 38, 43, 46, 49, 52, 53, 58, 60, 65, 74, 80, 81, 82, 93, 97, 106, 108, 109, 120, 127, 130, 131, 143, 158, 164, 166, 172, 173, 177, 194, 196, 197, 209, 212, 218, 222, 226, 230, 235, 238, 242, 244, 245, 251, 258, 260, 266, 273, 274, 277, 278, 281, 283, 285, 287, 291, 293, 302, 303, 309, 312, 315, 332, 339, 343, 351, 355, 356, 360, 372, 374, 383, 395, 398, 402, 403, 412, 422, 423, 427, 434, 435, 437, 439, 445, 459, 462, 466, 470, 476, 477, 482, 490, 493];
validation_participants = [5, 7, 9, 14, 16, 17, 21, 28, 32, 34, 39, 40, 42, 44, 45, 47, 48, 56, 62, 66, 67, 71, 73, 76, 77, 79, 84, 85, 88, 89, 92, 95, 96, 99, 100, 101, 104, 105, 107,110, 112, 113, 114, 115, 117, 119, 121, 122, 123, 124, 125, 136,137, 140, 144, 145, 147, 148, 151, 155, 156, 160, 162, 165, 171,175, 176, 178, 179, 182, 183, 184, 187, 189, 190, 191, 192, 200,201, 204, 207, 210, 214, 215, 216, 217, 221, 224, 225, 228, 231,232, 234, 236, 239, 240, 241, 243, 248, 250, 252, 253, 256, 259,262, 264, 265, 267, 270, 271, 275, 276, 280, 286, 288, 290, 304,306, 313, 316, 317, 318, 319, 320, 321, 322, 323, 326, 327, 329,338, 340, 341, 345, 346, 347, 348, 349, 350, 352, 353, 354, 357,361, 365, 367, 368, 373, 375, 378, 379, 380, 381, 384, 385, 387,388, 392, 396, 400, 401, 405, 408, 411, 413, 414, 418, 421, 425,426, 428, 429, 430, 431, 438, 440, 443, 452, 455, 456, 463, 465,467, 468, 469, 471, 472, 473, 479, 481, 483, 484, 485, 486, 489,495, 496, 497, 499, 500]
test_participants = [3, 4, 6, 8, 10, 13, 15, 18, 19, 22, 24, 25, 26, 27, 29, 30, 31, 33, 35, 36, 37, 41, 50, 51, 54, 55, 57, 59, 61, 63, 64, 68, 69, 70, 72, 75, 78, 83, 86, 87, 90, 91, 94, 98, 102, 103, 111, 116, 118, 126, 128, 129, 132, 133, 134, 135, 138, 139, 141, 142, 146, 149, 150, 152, 153, 154, 157, 159, 161, 163, 167, 168, 169, 170, 174, 180, 181, 185, 186, 188, 193, 195, 198, 199, 202, 203, 205, 206, 208, 211, 213, 219, 220, 223, 227, 229, 233, 237, 246, 247, 249, 254, 255, 257, 261, 263, 268, 269, 272, 279, 282, 284, 289, 292, 294, 295, 296, 297, 298, 299, 300, 301, 305, 307, 308, 310, 311, 314, 324, 325, 328, 330, 331, 333, 334, 335, 336, 337, 342, 344, 358, 359, 362, 363, 364, 366, 369, 370, 371, 376, 377, 382, 386, 389, 390, 391, 393, 394, 397, 399, 404, 406, 407, 409, 410, 415, 416, 417, 419, 420, 424, 432, 433, 436, 441, 442, 444, 446, 447, 448, 449, 450, 451, 453, 454, 457, 458, 460, 461, 464, 474, 475, 478, 480, 487, 488, 491, 492, 494, 498]

###############################################
## LOAD DATA                                 ##
###############################################

#Load EEG data
EEG_datasets = ['data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e1_len100.csv',
                'data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e2_len100.csv',
                'data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e8_len100.csv']

for EEG_dataset in tqdm(EEG_datasets):
    EEG_data = np.genfromtxt(EEG_dataset, delimiter=',', skip_header=1)
    EEG_header = np.genfromtxt(EEG_dataset, delimiter=',', names=True).dtype.names

    ###############################################
    ## EXTRACT VALIDATION/TEST DATASETS          ##
    ###############################################

    #Create validation file of 200 participants
    validation_dataset = []
    for participant in validation_participants: 
        validation_dataset.extend(EEG_data[EEG_data[:,0]==participant,:])

    save_path = 'data/Reinforcement Learning/Validation and Test Datasets/'
    save_filename = EEG_dataset.split('/')[-1].replace('_len100.csv','_validation.csv')
    np.savetxt(f"{save_path}{save_filename}", validation_dataset, delimiter=",", fmt='%f', header=','.join(EEG_header), comments='')

    #Create test file of 200 participants
    test_dataset = []
    for participant in test_participants: 
        test_dataset.extend(EEG_data[EEG_data[:,0]==participant,:])

    save_path = 'data/Reinforcement Learning/Validation and Test Datasets/'
    save_filename = EEG_dataset.split('/')[-1].replace('_len100.csv','_test.csv')
    np.savetxt(f"{save_path}{save_filename}", test_dataset, delimiter=",", fmt='%f', header=','.join(EEG_header), comments='')

    ###############################################
    ## EXTRACT TRAINING DATASETS                 ##
    ###############################################

    #Iterate through different sample sizes to save data
    for sample_size in tqdm([5, 10, 15, 20, 30, 60, 100]):
        for run in tqdm(range(5)):

            #Determine samples of current sample size
            sample_participants = np.sort(rnd.sample(training_participants, sample_size))
            
            #Create new data frame with determined samples
            reduced_data = []
            for participant in sample_participants:
                reduced_data.extend(EEG_data[EEG_data[:,0]==participant,:])

            #Save file
            save_path = 'data/Reinforcement Learning/Training Datasets/'
            save_filename = EEG_dataset.split('/')[-1].replace('_len100.csv',f'_SS{str(sample_size).zfill(3)}_Run0{run}.csv')
            np.savetxt(f"{save_path}{save_filename}", reduced_data, delimiter=",", fmt='%f', header=','.join(EEG_header), comments='')