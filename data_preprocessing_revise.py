import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
import os

from detecta import detect_peaks, detect_onset
from glob import glob
from tqdm import tqdm
import scipy.signal as signal
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

class PandasStandardScaler(StandardScaler):
    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns, index=X.index)

    
def lowpass_filter(data, sr, cut_off, order):
    nyq = 0.5 * sr
    b, a = signal.butter(order, cut_off/nyq, btype = 'low')
    lp_df = signal.filtfilt(b, a, data)
    return lp_df

def ang_vel_made(data, sr):
    x = np.diff(data) / (1 / sr)
    x = pd.Series(x, index = np.arange(1,len(data)))
    return x

def process_angle(data):
    '''
    관절 각도 값이 -180 ~ 180 도까지 표현되어 그 범위 이상으로 가는 값을 전처리해주는 작업
    '''
    up = np.where((np.diff(data) > 300))[0].tolist()
    down = np.where((np.diff(data) < -300))[0].tolist()
    all = sorted([*up, *down])
    all_len = len(all)
    if (len(up) == 0) & (len(down) == 0):
        pass
    
    elif (len(up) == 1) & (len(down) == 0):
        data.iloc[all[0]+1:] = data.iloc[all[0]+1:] - 360

    elif (len(up) == 0) & (len(down) == 1):
        data.iloc[all[0]+1:] = data.iloc[all[0]+1:] + 360

    elif (len(up) >= 1) & (len(down) >= 1):
        if up[0] < down[0]:
            if all_len % 2 == 0:
                for i in range(0,all_len,2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] - 360
            elif all_len % 2 == 1:
                for i in range(0,all_len - 1, 2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] - 360
                data.iloc[all[-1]+1:] = data.iloc[all[-1]+1:] - 360                                               
        elif up[0] > down[0]:
            if all_len % 2 == 0:
                for i in range(0,all_len,2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] + 360
            elif all_len % 2 == 1:
                for i in range(0,all_len - 1, 2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] + 360
                data.iloc[all[-1]+1:] = data.iloc[all[-1]+1:] + 360
                
    return data

def dataframe_differentiate_vel(data, sr):
    '''
    data = 데이터 프레임 형태의 데이터
    sr = sampling rate 
    아웃풋 : 데이터 프레임 형태의 각도값을 미분을 통한 데이터 프레임 형태의 각속도 값으로 추출
    '''
    N = data.shape[0]
    cols = [f"{col}_VEL" for col in data.columns]
    velocity = (data - data.shift(1))/ (1/sr)
    velocity.columns = cols
    return velocity

def dataframe_differentiate_acc(data, sr):
    '''
    data = 데이터 프레임 형태의 데이터
    sr = sampling rate 
    아웃풋 : 데이터 프레임 형태의 각도값을 미분을 통한 데이터 프레임 형태의 각속도 값으로 추출
    '''
    N = data.shape[0]
    cols = [f"{col.replace('_VEL','_ACC')}" for col in data.columns]
    acc = (data - data.shift(1))/ (1/sr)
    acc.columns = cols
    return acc
    
def data_processing(file_list, file_name):
    dir_meta = os.path.join(os.getcwd(), 'metadata.csv').replace('\\','/')
    meta = pd.read_csv(dir_meta)
    poi = pd.read_csv('poi/poi_metrics.csv')
    right_jc_cols = ['COM_X', 'COM_Y', 'COM_Z',
                    'LEAD_HIP_X', 'LEAD_HIP_Y', 'LEAD_HIP_Z', 'LEAD_KJC_X', 'LEAD_KJC_Y', 'LEAD_KJC_Z',
                    'LEAD_AJC_X', 'LEAD_AJC_Y', 'LEAD_AJC_Z', 'REAR_SJC_X', 'REAR_SJC_Y', 'REAR_SJC_Z', 
                    'REAR_EJC_X', 'REAR_EJC_Y', 'REAR_EJC_Z', 'REAR_WJC_X', 'REAR_WJC_Y', 'REAR_WJC_Z', 
                    'REAR_HJC_X', 'REAR_HJC_Y', 'REAR_HJC_Z', 
                    'REAR_HIP_X', 'REAR_HIP_Y', 'REAR_HIP_Z', 'REAR_KJC_X', 'REAR_KJC_Y', 'REAR_KJC_Z', 
                    'REAR_AJC_X', 'REAR_AJC_Y', 'REAR_AJC_Z', 'LEAD_SJC_X', 'LEAD_SJC_Y', 'LEAD_SJC_Z', 
                    'LEAD_EJC_X', 'LEAD_EJC_Y', 'LEAD_EJC_Z', 'LEAD_WJC_X', 'LEAD_WJC_Y', 'LEAD_WJC_Z',
                    'LEAD_HJC_X', 'LEAD_HJC_Y', 'LEAD_HJC_Z']

    left_jc_cols = ['COM_X', 'COM_Y', 'COM_Z',
                    'REAR_HIP_X', 'REAR_HIP_Y', 'REAR_HIP_Z', 'REAR_KJC_X', 'REAR_KJC_Y', 'REAR_KJC_Z',
                    'REAR_AJC_X', 'REAR_AJC_Y', 'REAR_AJC_Z', 'LEAD_SJC_X', 'LEAD_SJC_Y', 'LEAD_SJC_Z', 
                    'LEAD_EJC_X', 'LEAD_EJC_Y', 'LEAD_EJC_Z', 'LEAD_WJC_X', 'LEAD_WJC_Y', 'LEAD_WJC_Z', 
                    'LEAD_HJC_X', 'LEAD_HJC_Y', 'LEAD_HJC_Z', 
                    'LEAD_HIP_X', 'LEAD_HIP_Y', 'LEAD_HIP_Z', 'LEAD_KJC_X', 'LEAD_KJC_Y', 'LEAD_KJC_Z', 
                    'LEAD_AJC_X', 'LEAD_AJC_Y', 'LEAD_AJC_Z', 'REAR_SJC_X', 'REAR_SJC_Y', 'REAR_SJC_Z', 
                    'REAR_EJC_X', 'REAR_EJC_Y', 'REAR_EJC_Z', 'REAR_WJC_X', 'REAR_WJC_Y', 'REAR_WJC_Z',
                    'REAR_HJC_X', 'REAR_HJC_Y', 'REAR_HJC_Z']

    ANGLE = {}
    FORCE = {}
    JOINT = {}
    wrong = {'angle':{},
             'force':{},
             'joint':{},
             'weight':{}}
    for file, fname in tqdm(zip(file_list, file_name)):
        try:
            df = pd.read_csv(file, sep='\t', encoding='cp949', header=[1,2,3,4])
            df.drop('Unnamed: 0_level_0', axis=1, inplace=True)
            df.drop('LAR_ROTMAT', axis=1, inplace=True)
            kinematic_len = len(df['FRAMES'].dropna())
            kinetic_len = kinematic_len * 3
            cols = []
            for col in df.columns:
                if col[0] not in ['FP1','FP2','FP3']:
                    cols.append(col)

            kine = df[cols].iloc[:kinematic_len, :]
            force = df[['FP1','FP2','FP3']].iloc[:kinetic_len,:]


            '''
            세션, 몸무게, 키, 주팔, 공속도
            '''
            session,weight,height = meta[meta['filename_new'] == f"{fname}.c3d"][['session_pitch','session_mass_kg','session_height_m']].values[0] 
            dominant, bal_vel = poi[poi['session_pitch'] == session][['p_throws','pitch_speed_mph']].values[0]


            '''
            지면 반력 컬럼 정리
            '''
            rear_fp_col = []
            lead_fp_col = []
            for col in force.columns:
                side, data, _, axis = col
                if side == 'FP2':
                    rear_fp_col.append(f"{side}_{data}_{axis}")
                if side in ['FP1', 'FP3']:
                    lead_fp_col.append(f"{side}_{data}_{axis}")
                    
            lead_fp = force[['FP1','FP3']]; lead_fp.columns = lead_fp_col
            rear_fp = force['FP2']; rear_fp.columns = rear_fp_col

            usecol = []
            for fp1, fp3 in zip(lead_fp_col[:3],lead_fp_col[7:10]):
                fp, data, axis = fp1.split('_')
                fp = 'LEAD'
                lead_fp[f"{fp}_{data}_{axis}"] = lead_fp[fp1] + lead_fp[fp3]
                usecol.append(f"{fp}_{data}_{axis}")

            lead_fp = lead_fp[usecol]
            rear_fp = rear_fp[['FP2_FORCE_X','FP2_FORCE_Y','FP2_FORCE_Z']]; rear_fp.columns = ['REAR_FORCE_X','REAR_FORCE_Y','REAR_FORCE_Z']
            fp = pd.concat([lead_fp, rear_fp], axis=1)

            '''
            각도와 관절점 컬럼 정리
            '''
            angle_cols = {
                'col' : [],
                'idx' : []
            }
            jc_cols = {
                'col' : [],
                'idx' : []
            }
            for idx, col in enumerate(kine.columns):
                c, _, _, ax = col
                if ("ANGLE" in c):
                    angle_cols['col'].append(f"{c}_{ax}")
                    angle_cols['idx'].append(idx)
                elif ("JC" in c) | ("RIGHT_HIP" in c) | ("LEFT_HIP" in c) | ("CenterOfMass" in c):
                    jc_cols['col'].append(f"{c}_{ax}")
                    jc_cols['idx'].append(idx)
                    
            angle = kine.iloc[:,angle_cols['idx']]; angle.columns = angle_cols['col']
            joint = kine.iloc[:,jc_cols['idx']]; joint.columns = jc_cols['col']

            kine_cols = angle.columns.tolist()
            for idx ,col in enumerate(kine_cols):
                if 'R' in dominant: # 오른손 투수
                    pit_type = 'R'
                    if ('ANKLE' in col) | ('HIP' in col) | ('KNEE' in col) | ('HEEL' in col):
                        split_col = col.split('_')
                        
                        if 'R' in split_col[0]: # 오른손 투수의 오른쪽 다리 => REAR
                            split_col[0] = 'REAR'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col
                        elif 'L' in split_col[0]: # 오른손 투수의 왼쪽 다리 => LEAD
                            split_col[0] = 'LEAD'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col

                    elif ('ELBOW' in col) | ('SHOULDER' in col) | ('WRIST' in col):
                        split_col = col.split('_')
                        
                        if 'R' in col.split('_')[0]: # 오른손 투수의 오른쪽 팔 => LEAD
                            split_col[0] = 'LEAD'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col
                        elif 'L' in col.split('_')[0]: # 오른손 투수의 왼쪽 팔 => REAR
                            split_col[0] = 'REAR'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col
                            
                elif 'L' in dominant: # 왼손 투수
                    pit_type = 'L'
                    if ('ANKLE' in col) | ('HIP' in col) | ('KNEE' in col) | ('HEEL' in col):
                        split_col = col.split('_')
                        
                        if 'R' in split_col[0]: # 왼쪽 투수의 오른쪽 다리 => LEAD
                            split_col[0] = 'LEAD'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col
                        elif 'L' in split_col[0]: # 왼쪽 투수의 왼쪽 다리 => REAR
                            split_col[0] = 'REAR'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col
                            
                    elif ('ELBOW' in col) | ('SHOULDER' in col) | ('WRIST' in col):
                        split_col = col.split('_')
                        
                        if 'R' in col.split('_')[0]: # 왼쪽 투수의 오른쪽 팔 => REAR
                            split_col[0] = 'REAR'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col
                        elif 'L' in col.split('_')[0]: # 왼쪽 투수의 왼쪽 팔 => LEAD
                            split_col[0] = 'LEAD'
                            split_col = '_'.join(split_col)
                            kine_cols[idx] = split_col

            angle.columns = kine_cols
            for col in angle.columns:
                angle[col] = process_angle(angle[col])

            if dominant == 'R':
                joint.columns = right_jc_cols
            elif dominant == 'L':
                joint.columns = left_jc_cols
            
            
            '''
            왼손 오른손 부호 변경
            '''
            angle_change_plus_minus = []
            if dominant == 'L':
                for col in angle.columns:
                    if ('_Y' in col) | ('_Z' in col):
                        angle_change_plus_minus.append(col)
                angle[angle_change_plus_minus] = - angle[angle_change_plus_minus]
            
            if dominant == 'L':
                fp[['LEAD_FORCE_Y','REAR_FORCE_Y']] = - fp[['LEAD_FORCE_Y','REAR_FORCE_Y']]

            joint_change_plus_minus = []
            if dominant == 'L':
                for col in joint.columns:
                    if '_Y' in col:
                        joint_change_plus_minus.append(col)
                joint[joint_change_plus_minus] = - joint[joint_change_plus_minus]
                
            angle['LEAD_SHOULDER_ANGLE_Z'] = - angle['LEAD_SHOULDER_ANGLE_Z']
            angle['TORSO_ANGLE_Z'] = angle['TORSO_ANGLE_Z'] - angle['TORSO_ANGLE_Z'][0]
            angle['LEAD_ANKLE_VIRTUAL_LAB_ANGLE_Z'] = angle['LEAD_ANKLE_VIRTUAL_LAB_ANGLE_Z'] - angle['LEAD_ANKLE_VIRTUAL_LAB_ANGLE_Z'][0]
            angle['PELVIS_ANGLE_Z'] = angle['PELVIS_ANGLE_Z'] - angle['PELVIS_ANGLE_Z'][0]
            
            
            '''
            시점 찾기
            '''
            kinematic_sr = 360
            force_sr = 1080
            
            elbow_wrist_dis = joint['LEAD_EJC_X'] - joint['LEAD_WJC_X']
            lead_force_result = np.sqrt(fp['LEAD_FORCE_X']**2 + fp['LEAD_FORCE_Y']**2 + fp['LEAD_FORCE_Z']**2)
            rear_force_result = np.sqrt(fp['REAR_FORCE_X']**2 + fp['REAR_FORCE_Y']**2 + fp['REAR_FORCE_Z']**2)
            
            plot = False
            kh= detect_peaks(joint['LEAD_KJC_Z'], mph = 0.5, mpd=kinematic_sr*3, show=plot)[0]
            ic = detect_onset(lead_force_result, threshold = 10, n_above=force_sr*0.1,n_below=20, show=plot)[-1][0]
            fc = detect_onset((lead_force_result/9.81)/weight, threshold = 0.1, n_above=force_sr*0.1,n_below=force_sr*0.05, show=plot)[-1][0]
            mer = detect_peaks(angle['LEAD_SHOULDER_ANGLE_Z'], mph = 130, mpd=kinematic_sr*3,show=plot)[0]
            br = detect_onset(elbow_wrist_dis.iloc[mer:], threshold=0, n_above = kinematic_sr*0.01, n_below = kinematic_sr*0.1, show=plot) + mer
            ic = int(ic/3)
            fc = int(fc/3)
            br = br[0][-1] + 4

            if kh < ic <= fc < mer < br:
                pass
            else:
                wrong['angle'][file] = angle
                wrong['force'][file] = fp
                wrong['joint'][file] = joint
                wrong['weight'][file] = weight
                wrong['force'[file]] = fp
            
            '''
            데이터 저장
            '''
            fp['LEAD_FORCE_R'] = lead_force_result
            fp['REAR_FORCE_R'] = rear_force_result
            
            
            angle['kh_frame'] = kh
            angle['ic_frame'] = ic
            angle['fc_frame'] = fc
            angle['mer_frame'] = mer
            angle['br_frame'] = br
            angle['height'] = height
            angle['weight'] = weight
            angle['ballspeed'] = bal_vel
            angle['dominant'] = dominant
            
            fp['kh_frame'] = kh * 3
            fp['ic_frame'] = ic * 3
            fp['fc_frame'] = fc * 3
            fp['mer_frame'] = mer * 3
            fp['br_frame'] = br * 3
            fp['height'] = height
            fp['weight'] = weight
            fp['ballspeed'] = bal_vel
            fp['dominant'] = dominant
            
            joint['kh_frame'] = kh
            joint['ic_frame'] = ic
            joint['fc_frame'] = fc
            joint['mer_frame'] = mer
            joint['br_frame'] = br
            joint['height'] = height
            joint['weight'] = weight
            joint['ballspeed'] = bal_vel
            joint['dominant'] = dominant
            
            FORCE[session] = fp
            ANGLE[session] = angle
            JOINT[session] = joint
        except Exception as e:
            pass
            # print(file, ':', e)
            # wrong['angle'][file] = angle
            # wrong['force'][file] = fp
            # wrong['joint'][file] = joint
            # wrong['weight'][file] = weight
    
    '''
    데이터 이상치 제거
    '''
    delete = [
              '2919_5','2918_1','2918_2','2918_4','2923_1','2923_2','2923_3','2923_5',
              '1371_2','1371_3', 
              '2996_2','2996_3','1615_2','3132_5','2655_2',
              '2655_5','3035_5','2818_1','2818_2','2818_3','2818_4',
              '2818_5','2860_1','2860_5','2905_1','2905_3','3232_2',
              ]
    for d in delete:
        del FORCE[d]
        del ANGLE[d]
        del JOINT[d]
        
    return FORCE, ANGLE, JOINT

def split_train_test(JOINT, random):
    from sklearn.model_selection import train_test_split
    sessions = list(JOINT.keys())
    players_trial = {}
    for session in sessions:
        if session.split('_')[0] not in players_trial:
            players_trial[session.split('_')[0]] = []
            
        players_trial[session.split('_')[0]].append(session)

    player = list(players_trial.keys())

    train, test = train_test_split(player, test_size=0.1, random_state=random)
    TRAIN = []
    TEST = []
    for trn in train:
        TRAIN.extend(players_trial[trn])
    for tst in test:
        TEST.extend(players_trial[tst])
        
    return players_trial, train, test


def input_target(JOINT, FORCE, train, test, joint_cut_off, force_cut_off):
    inputs_sc = {
        'train' : {
            'rear' : {},
            'lead' : {},
        },
        'test' : {
            'rear' : {},
            'lead' : {},
        },
    }

    targets = {
        'train' : {
            'rear' : {},
            'lead' : {},
        },
        'test' : {
            'rear' : {},
            'lead' : {},
        },
    }

    force_col = ['LEAD_FORCE_X','LEAD_FORCE_Y','LEAD_FORCE_Z','REAR_FORCE_X','REAR_FORCE_Y','REAR_FORCE_Z']
    


    jc_cols =  ['LEAD_HIP_X','LEAD_HIP_Y','LEAD_HIP_Z', 'LEAD_KJC_X','LEAD_KJC_Y','LEAD_KJC_Z',
                'LEAD_AJC_X','LEAD_AJC_Y','LEAD_AJC_Z', 'REAR_SJC_X','REAR_SJC_Y','REAR_SJC_Z',
                'REAR_EJC_X','REAR_EJC_Y','REAR_EJC_Z', 'REAR_WJC_X','REAR_WJC_Y','REAR_WJC_Z',
                'REAR_HIP_X','REAR_HIP_Y','REAR_HIP_Z', 'REAR_KJC_X','REAR_KJC_Y','REAR_KJC_Z',
                'REAR_AJC_X','REAR_AJC_Y','REAR_AJC_Z', 'LEAD_SJC_X','LEAD_SJC_Y','LEAD_SJC_Z',
                'LEAD_EJC_X','LEAD_EJC_Y','LEAD_EJC_Z', 'LEAD_WJC_X','LEAD_WJC_Y','LEAD_WJC_Z']

    new_jc_cols = [
                    'LEAD_HIP_X','LEAD_HIP_Y','LEAD_HIP_Z', 'LEAD_KJC_X','LEAD_KJC_Y','LEAD_KJC_Z',
                    'LEAD_AJC_X','LEAD_AJC_Y','LEAD_AJC_Z', 'REAR_SJC_X','REAR_SJC_Y','REAR_SJC_Z',
                    'REAR_EJC_X','REAR_EJC_Y','REAR_EJC_Z', 'REAR_WJC_X','REAR_WJC_Y','REAR_WJC_Z',
                    'REAR_HIP_X','REAR_HIP_Y','REAR_HIP_Z', 'REAR_KJC_X','REAR_KJC_Y','REAR_KJC_Z',
                    'REAR_AJC_X','REAR_AJC_Y','REAR_AJC_Z', 'LEAD_SJC_X','LEAD_SJC_Y','LEAD_SJC_Z',
                    'LEAD_EJC_X','LEAD_EJC_Y','LEAD_EJC_Z', 'LEAD_WJC_X','LEAD_WJC_Y','LEAD_WJC_Z',
                    
                    'LEAD_HIP_X_VEL', 'LEAD_HIP_Y_VEL', 'LEAD_HIP_Z_VEL', 'LEAD_KJC_X_VEL','LEAD_KJC_Y_VEL', 'LEAD_KJC_Z_VEL',
                    'LEAD_AJC_X_VEL', 'LEAD_AJC_Y_VEL', 'LEAD_AJC_Z_VEL', 'REAR_SJC_X_VEL','REAR_SJC_Y_VEL', 'REAR_SJC_Z_VEL',
                    'REAR_EJC_X_VEL', 'REAR_EJC_Y_VEL', 'REAR_EJC_Z_VEL', 'REAR_WJC_X_VEL','REAR_WJC_Y_VEL', 'REAR_WJC_Z_VEL',
                    'REAR_HIP_X_VEL', 'REAR_HIP_Y_VEL', 'REAR_HIP_Z_VEL', 'REAR_KJC_X_VEL','REAR_KJC_Y_VEL', 'REAR_KJC_Z_VEL',
                    'REAR_AJC_X_VEL', 'REAR_AJC_Y_VEL', 'REAR_AJC_Z_VEL', 'LEAD_SJC_X_VEL','LEAD_SJC_Y_VEL', 'LEAD_SJC_Z_VEL',
                    'LEAD_EJC_X_VEL', 'LEAD_EJC_Y_VEL', 'LEAD_EJC_Z_VEL', 'LEAD_WJC_X_VEL','LEAD_WJC_Y_VEL', 'LEAD_WJC_Z_VEL',
                    
                    'LEAD_HIP_X_ACC', 'LEAD_HIP_Y_ACC', 'LEAD_HIP_Z_ACC', 'LEAD_KJC_X_ACC','LEAD_KJC_Y_ACC', 'LEAD_KJC_Z_ACC',
                    'LEAD_AJC_X_ACC', 'LEAD_AJC_Y_ACC', 'LEAD_AJC_Z_ACC', 'REAR_SJC_X_ACC','REAR_SJC_Y_ACC', 'REAR_SJC_Z_ACC',
                    'REAR_EJC_X_ACC', 'REAR_EJC_Y_ACC', 'REAR_EJC_Z_ACC', 'REAR_WJC_X_ACC','REAR_WJC_Y_ACC', 'REAR_WJC_Z_ACC',
                    'REAR_HIP_X_ACC', 'REAR_HIP_Y_ACC', 'REAR_HIP_Z_ACC', 'REAR_KJC_X_ACC','REAR_KJC_Y_ACC', 'REAR_KJC_Z_ACC',
                    'REAR_AJC_X_ACC', 'REAR_AJC_Y_ACC', 'REAR_AJC_Z_ACC', 'LEAD_SJC_X_ACC','LEAD_SJC_Y_ACC', 'LEAD_SJC_Z_ACC',
                    'LEAD_EJC_X_ACC', 'LEAD_EJC_Y_ACC', 'LEAD_EJC_Z_ACC', 'LEAD_WJC_X_ACC','LEAD_WJC_Y_ACC', 'LEAD_WJC_Z_ACC']
    for session in JOINT:
        joint = JOINT[session]
        force = FORCE[session][force_col]

        weight = joint['weight'][0]        
        kh = joint['kh_frame'][0]
        ic = joint['ic_frame'][0]
        fc = joint['fc_frame'][0]
        br = joint['br_frame'][0]

        f = force.copy()

        j = joint[jc_cols]
        for c in j.columns:
            j[c] = lowpass_filter(j[c], 360, joint_cut_off, 4)
        
        j_vel = dataframe_differentiate_vel(j, 360)
        j_acc = dataframe_differentiate_acc(j_vel, 360)
        j = pd.concat([j, j_vel, j_acc], axis=1)
        
        j = j[new_jc_cols]
        
        for c in f.columns:
            f[c] = lowpass_filter(f[c], 1080, force_cut_off, 4)
        
        res_force = 100 * (f[::3].reset_index(drop=True)/9.81) / weight

        force_use_col_rear = ['REAR_FORCE_X','REAR_FORCE_Y','REAR_FORCE_Z']
        force_use_col_lead = ['LEAD_FORCE_X','LEAD_FORCE_Y','LEAD_FORCE_Z']

        if session.split('_')[0] in train:
            targets['train']['rear'][session] = res_force.iloc[kh:fc+1, :][force_use_col_rear]
            targets['train']['lead'][session] = res_force.iloc[ic:br+1, :][force_use_col_lead]
        
        elif session.split('_')[0] in test:
            targets['test']['rear'][session] = res_force.iloc[kh:fc+1, :][force_use_col_rear]
            targets['test']['lead'][session] = res_force.iloc[fc:br+1, :][force_use_col_lead]


        sc = PandasStandardScaler()
        sc.fit(j.loc[int(kh-360*0.05):int(br+360*0.05),:])
        j_sc = sc.transform(j.loc[int(kh-360*0.05):int(br+360*0.05),:])
        
        if session.split('_')[0] in train:

            inputs_sc['train']['rear'][session] = j_sc.loc[int(kh-360*0.05):int(fc+360*0.05), :]
            inputs_sc['train']['lead'][session] = j_sc.loc[int(ic-360*0.05):int(br+360*0.05), :]
            
    
        elif session.split('_')[0] in test:

            inputs_sc['test']['rear'][session] = j_sc.loc[int(kh-360*0.05):int(fc+360*0.05), :]
            inputs_sc['test']['lead'][session] = j_sc.loc[int(fc-360*0.05):int(br+360*0.05), :]

    
    return inputs_sc, targets

def input_target_grouping(inputs_sc, targets):
    input_target_sc_np = {
        'train' : {
            'rear' : {
                'input' : {},
                'target' : {}
                },
            'lead' : {
                'input' : {},
                'target' : {}
                }
            },
        
        'test' : {
            'rear' : {
                'input' : {},
                'target' : {}
                },
            'lead' : {
                'input' : {},
                'target' : {}
                },
        },
    }

    for session in inputs_sc['train']['lead']:
        col_x = []
        col_z = []
        col_y = []
        for c in inputs_sc['train']['lead'][session].columns:
            if '_X' in c:
                col_x.append(c)
            elif '_Z' in c:
                col_z.append(c)
            elif '_Y' in c:
                col_y.append(c)

  
        lead_sc_arr = []
        lead_sc_arr.append(inputs_sc['train']['lead'][session][col_x].values)
        lead_sc_arr.append(inputs_sc['train']['lead'][session][col_y].values)
        lead_sc_arr.append(inputs_sc['train']['lead'][session][col_z].values)

        rear_sc_arr = []
        rear_sc_arr.append(inputs_sc['train']['rear'][session][col_x].values)
        rear_sc_arr.append(inputs_sc['train']['rear'][session][col_y].values)
        rear_sc_arr.append(inputs_sc['train']['rear'][session][col_z].values)

            
        ## TRAIN (LEAD FP)
        input_sc_lead = np.array(lead_sc_arr)

        target_lead = targets['train']['lead'][session]
        target_N_lead = len(target_lead)
        
        input_sc_np_lead = np.zeros((target_N_lead, 3, 36, 36))
        for i in range(target_N_lead):
            input_sc_np_lead[i,:,:,:] = input_sc_lead[:, i:i+36, :]
    
        input_target_sc_np['train']['lead']['input'][session] = input_sc_np_lead
        input_target_sc_np['train']['lead']['target'][session] = np.array(target_lead)
        
        ## TRAIN (REAR FP)
        input_sc_rear = np.array(rear_sc_arr)

        target_rear = targets['train']['rear'][session]
        target_N_rear = len(target_rear)
        
        input_sc_np_rear = np.zeros((target_N_rear, 3, 36, 36))
        for i in range(target_N_rear):
            input_sc_np_rear[i,:,:,:] = input_sc_rear[:, i:i+36, :]

        input_target_sc_np['train']['rear']['input'][session] = input_sc_np_rear
        input_target_sc_np['train']['rear']['target'][session] = np.array(target_rear)

        
    for session in inputs_sc['test']['lead']:


        lead_sc_arr = []
        rear_sc_arr = []
        lead_sc_arr.append(inputs_sc['test']['lead'][session][col_x].values)
        lead_sc_arr.append(inputs_sc['test']['lead'][session][col_y].values)
        lead_sc_arr.append(inputs_sc['test']['lead'][session][col_z].values)
        rear_sc_arr.append(inputs_sc['test']['rear'][session][col_x].values)
        rear_sc_arr.append(inputs_sc['test']['rear'][session][col_y].values)
        rear_sc_arr.append(inputs_sc['test']['rear'][session][col_z].values)

        ## TEST (LEAD FP)
        input_sc_lead = np.array(lead_sc_arr)
        target_lead = targets['test']['lead'][session]
        target_N_lead = len(target_lead)
        
        input_sc_np_lead = np.zeros((target_N_lead, 3, 36, 36))
        for i in range(target_N_lead):
            input_sc_np_lead[i,:,:,:] = input_sc_lead[:, i:i+36, :]
        input_target_sc_np['test']['lead']['input'][session] = input_sc_np_lead
        input_target_sc_np['test']['lead']['target'][session] = np.array(target_lead)

        
        ## TEST (REAR FP)
        input_sc_rear = np.array(rear_sc_arr)
        target_rear = targets['test']['rear'][session]
        target_N_rear = len(target_rear)
        
        input_sc_np_rear = np.zeros((target_N_rear, 3, 36, 36))
        for i in range(target_N_rear):
            input_sc_np_rear[i,:,:,:] = input_sc_rear[:, i:i+36, :]

        input_target_sc_np['test']['rear']['input'][session] = input_sc_np_rear
        input_target_sc_np['test']['rear']['target'][session] = np.array(target_rear)

    return input_target_sc_np