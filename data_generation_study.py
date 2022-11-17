import os
import librosa
import numpy as np

def wav_split(wav, frame_length,strid):
    wav_slices = []
    wav_length = len(wav)
    if wav_length > frame_length:
        for index_end in range(frame_length, wav_length, strid):
            start = index_end - frame_length
            wav_slice = wav[start:index_end]
            wav_slices.append(wav_slice)

    wav_slices.append(wav[-frame_length:])
    return wav_slices

def save_slices(slices, name):
    name_list = []

    if len(slices) > 0:
        for index, wav_slice in enumerate(slices):
            wav_slice_name = name + "_" + str(index) + ".npy"
            np.save(wav_slice_name, wav_slice)
            name_list = np.append(name_list, wav_slice_name)

    return name_list

if __name__ == "__main__":
    clean_wav_train_path = "./dataset/clean_trainset_wav"
    noisy_wav_train_path = "./dataset/noisy_trainset_wav"
    clean_wav_test_path = "./dataset/clean_testset_wav"
    noisy_wav_test_path = "./dataset/noisy_testset_wav"

    catch_clean_wav_train_path = "./dataset_catch/train/clean"
    catch_noisy_wav_train_path = "./dataset_catch/train/noisy"
    catch_clean_wav_test_path = "./dataset_catch/test/clean"
    catch_noisy_wav_test_path = "./dataset_catch/test/noisy"

    scp_train_path = "./scp/train_segan.scp"
    scp_test_path = "./scp/test_segan.scp"

    """
    os.makedirs(catch_clean_wav_train_path, exist_ok=True)
    os.makedirs(catch_noisy_wav_train_path, exist_ok=True)
    os.makedirs(catch_clean_wav_test_path, exist_ok=True)
    os.makedirs(catch_noisy_wav_test_path, exist_ok=True)
    """

    frame_length = 16384
    strid = int(frame_length/2)
    #train_set
    with open("scp/train_segan.scp", 'wt') as f:
        for root, dirs, files in os.walk(clean_wav_train_path):
            for file in files:
                file_clean_name_path = os.path.join(root, file)
                file_clean_name = os.path.split(file_clean_name_path)[-1]
                if file_clean_name.endswith("wav"):

                    file_noisy_name_path = os.path.join(noisy_wav_train_path, file_clean_name)
                    print("正在加载文件:{}".format(file_noisy_name_path))

                    if not os.path.exists(file_noisy_name_path):
                        print("{}不存在".format(file_noisy_name_path))
                        continue

                    clean_data, _ = librosa.load(file_clean_name_path, sr=16000)
                    noisy_data, _ = librosa.load(file_noisy_name_path, sr=16000)

                    if not len(clean_data) == len(noisy_data):
                        print("文件长度不匹配")
                        continue

                    clean_slices = wav_split(clean_data, frame_length, strid)
                    noisy_slices = wav_split(noisy_data, frame_length, strid)

                    clean_nameList = save_slices(clean_slices, os.path.join(catch_clean_wav_train_path, file_clean_name))
                    noisy_nameList = save_slices(noisy_slices, os.path.join(catch_noisy_wav_train_path, file_clean_name))

                    for clean_catch_name, noisy_catch_name in zip(clean_nameList, noisy_nameList):
                        f.write("{} {}\n".format(clean_catch_name, noisy_catch_name))
    #testset
    with open("scp/test_segan.scp", 'wt') as f:
        for root, dirs, files in os.walk(clean_wav_test_path):
            for file in files:
                file_clean_name_path = os.path.join(root, file)
                file_clean_name = os.path.split(file_clean_name_path)[-1]
                if file_clean_name.endswith("wav"):

                    file_noisy_name_path = os.path.join(noisy_wav_test_path, file_clean_name)
                    print("正在加载文件:{}".format(file_noisy_name_path))

                    if not os.path.exists(file_noisy_name_path):
                        print("{}不存在".format(file_noisy_name_path))
                        continue

                    clean_data, _ = librosa.load(file_clean_name_path, sr=16000)
                    noisy_data, _ = librosa.load(file_noisy_name_path, sr=16000)

                    if not len(clean_data) == len(noisy_data):
                        print("文件长度不匹配")
                        continue

                    clean_slices = wav_split(clean_data, frame_length, strid)
                    noisy_slices = wav_split(noisy_data, frame_length, strid)

                    clean_nameList = save_slices(clean_slices,
                                                 os.path.join(catch_clean_wav_test_path, file_clean_name))
                    noisy_nameList = save_slices(noisy_slices,
                                                 os.path.join(catch_noisy_wav_test_path, file_clean_name))

                    for clean_catch_name, noisy_catch_name in zip(clean_nameList, noisy_nameList):
                        f.write("{} {}\n".format(clean_catch_name, noisy_catch_name))