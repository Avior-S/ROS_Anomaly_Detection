
from collections import namedtuple
import numpy as np
from Style import Configure as Conf
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
ROWS_INDEX = 0
COLOMUNS_INDEX = 1

DataPred = namedtuple("DataPred", "paths datasets predictions")
class Datasets:

    def __init__(self, positives_dir_path, negatives_dir_path, test_size=0.3, delete=0, nfolds=0,i=-1):

        # The files will be inside two directory, 1 for "normal" and 2 for the type of the normal run
        positives_paths = []
        trainings_paths = []
        for pos_dir_path in self.get_dirs_names(positives_dir_path):
            files = Datasets.get_files_name(pos_dir_path)
            if nfolds == 0:
                trainings_files, positives_files = Datasets.__test_split(files, test_size)
            else:
                fold_size = len(files) // nfolds
                trainings_files = files[:i * fold_size] + files[(i + 1) * fold_size:]
                positives_files = files[i * fold_size:(i + 1) * fold_size]
            positives_paths.extend(positives_files)
            trainings_paths.extend(trainings_files)

        if not positives_paths:
            print("positives_paths: empty")

        if not trainings_paths:
            print("trainings_paths: empty")

        # The files will be inside two directory, 1 for "abnormal" and 2 for the type of the abnormal run
        negatives_paths = []
        for neg_dir_path in  self.get_dirs_names(negatives_dir_path):
            files = Datasets.get_files_name(neg_dir_path)
            negatives_paths.extend(files)

        if not negatives_paths:
            print("negatives_paths: empty")

        trainings, positives, negatives = Datasets.read_csv_from_paths(trainings_paths, positives_paths, negatives_paths)
        trainings, positives, negatives = Datasets.remove_rows_from(delete, trainings, positives, negatives)

        self.trainings_preds = DataPred(trainings_paths, trainings, self.__create_predictions(trainings))
        self.positives_preds = DataPred(positives_paths, positives, self.__create_predictions(positives))
        self.negatives_preds = DataPred(negatives_paths, negatives, self.__create_predictions(negatives))

    @staticmethod
    def get_dirs_names(path):
        import glob
        files = glob.glob(path + "*")
        return [f + "/" for f in files]

    @staticmethod
    def remove_rows_from(delete, *datasets_types):
        def remove_from(df):
            return df[delete:]# -delete]
        if delete == 0:
            return datasets_types
        ret = []
        for dfs in datasets_types:
            ret.append(list(map(lambda df: remove_from(df), dfs)))
        return ret

    def get_predictions(self):
        trainings_preds = self.trainings_preds.predictions
        positives_preds = self.positives_preds.predictions
        negatives_preds = self.negatives_preds.predictions
        return trainings_preds, positives_preds, negatives_preds

    def get_names(self):
        trainings_names = list(map(lambda p: Datasets.__get_filename_from_path(p), self.trainings_preds.paths))
        positives_names = list(map(lambda p: Datasets.__get_filename_from_path(p), self.positives_preds.paths))
        negatives_names = list(map(lambda p: Datasets.__get_filename_from_path(p), self.negatives_preds.paths))
        return trainings_names, positives_names, negatives_names

    def update_predictions(self, trainings_preds, positives_preds, negatives_preds):
        self.trainings_preds = DataPred(self.trainings_preds.paths, self.trainings_preds.datasets, list(map(lambda x,y: self.intersection_labeling(x,y), trainings_preds, self.trainings_preds.predictions)))
        self.positives_preds = DataPred(self.positives_preds.paths, self.positives_preds.datasets, list(map(lambda x,y: self.intersection_labeling(x,y), positives_preds, self.positives_preds.predictions)))
        if negatives_preds:
            self.negatives_preds = DataPred(self.negatives_preds.paths, self.negatives_preds.datasets, list(map(lambda x,y: self.intersection_labeling(x,y), negatives_preds, self.negatives_preds.predictions)))

    def set_predictions(self, trainings_preds, positives_preds, negatives_preds):
        self.trainings_preds = DataPred(self.trainings_preds.paths, self.trainings_preds.datasets, trainings_preds)
        self.positives_preds = DataPred(self.positives_preds.paths, self.positives_preds.datasets, positives_preds)
        if negatives_preds:
            self.negatives_preds = DataPred(self.negatives_preds.paths, self.negatives_preds.datasets, negatives_preds)

    def get_copied_datasets(self):
        trainings = [df.copy() for df in self.trainings_preds.datasets]
        positives = [df.copy() for df in self.positives_preds.datasets]
        negatives = [df.copy() for df in self.negatives_preds.datasets]
        return trainings, positives, negatives

    def get_datasets(self):
        trainings = self.trainings_preds.datasets
        positives = self.positives_preds.datasets
        negatives = self.negatives_preds.datasets
        return trainings, positives, negatives

    def get_mic_topics(self):
        topics = []
        if "panda" in self.trainings_preds.paths[0]:
            topics = []
            for epv in ['effort', 'position', 'velocity']:
                for i in range(1, 8):
                    topics.append('panda_joint' + str(i) + ' ' + epv)
                for i in range(1, 3):
                    topics.append('panda_finger_joint' + str(i) + ' ' + epv)
        else:
            topics = ['linear velocity x', 'linear velocity y', 'angular velocity z']
        return topics

    def get_mic_normalize_datasets(self):
        mic_topics = self.get_mic_topics()
        mic_dfs = self.get_copied_datasets()
        flt_mic_dfs = self.filter_by_columns(mic_topics)
        return self.normalization(flt_mic_dfs)

    def get_topics_statistics_normalize_datasets(self):
        similar_columns = self.find_similar_columns_in_training()
        self.filter_by_columns(similar_columns)
        mac_dfs = self.get_copied_datasets()
        d_mac_dfs = self.topic_drop(mac_dfs, self.get_mic_topics(), 'Active')
        return self.normalization()

    def normalization(self):
        trainings, positives, negatives = self.get_datasets()
        t_df = pd.concat(trainings, ignore_index=True)
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(t_df)
        n_trainings = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), trainings))
        n_positives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), positives))
        n_negatives = list(map(lambda df: pd.DataFrame(min_max_scaler.transform(df), columns=t_df.columns), negatives))
        self.set_datasets(n_trainings, n_positives, n_negatives)
        return n_trainings, n_positives, n_negatives

    def find_similar_columns_in_training(self):
        trainings, positives, negatives = self.get_datasets()
        all_trn = pd.concat(trainings, join='inner')
        all_pos = pd.concat(positives, join='inner')
        all_neg = pd.concat(negatives, join='inner')
        all = pd.concat([all_trn, all_pos, all_neg], join='inner')
        return all.columns

    def find_columns_contains(self, str):
        all_columns = self.find_similar_columns_in_training()
        return [i for i in all_columns if str in i]

    def filter_by_columns(self, columns):
        trainings = self.trainings_preds.datasets
        positives = self.positives_preds.datasets
        negatives = self.negatives_preds.datasets
        flt_training = list(map(lambda df: df[columns], trainings))
        flt_positive = list(map(lambda df: df[columns], positives))
        flt_negative = list(map(lambda df: df[columns], negatives))
        self.set_datasets(flt_training, flt_positive, flt_negative)

    def set_datasets(self,training_datasets, positive_datasets, negative_datasets):
        self.trainings_preds = DataPred(self.trainings_preds.paths, training_datasets, self.trainings_preds.predictions)
        self.positives_preds = DataPred(self.positives_preds.paths, positive_datasets, self.positives_preds.predictions)
        if negative_datasets:
            self.negatives_preds = DataPred(self.negatives_preds.paths, negative_datasets, self.negatives_preds.predictions)

    def __create_predictions(self, datasets):
        ret = []
        for df in datasets:
            df_len = df.shape[ROWS_INDEX]
            preds = [Conf.POSITIVE_LABEL] * df_len
            ret.append(preds)
        return ret

    def intersection_labeling(self, labeling, *labelings):
        new_labeling = []
        labelings = list(labelings)
        labelings.append(labeling)
        for i in range(len(labeling)):
            lbl = True
            for labeling in labelings:
                lbl = lbl and (labeling[i] == Conf.POSITIVE_LABEL)
            new_labeling.append(Conf.POSITIVE_LABEL if lbl else Conf.NEGATIVE_LABEL)
        # print new_labeling
        return new_labeling

    @staticmethod
    def get_files_name(path):
        import glob
        csv_suffix = ".csv"
        if path[-len(csv_suffix):] == csv_suffix:
            files = [path]
        else:
            files = glob.glob(path + ("*" + csv_suffix))
        return files

    @staticmethod
    def read_csv_from_paths(*csvs_paths):
        import pandas as pd
        ret = []
        for csvs_path in csvs_paths:
            # dfs = [pd.read_csv(file, header=0) for file in csvs_path]
            dfs = list(map(lambda x: pd.read_csv(x, header=0), csvs_path))
            ret.append(dfs)
        return ret

    @staticmethod
    def __test_split(paths, _test_size):
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(paths, test_size=_test_size)
        return train, test

    @staticmethod
    def __get_filename_from_path(path):
        return path.split('/')[-1]

    def convert_f_concat_t_datasets(self, Xdf, Ydf):
        count = 0
        trainings, positives, negatives = self.get_datasets()
        new_trainings_dataset = []
        new_trainings_predicts = []
        for train in trainings:
            dataset = Xdf.iloc[count:train.shape[0] + count, :]
            predicts = Ydf[count:train.shape[0] + count]
            new_trainings_dataset.append(dataset)
            new_trainings_predicts.append(predicts)
            count += train.shape[0]
        new_positives_dataset = []
        new_positives_predicts = []
        for pos in positives:
            dataset = Xdf.iloc[count:pos.shape[0] + count, :]
            predicts = Ydf[count:pos.shape[0] + count]
            new_positives_dataset.append(dataset)
            new_positives_predicts.append(predicts)
            count += pos.shape[0]
        new_negatives_dataset = []
        new_neggatives_predicts = []
        for neg in negatives:
            dataset = Xdf.iloc[count:neg.shape[0] + count, :]
            predicts = Ydf[count:neg.shape[0] + count]
            new_negatives_dataset.append(dataset)
            new_neggatives_predicts.append(predicts)
            count += neg.shape[0]
        self.update_predictions(new_trainings_predicts, new_positives_predicts, new_neggatives_predicts)
        self.set_datasets(new_trainings_dataset, new_positives_dataset, new_negatives_dataset)
        return self

    def __str__(self):
        def get_dir_path(path):
            return '/'.join(path.split('/')[:-1])
        def count_values(val, *preds):
            count = 0
            for pred in preds:
                count += (np.array(pred) == val).sum()
            return count
        def create_info(title, paths, results):
            if len(paths) < 1:
                return ""
            directory = get_dir_path(paths[0])
            information = "%s directory: %s\n" % (title, directory)
            # information += "%s files: %s\n" % (title, str(map(Datasets.__get_filename_from_path, paths)))
            pos_count = count_values(Conf.POSITIVE_LABEL, *results)
            neg_count = count_values(Conf.NEGATIVE_LABEL, *results)
            information += "positive predictions: %s/%s, negative predictions: %s/%s\n" % (pos_count, pos_count+neg_count, neg_count, pos_count+neg_count)
            return information
        trainings_preds, positives_preds, negatives_preds = self.get_predictions()
        trainings_location = create_info("trainings", self.trainings_preds.paths, trainings_preds)
        positives_location = create_info("positives", self.positives_preds.paths, positives_preds)
        negatives_location = create_info("negatives", self.negatives_preds.paths, negatives_preds)

        return "%s\n%s\n%s" % (trainings_location, positives_location, negatives_location)

    @staticmethod
    def remove_columns(dataset, columns):
        return dataset.drop(columns, COLOMUNS_INDEX)


def drop_topics(datasets, topics=None, st='Active'):
    if topics is None:
        topics = []
    trainings, positives, negatives = datasets
    drp_trainings = list(map(lambda df: df.drop(topics, axis=1), trainings))
    drp_positives = list(map(lambda df: df.drop(topics, axis=1), positives))
    drp_negatives = list(map(lambda df: df.drop(topics, axis=1), negatives))
    if st:
        drp_trainings = list(map(lambda df: df.drop(list(df.filter(regex=st)), axis=1), drp_trainings))
        drp_positives = list(map(lambda df: df.drop(list(df.filter(regex=st)), axis=1), drp_positives))
        drp_negatives = list(map(lambda df: df.drop(list(df.filter(regex=st)), axis=1), drp_negatives))
    return drp_trainings, drp_positives, drp_negatives

# d = Datasets("data/real_panda/normal/","data/real_panda/test/")
# print(d)