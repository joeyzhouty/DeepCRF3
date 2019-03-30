import json
import codecs
import random
import math


def boolean_string(bool_str):
    bool_str = bool_str.lower()
    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")
    return bool_str == "true"


class Dataset:
    def __init__(self, train_file, dev_file, test_file, batch_size=20, fold=None, shuffle=True):
        # load dataset
        if fold is None:
            self.train_dataset = self._load_dataset(train_file)
            self.dev_dataset = self._load_dataset(dev_file) if dev_file is not None else None
            self.test_dataset = self._load_dataset(test_file) if test_file is not None else None
        else:
            train_dataset = self._load_dataset(train_file)
            self.train_dataset = []
            for idx in range(fold[0], fold[1]):
                self.train_dataset += train_dataset["fold_{}".format(idx)]
            self.dev_dataset = self._load_dataset(dev_file) if dev_file is not None else None
            self.test_dataset = self._load_dataset(test_file) if test_file is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_batches = None
        self.dev_batches = None
        self.test_batches = None

    def get_dataset(self, name="train"):
        if name.lower() == "train":
            return self.train_dataset
        elif name.lower() == "dev":
            return self.dev_dataset
        elif name.lower() == "test":
            return self.test_dataset
        else:
            raise ValueError("Unknown dataset {}".format(name))

    def get_data_batches(self, name="train"):  # return the whole dataset batches
        if name.lower() == "train":
            if self.train_batches is None or len(self.train_batches) < self.get_num_batches():
                self.train_batches = self._sample_batches(self.train_dataset, shuffle=self.shuffle)
            return self.train_batches
        elif name.lower() == "dev":
            if self.dev_batches is None:
                self.dev_batches = self._sample_batches(self.dev_dataset) if self.dev_dataset is not None else None
            return self.dev_batches
        elif name.lower() == "test":
            if self.test_batches is None:
                self.test_batches = self._sample_batches(self.test_dataset) if self.test_dataset is not None else None
            return self.test_batches
        else:
            raise ValueError("Unknown dataset {}".format(name))

    def get_dataset_size(self, name="train"):
        if name.lower() == "train":
            return len(self.train_dataset)
        elif name.lower() == "dev":
            return len(self.dev_dataset) if self.dev_dataset is not None else 0
        elif name.lower() == "test":
            return len(self.test_dataset) if self.test_dataset is not None else 0
        else:
            raise ValueError("Unknown dataset {}".format(name))

    def get_num_batches(self, name="train"):
        return math.ceil(float(self.get_dataset_size(name)) / float(self.batch_size))

    def get_batch_size(self):
        return self.batch_size

    def get_next_train_batch(self):
        if self.train_batches is None or len(self.train_batches) == 0:
            self.train_batches = self._sample_batches(self.train_dataset, shuffle=self.shuffle)
        batch = self.train_batches.pop()
        return batch

    def _sample_batches(self, dataset, shuffle=False):
        if shuffle:
            random.shuffle(dataset)
        data_batches = []
        dataset_size = len(dataset)
        for i in range(0, dataset_size, self.batch_size):
            batch_data = dataset[i: i + self.batch_size]
            batch_data = self._process_batch(batch_data)
            data_batches.append(batch_data)
        return data_batches

    def _process_batch(self, batch_data):
        batch_words, batch_chars, batch_labels = [], [], []
        for data in batch_data:
            batch_words.append(data["words"])
            batch_chars.append(data["chars"])
            if "labels" in data:
                batch_labels.append(data["labels"])
        b_words, b_words_len = self.pad_sequences(batch_words)
        b_chars, b_chars_len = self.pad_char_sequences(batch_chars)
        if len(batch_labels) == 0:
            return {"words": b_words, "chars": b_chars, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                    "batch_size": len(b_words)}
        else:
            b_labels, _ = self.pad_sequences(batch_labels)
            return {"words": b_words, "chars": b_chars, "labels": b_labels, "seq_len": b_words_len,
                    "char_seq_len": b_chars_len, "batch_size": len(b_words)}

    @staticmethod
    def _load_dataset(filename):
        with codecs.open(filename, mode='r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset

    @staticmethod
    def pad_sequences(sequences, pad_tok=None, max_length=None):
        if pad_tok is None:
            pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
        if max_length is None:
            max_length = max([len(seq) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded.append(seq_)
            sequence_length.append(min(len(seq), max_length))
        return sequence_padded, sequence_length

    def pad_char_sequences(self, sequences, max_length=None, max_length_2=None):
        sequence_padded, sequence_length = [], []
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))
        if max_length_2 is None:
            max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        for seq in sequences:
            sp, sl = self.pad_sequences(seq, max_length=max_length_2)
            sequence_padded.append(sp)
            sequence_length.append(sl)
        sequence_padded, _ = self.pad_sequences(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
        sequence_length, _ = self.pad_sequences(sequence_length, max_length=max_length)
        return sequence_padded, sequence_length


def align_data(data):
    """Given dict with lists, creates aligned strings
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                             data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()]) for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()
    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)
        data_aligned[key] = str_aligned
    return data_aligned
